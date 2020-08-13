# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""timecast/modules/core.py

Todos:
  - Tree flatten is very crude (only applies to params)
  - How to identify params (right now just ndarray)
  - Users can do bad things with naming
  - Think about jnp/np shim layer beyond tree_jnpify
  - Why are submodule params getting stuck in params dict after unflatten
  - Apply via tree traversal (E.g., bias only)
  - set_param_tree modifies self; ideally should return new object
"""
import inspect
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np


def is_param(x):
    """Definition of parameter"""
    return isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray)


def is_module(x):
    """Definition of module

    NOTE: this is a kludge to deal with different id(Module)
    """
    return "modules" in dir(x)


def tree_jnpify(param_tree):
    """Turn numpy arrays into jax arrays"""
    if is_param(param_tree):
        return jnp.asarray(param_tree)

    for key, val in param_tree.items():
        param_tree[key] = tree_jnpify(val)

    return param_tree


def tree_flatten(module):
    """Flatten module parameters for Jax"""
    leaves, aux = jax.tree_util.tree_flatten(tree_jnpify(module.get_param_tree()))
    aux = {
        "treedef": aux,
        "arguments": module.arguments,
        "attrs": module.attrs,
        "class": module.__class__,
    }
    return leaves, aux


def tree_unflatten(aux, leaves):
    """Unflatten module parameters for Jax"""
    module = aux["class"](*aux["arguments"].args, **aux["arguments"].kwargs)
    module.set_param_tree(jax.tree_util.tree_unflatten(aux["treedef"], leaves))
    for attr in aux["attrs"]:
        if attr in module.__dict__["params"]:
            module.__setattr__(attr, module.__dict__["params"][attr])
    return module


class Module:
    """Core module class"""

    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = object.__new__(cls)
        obj.__setattr__("attrs", set())
        obj.__setattr__("modules", {})
        obj.__setattr__("params", {})
        obj.__setattr__("arguments", inspect.signature(obj.__init__).bind(*args, **kwargs))
        obj.arguments.apply_defaults()

        return obj

    @property
    def name(self):
        """Name of Module"""
        return self.__class__.__name__

    @classmethod
    def __init_subclass__(cls, *args, **kwargs):
        """For avoiding a decorator for each subclass"""
        super().__init_subclass__(*args, **kwargs)
        jax.tree_util.register_pytree_node(cls, tree_flatten, tree_unflatten)

    def __setattr__(self, name, value):
        """Setting attributes

        Notes:
            * Any attribute of type Module is added to a modules dict
            * Any attribute of type jnp.ndarray is added to a params dict
        """
        self.__dict__[name] = value
        if name[0] != "_":
            self.attrs.add(name)

        if is_module(value):
            self.__dict__["modules"][name] = value
        elif is_param(value):
            self.__dict__["params"][name] = value

    def get_param_tree(self):
        """Return recursed parameter tree"""
        params = self.params
        for name, module in self.modules.items():
            params[name] = module.get_param_tree()
        return params

    def set_param_tree(self, tree=None, func=None, filter=None, path="/"):
        """Apply parameter tree"""
        tree = tree or self
        func = func or (lambda old, new: new)
        filter = filter or (lambda path: True)

        is_dict = isinstance(tree, dict)
        tree_params = tree if is_dict else tree.params
        tree_modules = tree if is_dict else tree.modules

        for name, param in self.params.items():
            if name not in self.modules and filter(os.path.join(path, name)):
                param = func(param, tree_params[name])
                self.params[name] = param
                self.__dict__[name] = param

        for name, module in self.modules.items():
            module.set_param_tree(
                tree=tree_modules[name], func=func, filter=filter, path=os.path.join(path, name)
            )

    def add_module(self, module, name=None):
        """Add module outside attributes"""
        counter = 0
        while name is None or name in self.__dict__["modules"]:
            name = "{}_{}".format(type(module).__name__, counter)
            counter += 1
        self.__dict__["modules"][name] = module

    def add_param(self, param, name):
        """Add parameter outside attributes"""
        counter = 0
        while name is None or name in self.__dict__["params"]:
            name = "{}_{}".format(name, counter)
            counter += 1
        self.__dict__["params"][name] = param

    def save(self, path):
        """Save module"""
        dirname = os.path.abspath(os.path.dirname(path))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path):
        """Load module"""
        return pickle.load(open(path, "rb"))
