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
"""timecast/modules/core.py"""
# TODO
# - Optimizers should apply to all children unless children have specified version
# - hierarchical parameters?
# - Tree flatten is very crude (only applies to params)
# - How to identify params (right now just ndarray)
# - Pass class directly to jax
# - How to handle buffers vs parameters
# - Users can do bad things with naming
import inspect

import jax
import jax.numpy as jnp


def tree_flatten(module):
    """Flatten module parameters for Jax"""
    leaves, aux = jax.tree_util.tree_flatten(module.get_param_tree())
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
            module.__dict__[attr] = module.__dict__["params"][attr]
    return module


class Module:
    """Core module class"""

    def __new__(cls, *args, **kwargs):
        """For avoiding super().__init__()"""
        obj = object.__new__(cls)
        obj.__setattr__("attrs", set())
        obj.__setattr__("modules", {})
        obj.__setattr__("params", {})
        obj.__setattr__("arguments", inspect.signature(obj.__init__).bind(*args))
        obj.arguments.apply_defaults()

        return obj

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
        self.attrs.add(name)

        if isinstance(value, Module):
            self.__dict__["modules"][name] = value
        elif isinstance(value, jnp.ndarray):
            self.__dict__["params"][name] = value

    def get_param_tree(self):
        """Return recursed parameter tree"""
        params = self.params
        for name, module in self.modules.items():
            params[name] = module.get_param_tree()
        return params

    def set_param_tree(self, tree):
        """Apply parameter tree"""
        for param in self.params:
            self.params[param] = tree[param]
            self.__dict__[param] = tree[param]
        for name, module in self.modules.items():
            module.set_param_tree(tree[name])

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
