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
"""timecast/optim/_sgd.py"""
import jax
import jax.numpy as jnp


class SGD:
    """Stochastic gradient descent"""

    def __init__(
        self, loss_fn=lambda pred, true: jnp.square(pred - true).mean(), learning_rate=0.0001,
    ):
        """init"""
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def __call__(self, module, x, y):
        """call"""
        grad = jax.jit(jax.grad(lambda module, x, y: self.loss_fn(module(x), y)))(module, x, y)
        module.set_param_tree(grad, lambda old, new: old - self.learning_rate * new)

        return module
