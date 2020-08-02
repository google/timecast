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
"""timecast/optim/_multiplicative_weights.py"""
import jax
import jax.numpy as jnp


class MultiplicativeWeights:
    """Multiplicative weights"""

    def __init__(self, eta=0.008):
        """init"""
        self.eta = eta
        self.grad = jax.jit(jax.grad(lambda W, preds, y: jnp.square(jnp.dot(W, preds) - y).sum()))

    def __call__(self, module, x, y):
        """call"""
        grad = self.grad(module.W, x, y)
        module.W = module.W * jnp.exp(-1 * self.eta * grad)
        module.W /= module.W.sum()
        return module
