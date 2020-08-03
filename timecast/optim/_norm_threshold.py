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
"""timecast/optim/_threshold.py"""
import jax
import jax.numpy as jnp


def clip_norm(old, new, threshold):
    """Helper function"""
    norm = jnp.linalg.norm(old)
    return jax.lax.cond(norm > threshold, old, lambda x: threshold / norm * x, old, lambda x: x)


class NormThreshold:
    """Norm threshold"""

    def __init__(self, threshold, filter=None):
        """init"""
        self.threshold = threshold
        self.filter = filter
        self.clip_norm = lambda old, new: clip_norm(old, new, threshold)

    def __call__(self, module, x=None, y=None):
        """call"""
        module.set_param_tree(func=self.clip_norm, filter=self.filter)

        return module
