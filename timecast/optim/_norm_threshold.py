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
"""timecast/optim/_norm_threshold.py"""
import jax
import jax.numpy as jnp


class NormThreshold:
    """Norm threshold"""

    def __init__(self, norm_threshold=None):
        """init"""
        self.norm_threshold = norm_threshold or None

    def __call__(self, module, params, x, y):
        """call"""
        new_params = {}
        for k in params():
            norm = jnp.linalg.norm(params[k])
            new_params[k] = jax.lax.cond(
                norm > self.norm_threshold[k],
                params[k],
                lambda x: (self.norm_threshold[k] / norm) * x,
                params[k],
                lambda x: x,
            )

        return params
