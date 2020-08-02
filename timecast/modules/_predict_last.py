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
"""timecast/modules/_predict_last.py"""
import jax.numpy as jnp
import numpy as np

from timecast.modules.core import Module


class PredictLast(Module):
    """Predict last value"""

    def __init__(self, steps=1):
        """init"""
        self.steps = steps
        self.history = np.zeros(steps)

    def __call__(self, x):
        """call"""
        if isinstance(x, jnp.ndarray):
            self.history = jnp.roll(self.history, shift=1)
            self.history = self.history.at[0].set(x)
        else:
            self.history = np.roll(self.history, shift=1)
            self.history[0] = x
        return self.history[self.steps - 1]
