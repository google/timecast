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
"""timecast/modules/_linear.py

Todos:
  - Implement batching
  - Deal with array vs scalar
"""
from typing import Iterable

import jax.numpy as jnp
import numpy as np

from timecast.modules.core import Module


class Linear(Module):
    """Predict linear combination"""

    def __init__(self, input_shape, output_shape):
        """init"""
        if not isinstance(input_shape, Iterable):
            input_shape = (input_shape,)
        if not isinstance(output_shape, Iterable):
            output_shape = (output_shape,)

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.kernel = np.zeros(input_shape + output_shape)
        self.bias = np.zeros(output_shape)

    def __call__(self, x):
        """call"""
        numpy = jnp if isinstance(x, jnp.ndarray) else np
        axes = tuple(range(x.ndim))
        return numpy.tensordot(self.kernel, x, (axes, axes)) + self.bias
