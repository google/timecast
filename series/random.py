# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""timecast/series/random.py"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np


def generate(n: int = 1000, loc: float = 0.0, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description: outputs a timeline randomly distributed i.i.d. from gaussian
    with mean `loc`, standard deviation `scale`
    """
    X = np.random.normal(loc=loc, scale=scale, size=(n + 1))

    return jnp.asarray(X)[:-1].reshape(-1, 1), jnp.asarray(X)[1:].reshape(-1, 1)
