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
"""timecast/series/crypto.py"""
import os
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd


def generate(path=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description: Outputs the daily price of bitcoin from 2013-04-28 to 2018-02-10
    """

    data = pd.read_csv(
        path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/crypto.csv")
    )

    data = jnp.asarray(data[data.Currency == "bitcoin"].Price)

    return data[:-1].reshape(-1, 1), data[1:].reshape(-1, 1)
