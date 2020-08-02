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
"""timecast/series/core.py"""
import jax.numpy as jnp
import pandas as pd


def generate_timeline(path: str, name=None, delimiter=","):
    """Convenience function to grab a single time series and convert to X, y pair"""
    data = pd.read_csv(path, delimiter=delimiter)

    if name is not None:
        data = data[[name]]

    return jnp.asarray(data)[:-1].reshape(-1, 1), jnp.asarray(data)[1:].reshape(-1, 1)
