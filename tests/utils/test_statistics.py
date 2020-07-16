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
"""timecast/tests/utils/test_statistics.py"""
import jax
import numpy as np
import pytest

from timecast.utils import random
from timecast.utils.statistics import OnlineStatistics


@pytest.mark.parametrize("n", [2, 10, 50, 100])
@pytest.mark.parametrize("j", [1, 10, 100])
@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("func", ["sum", "mean", "std", "var", "observations", "zscore"])
def test_online_sum(n, j, k, func):
    """Test online statistics"""
    stats = OnlineStatistics(dim=k)
    X = jax.random.uniform(random.generate_key(), shape=(n, j * k))

    for i in X:
        stats.update(i.reshape(j, k))

    if func == "zscore":
        np.testing.assert_array_almost_equal(
            stats.zscore(X[0, :].reshape(j, k)), (X[0, :].reshape(j, k) - stats.mean) / stats.std,
        )
    elif func != "observations":
        result = getattr(stats, func)
        np.testing.assert_array_almost_equal(
            result, getattr(X.reshape(n * j, k), func)(axis=0).reshape(1, -1), decimal=2
        )
    else:
        assert n * j == stats.observations
