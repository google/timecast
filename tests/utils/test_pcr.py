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
"""timecast/tests/utils/test_pcr.py"""
import jax
import numpy as np
import pytest
from sklearn.decomposition import PCA

from timecast.utils import random
from timecast.utils.pcr import compute_projection


@pytest.mark.parametrize("shape", [(1, 1), (10, 1), (2, 10), (10, 2)])
def test_compute_projection(shape):
    """Test PCA projection of X vs X.T @ X"""
    X = jax.random.uniform(random.generate_key(), shape=shape)
    XTX = X.T @ X

    k = 1 if X.ndim == 1 else min(X.shape)
    p1 = compute_projection(X, k)
    p2 = compute_projection(XTX, k)

    np.testing.assert_array_almost_equal(abs(p1), abs(p2), decimal=3)


@pytest.mark.parametrize("shape", [(1, 1), (10, 1), (1, 10), (10, 10)])
def test_compute_projection_sklearn(shape):
    """Test PCA projection of X vs sklearn"""
    X = jax.random.uniform(random.generate_key(), shape=shape)

    projection = compute_projection(X, 1, center=True)

    pca = PCA(n_components=1)
    pca.fit(X)

    np.testing.assert_array_almost_equal(abs(projection), abs(pca.components_.T), decimal=3)
