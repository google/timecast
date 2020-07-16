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
"""timecast/tests/utils/test_gram.py"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from timecast.utils import random
from timecast.utils.gram import OnlineGram


dims = [1, 10]


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("Y_dim", dims)
@pytest.mark.parametrize("n", [1, 10])
def test_gram_update(X_dim, Y_dim, n):
    """Test update"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, Y_dim))

    gram = OnlineGram(X_dim, Y_dim)
    gram.update(X, Y)

    np.testing.assert_array_almost_equal(gram.matrix(normalize=False, fit_intercept=False), X.T @ Y)


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("Y_dim", dims)
@pytest.mark.parametrize("n", [1, 10])
def test_gram_update_iterative(X_dim, Y_dim, n):
    """Test update iteratively"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, Y_dim))

    gram = OnlineGram(X_dim, Y_dim)

    for x, y in zip(X, Y):
        gram.update(x.reshape(1, -1), y.reshape(1, -1))

    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=False, fit_intercept=False), X.T @ Y, decimal=3
    )


def test_gram_update_value_error():
    """Test update value error"""
    gram = OnlineGram(1, 1)
    with pytest.raises(ValueError):
        gram.update(
            jax.random.uniform(random.generate_key(), shape=(4, 1)),
            jax.random.uniform(random.generate_key(), shape=(1, 1)),
        )

    with pytest.raises(ValueError):
        gram.update(jax.random.uniform(random.generate_key(), shape=(4, 1)))

    gram = OnlineGram(1)
    with pytest.raises(ValueError):
        gram.update(np.ones((1, 1)), np.ones((1, 1)))


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("Y_dim", dims)
@pytest.mark.parametrize("n", [2, 10])
def test_gram_normalize(X_dim, Y_dim, n):
    """Test normalize"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, Y_dim))

    gram = OnlineGram(X_dim, Y_dim)
    gram.update(X, Y)

    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_norm = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=True, fit_intercept=False), X_norm.T @ Y_norm, decimal=1
    )


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("normalize", [True, False])
def test_gram_intercept_xtx(X_dim, n, normalize):
    """Test intercept on X.T @ X"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))

    gram = OnlineGram(X_dim)
    gram.update(X)

    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    X = jnp.hstack((jnp.ones((n, 1)), X))

    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=normalize, fit_intercept=True), X.T @ X, decimal=1
    )


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("k", [1, 2, 5])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_gram_projection_xtx(X_dim, n, k, normalize, fit_intercept):
    """Test projection on X.T @ X"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    k = min(k, X_dim)
    projection = jax.random.uniform(random.generate_key(), shape=(X_dim, k))

    gram = OnlineGram(X_dim)
    gram.update(X)

    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)

    expected = projection.T @ X.T @ X @ projection
    if fit_intercept:
        expected = gram.fit_intercept(expected, normalize=normalize, projection=projection)

    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=normalize, projection=projection, fit_intercept=fit_intercept),
        expected,
        decimal=1,
    )


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("Y_dim", dims)
@pytest.mark.parametrize("n", [4, 10])
@pytest.mark.parametrize("normalize", [True, False])
def test_gram_intercept_xty(X_dim, Y_dim, n, normalize):
    """Test intercept on X.T @ Y"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, Y_dim))

    gram = OnlineGram(X_dim, Y_dim)

    gram.update(X, Y)

    np.testing.assert_array_almost_equal(gram.mean.squeeze(), X.mean(axis=0))
    np.testing.assert_array_almost_equal(gram.std.squeeze(), X.std(axis=0))
    assert gram.observations == X.shape[0]

    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    X = jnp.hstack((jnp.ones((n, 1)), X))

    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=normalize, fit_intercept=True), X.T @ Y, decimal=1
    )


def test_gram_intercept_constrained_projection():
    """Constrained projection should error"""
    X = jax.random.uniform(random.generate_key(), shape=(5, 10))
    XTX = OnlineGram(10)
    XTX.update(X)

    with pytest.raises(ValueError):
        XTX.fit_intercept(projection=1, input_dim=2)


def test_gram_projection_no_projection():
    """Test empty projection"""
    X = jax.random.uniform(random.generate_key(), shape=(5, 10))
    XTX = OnlineGram(10)
    XTX.update(X)
    np.testing.assert_array_almost_equal(X.T @ X, XTX.project())


@pytest.mark.parametrize("X_dim", dims)
@pytest.mark.parametrize("Y_dim", dims)
@pytest.mark.parametrize("n", [3, 10])
@pytest.mark.parametrize("k", [1, 5])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_gram_projection_xty(X_dim, Y_dim, n, k, normalize, fit_intercept):
    """Test projection on X.T @ Y"""
    X = jax.random.uniform(random.generate_key(), shape=(n, X_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, Y_dim))

    k = min(k, X_dim)
    projection = jax.random.uniform(random.generate_key(), shape=(X_dim, k))

    gram = OnlineGram(X_dim, Y_dim)
    gram.update(X, Y)

    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)

    expected = projection.T @ X.T @ Y
    if fit_intercept:
        expected = gram.fit_intercept(expected, normalize=normalize, projection=projection)

    np.testing.assert_array_almost_equal(
        gram.matrix(normalize=normalize, projection=projection, fit_intercept=fit_intercept),
        expected,
        decimal=1,
    )
