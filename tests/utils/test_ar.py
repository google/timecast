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
"""timecast/tests/utils/test_ar.py"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from timecast.utils import random
from timecast.utils.ar import _compute_xtx_inverse
from timecast.utils.ar import _fit_constrained
from timecast.utils.ar import _fit_unconstrained
from timecast.utils.ar import _form_constraints
from timecast.utils.ar import compute_gram
from timecast.utils.ar import fit_gram
from timecast.utils.ar import historify
from timecast.utils.gram import OnlineGram


def _compute_kernel_bias(X: np.ndarray, Y: np.ndarray, fit_intercept=True, alpha: float = 0.0):
    """Compute linear regression parameters"""
    num_samples, num_features = X.shape

    if fit_intercept:
        if num_features >= num_samples:
            X -= X.mean(axis=0)
        X = jnp.hstack((jnp.ones((X.shape[0], 1)), X))

    reg = alpha * jnp.eye(X.shape[0 if num_features >= num_samples else 1])
    if fit_intercept:
        reg = jax.ops.index_update(reg, [0, 0], 0)

    if num_features >= num_samples:
        beta = X.T @ jnp.linalg.inv(X @ X.T + reg) @ Y
    else:
        beta = jnp.linalg.inv(X.T @ X + reg) @ X.T @ Y

    if fit_intercept:
        return beta[1:], beta[0]
    else:
        return beta, [0]


@pytest.mark.parametrize("m", [1, 10])
@pytest.mark.parametrize("n", [1, 10])
@pytest.mark.parametrize("history_len", [-1, 0, 1, 10])
def test_historify(m, n, history_len):
    """Test history-making"""
    X = jax.random.uniform(random.generate_key(), shape=(m, n))

    if history_len < 1 or X.shape[0] - history_len < 0:
        with pytest.raises(ValueError):
            historify(X, history_len)

    else:
        batched = historify(X, history_len)
        batched = batched.reshape(batched.shape[0], -1)

        for i, batch in enumerate(batched):
            np.testing.assert_array_almost_equal(X[i : i + history_len].ravel().squeeze(), batch)


@pytest.mark.parametrize("n", [100])
@pytest.mark.parametrize("input_dim", [1, 5])
@pytest.mark.parametrize("output_dim", [1, 4])
@pytest.mark.parametrize("history_len", [1, 3])
def test_compute_gram(n, input_dim, output_dim, history_len):
    """Test compouting gram matrices"""
    X = jax.random.uniform(random.generate_key(), shape=(n, input_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, output_dim))

    XTX, XTY = compute_gram([(X, Y, None)], input_dim, output_dim, history_len)

    history = historify(X, history_len)
    history = history.reshape(history.shape[0], -1)
    np.testing.assert_array_almost_equal(history.T @ history, XTX.matrix(), decimal=4)
    np.testing.assert_array_almost_equal(history.T @ Y[history_len - 1 :], XTY.matrix(), decimal=4)


def test_compute_gram_no_data():
    """Test no data"""
    with pytest.raises(ValueError):
        compute_gram([(jnp.zeros((0, 1)), jnp.zeros((0, 1)), None)], 1, 1, 1)

    with pytest.raises(IndexError):
        compute_gram([], 1, 1, 1)


def test_compute_gram_underdetermined():
    """Test underdetermined"""
    data = jnp.ones((13, 10))
    with pytest.raises(ValueError):
        compute_gram([(data, data, None)], 10, 10, 10)


def test_fit_gram_underdetermined():
    """Test underdetermined"""
    XTX = OnlineGram(1)
    XTY = XTX

    with pytest.raises(ValueError):
        fit_gram(XTX, XTY)


@pytest.mark.parametrize(
    "history_len,input_dim,output_dim,fit_intercept,expected_R,expected_r",
    [
        (
            3,
            2,
            1,
            False,
            np.array(
                [
                    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                ]
            ),
            np.zeros(3),
        ),
        (
            3,
            2,
            1,
            True,
            np.array(
                [
                    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                ]
            ),
            np.zeros(4),
        ),
        (4, 1, 1, True, np.zeros((0, 5)), np.zeros((0))),
        (
            3,
            2,
            2,
            True,
            np.array(
                [
                    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
                ]
            ),
            np.zeros((4, 2)),
        ),
    ],
)
def test_form_constraints(
    history_len, input_dim, output_dim, fit_intercept, expected_R, expected_r
):
    """Test forming constraints"""
    R, r = _form_constraints(input_dim, output_dim, history_len, fit_intercept)

    r = r.squeeze()

    assert np.array_equal(expected_R, R)
    assert np.array_equal(expected_r, r)


@pytest.mark.parametrize("n", [40, 1000])
@pytest.mark.parametrize("input_dim", [1, 5])
@pytest.mark.parametrize("output_dim", [1, 4])
@pytest.mark.parametrize("history_len", [1, 3])
def test_fit_unconstrained(n, input_dim, output_dim, history_len):
    """Fit unconstrained regression"""
    # NOTE: we use random data because we want to test dimensions and
    # correctness vs a second implementation
    X = jax.random.uniform(random.generate_key(), shape=(n, input_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, output_dim))

    XTX, XTY = compute_gram([(X, Y, None)], input_dim, output_dim, history_len)

    kernel, bias = fit_gram(XTX, XTY)
    n - history_len + 1
    history = historify(X, history_len)
    history = history.reshape(history.shape[0], -1)

    expected_kernel, expected_bias = _compute_kernel_bias(history, Y[history_len - 1 :], alpha=1.0)
    expected_kernel = expected_kernel.reshape(1, history_len * input_dim, -1)

    np.testing.assert_array_almost_equal(expected_kernel, kernel, decimal=3)
    np.testing.assert_array_almost_equal(expected_bias, bias, decimal=3)


@pytest.mark.parametrize("n", [1000])
@pytest.mark.parametrize("input_dim", [10, 12])
@pytest.mark.parametrize("output_dim", [1, 10])
@pytest.mark.parametrize("history_len", [2])
def test_fit_constrained(n, input_dim, output_dim, history_len):
    """Fit constrained regression"""
    # NOTE: we use random data because we want to test dimensions and
    # correctness vs a second implementation
    X = jax.random.uniform(random.generate_key(), shape=(n, input_dim))
    Y = jax.random.uniform(random.generate_key(), shape=(n, output_dim))

    XTX, XTY = compute_gram([(X, Y, None)], input_dim, output_dim, history_len)
    result = fit_gram(XTX, XTY, input_dim=input_dim)

    # Next, check that each chunk of input_dim features have the same coefficient
    # result = fit_gram(XTX, XTY, input_dim=input_dim)
    R, r = _form_constraints(
        input_dim=input_dim, output_dim=output_dim, history_len=history_len, fit_intercept=True,
    )

    XTX = XTX.matrix(fit_intercept=True, input_dim=input_dim)
    XTY = XTY.matrix(fit_intercept=True, input_dim=input_dim)
    inv = _compute_xtx_inverse(XTX, alpha=1.0)
    beta = _fit_unconstrained(inv, XTY)
    beta = _fit_constrained(beta, inv, R, r)
    beta = beta.reshape(history_len + 1, input_dim, -1)
    assert np.sum([np.abs(x - x[0]) for x in beta]) < 1e-4

    # Finally, check that resulting vector is of the correct length and the
    # values are self-consistent
    assert len(beta) == history_len + 1

    beta = beta[:, 0]
    beta = beta[1:], beta[0]

    # Check final results
    np.testing.assert_array_almost_equal(beta[0], result[0])
    np.testing.assert_array_almost_equal(beta[1], result[1])


def test_fit_constrained_bad_input_dim():
    """Bad input for constrained"""
    XTX = OnlineGram(10)
    XTY = OnlineGram(5)

    XTX.update(jax.random.uniform(random.generate_key(), shape=(100, 10)))
    XTY.update(jax.random.uniform(random.generate_key(), shape=(100, 5)))

    with pytest.raises(ValueError):
        fit_gram(XTX, XTY, input_dim=7)
