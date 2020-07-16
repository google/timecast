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
"""timecast/utils/ar.py"""
from typing import Any
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from timecast.utils import internalize
from timecast.utils.gram import OnlineGram


def historify(
    X: np.ndarray, history_len: int,
):
    """Generate (num_histories, history_len, input_dim) history from time series data

    Todo:
        * Implement striding

    Warning:
        * This converts back and forth between jnp and np, which is fine for
        CPU, but may cause issues if we need to move off CPU

    Args:
        X (np.ndarray): first axis is time, remainder are feature dimensions
        history_len: length of a history window
        number of histories possible

    Returns:
        np.ndarray: nD array organized as (num_histories, history_len) + feature_shape
    """
    num_histories = X.shape[0] - history_len + 1

    if num_histories < 1 or history_len < 1:
        raise ValueError("Must have positive history_len and at least one window")
    if X.shape[0] < num_histories + history_len - 1:
        raise ValueError(
            "Not enough history ({}) to produce {} windows of length {}".format(
                X.shape[0], num_histories, history_len
            )
        )

    if num_histories == 1:
        return X[None, :]

    X = np.asarray(X)
    shape = (num_histories, history_len) + X.shape[1:]
    strides = (X.strides[0],) + X.strides
    return jnp.asarray(np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides))


def _form_constraints(
    input_dim: int, output_dim: int, history_len: int, fit_intercept: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Description: Sets the linear constraint matrix and vector R and r,
    respectively for constraining the `_params` vector. Specifically, each row
    in R is a linearly independent constraint where we force each dimension in
    the time series value to share a coefficient across the window.

    References: https://www.le.ac.uk/users/dsgp1/COURSES/TOPICS/restrict.pdf

    Args:
        input_dim (int): number of dimensions in the time series value
        output_dim (int): number of dimensions in the output
        history_len (int): window size for the auto reggressor
        fit_intercept (bool, optional): whether or not to fit an intercept.
        Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: RB = r
    """
    if fit_intercept:
        history_len += 1

    num_constraints = history_len * (input_dim - 1)

    R = np.zeros((num_constraints, input_dim * history_len))
    r = np.zeros((num_constraints, output_dim))

    # To share a parameter p across features x1 and x2, create a new row in R
    # with index 1 set to 1, index 2 set to -1, the rest set to 0 and the
    # corresponding row in r set to 0 (i.e., x1 + x2 = 0)
    for i in range(history_len):
        for j in range(input_dim - 1):

            # Set the constraint row
            row = i * (input_dim - 1) + j

            # Column indices for the first and second feature to tie. Note that
            # within the inner for loop, we tie multiple features to the same
            # parameter
            col1 = i * input_dim + j
            col2 = i * input_dim + j + 1

            # Update R appropriately
            R[row, col1] = 1
            R[row, col2] = -1

    return R, r


def _fit_constrained(beta, inv, R, r):
    """Fit constrained"""
    return beta - inv @ (R.T @ np.linalg.inv(R @ (inv @ R.T))) @ (R @ beta - r)


def _compute_xtx_inverse(XTX, alpha):
    """Compute inverse of X.T @ X"""
    reg = alpha * jnp.eye(XTX.shape[0])
    reg = jax.ops.index_update(reg, [0, 0], 0)
    inv = jnp.linalg.inv(XTX + reg)
    return inv


def _fit_unconstrained(inv, XTY):
    """Fit unconstrained"""
    return inv @ XTY


def fit_gram(
    XTX: OnlineGram,
    XTY: OnlineGram,
    alpha: float = 1.0,
    normalize: bool = False,
    projection: np.ndarray = None,
    input_dim: int = None,
):
    """Compute linear regression parameters from gram matrix

    Notes:
        * Assumes over-determined systems
        * Assumes we always fit an intercept
    """

    fit_intercept = True
    feature_dim = XTX.feature_dim if projection is None else projection.shape[1]
    output_dim = XTY.output_dim

    if input_dim is None:
        history_len = None
    else:
        if feature_dim % input_dim != 0:
            raise ValueError("Original input dimension must evenly divide feature dimensions")
        history_len = feature_dim // input_dim

    if XTX.observations <= feature_dim:
        raise ValueError(
            "Underdetermined systems not currently supported (observations: {},"
            "features: {})".format(XTX.observations, feature_dim)
        )

    # Finalize gram matrices
    XTX = XTX.matrix(
        normalize=normalize,
        projection=projection,
        fit_intercept=fit_intercept,
        input_dim=input_dim,
    )
    XTY = XTY.matrix(
        normalize=normalize,
        projection=projection,
        fit_intercept=fit_intercept,
        input_dim=input_dim,
    )
    inv = _compute_xtx_inverse(XTX, alpha)
    beta = _fit_unconstrained(inv, XTY)

    # Ignore constrained regression if we project
    if input_dim is not None:
        R, r = _form_constraints(
            input_dim=input_dim,
            output_dim=output_dim,
            history_len=history_len,
            fit_intercept=fit_intercept,
        )
        beta = _fit_constrained(beta, inv, R, r)
        beta = beta.take(jnp.arange(0, len(beta), input_dim), axis=0)
        return beta[1:], beta[0]

    return beta[1:].reshape(1, feature_dim, -1), beta[0]


def compute_gram(
    data: Tuple[np.ndarray, np.ndarray, Any], input_dim: int, output_dim: int, history_len: int,
) -> Tuple[OnlineGram, OnlineGram]:
    """Compute X.T @ X and X.T @ Y on history windows incrementally"""
    num_features = input_dim * history_len
    XTX = OnlineGram(num_features)
    XTY = OnlineGram(num_features, output_dim)

    for X, Y, _ in data:
        X = internalize(X, input_dim)[0]
        Y = internalize(Y, output_dim)[0]

        if X.shape[0] != Y.shape[0]:
            raise ValueError("Input and output data must have the same number of observations")

        # Expand input time series X into histories, whic should result in a
        # (num_histories, history_len * input_dim)-shaped array
        history = historify(X, history_len=history_len)
        history = history.reshape(history.shape[0], -1)

        XTX.update(history)
        XTY.update(history, Y[history_len - 1 :])

    if XTX.observations == 0:
        raise IndexError("No data to fit")

    if XTX.observations <= num_features:
        raise ValueError(
            "Underdetermined systems not currently supported (observations: {},"
            "features: {})".format(XTX.observations, num_features)
        )

    return XTX, XTY
