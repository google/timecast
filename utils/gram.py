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
"""timecast/utils/gram.py"""
import jax
import jax.numpy as jnp

from timecast.utils.statistics import OnlineStatistics


class OnlineGram:
    """Compute gram matrix of observations online

    Notes:
        * Computes X.T @ X, where axis 0 is time and axis 1 is
        features
        * In the case where we supply a second matrix, this is no
        longer a gram matrix; X.T @ Y
    """

    def __init__(self, feature_dim: int = 1, output_dim: int = None,) -> None:
        """Initialize OnlineGram

        Args:
            feature_dim (int): feature dimension (typically X.shape[1])
            output_dim(int): same as feature_dim unless we supply a
            second matrix with different dimensions
        """
        self._feature_dim = feature_dim
        self._output_dim = output_dim or feature_dim

        self._matrix = jnp.zeros((self._feature_dim, self._output_dim))
        self._feature_stats = OnlineStatistics(dim=self._feature_dim)
        self._output_stats = None if output_dim is None else OnlineStatistics(dim=self._output_dim)

    def matrix(
        self,
        normalize: bool = False,
        projection: jnp.ndarray = None,
        fit_intercept: bool = False,
        input_dim: int = None,
    ):
        """Property to retrieve processed gram matrix

        Args:
            normalize (bool): normalize when finalized?
            projection (jnp.ndarray): projection matrix
            fit_intercept (bool): fit intercept when finalized?
            input_dim (int): for constrained linear regression TODO: needs
            thorough explanation
        """
        result = self._matrix
        if normalize:
            result = self.normalize(matrix=result)

        if projection is not None:
            result = self.project(matrix=result, projection=projection)

        if fit_intercept:
            result = self.fit_intercept(
                matrix=result, normalize=normalize, projection=projection, input_dim=input_dim,
            )
        return result

    @property
    def feature_dim(self):
        """Feature dim"""
        return self._feature_dim

    @property
    def output_dim(self):
        """Output dim"""
        return self._output_dim

    @property
    def observations(self):
        """Observations"""
        return self._feature_stats.observations

    @property
    def mean(self):
        """Mean"""
        return self._feature_stats.mean

    @property
    def std(self):
        """Std"""
        return self._feature_stats.std

    def update(self, X: jnp.ndarray, Y: jnp.ndarray = None) -> None:
        """Update Gram matrix bassed on new data"""
        if Y is None and self._output_stats is not None:
            raise ValueError(
                "Expected second matrix since output_dim of {} was initialized".format(
                    self._output_dim
                )
            )
        if Y is not None and self._output_stats is None:
            raise ValueError("Unexpected second matrix since output_dim is None")

        Y = X if Y is None else Y

        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                "Inputs must have the same number of observations; got {}, {}".format(
                    X.shape[0], Y.shape[0]
                )
            )
        self._matrix += X.T @ Y
        self._feature_stats.update(X)
        if self._output_stats is not None:
            self._output_stats.update(Y)

    def normalize(self, matrix: jnp.ndarray = None) -> jnp.ndarray:
        """Z-score across feature dimensions"""
        X_stats = self._feature_stats
        Y_stats = self._output_stats or self._feature_stats
        matrix = self._matrix if matrix is None else matrix

        n = X_stats.observations

        result = (
            matrix
            + n * jnp.outer(X_stats.mean, Y_stats.mean)
            - jnp.outer(X_stats.mean, Y_stats.sum)
            - jnp.outer(X_stats.sum, Y_stats.mean)
        ) / jnp.outer(X_stats.std, Y_stats.std)

        # Replace nans with 0s because std of some columns may be 0
        result = jnp.nan_to_num(result)

        return result

    def project(self, matrix: jnp.ndarray = None, projection: jnp.ndarray = None) -> jnp.ndarray:
        """Project gram matrix"""
        matrix = self._matrix if matrix is None else matrix

        if projection is None:
            return matrix

        # (X @ p).T @ (X @ P) = p.T @ X.T @ X @ p
        # (X @ p).T @ Y = p.T @ X.T @ Y
        result = projection.T @ matrix
        if self._output_stats is None:
            result @= projection

        return result

    def fit_intercept(
        self,
        matrix: jnp.ndarray = None,
        normalize: bool = False,
        projection: jnp.ndarray = None,
        input_dim: int = None,
    ) -> jnp.ndarray:
        """Add an intercept to X for X.T @ Y

        Notes:
            * This method returns rather than modifies _matrix because
            the change in dimension would mean we can no longer update
        """
        matrix = self._matrix if matrix is None else matrix

        X_stats = self._feature_stats
        Y_stats = self._output_stats

        if input_dim is not None and projection is not None:
            raise ValueError("Constrained regression not supported with projection")

        num_rc = input_dim or 1

        result = jnp.concatenate((jnp.zeros((num_rc, matrix.shape[1])), matrix), axis=0)

        stats = X_stats if Y_stats is None else Y_stats
        sumcol = jnp.tile(stats.sum, (num_rc, 1))

        if normalize:
            sumcol = jnp.zeros_like(sumcol)

        if Y_stats is None:
            if projection is not None and input_dim is None:
                sumcol @= projection
            result = jnp.concatenate((jnp.zeros((result.shape[0], num_rc)), result), axis=1)
            result = jax.ops.index_update(
                result, jax.ops.index[:num_rc, :num_rc], stats.observations
            )
            result = jax.ops.index_update(result, jax.ops.index[:num_rc, num_rc:], sumcol)
            result = jax.ops.index_update(result, jax.ops.index[num_rc:, :num_rc], sumcol.T)
        else:
            result = jax.ops.index_update(result, jax.ops.index[:num_rc, :], sumcol)

        return result
