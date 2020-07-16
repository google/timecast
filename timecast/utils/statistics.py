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
"""timecast/utils/statistics.py"""
from numbers import Real
from typing import Union

import jax.numpy as jnp

from timecast.utils import internalize


class OnlineStatistics:
    """Compute summary statistics online"""

    def __init__(self, dim: int = 1):
        """Initialize OnlineStatistics"""
        self._dim = dim
        self._mean = jnp.zeros((1, dim))
        self._var = jnp.zeros((1, dim))
        self._sum = jnp.zeros((1, dim))
        self._observations = 0

    def update(self, observation: Union[Real, jnp.ndarray]) -> None:
        """Update with new observation"""
        observation, is_value, _, _ = internalize(observation, self._dim)

        num_observations = observation.shape[0]

        prev_mean = self._mean
        curr_mean = observation if is_value else observation.mean(axis=0)
        self._mean = (self._observations * prev_mean + num_observations * curr_mean) / (
            self._observations + num_observations
        )

        prev_var = self._var
        curr_var = 0 if is_value else observation.var(axis=0)

        self._var = (
            self._observations * prev_var
            + num_observations * curr_var
            + self._observations * ((prev_mean - self._mean) ** 2)
            + num_observations * ((curr_mean - self._mean) ** 2)
        ) / (self._observations + num_observations)

        self._sum += observation if is_value else observation.sum(axis=0)

        self._observations += num_observations

    @property
    def mean(self) -> Union[Real, jnp.ndarray]:
        """Mean"""
        return self._mean

    @property
    def var(self) -> Union[Real, jnp.ndarray]:
        """Variance"""
        return self._var

    @property
    def std(self) -> Union[Real, jnp.ndarray]:
        """Standard deviation"""
        return jnp.sqrt(self._var)

    @property
    def sum(self) -> Union[Real, jnp.ndarray]:
        """Sum"""
        return self._sum

    @property
    def observations(self) -> int:
        """Number of observations"""
        return self._observations

    def zscore(self, data: jnp.ndarray) -> Union[Real, jnp.ndarray]:
        """Zscore an observation based on current statistics"""
        return (data - self.mean) / self.std
