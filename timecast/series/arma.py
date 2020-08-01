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
"""timecast/series/arma.py"""
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def generate(
    n: int = 1000, p: int = 3, q: int = 3, dim: int = 1, c: float = 0.0, mag: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description: Randomly initialize the hidden dynamics of the system.
    Args:
        n (int): number of values to generate
        p (int): Autoregressive dynamics. If type int
        then randomly initializes a Gaussian length-p vector with L1-norm
        bounded by 1.0.  If p is a 1-dimensional numpy.ndarray then uses it
        as dynamics vector.
        q (int): Moving-average dynamics. If type int then
        randomly initializes a Gaussian length-q vector (no bound on norm).
        If p is a 1-dimensional numpy.ndarray then uses it as dynamics
        vector.
        dim (int): Dimension of values.
        c (float): Default value follows a normal distribution. The ARMA
        dynamics follows the equation x_t = c + AR-part + MA-part + noise,
        and thus tends to be centered around mean c.
        mag (float): Noise magnitude
    Returns:
        Tuple[np.ndarray, np.ndarray]: X, y pair
    """
    phi = jnp.asarray(np.random.rand(p))
    phi = 0.99 * phi / jnp.linalg.norm(phi, ord=1)

    psi = jnp.asarray(np.random.rand(q))

    c = jnp.asarray(np.random.rand(dim)) if c is None else c
    x = jnp.asarray(np.random.rand(p, dim))

    noise = mag * jnp.asarray(np.random.rand(q, dim))

    def step(carry, eps):
        """Internal step function for ARMA"""
        x, noise = carry
        x_ar = jnp.dot(x.T, phi)

        x_ma = jnp.dot(noise.T, psi)

        y = c + x_ar + x_ma + eps
        next_x = jnp.roll(x, n)
        next_noise = jnp.roll(noise, n)

        next_x = jax.ops.index_update(next_x, 0, y)
        next_noise = jax.ops.index_update(next_noise, 0, eps)

        return (next_x, next_noise), y

    EPS = jnp.asarray(np.random.rand(n + 1, dim))

    _, y = jax.lax.scan(step, (x, noise), EPS)

    return y[:-1].reshape(-1, 1), y[1:].reshape(-1, 1)
