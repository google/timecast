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
"""timecast/utils/__init__.py"""
from numbers import Real
from random import randint
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np


def internalize(x: Union[np.ndarray, jnp.ndarray, Real], dim: int) -> Tuple[bool, jnp.ndarray, int]:
    """ Takes user input data and converts into a consistent internal form
    Args:
        x (Union[np.ndarray, jnp.ndarray, Real]): some input data to be
        converted dim (int): The input dimension specified at
        instantiation

    Notes:
        * Inputs can be scalar, 0D, 1D, or 2D.
        * While internally, we may have >2D arrays, we consider that an error
        if coming from the user
        * Internal representation is 2D array where dimension 0 is time and
        dimension 1 is feature
        * Inputs can be single time series values (value) or a sequence of time
        series values (batch)
        * Scalar and 0D inputs are values if dim is 1
        * 1D input is a value if its length matches the dim
        * 1D input is a batch if dim is 1 (previous rule takes
        precedence)
        * 2D input is a value if dimension 0 is 1 and dimension 1 is dim

        * 2D input is a batch if dimension 1 equals dim
    Warning:
        * When x is 2D, dimensions are always (time, feature)
        * We don't check whether the NumPy array is well-formed or not (e.g.,

        single data type, np.product(x.shape) == np.len(x.ravel()))
    Returns:
        x (jnp.ndarray): internalized version of input
        is_value (bool): whether the input was a value or a batch
        dim (int): number of dimensions of original input. -1 means
        np.isscalar(x) is True
        was_jax (bool): whether or not the input array was a
        jax.numpy.DeviceArray

    Raises:
        TypeError: x is not NumPy array or real number, or has more than two
        dimensions
        ValueError: input x does not match dim
    """
    was_jax = isinstance(x, jnp.DeviceArray)
    is_scalar, x = jnp.isscalar(x), jnp.asarray(x)

    if x.ndim > 2:
        raise TypeError("Input cannot have more than two dimensions")

    # -1 if scalar, otherwise take original NumPy dimension
    ndim = -1 if is_scalar else x.ndim

    # If Python or 0D NumPy array, x is a value
    if ndim <= 0 and dim == 1:
        return jnp.asarray([[x]]), True, ndim, was_jax

    # If 1D array and length is equal to dim, x is a value
    if ndim == 1 and len(x.ravel()) == dim:
        return x.reshape(1, -1), True, ndim, was_jax

    # If 1D array and dim is 1, x is a batch
    if ndim == 1 and dim == 1:
        return x.reshape(-1, 1), False, ndim, was_jax

    # If 2D array and ndimension 1 is equal to dim, x is a batch
    if ndim == 2 and x.shape[1] == dim:
        return x, x.shape[0] == 1, ndim, was_jax

    raise ValueError("Input shape {} does not match dim {}".format(x.shape, dim))


def externalize(x: jnp.ndarray, dim: int, was_jax: bool = False) -> Union[np.ndarray, jnp.ndarray]:
    """Takes internal representation of 2D (time, feature) and returns to user
    Args:
        x (Union[jnp.ndarray, Real]): some input data to be converted
        dim (int): The original number of dimensions (-1 for scalar, n for nD
        NumPy array)
        was_jax (bool): whether original array was jax.numpy.DeviceArray

    Warning:
        * Transformations may not always be from R^n to R^n, so we have to make
        some guesses
        * We don't check whether the NumPy array is well-formed or not (e.g.,
        single data type, jnp.product(x.shape) == jnp.len(x.ravel()))

    Notes:
        * We assume if the original input was less than 2, we should squeeze
        out as many dimensions as we can
        * We leave input as is if original dimension was 2
        * It's not clear when to use this, especially if users are not
        consistent with their inputs

    Returns:
        x (Union[np.ndarray, jnp.ndarray]): externalized version of input

    Raises:
        TypeError: x is not a NumPy array with two dimensions
        ValueError: dim is not -1, 0, 1, or 2; input x does not match dim
        or dim
    """
    if not isinstance(x, jnp.ndarray):
        raise TypeError("x has incorrect type {}".format(type(x)))

    if was_jax:
        x = np.asarray(x)
    else:
        x = jnp.asarray(x)

    if x.ndim != 2:
        raise TypeError("x does not have two dimensions")
    if dim < -1 or dim > 2:
        raise ValueError("Original dimension must be -1, 0, 1, or 2")

    # If the original dimension was 2, just return
    if dim == 2:
        return x

    # Need to be careful with x with one item
    if len(x.ravel()) == 1:

        # If the original was a scalar, return a scalar
        if dim == -1:
            return x.item()

        # If the original was 1D, return 1D
        if dim == 1:
            return x.ravel()

    # Otherwise, try to squeeze as many dimensions as we can
    if dim < 2:
        return x.squeeze()


class random:
    """Singleton class to manage global jax randomness keys"""

    class __random:
        """Nested class to define instance"""

        def __init__(self) -> None:
            """Initialize instance"""
            self._key = jax.random.PRNGKey(0)

        def set_key(self, seed=None) -> None:
            """Fix global random key to ensure reproducibility of results.
            Args:
                seed (int): key that determines reproducible output
            """
            if seed is None:
                seed = randint(1, 9223372036854775807 + 1)
            assert type(seed) == int
            self._key = jax.random.PRNGKey(seed)

        def get_key(self) -> jnp.ndarray:
            """Return current global key"""
            return self._key

        def generate_key(self) -> jnp.ndarray:
            """Generates random subkey"""
            self._key, subkey = jax.random.split(self._key)
            return subkey

    __instance = __random()

    @staticmethod
    def set_key(seed=None) -> None:
        """set_key"""
        random.__instance.set_key(seed)

    @staticmethod
    def get_key() -> jnp.ndarray:
        """get_key"""
        return random.__instance.get_key()

    @staticmethod
    def generate_key() -> jnp.ndarray:
        """generate_key"""
        return random.__instance.generate_key()
