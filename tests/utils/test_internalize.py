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
"""timecast/tests/utils/test_internalize.py"""
import jax.numpy as jnp
import numpy as np
import pytest

from timecast.utils import internalize


@pytest.mark.parametrize(
    "n,shape,expected",
    [
        (1, (1,), (True, 1)),
        (1, (10,), (False, jnp.ones((10, 1)))),
        (1, (1, 1), (True, 1)),
        (1, (10, 1), (False, jnp.ones((10, 1)))),
        (2, (2,), (True, jnp.ones((1, 2)))),
        (2, (1, 2), (True, jnp.ones((1, 2)))),
        (2, (4, 2), (False, jnp.ones((4, 2)))),
        (1, (), (True, 1)),
    ],
)
def test_internalize(n, shape, expected):
    """Test internalize"""
    X = jnp.ones(shape)

    X, is_value, dim, _ = internalize(X, n)

    assert is_value == expected[0]
    np.testing.assert_array_equal(X, expected[1])


@pytest.mark.parametrize(
    "n,shape",
    [(2, (1, 3)), (2, (3, 1)), (2, (2, 4)), (2, (2, 1)), (2, (5,)), (1, (2, 2)), (1, (1, 10))],
)
def test_internalize_value_error(n, shape):
    """Test value error"""
    with pytest.raises(ValueError):
        internalize(jnp.ones(shape), n)


def test_internalize_type_error():
    """Test type error"""
    with pytest.raises(TypeError):
        internalize(jnp.ones((1, 2, 3)), 10)


@pytest.mark.parametrize("n,expected", [(1, True), (10, False)])
def test_internalize_scalar(n, expected):
    """Test scalar"""
    if expected:
        _, is_value, _, _ = internalize(4, n)
        assert is_value == expected
    else:
        with pytest.raises(ValueError):
            internalize(4, n)
