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
"""timecast/tests/utils/test_random.py"""
import jax.numpy as jnp

from timecast.utils import random


def test_random():
    """Test random singleton"""
    random.set_key()
    random.set_key(0)
    assert jnp.array_equal(jnp.array([0, 0]), random.get_key())
    expected = jnp.array([2718843009, 1272950319], dtype=jnp.uint32)
    assert jnp.array_equal(random.generate_key(), expected)
    expected = jnp.array([4146024105, 967050713], dtype=jnp.uint32)
    assert jnp.array_equal(random.get_key(), expected)
