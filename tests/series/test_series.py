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
"""timecast/tests/series/test_series.py"""
import jax.numpy as jnp
import pytest

from timecast.series import arma
from timecast.series import crypto
from timecast.series import enso
from timecast.series import lds
from timecast.series import lstm
from timecast.series import random
from timecast.series import rnn
from timecast.series import sp500
from timecast.series import uci
from timecast.series import unemployment


@pytest.mark.parametrize(
    "module", [arma, crypto, enso, lds, lstm, random, rnn, sp500, uci, unemployment]
)
def test_series(module):
    """Test outputs, shapes of data time series"""
    X, y = getattr(module, "generate")()  # noqa: B009

    assert isinstance(X, jnp.ndarray)
    assert isinstance(y, jnp.ndarray)
    assert X.shape[0] == y.shape[0]
