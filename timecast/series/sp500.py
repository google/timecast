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
"""timecast/series/sp500.py"""
import os
from typing import Tuple

import numpy as np

from timecast.series.core import generate_timeline


def generate(path=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description: Outputs the log-normalized daily opening price of the S&P 500
    stock market index from January 3, 1986 to June 29, 2018.
    """

    return generate_timeline(
        path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data/sp500.csv")
    )
