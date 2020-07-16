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
"""timecast/modules/_predict_constant.py"""
from timecast.modules.core import Module


class PredictConstant(Module):
    """Predict constant value"""

    def __init__(self, constant: int = 0.0):
        """init"""
        self.constant = constant

    def __call__(self, x):
        """call"""
        return self.constant
