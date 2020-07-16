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
"""timecast/optim/__init__.py

Todo:
    * Document available optimizers
"""
from timecast.optim._multiplicative_weights import MultiplicativeWeights
from timecast.optim._norm_threshold import NormThreshold
from timecast.optim._sgd import SGD


__all__ = ["SGD", "MultiplicativeWeights", "NormThreshold"]
