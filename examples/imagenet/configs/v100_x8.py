# Copyright 2023 The Flax Authors.
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

"""Hyperparameter configuration to run the example on 8 x Nvidia V100 GPUs."""

from configs import default as default_lib


def get_config():
  """Get the hyperparameter configuration to train on 8 x Nvidia V100 GPUs."""
  # Override default configuration to avoid duplication of field definition.
  config = default_lib.get_config()

  config.batch_size = 512
  config.cache = True

  return config


metrics = default_lib.metrics
