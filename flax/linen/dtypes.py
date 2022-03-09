# Copyright 2022 The Flax Authors.
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

"""Tools for working with dtypes."""

from typing import Any, Callable, Optional, Tuple, Type

import jax.numpy as jnp
import numpy as np


Array = Any    # pylint: disable=invalid-name
PRNGKey = Any    # pylint: disable=invalid-name
Shape = Tuple[int, ...]
FloatingDType = Type[jnp.floating]
GenericDType = Type[np.generic]
InexactDType = Type[jnp.inexact]
NumericDType = Type[jnp.number]
Initializer = Callable[[PRNGKey, Shape, InexactDType], Array]


def canonicalize_inexact_dtypes(
    computation_dtype: Optional[InexactDType],
    *input_dtypes: Union[Array, InexactDType]) -> InexactDType:
    """
    Args:
      computation_dtype: The dtype that the module will do computations in, or
        None if it is to be inferred.
      input_dtypes: The dtype of the module inputs and parameters from which the
        computation_dtype will be inferred.
    """
  dtype = (jnp.result_type(*input_dtypes)
           if computation_dtype is None else computation_dtype)
  assert jnp.issubdtype(dtype, jnp.inexact)
  return dtype


def canonicalize_numeric_dtypes(
    computation_dtype: Optional[NumericDType],
    *input_dtypes: Union[Array, NumericDType]) -> NumericDType:
    """
    Args:
      computation_dtype: The dtype that the module will do computations in, or
        None if it is to be inferred.
      input_dtypes: The dtype of the module inputs and parameters from which the
        computation_dtype will be inferred.
    """
  dtype = (jnp.result_type(*input_dtypes)
           if computation_dtype is None else computation_dtype)
  assert jnp.issubdtype(dtype, jnp.number)
  return dtype
