# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from cuda.lang import _datatype as datatype
from cuda.lang._execution import stub
from cuda.tile._symbolic import SymbolicTile


def satisfies_scalar_integral_constraint(value):
    return (
        type(value) is int
        or (
            isinstance(value, SymbolicTile)
            and value.ndim == 0
            and datatype.is_integral(value.dtype)
        )
    )


def satisfies_scalar_or_vector_constraint(value):
    return isinstance(value, SymbolicTile) and value.ndim in (0, 1)


def satisfies_vector_constraint(value):
    return isinstance(value, SymbolicTile) and value.ndim == 1


@stub
def _raw_nvvm_intrinsic(
    intrinsic: str,
    result_dtypes: tuple[datatype.TypeSpec, ...],
    operands: tuple[Any, ...],
):
    '''
    Call an NVVM intrinsic directly.
    '''
