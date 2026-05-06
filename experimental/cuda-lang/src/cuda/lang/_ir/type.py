# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from enum import Enum

from cuda.lang._ir.ir import LocalArrayContextManagerValue
from cuda.tile._ir.type import (
    Type,
    TupleTy,
    ArrayTy,
    TileTy,
    StringTy,
    FunctionTy,
    DTypeConstructor,
    NoneType,
    ModuleTy,
    PointerTy,
    TokenTy,
    TypeTy,
    EnumTy,
    make_tile_ty,
    ContextManagerTy,
    ContextManagerState,
)
from cuda.tile._datatype import DType
from cuda.tile._ir.ir import Var, AggregateValue
from cuda.lang._exception import TileTypeError


class MemorySpace(Enum):
    GENERIC = 0
    GLOBAL = 1
    SHARED = 3
    CONSTANT = 4
    LOCAL = 5
    TENSOR = 6
    SHARED_CLUSTER = 7


def _is_power_of_2(value: int) -> bool:
    assert isinstance(value, int)
    return value > 0 and value & (value - 1) == 0


def is_vector_ty(ty: Type) -> bool:
    return (
        isinstance(ty, TileTy)
        and len(ty.shape) == 1
        and _is_power_of_2(ty.shape[0])
    )


def make_vector_ty(dtype: DType, length: int) -> TileTy:
    if not isinstance(length, int):
        raise TileTypeError(
            f"Expected vector length to be an int, got {type(length).__name__}"
        )
    if not _is_power_of_2(length):
        raise TileTypeError(
            f"Expected vector length to be a positive power of two, got {length}"
        )
    return make_tile_ty(dtype, (length,))


@dataclass(frozen=True, eq=True)
class OpaquePointerTy(Type):
    memory_space: MemorySpace = MemorySpace.GENERIC

    def __str__(self) -> str:
        return f'OpaquePtr[{self.memory_space}]'

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True)
class LocalArrayContextManagerTy(ContextManagerTy):
    dtype: DType
    shape: tuple[int, ...]
    alignment: int | None
    state: ContextManagerState

    def is_aggregate(self) -> bool:
        return True

    def aggregate_item_types(self) -> tuple[Type, ...]:
        return ()

    def make_aggregate_value(self, items: tuple[Var, ...]) -> AggregateValue:
        return LocalArrayContextManagerValue()

    def get_context_manager_state(self) -> ContextManagerState:
        return self.state


__all__ = (
    "Type",
    "TupleTy",
    "ArrayTy",
    "TileTy",
    "StringTy",
    "FunctionTy",
    "DTypeConstructor",
    "NoneType",
    "OpaquePointerTy",
    "ModuleTy",
    "PointerTy",
    "TokenTy",
    "TypeTy",
    "EnumTy",
    "make_tile_ty",
    "make_vector_ty",
    "is_vector_ty",
    "MemorySpace",
)
