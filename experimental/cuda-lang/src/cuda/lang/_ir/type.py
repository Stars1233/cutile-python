# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum

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
)


class MemorySpace(Enum):
    GENERIC = 0
    GLOBAL = 1
    SHARED = 3
    CONSTANT = 4
    LOCAL = 5
    TENSOR = 6
    SHARED_CLUSTER = 7


class VectorTy(TileTy):
    def __init__(self, dtype, length: int):
        super().__init__(dtype, (length,))

    def __str__(self):
        return f"VectorTy[{self.dtype},{self.num_elements}]"

    @property
    def num_elements(self):
        return self.shape[0]


@dataclass(frozen=True, eq=True)
class OpaquePointerTy(Type):
    memory_space: MemorySpace = MemorySpace.GENERIC

    def __str__(self) -> str:
        return f'OpaquePtr[{self.memory_space}]'

    def __repr__(self) -> str:
        return str(self)


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
    "VectorTy",
    "ModuleTy",
    "PointerTy",
    "TokenTy",
    "TypeTy",
    "EnumTy",
    "make_tile_ty",
    "MemorySpace",
)
