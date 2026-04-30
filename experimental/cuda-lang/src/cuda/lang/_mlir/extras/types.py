# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .. import (
    IntegerType,
    IndexType,
    Float16Type,
    Float32Type,
    Float64Type,
)
from ..llvm import LLVMPointerType


def index() -> IntegerType:
    return IndexType()


def i1() -> IntegerType:
    return IntegerType.signless(1)


def i8() -> IntegerType:
    return IntegerType.signless(8)


def i16() -> IntegerType:
    return IntegerType.signless(16)


def i32() -> IntegerType:
    return IntegerType.signless(32)


def i64() -> IntegerType:
    return IntegerType.signless(64)


def f16() -> Float16Type:
    return Float16Type()


def f32() -> Float32Type:
    return Float32Type()


def f64() -> Float64Type:
    return Float64Type()


def ptr() -> LLVMPointerType:
    return LLVMPointerType()


__all__ = (
    "index",
    "i1",
    "i8",
    "i16",
    "i32",
    "i64",
    "f16",
    "f32",
    "f64",
    "ptr",
)
