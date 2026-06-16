# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

import cuda.lang._mlir as mlir
from cuda.tile._memory_model import MemoryOrder
from cuda.tile._ir.ir import MemoryEffect
from .ir import Operation, Var, attribute, operand
from .type import VectorTy, ScalarTy


@dataclass(eq=False)
class RawNVVMIntrinsic(
    Operation, opcode="nvvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class RawMLIROperation(
    Operation, opcode="mlir.operation", memory_effect=MemoryEffect.STORE
):
    op_name: str = attribute()
    operands_: tuple[Var, ...] = operand()
    mlir_attributes: tuple[tuple[str, mlir.Attribute], ...] = attribute(default=())


@dataclass(eq=False)
class ForeignFunction(
    Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE
):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class VectorGetItem(
    Operation, opcode="vector_getitem", memory_effect=MemoryEffect.LOAD
):
    x: Var[VectorTy] = operand()
    index: Var[ScalarTy] = operand()


@dataclass(eq=False)
class BitCast(Operation, opcode="bitcast"):
    x: Var = operand()


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: Optional[int] = attribute()
    volatile: bool = attribute(default=False)
    ordering: Optional[MemoryOrder] = attribute(default=None)

    valid_orderings = (
        None,
        MemoryOrder.WEAK,
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: Optional[int] = attribute()
    volatile: bool = attribute(default=False)
    ordering: Optional[MemoryOrder] = attribute(default=None)

    valid_orderings = (
        None,
        MemoryOrder.WEAK,
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()
