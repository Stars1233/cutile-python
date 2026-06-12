# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from cuda.lang._ir.ir import Operation, Var, attribute, operand
from cuda.tile._ir.ir import MemoryEffect


@dataclass(eq=False)
class RawNVVMIntrinsic(
    Operation, opcode="nvvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()
