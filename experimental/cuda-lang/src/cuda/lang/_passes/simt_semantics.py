# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._ir.ops import AllocDynSharedMemory, IfElse, Loop, AllocStaticSharedMemory
from cuda.lang._ir.ir import IRContext, Block, TileBlock, Operation
from cuda.lang._exception import TileError


def visit_block(block: Block, in_control_flow: bool) -> None:
    for op in block.operations:
        visit_op(op, in_control_flow)


def visit_op(op: Operation, in_control_flow: bool) -> None:
    match op:
        case IfElse():
            visit_block(op.then_block, True)
            visit_block(op.else_block, True)
        case Loop():
            visit_block(op.body, True)
        case (AllocStaticSharedMemory() | AllocDynSharedMemory()) if in_control_flow:
            raise TileError(
                "Memory allocated in dynamic control flow",
                op.loc
            )
        case _:
            pass


def simt_semantic_analysis(body: TileBlock, ctx: IRContext) -> None:
    visit_block(body, False)


__all__ = ("simt_semantic_analysis",)
