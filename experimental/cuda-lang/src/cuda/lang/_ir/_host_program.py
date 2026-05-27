# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Literal

from cuda.lang._ir import ir
from cuda.tile._ir.ops import (
    loosely_typed_const, binary_arithmetic_tensorlike_raw,
    strictly_typed_const, unary, _UNARY_BOOL_INT, TypedConst, AssumeBounded, AssumeDivBy, Assign,
    binary_bitwise_tensorlike_raw
)
from cuda.tile._ir.typing_support import I32_TY, I64_TY

HostOpcode = Literal["Const", "KernelArgI32", "KernelArgI64", "Mul", "Add", "RoundUpToPow2"]


@dataclass
class HostProgram:
    """
    Bytecode for a simple stack machine that runs on the host at launch time, in order to derive
    some launch parameters (e.g. dynamic shared memory size) from user-supplied kernel arguments.
    """

    opcodes: list[HostOpcode] = dataclasses.field(default_factory=list)

    # attributes for "Const", "KernelArgI32", "KernelArgI64" and "RoundUpToPow2" opcodes
    op_attrs: list[int] = dataclasses.field(default_factory=list)

    def extend(self, other: "HostProgram"):
        self.opcodes.extend(other.opcodes)
        self.op_attrs.extend(other.op_attrs)

    def as_const(self) -> int | None:
        if len(self.opcodes) == 1 and self.opcodes[0] == "Const":
            assert len(self.op_attrs) == 1
            return self.op_attrs[0]
        else:
            return None


def get_host_programs_by_var(kernel_body: ir.Block) -> dict[str, HostProgram]:
    ret = dict()
    for i, p in enumerate(kernel_body.params):
        ty = p.get_type()
        if ty == I32_TY:
            ret[p.name] = HostProgram(opcodes=["KernelArgI32"], op_attrs=[i])
        elif ty == I64_TY:
            ret[p.name] = HostProgram(opcodes=["KernelArgI64"], op_attrs=[i])

    for op in kernel_body:
        if isinstance(op, TypedConst) and isinstance(op.value, int):
            # FIXME: check that we fit in int64 range
            ret[op.result_var.name] = HostProgram(opcodes=["Const"], op_attrs=[op.value])
        elif isinstance(op, Assign | AssumeBounded | AssumeDivBy):
            [x] = op.all_inputs()
            if x.name in ret:
                ret[op.result_var.name] = ret[x.name]
        # TODO: do we need to descend into nested blocks?

    return ret


def host_program_to_ir(program: HostProgram, kernel_params: tuple[ir.Var, ...]) -> ir.Var:
    attrs = iter(program.op_attrs)
    stack: list[ir.Var] = []
    for opcode in program.opcodes:
        match opcode:
            case "Const": stack.append(loosely_typed_const(next(attrs)))
            case "KernelArgI32": stack.append(kernel_params[next(attrs)])
            case "KernelArgI64": stack.append(kernel_params[next(attrs)])
            case "Mul":
                b = stack.pop()
                stack[-1] = binary_arithmetic_tensorlike_raw("mul", stack[-1], b)
            case "Add":
                b = stack.pop()
                stack[-1] = binary_arithmetic_tensorlike_raw("add", stack[-1], b)
            case "RoundUpToPow2":
                stack[-1] = _round_up_ir(stack[-1], next(attrs))
            case _:
                assert False
    assert next(attrs, None) is None
    assert len(stack) == 1
    return stack[0]


def _round_up_ir(value: ir.Var, alignment: int) -> ir.Var:
    value_ty = value.get_type()
    mask = strictly_typed_const(alignment - 1, value_ty)
    value_plus_mask = binary_arithmetic_tensorlike_raw("add", value, mask)
    neg_mask = unary('neg', _UNARY_BOOL_INT, mask)
    rounded = binary_bitwise_tensorlike_raw('and_', value_plus_mask, neg_mask)
    return rounded
