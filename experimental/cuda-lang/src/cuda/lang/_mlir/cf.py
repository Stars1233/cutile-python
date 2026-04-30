# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import DenseI32ArrayAttr
from . import DenseIntElementsAttr
from . import StringAttr
from ._builtins import Attribute
from ._builtins import BlockLabel
from ._builtins import Value
from ._builtins import add_operation
from typing import Optional
from typing import Sequence


# ========= 'cf' dialect of MLIR ==========


# ---- Operators ----


def add_AssertOp(
    *,
    arg: Value,
    msg: str,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('msg', StringAttr(value=msg)))
    return add_operation(
        name="cf.assert",
        result_type=None,
        operands=[arg],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BranchOp(
    *,
    destOperands: Sequence[Value],
    dest: BlockLabel,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="cf.br",
        result_type=None,
        operands=list(destOperands),
        properties=all_props,
        attributes=extra_attributes,
        successors=[dest],
    )


def add_CondBranchOp(
    *,
    condition: Value,
    trueDestOperands: Sequence[Value],
    falseDestOperands: Sequence[Value],
    branch_weights: Optional[Sequence[int]] = None,
    trueDest: BlockLabel,
    falseDest: BlockLabel,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if branch_weights is not None:
        all_props.append(('branch_weights', DenseI32ArrayAttr(branch_weights)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(trueDestOperands), len(falseDestOperands)])))
    return add_operation(
        name="cf.cond_br",
        result_type=None,
        operands=[condition, *trueDestOperands, *falseDestOperands],
        properties=all_props,
        attributes=extra_attributes,
        successors=[trueDest, falseDest],
    )


def add_SwitchOp(
    *,
    flag: Value,
    defaultOperands: Sequence[Value],
    caseOperands: Sequence[Value],
    case_values: Optional[DenseIntElementsAttr] = None,
    case_operand_segments: Sequence[int],
    defaultDestination: BlockLabel,
    caseDestinations: Sequence[BlockLabel],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if case_values is not None:
        all_props.append(('case_values', case_values))
    all_props.append(('case_operand_segments', DenseI32ArrayAttr(case_operand_segments)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(defaultOperands), len(caseOperands)])))
    return add_operation(
        name="cf.switch",
        result_type=None,
        operands=[flag, *defaultOperands, *caseOperands],
        properties=all_props,
        attributes=extra_attributes,
        successors=[defaultDestination, *caseDestinations],
    )
