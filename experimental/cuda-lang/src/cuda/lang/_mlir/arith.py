# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import IntegerAttr
from . import IntegerType
from . import TypedAttr
from . import UnitAttr
from . import _util
from ._builtins import APInt
from ._builtins import Attribute
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import enum


# ========= 'arith' dialect of MLIR ==========


# ---- Interfaces ----


class ArithFastMathInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ArithIntegerOverflowFlagsInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ArithNonNegFlagInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class ArithRoundingModeInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Enums ----


class AtomicRMWKind(enum.Enum):
    addf = 0
    addi = 1
    andi = 2
    assign = 3
    maximumf = 4
    maxnumf = 5
    maxs = 6
    maxu = 7
    minimumf = 8
    minnumf = 9
    mins = 10
    minu = 11
    mulf = 12
    muli = 13
    ori = 14
    xori = 15

    def _print_mlir_unqualified(self, p):
        p(("addf", "addi", "andi", "assign", "maximumf", "maxnumf", "maxs", "maxu", "minimumf",
           "minnumf", "mins", "minu", "mulf", "muli", "ori", "xori",)[self._value_])


class AtomicRMWKindAttr(IntegerAttr):
    def __init__(self, value: AtomicRMWKind):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class CmpFPredicate(enum.Enum):
    AlwaysFalse = 0
    OEQ = 1
    OGT = 2
    OGE = 3
    OLT = 4
    OLE = 5
    ONE = 6
    ORD = 7
    UEQ = 8
    UGT = 9
    UGE = 10
    ULT = 11
    ULE = 12
    UNE = 13
    UNO = 14
    AlwaysTrue = 15

    def _print_mlir_unqualified(self, p):
        p(("false", "oeq", "ogt", "oge", "olt", "ole", "one", "ord", "ueq", "ugt", "uge", "ult",
           "ule", "une", "uno", "true",)[self._value_])


class CmpFPredicateAttr(IntegerAttr):
    def __init__(self, value: CmpFPredicate):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class CmpIPredicate(enum.Enum):
    eq = 0
    ne = 1
    slt = 2
    sle = 3
    sgt = 4
    sge = 5
    ult = 6
    ule = 7
    ugt = 8
    uge = 9

    def _print_mlir_unqualified(self, p):
        p(("eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge",)[self._value_])


class CmpIPredicateAttr(IntegerAttr):
    def __init__(self, value: CmpIPredicate):
        super().__init__(type=IntegerType.signless(64), value=APInt(value._value_, 64))


class FastMathFlags(enum.IntFlag):
    none = 0x0
    reassoc = 0x1
    nnan = 0x2
    ninf = 0x4
    nsz = 0x8
    arcp = 0x10
    contract = 0x20
    afn = 0x40
    fast = 0x7f

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, ((0x7f, 'fast'),),
                         ((0x1, 'reassoc'), (0x2, 'nnan'), (0x4, 'ninf'), (0x8, 'nsz'),
                          (0x10, 'arcp'), (0x20, 'contract'), (0x40, 'afn'),))


class IntegerOverflowFlags(enum.IntFlag):
    none = 0x0
    nsw = 0x1
    nuw = 0x2

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, (), ((0x1, 'nsw'), (0x2, 'nuw'),))


class RoundingMode(enum.Enum):
    to_nearest_even = 0
    downward = 1
    upward = 2
    toward_zero = 3
    to_nearest_away = 4

    def _print_mlir_unqualified(self, p):
        p(("to_nearest_even", "downward", "upward", "toward_zero",
           "to_nearest_away",)[self._value_])


class RoundingModeAttr(IntegerAttr):
    def __init__(self, value: RoundingMode):
        super().__init__(type=IntegerType.signless(32), value=APInt(value._value_, 32))


# ---- Attributes ----


@dataclass(kw_only=True)
class FastMathFlagsAttr(Attribute, dialect='arith', mnemonic='fastmath'):
    value: "FastMathFlags"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


@dataclass(kw_only=True)
class IntegerOverflowFlagsAttr(Attribute, dialect='arith', mnemonic='overflow'):
    value: "IntegerOverflowFlags"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")


# ---- Operators ----


def add_AddFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    roundingmode: Optional[RoundingMode] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    return add_operation(
        name="arith.addf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AddIOp(
    *,
    lhs: Value,
    rhs: Value,
    overflowFlags: IntegerOverflowFlags = IntegerOverflowFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('overflowFlags', IntegerOverflowFlagsAttr(value=overflowFlags)))
    return add_operation(
        name="arith.addi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AddUIExtendedOp(
    *,
    sum_type: Type,
    overflow_type: Type,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value]:
    all_props = []
    return add_operation(
        name="arith.addui_extended",
        result_type=(sum_type, overflow_type),
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AndIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.andi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_BitcastOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.bitcast",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CeilDivSIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.ceildivsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CeilDivUIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.ceildivui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CmpFOp(
    *,
    predicate: CmpFPredicate,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = _util.get_i1_same_shape(lhs.type)
    all_props = []
    all_props.append(('predicate', CmpFPredicateAttr(predicate)))
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.cmpf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CmpIOp(
    *,
    predicate: CmpIPredicate,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = _util.get_i1_same_shape(lhs.type)
    all_props = []
    all_props.append(('predicate', CmpIPredicateAttr(predicate)))
    return add_operation(
        name="arith.cmpi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConstantOp(
    *,
    value: TypedAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = value.get_type()
    all_props = []
    all_props.append(('value', value))
    return add_operation(
        name="arith.constant",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ConvertFOp(
    *,
    out_type: Type,
    in_: Value,
    roundingmode: Optional[RoundingMode] = None,
    fastmath: Optional[FastMathFlags] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    if fastmath is not None:
        all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.convertf",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DivFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    roundingmode: Optional[RoundingMode] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    return add_operation(
        name="arith.divf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DivSIOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="arith.divsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DivUIOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="arith.divui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtFOp(
    *,
    out_type: Type,
    in_: Value,
    fastmath: Optional[FastMathFlags] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if fastmath is not None:
        all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.extf",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtSIOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.extsi",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtUIOp(
    *,
    out_type: Type,
    in_: Value,
    nonNeg: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if nonNeg:
        all_props.append(('nonNeg', UnitAttr()))
    return add_operation(
        name="arith.extui",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPToSIOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.fptosi",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FPToUIOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.fptoui",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FloorDivSIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.floordivsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FlushDenormalsOp(
    *,
    operand: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = operand.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.flush_denormals",
        result_type=result_type,
        operands=[operand],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IndexCastOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.index_cast",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_IndexCastUIOp(
    *,
    out_type: Type,
    in_: Value,
    nonNeg: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if nonNeg:
        all_props.append(('nonNeg', UnitAttr()))
    return add_operation(
        name="arith.index_castui",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaxNumFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.maxnumf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaxSIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.maxsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaxUIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.maxui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaximumFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.maximumf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinNumFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.minnumf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinSIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.minsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinUIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.minui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MinimumFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.minimumf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MulFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    roundingmode: Optional[RoundingMode] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    return add_operation(
        name="arith.mulf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MulIOp(
    *,
    lhs: Value,
    rhs: Value,
    overflowFlags: IntegerOverflowFlags = IntegerOverflowFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('overflowFlags', IntegerOverflowFlagsAttr(value=overflowFlags)))
    return add_operation(
        name="arith.muli",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MulSIExtendedOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value]:
    low_type = rhs.type
    high_type = rhs.type
    all_props = []
    return add_operation(
        name="arith.mulsi_extended",
        result_type=(low_type, high_type),
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MulUIExtendedOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value]:
    low_type = rhs.type
    high_type = rhs.type
    all_props = []
    return add_operation(
        name="arith.mului_extended",
        result_type=(low_type, high_type),
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_NegFOp(
    *,
    operand: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = operand.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.negf",
        result_type=result_type,
        operands=[operand],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_OrIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.ori",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RemFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.remf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RemSIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.remsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RemUIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.remui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SIToFPOp(
    *,
    out_type: Type,
    in_: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="arith.sitofp",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ScalingExtFOp(
    *,
    out_type: Type,
    in_: Value,
    scale: Value,
    fastmath: Optional[FastMathFlags] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if fastmath is not None:
        all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.scaling_extf",
        result_type=out_type,
        operands=[in_, scale],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ScalingTruncFOp(
    *,
    out_type: Type,
    in_: Value,
    scale: Value,
    roundingmode: Optional[RoundingMode] = None,
    fastmath: Optional[FastMathFlags] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    if fastmath is not None:
        all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.scaling_truncf",
        result_type=out_type,
        operands=[in_, scale],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SelectOp(
    *,
    condition: Value,
    true_value: Value,
    false_value: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = false_value.type
    all_props = []
    return add_operation(
        name="arith.select",
        result_type=result_type,
        operands=[condition, true_value, false_value],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShLIOp(
    *,
    lhs: Value,
    rhs: Value,
    overflowFlags: IntegerOverflowFlags = IntegerOverflowFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('overflowFlags', IntegerOverflowFlagsAttr(value=overflowFlags)))
    return add_operation(
        name="arith.shli",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShRSIOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="arith.shrsi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ShRUIOp(
    *,
    lhs: Value,
    rhs: Value,
    isExact: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    if isExact:
        all_props.append(('isExact', UnitAttr()))
    return add_operation(
        name="arith.shrui",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubFOp(
    *,
    lhs: Value,
    rhs: Value,
    fastmath: FastMathFlags = FastMathFlags(0),
    roundingmode: Optional[RoundingMode] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    return add_operation(
        name="arith.subf",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubIOp(
    *,
    lhs: Value,
    rhs: Value,
    overflowFlags: IntegerOverflowFlags = IntegerOverflowFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    all_props.append(('overflowFlags', IntegerOverflowFlagsAttr(value=overflowFlags)))
    return add_operation(
        name="arith.subi",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TruncFOp(
    *,
    out_type: Type,
    in_: Value,
    roundingmode: Optional[RoundingMode] = None,
    fastmath: Optional[FastMathFlags] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if roundingmode is not None:
        all_props.append(('roundingmode', RoundingModeAttr(roundingmode)))
    if fastmath is not None:
        all_props.append(('fastmath', FastMathFlagsAttr(value=fastmath)))
    return add_operation(
        name="arith.truncf",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TruncIOp(
    *,
    out_type: Type,
    in_: Value,
    overflowFlags: IntegerOverflowFlags = IntegerOverflowFlags(0),
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('overflowFlags', IntegerOverflowFlagsAttr(value=overflowFlags)))
    return add_operation(
        name="arith.trunci",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_UIToFPOp(
    *,
    out_type: Type,
    in_: Value,
    nonNeg: bool = False,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if nonNeg:
        all_props.append(('nonNeg', UnitAttr()))
    return add_operation(
        name="arith.uitofp",
        result_type=out_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_XOrIOp(
    *,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = lhs.type
    all_props = []
    return add_operation(
        name="arith.xori",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )
