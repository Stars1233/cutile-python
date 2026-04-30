# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import DataLayoutTypeInterface
from . import MemRefElementTypeInterface
from . import PtrLikeTypeInterface
from . import StringAttr
from . import TypeAttr
from . import TypedAttr
from . import VectorElementTypeInterface
from ._builtins import APInt
from ._builtins import Attribute
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
import enum


# ========= 'ptr' dialect of MLIR ==========


# ---- Interfaces ----


class MemorySpaceAttrInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Enums ----


class AtomicBinOp(enum.Enum):
    xchg = 0
    add = 1
    sub = 2
    _and = 3
    nand = 4
    _or = 5
    _xor = 6
    max = 7
    min = 8
    umax = 9
    umin = 10
    fadd = 11
    fsub = 12
    fmax = 13
    fmin = 14
    uinc_wrap = 15
    udec_wrap = 16

    def _print_mlir_unqualified(self, p):
        p(("xchg", "add", "sub", "_and", "nand", "_or", "_xor", "max", "min", "umax", "umin",
           "fadd", "fsub", "fmax", "fmin", "uinc_wrap", "udec_wrap",)[self._value_])


class AtomicOrdering(enum.Enum):
    not_atomic = 0
    unordered = 1
    monotonic = 2
    acquire = 3
    release = 4
    acq_rel = 5
    seq_cst = 6

    def _print_mlir_unqualified(self, p):
        p(("not_atomic", "unordered", "monotonic", "acquire", "release", "acq_rel",
           "seq_cst",)[self._value_])


class PtrAddFlags(enum.Enum):
    none = 0
    nusw = 1
    nuw = 2
    inbounds = 3

    def _print_mlir_unqualified(self, p):
        p(("none", "nusw", "nuw", "inbounds",)[self._value_])


class PtrDiffFlags(enum.IntFlag):
    none = 0x0
    nuw = 0x1
    nsw = 0x2

    def _print_mlir_unqualified(self, p):
        value = int(self._value_)
        if value == 0:
            p('none')
            return
        p.print_bit_enum(value, (), ((0x0, 'none'), (0x1, 'nuw'), (0x2, 'nsw'),))


# ---- Attributes ----


@dataclass(kw_only=True)
class AddressAttr(Attribute, TypedAttr, dialect='ptr', mnemonic='address'):
    type: "PtrType"
    value: "APInt"

    def _print_mlir_unqualified(self, p):
        p("<")
        self.value._print_mlir_unqualified(p)
        p(">")

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class GenericSpaceAttr(Attribute, MemorySpaceAttrInterface, dialect='ptr',
                       mnemonic='generic_space'):

    def _print_mlir_unqualified(self, p):
        pass


@dataclass(kw_only=True)
class NullAttr(Attribute, TypedAttr, dialect='ptr', mnemonic='null'):
    type: "PtrType"

    def _print_mlir_unqualified(self, p):
        pass

    def get_type(self):
        return self.type


@dataclass(kw_only=True)
class SpecAttr(Attribute, dialect='ptr', mnemonic='spec'):
    size: "int"
    abi: "int"
    preferred: "int"
    index: "int" = 4294967295

    def _print_mlir_unqualified(self, p):
        p("<size = ")
        p(str(self.size))
        p(", abi = ")
        p(str(self.abi))
        p(", preferred = ")
        p(str(self.preferred))
        if self.index != 4294967295:
            p(", index = ")
            p(str(self.index))
        p(">")


# ---- Types ----


@dataclass(kw_only=True)
class PtrMetadataType(Type, dialect='ptr', mnemonic='ptr_metadata'):
    type: "PtrLikeTypeInterface"

    def _print_mlir_unqualified(self, p):
        p("<")
        p.print_qualified_attr_or_type(self.type)
        p(">")


@dataclass(kw_only=True)
class PtrType(Type, MemRefElementTypeInterface, PtrLikeTypeInterface, VectorElementTypeInterface,
              DataLayoutTypeInterface, dialect='ptr', mnemonic='ptr'):
    memorySpace: "MemorySpaceAttrInterface"

    def _print_mlir_unqualified(self, p):
        p("<")
        p.print_qualified_attr_or_type(self.memorySpace)
        p(">")


# ---- Operators ----


def add_ConstantOp(
    *,
    value: TypedAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = value.get_type()
    all_props = []
    all_props.append(('value', value))
    return add_operation(
        name="ptr.constant",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_FromPtrOp(
    *,
    result_type: PtrLikeTypeInterface,
    ptr: Value,
    metadata: Optional[Value] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="ptr.from_ptr",
        result_type=result_type,
        operands=[ptr, *([] if metadata is None else [metadata])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GatherOp(
    *,
    ptrs: Value,
    mask: Value,
    passthrough: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = passthrough.type
    all_props = []
    return add_operation(
        name="ptr.gather",
        result_type=result_type,
        operands=[ptrs, mask, passthrough],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GetMetadataOp(
    *,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = PtrMetadataType(type=ptr.type)
    all_props = []
    return add_operation(
        name="ptr.get_metadata",
        result_type=result_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_LoadOp(
    *,
    value_type: Type,
    ptr: Value,
    syncscope: Optional[str] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    return add_operation(
        name="ptr.load",
        result_type=value_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaskedLoadOp(
    *,
    ptr: Value,
    mask: Value,
    passthrough: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = passthrough.type
    all_props = []
    return add_operation(
        name="ptr.masked_load",
        result_type=result_type,
        operands=[ptr, mask, passthrough],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MaskedStoreOp(
    *,
    value: Value,
    ptr: Value,
    mask: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="ptr.masked_store",
        result_type=None,
        operands=[value, ptr, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrAddOp(
    *,
    result_type: Type,
    base: Value,
    offset: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="ptr.ptr_add",
        result_type=result_type,
        operands=[base, offset],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PtrDiffOp(
    *,
    result_type: Type,
    lhs: Value,
    rhs: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="ptr.ptr_diff",
        result_type=result_type,
        operands=[lhs, rhs],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ScatterOp(
    *,
    value: Value,
    ptrs: Value,
    mask: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="ptr.scatter",
        result_type=None,
        operands=[value, ptrs, mask],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StoreOp(
    *,
    value: Value,
    ptr: Value,
    syncscope: Optional[str] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if syncscope is not None:
        all_props.append(('syncscope', StringAttr(value=syncscope)))
    return add_operation(
        name="ptr.store",
        result_type=None,
        operands=[value, ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ToPtrOp(
    *,
    result_type: PtrType,
    ptr: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="ptr.to_ptr",
        result_type=result_type,
        operands=[ptr],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TypeOffsetOp(
    *,
    result_type: Type,
    elementType: Type,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('elementType', TypeAttr(value=elementType)))
    return add_operation(
        name="ptr.type_offset",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )
