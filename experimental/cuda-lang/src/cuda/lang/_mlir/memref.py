# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from . import AffineMapAttr
from . import ArrayAttr
from . import BaseMemRefType
from . import BoolAttr
from . import DenseI32ArrayAttr
from . import DenseI64ArrayAttr
from . import FlatSymbolRefAttr
from . import IndexType
from . import IntegerAttr
from . import IntegerType
from . import MemRefType
from . import StringAttr
from . import TypeAttr
from . import UnitAttr
from . import arith
from ._builtins import AffineMap
from ._builtins import Attribute
from ._builtins import Region
from ._builtins import Type
from ._builtins import Value
from ._builtins import add_operation
from typing import Optional
from typing import Sequence


# ========= 'memref' dialect of MLIR ==========


# ---- Interfaces ----


class IndexedAccessOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


class IndexedMemCopyOpInterface:
    def __init__(self):
        raise NotImplementedError('Interfaces cannot be instantiated')


# ---- Operators ----


def add_AssumeAlignmentOp(
    *,
    memref: Value,
    alignment: int,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = memref.type
    all_props = []
    all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(32), alignment)))
    return add_operation(
        name="memref.assume_alignment",
        result_type=result_type,
        operands=[memref],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AtomicRMWOp(
    *,
    kind: arith.AtomicRMWKind,
    value: Value,
    memref: Value,
    indices: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = value.type
    all_props = []
    all_props.append(('kind', arith.AtomicRMWKindAttr(kind)))
    return add_operation(
        name="memref.atomic_rmw",
        result_type=result_type,
        operands=[value, memref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AtomicYieldOp(
    *,
    result: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.atomic_yield",
        result_type=None,
        operands=[result],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CopyOp(
    *,
    source: Value,
    target: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.copy",
        result_type=None,
        operands=[source, target],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DistinctObjectsOp(
    *,
    results_types: Sequence[MemRefType],
    operands: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, ...]:
    all_props = []
    return add_operation(
        name="memref.distinct_objects",
        result_type=results_types,
        operands=list(operands),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GenericAtomicRMWOp(
    *,
    memref: Value,
    indices: Sequence[Value],
    atomic_body: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = memref.type.get_element_type()
    all_props = []
    return add_operation(
        name="memref.generic_atomic_rmw",
        result_type=result_type,
        operands=[memref, *indices],
        properties=all_props,
        attributes=extra_attributes,
        regions=[atomic_body],
    )


def add_LoadOp(
    *,
    memref: Value,
    indices: Sequence[Value],
    nontemporal: bool = False,
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = memref.type.get_element_type()
    all_props = []
    if nontemporal:
        all_props.append(('nontemporal', BoolAttr(value=nontemporal)))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    return add_operation(
        name="memref.load",
        result_type=result_type,
        operands=[memref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AllocOp(
    *,
    memref_type: MemRefType,
    dynamicSizes: Sequence[Value],
    symbolOperands: Sequence[Value],
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(dynamicSizes), len(symbolOperands)])))
    return add_operation(
        name="memref.alloc",
        result_type=memref_type,
        operands=[*dynamicSizes, *symbolOperands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AllocaOp(
    *,
    memref_type: MemRefType,
    dynamicSizes: Sequence[Value],
    symbolOperands: Sequence[Value],
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([len(dynamicSizes), len(symbolOperands)])))
    return add_operation(
        name="memref.alloca",
        result_type=memref_type,
        operands=[*dynamicSizes, *symbolOperands],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_AllocaScopeOp(
    *,
    results_types: Sequence[Type],
    bodyRegion: Region,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, ...]:
    all_props = []
    return add_operation(
        name="memref.alloca_scope",
        result_type=results_types,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
        regions=[bodyRegion],
    )


def add_AllocaScopeReturnOp(
    *,
    results: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.alloca_scope.return",
        result_type=None,
        operands=list(results),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CastOp(
    *,
    dest_type: BaseMemRefType,
    source: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="memref.cast",
        result_type=dest_type,
        operands=[source],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_CollapseShapeOp(
    *,
    result_type: MemRefType,
    src: Value,
    reassociation: ArrayAttr,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('reassociation', reassociation))
    return add_operation(
        name="memref.collapse_shape",
        result_type=result_type,
        operands=[src],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DeallocOp(
    *,
    memref: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.dealloc",
        result_type=None,
        operands=[memref],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DimOp(
    *,
    source: Value,
    index: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    return add_operation(
        name="memref.dim",
        result_type=result_type,
        operands=[source, index],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DmaStartOp(
    *,
    operands: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.dma_start",
        result_type=None,
        operands=list(operands),
        properties=all_props,
        attributes=extra_attributes,
    )


def add_DmaWaitOp(
    *,
    tagMemRef: Value,
    tagIndices: Sequence[Value],
    numElements: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    return add_operation(
        name="memref.dma_wait",
        result_type=None,
        operands=[tagMemRef, *tagIndices, numElements],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExpandShapeOp(
    *,
    result_type: MemRefType,
    src: Value,
    reassociation: ArrayAttr,
    output_shape: Sequence[Value],
    static_output_shape: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('reassociation', reassociation))
    all_props.append(('static_output_shape', DenseI64ArrayAttr(static_output_shape)))
    return add_operation(
        name="memref.expand_shape",
        result_type=result_type,
        operands=[src, *output_shape],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtractAlignedPointerAsIndexOp(
    *,
    source: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    aligned_pointer_type = IndexType()
    all_props = []
    return add_operation(
        name="memref.extract_aligned_pointer_as_index",
        result_type=aligned_pointer_type,
        operands=[source],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ExtractStridedMetadataOp(
    *,
    base_buffer_type: Type,
    offset_type: IndexType,
    sizes_types: Sequence[IndexType],
    strides_types: Sequence[IndexType],
    source: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> tuple[Value, Value, tuple[Value, ...], tuple[Value, ...]]:
    all_props = []
    return add_operation(
        name="memref.extract_strided_metadata",
        result_type=(base_buffer_type, offset_type, sizes_types, strides_types),
        operands=[source],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GetGlobalOp(
    *,
    result_type: MemRefType,
    name: str,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('name', FlatSymbolRefAttr(name)))
    return add_operation(
        name="memref.get_global",
        result_type=result_type,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_GlobalOp(
    *,
    sym_name: str,
    sym_visibility: Optional[str] = None,
    type: MemRefType,
    initial_value: Optional[Attribute] = None,
    constant: bool = False,
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('sym_name', StringAttr(value=sym_name)))
    if sym_visibility is not None:
        all_props.append(('sym_visibility', StringAttr(value=sym_visibility)))
    all_props.append(('type', TypeAttr(value=type)))
    if initial_value is not None:
        all_props.append(('initial_value', initial_value))
    if constant:
        all_props.append(('constant', UnitAttr()))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    return add_operation(
        name="memref.global",
        result_type=None,
        operands=[],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_MemorySpaceCastOp(
    *,
    dest_type: BaseMemRefType,
    source: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="memref.memory_space_cast",
        result_type=dest_type,
        operands=[source],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_PrefetchOp(
    *,
    memref: Value,
    indices: Sequence[Value],
    isWrite: bool,
    localityHint: int,
    isDataCache: bool,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    all_props.append(('isWrite', BoolAttr(value=isWrite)))
    all_props.append(('localityHint', IntegerAttr.make(IntegerType.signless(32), localityHint)))
    all_props.append(('isDataCache', BoolAttr(value=isDataCache)))
    return add_operation(
        name="memref.prefetch",
        result_type=None,
        operands=[memref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_RankOp(
    *,
    memref: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    result_type = IndexType()
    all_props = []
    return add_operation(
        name="memref.rank",
        result_type=result_type,
        operands=[memref],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReallocOp(
    *,
    result_type: MemRefType,
    source: Value,
    dynamicResultSize: Optional[Value] = None,
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    return add_operation(
        name="memref.realloc",
        result_type=result_type,
        operands=[source, *([] if dynamicResultSize is None else [dynamicResultSize])],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReinterpretCastOp(
    *,
    result_type: MemRefType,
    source: Value,
    offsets: Sequence[Value],
    sizes: Sequence[Value],
    strides: Sequence[Value],
    static_offsets: Sequence[int],
    static_sizes: Sequence[int],
    static_strides: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('static_offsets', DenseI64ArrayAttr(static_offsets)))
    all_props.append(('static_sizes', DenseI64ArrayAttr(static_sizes)))
    all_props.append(('static_strides', DenseI64ArrayAttr(static_strides)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(offsets), len(sizes), len(strides)])))
    return add_operation(
        name="memref.reinterpret_cast",
        result_type=result_type,
        operands=[source, *offsets, *sizes, *strides],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ReshapeOp(
    *,
    result_type: BaseMemRefType,
    source: Value,
    shape: Value,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="memref.reshape",
        result_type=result_type,
        operands=[source, shape],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_StoreOp(
    *,
    value: Value,
    memref: Value,
    indices: Sequence[Value],
    nontemporal: bool = False,
    alignment: Optional[int] = None,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> None:
    all_props = []
    if nontemporal:
        all_props.append(('nontemporal', BoolAttr(value=nontemporal)))
    if alignment is not None:
        all_props.append(('alignment', IntegerAttr.make(IntegerType.signless(64), alignment)))
    return add_operation(
        name="memref.store",
        result_type=None,
        operands=[value, memref, *indices],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_TransposeOp(
    *,
    result_type: MemRefType,
    in_: Value,
    permutation: AffineMap,
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('permutation', AffineMapAttr(value=permutation)))
    return add_operation(
        name="memref.transpose",
        result_type=result_type,
        operands=[in_],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_ViewOp(
    *,
    result_type: MemRefType,
    source: Value,
    byte_shift: Value,
    sizes: Sequence[Value],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    return add_operation(
        name="memref.view",
        result_type=result_type,
        operands=[source, byte_shift, *sizes],
        properties=all_props,
        attributes=extra_attributes,
    )


def add_SubViewOp(
    *,
    result_type: MemRefType,
    source: Value,
    offsets: Sequence[Value],
    sizes: Sequence[Value],
    strides: Sequence[Value],
    static_offsets: Sequence[int],
    static_sizes: Sequence[int],
    static_strides: Sequence[int],
    extra_attributes: Sequence[tuple[str, Attribute]] = (),
) -> Value:
    all_props = []
    all_props.append(('static_offsets', DenseI64ArrayAttr(static_offsets)))
    all_props.append(('static_sizes', DenseI64ArrayAttr(static_sizes)))
    all_props.append(('static_strides', DenseI64ArrayAttr(static_strides)))
    all_props.append(('operandSegmentSizes',
                      DenseI32ArrayAttr([1, len(offsets), len(sizes), len(strides)])))
    return add_operation(
        name="memref.subview",
        result_type=result_type,
        operands=[source, *offsets, *sizes, *strides],
        properties=all_props,
        attributes=extra_attributes,
    )
