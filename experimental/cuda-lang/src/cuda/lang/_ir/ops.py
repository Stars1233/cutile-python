# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
import re
import operator
from dataclasses import dataclass
from enum import Enum, auto

from cuda.tile._ir.op_impl import (
    require_tuple_type,
    require_constant_str,
    require_dtype_spec,
    require_constant_int_tuple,
    require_constant_int,
    require_constant_enum,
    require_index_or_index_tuple_type,
    require_array_type,
    WILDCARD,
    require_tile_type, require_constant_bool,
)
from cuda.lang._ir.type import OpaquePointerTy
from cuda.tile._ir.ops import (
    tile_impl_registry,
    bind_method,
    build_tuple,
    strictly_typed_const,
    astype,
    raw_binary_arithmetic,
    unflatten_aggregates,
    Return,
    return_,
    Assign,
    AssumeBounded,
    AssumeDivBy,
    MakeTensorView,
    RawBinaryArithmeticOperation,
    RawComparisonOperation,
    RawBinaryBitwiseOperation,
    RawBitwiseShiftOperation,
    TileAsType,
    TypedConst,
    IfElse,
    RawWhereOperation,
    EndBranch,
    Loop,
    Continue,
    Break,
    PointerOffset,
    LoadPointer,
    StorePointer,
    TilePrintf,
    printf_impl,
    Unary,
)
from cuda.tile._ir.ir import ArrayValue, MemoryEffect, make_aggregate
from cuda.lang._exception import TileTypeError
import cuda.lang._datatype as datatype
import cuda.lang._mlir as mlir

from .. import _stub as stub

from .type import (
    MemorySpace,
    Type,
    make_tile_ty,
    ArrayTy,
    PointerTy,
    TileTy,
    TupleTy,
    TokenTy,
)

from .ir import (
    Operation,
    Block,
    attribute,
    operand,
    Var,
    add_operation,
    format_var,
    TupleValue,
)


cuda_lang_impl_registry = tile_impl_registry.clone()
impl = cuda_lang_impl_registry.impl
overload_dispatcher = cuda_lang_impl_registry.overload_dispatcher


@dataclass(eq=False)
class ArrayGetItem(Operation, opcode="array_getitem", memory_effect=MemoryEffect.LOAD):
    x: Var = operand()
    indices: tuple[Var, ...] = operand()


@dataclass(eq=False)
class ArraySetItem(Operation, opcode="array_setitem", memory_effect=MemoryEffect.STORE):
    x: Var = operand()
    indices: tuple[Var, ...] = operand()
    value: Var = operand()


@dataclass(eq=False)
class AddrSpaceCast(Operation, opcode="address_space_cast"):
    pointer: Var = operand()


@dataclass(eq=False)
class ReinterpretPointer(Operation, opcode="reinterpret_pointer"):
    pointer: Var = operand()


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


def require_array_indices(array: Var, indices: Var) -> tuple[Var, ...]:
    array_ty = require_array_type(array)
    key_ty = require_index_or_index_tuple_type(indices)
    if isinstance(key_ty, TileTy):
        if key_ty.ndim == 0:
            return (indices,)
        raise TileTypeError(
            "Cannot index an array with a tile, use a tuple of indices or a scalar instead"
        )
    if isinstance(key_ty, TupleTy):
        if len(key_ty.value_types) != array_ty.ndim:
            raise TileTypeError(
                f"Expected {array_ty.ndim} indices but got {len(key_ty.value_types)}"
            )
        tuple_value = indices.get_aggregate()
        assert isinstance(tuple_value, TupleValue)
        return tuple_value.items
    raise TileTypeError(
        f"Expected a tuple of indices or a single index, but got {indices.get_type()}"
    )


def require_matching_array_value_type(array: Var, value: Var) -> tuple[ArrayTy, TileTy]:
    array_ty = require_array_type(array)
    value_ty = require_tile_type(value)
    if array_ty.dtype != value_ty.dtype:
        raise TileTypeError(
            "Expected type of value to match element type of array"
            f", but got {array_ty.dtype=} != {value_ty.dtype=}"
        )
    return array_ty, value_ty


def _require_constant_memory_space(var: Var) -> MemorySpace:
    return require_constant_enum(var, MemorySpace)


def require_any_pointer_var(var: Var) -> TileTy:
    ty = require_tile_type(var)
    if ty.shape != () or not isinstance(ty.dtype, (PointerTy, OpaquePointerTy)):
        raise TileTypeError(f"Expected a scalar pointer, got {ty}")
    return ty


def _array_base_pointer_type(array_ty: ArrayTy) -> TileTy:
    return make_tile_ty(
        PointerTy(make_tile_ty(array_ty.dtype, ()), array_ty.memory_space), ()
    )


def _get_array_base_pointer(array: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)
    base_ptr = array_val.base_ptr
    expected_type = _array_base_pointer_type(array_ty)
    if base_ptr.get_type() != expected_type:
        raise TileTypeError(
            "Array base pointer type does not match expected type: "
            f"{expected_type=}, got {base_ptr.get_type()}"
        )
    return base_ptr


def _array_linear_offset(array: Var, indices: tuple[Var, ...]) -> Var:
    array_val = array.get_aggregate()
    zero = strictly_typed_const(0, make_tile_ty(datatype.uint64, ()))
    offset = zero
    if len(indices) != len(array_val.strides):
        raise TileTypeError(
            f"Expected {len(array_val.strides)} indices but got {len(indices)}"
        )
    for index, stride in zip(indices, array_val.strides, strict=True):
        index = astype(index, datatype.uint64)
        stride = astype(stride, datatype.uint64)
        scaled = raw_binary_arithmetic("mul", index, stride)
        offset = raw_binary_arithmetic("add", offset, scaled)
    return offset


def _array_get_element_pointer(array: Var, indices: tuple[Var, ...]) -> Var:
    base_pointer = _get_array_base_pointer(array)
    offset = _array_linear_offset(array, indices)
    return add_operation(
        PointerOffset,
        base_pointer.get_type(),
        pointer=base_pointer,
        offset=offset,
    )


def _reinterpret_pointer(pointer: Var, result_ty: TileTy) -> Var:
    require_any_pointer_var(pointer)
    return add_operation(
        ReinterpretPointer,
        result_ty,
        pointer=pointer,
    )


@impl(getattr, overload=(ArrayTy, "get_base_pointer"))
def array_get_base_pointer_impl(object: Var, name: Var):
    return bind_method(object, stub._m_array_get_base_pointer)


@impl(stub._m_array_get_base_pointer)
def _m_array_get_base_pointer_impl(array: Var) -> Var:
    return _get_array_base_pointer(array)


@impl(getattr, overload=(ArrayTy, "get_element_pointer"))
def array_get_element_pointer_impl(object: Var, name: Var):
    return bind_method(object, stub._m_array_get_element_pointer)


@impl(stub._m_array_get_element_pointer)
def _m_array_get_element_pointer_impl(array: Var, indices: Var) -> Var:
    return _array_get_element_pointer(array, require_array_indices(array, indices))


@impl(operator.getitem, overload=(ArrayTy, WILDCARD))
def array_getitem(object: Var, key: Var) -> Var:
    array_ty = require_array_type(object)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    [result, _token] = add_operation(
        LoadPointer,
        (make_tile_ty(array_ty.dtype, ()), TokenTy()),
        pointer=pointer,
        mask=None,
        padding_value=None,
        token=None,
        latency=None,
    )
    return result


@impl(operator.setitem, overload=(ArrayTy, WILDCARD, WILDCARD))
def array_setitem(object: Var, key: Var, value: Var):
    array_ty, value_ty = require_matching_array_value_type(object, value)
    indices = require_array_indices(object, key)
    if array_ty.dtype != value_ty.dtype:
        raise TileTypeError(
            f"Expected value of type {array_ty.dtype} on "
            f"right-hand side of assignment but got {value_ty.dtype}"
        )
    pointer = _array_get_element_pointer(object, indices)
    add_operation(
        StorePointer,
        (TokenTy(),),
        pointer=pointer,
        value=value,
        mask=None,
        token=None,
        latency=None,
    )


class AtomicRMWKind(Enum):
    ADD = auto()
    SUB = auto()
    AND = auto()
    OR = auto()
    XOR = auto()
    MIN = auto()
    MAX = auto()
    INC = auto()
    DEC = auto()


@dataclass(eq=False)
class AtomicRMW(Operation, opcode="atomic_rmw", memory_effect=MemoryEffect.STORE):
    kind: AtomicRMWKind = attribute()
    pointer: Var = operand()
    value: Var = operand()
    memory_order: int = attribute()


@dataclass(eq=False)
class AtomicExchange(Operation, opcode="atomic_exch", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    memory_order: int = attribute()


@dataclass(eq=False)
class AtomicCAS(Operation, opcode="atomic_cas", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    compare: Var = operand()
    value: Var = operand()
    success_memory_order: int = attribute()
    failure_memory_order: int = attribute()


@impl(stub.atomic_add, fixed_args=[AtomicRMWKind.ADD])
@impl(stub.atomic_sub, fixed_args=[AtomicRMWKind.SUB])
@impl(stub.atomic_and, fixed_args=[AtomicRMWKind.AND])
@impl(stub.atomic_or, fixed_args=[AtomicRMWKind.OR])
@impl(stub.atomic_xor, fixed_args=[AtomicRMWKind.XOR])
@impl(stub.atomic_min, fixed_args=[AtomicRMWKind.MIN])
@impl(stub.atomic_max, fixed_args=[AtomicRMWKind.MAX])
@impl(stub.atomic_inc, fixed_args=[AtomicRMWKind.INC])
@impl(stub.atomic_dec, fixed_args=[AtomicRMWKind.DEC])
def atomic_rmw_dispatch_impl(kind: AtomicRMWKind, A: Var, idx: Var, val: Var) -> Var:
    array_ty, _ = require_matching_array_value_type(A, val)
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = make_tile_ty(array_ty.dtype, ())
    memory_order = mlir.llvm.AtomicOrdering.acq_rel
    return add_operation(
        AtomicRMW,
        result_ty,
        kind=kind,
        pointer=pointer,
        value=val,
        memory_order=memory_order,
    )


@impl(stub.atomic_exch)
def atomic_exch_impl(A: Var, idx: Var, val: Var) -> Var:
    array_ty, _ = require_matching_array_value_type(A, val)
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = make_tile_ty(array_ty.dtype, ())
    memory_order = mlir.llvm.AtomicOrdering.acq_rel
    return add_operation(
        AtomicExchange,
        result_ty,
        pointer=pointer,
        value=val,
        memory_order=memory_order,
    )


@impl(stub.atomic_cas)
def atomic_cas_impl(A: Var, idx: Var, old: Var, val: Var) -> Var:
    array_ty, _ = require_matching_array_value_type(A, val)
    compare_ty = require_tile_type(old)
    if array_ty.dtype != compare_ty.dtype:
        raise TileTypeError(
            f"Expected atomic compare value of type {array_ty.dtype}, got {compare_ty.dtype}"
        )
    indices = require_array_indices(A, idx)
    pointer = _array_get_element_pointer(A, indices)
    result_ty = make_tile_ty(array_ty.dtype, ())
    success_memory_order = mlir.llvm.AtomicOrdering.acq_rel
    failure_memory_order = mlir.llvm.AtomicOrdering.monotonic
    return add_operation(
        AtomicCAS,
        result_ty,
        pointer=pointer,
        compare=old,
        value=val,
        success_memory_order=success_memory_order,
        failure_memory_order=failure_memory_order,
    )


def require_pointer_var(var: Var) -> TileTy:
    ty = require_tile_type(var)
    if ty.shape != () or not isinstance(ty.dtype, PointerTy):
        raise TileTypeError(f"Expected a scalar pointer, got {ty}")
    return ty


def _pointer_load(pointer: Var) -> Var:
    pointer_tile_ty = require_pointer_var(pointer)
    pointee = pointer_tile_ty.dtype.pointee_type
    assert isinstance(pointee, TileTy)
    assert pointee.shape == ()
    result_ty = TileTy(pointee.dtype, pointer_tile_ty.shape)
    [result, _token] = add_operation(
        LoadPointer,
        (result_ty, TokenTy()),
        pointer=pointer,
        mask=None,
        padding_value=None,
        token=None,
        latency=None,
    )
    return result


def _pointer_store(pointer: Var, value: Var) -> None:
    add_operation(
        StorePointer,
        (TokenTy(),),
        pointer=pointer,
        value=value,
        mask=None,
        token=None,
        latency=None,
    )


def _pointer_with_offset(pointer: Var, offset: Var) -> Var:
    require_pointer_var(pointer)
    offset = astype(offset, datatype.uint64)
    return add_operation(
        PointerOffset,
        pointer.get_type(),
        pointer=pointer,
        offset=offset,
    )


@impl(stub.address_space_cast)
def address_space_cast_impl(value: Var, memory_space: Var) -> Var:
    pointer_tile_ty = require_any_pointer_var(value)
    memory_space = _require_constant_memory_space(memory_space)
    match pointer_tile_ty.dtype:
        case PointerTy():
            result_ty = make_tile_ty(
                dataclasses.replace(pointer_tile_ty.dtype, memory_space=memory_space),
                (),
            )
        case OpaquePointerTy():
            result_ty = make_tile_ty(OpaquePointerTy(memory_space), ())
        case _:
            raise TileTypeError(f"Expected a pointer type, got {pointer_tile_ty}")
    return add_operation(
        AddrSpaceCast,
        result_ty,
        pointer=value,
    )


@impl(stub.reinterpret_pointer_as_array)
def reinterpret_pointer_as_array_impl(pointer: Var, dtype: Var, shape: Var, strides: Var) -> Var:
    if not strides.is_constant() or strides.get_constant() is not None:
        raise TileTypeError(
            "Reinterpreting a pointer as an array with "
            "non-default strides is not yet implemented."
        )
    pointer_tile_ty = require_any_pointer_var(pointer)
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    strides = _contiguous_strides(shape)
    element_ty = make_tile_ty(dtype, ())
    memory_space = pointer_tile_ty.dtype.memory_space

    typed_pointer_ty = TileTy(PointerTy(element_ty, memory_space=memory_space), ())
    if pointer.get_type() != typed_pointer_ty:
        pointer = _reinterpret_pointer(pointer, typed_pointer_ty)
    index_dtype = datatype.int32
    array_ty = ArrayTy(
        element_ty,
        shape=shape,
        strides=strides,
        index_dtype=index_dtype,
        memory_space=memory_space,
    )
    result = add_operation(
        ReinterpretPointerAsArray,
        array_ty,
        pointer=pointer,
    )
    # FIXME: it seems that the index dtype should be derived from the dtype of shape/strides instead
    size_ty = TileTy(index_dtype, ())
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)
    result.set_aggregate(ArrayValue(pointer, shape_vars, stride_vars))
    return result


@impl(getattr, overload=(TileTy, "load"))
def pointer_load_impl(object: Var, name: Var):
    require_pointer_var(object)
    return bind_method(object, stub._m_pointer_load)


@impl(getattr, overload=(TileTy, "store"))
def pointer_store_impl(object: Var, name: Var):
    require_pointer_var(object)
    return bind_method(object, stub._m_pointer_store)


@impl(getattr, overload=(TileTy, "load_offset"))
def pointer_load_offset_impl(object: Var, name: Var):
    require_pointer_var(object)
    return bind_method(object, stub._m_pointer_load_offset)


@impl(getattr, overload=(TileTy, "store_offset"))
def pointer_store_offset_impl(object: Var, name: Var):
    require_pointer_var(object)
    return bind_method(object, stub._m_pointer_store_offset)


@impl(stub._m_pointer_load)
def _m_pointer_load_impl(pointer: Var) -> Operation:
    return _pointer_load(pointer)


@impl(stub._m_pointer_store)
def _m_pointer_store_impl(pointer: Var, value: Var) -> Operation:
    _pointer_store(pointer, value)
    return None


@impl(stub._m_pointer_load_offset)
def _m_pointer_load_offset_impl(pointer: Var, offset: Var) -> Operation:
    return _pointer_load(_pointer_with_offset(pointer, offset))


@impl(stub._m_pointer_store_offset)
def _m_pointer_store_offset_impl(pointer: Var, offset: Var, value: Var) -> Operation:
    _pointer_store(_pointer_with_offset(pointer, offset), value)
    return None


@dataclass(eq=False)
class Branch(Operation, opcode="br", terminator=True):
    target: Block = attribute()
    args: tuple[Var, ...] = operand()

    def _to_string_rhs(self) -> str:
        return f"{self.op} ^{self.target._name}({', '.join(format_var(arg) for arg in self.args)})"


def branch(target: Block, args: tuple[Var, ...]) -> Branch:
    return add_operation(Branch, (), target=target, args=args)


@dataclass(eq=False)
class CondBranch(Operation, opcode="cond_br", terminator=True):
    cond: Var = operand()
    true_args: tuple[Var, ...] = operand()
    false_args: tuple[Var, ...] = operand()
    true_target: Block = attribute()
    false_target: Block = attribute()

    def _to_string_rhs(self) -> str:
        formatted = f"{self.op} {format_var(self.cond)}"

        formatted += " ^" + self.true_target._name
        formatted += f"({', '.join(format_var(arg) for arg in self.true_args)})"

        formatted += " ^" + self.false_target._name
        formatted += f"({', '.join(format_var(arg) for arg in self.false_args)})"

        return formatted


def cond_branch(
    cond: Var,
    true_args: tuple[Var, ...],
    false_args: tuple[Var, ...],
    true_target: Block,
    false_target: Block,
) -> CondBranch:
    return add_operation(
        CondBranch,
        (),
        cond=cond,
        true_args=true_args,
        false_args=false_args,
        true_target=true_target,
        false_target=false_target,
    )


@dataclass(eq=False)
class MemoryAllocation(Operation, opcode="alloc", memory_effect=MemoryEffect.STORE):
    memory_space: MemorySpace = attribute()
    shape: tuple[int, ...] = attribute()
    element_type: datatype.DType = attribute()


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


@impl(stub.local_array)
def local_array_impl(shape: Var, dtype: Var) -> Operation:
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    strides = _contiguous_strides(shape)
    index_dtype = datatype.int32
    array_type = ArrayTy(
        TileTy(dtype, ()),
        shape=shape,
        strides=strides,
        index_dtype=index_dtype,
        memory_space=MemorySpace.LOCAL,
    )
    base_ptr = add_operation(
        MemoryAllocation,
        _array_base_pointer_type(array_type),
        memory_space=MemorySpace.LOCAL,
        shape=shape,
        element_type=dtype,
    )
    size_ty = TileTy(index_dtype, ())
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)
    [ret] = unflatten_aggregates(
        (base_ptr,) + shape_vars + stride_vars, (array_type,), (array_type,)
    )
    return ret


@dataclass(eq=False)
class GetDynSharedMemoryBasePtr(Operation, opcode="get_dyn_shared_memory_base_ptr"):
    pass


def get_dyn_shared_memory_base_ptr():
    u8_ty = TileTy(datatype.uint8, ())
    result_ty = TileTy(PointerTy(u8_ty, MemorySpace.SHARED), ())
    return add_operation(GetDynSharedMemoryBasePtr, result_ty)


@dataclass(eq=False)
class AllocDynSharedMemory(Operation, opcode="alloc_dyn_shared_memory",
                           memory_effect=MemoryEffect.STORE):
    shape: tuple[Var, ...] = operand()


@impl(stub.shared_array)
def shared_array_impl(shape: Var, dtype: Var, dynamic: Var) -> Operation:
    dynamic = require_constant_bool(dynamic)

    shape_ty = require_index_or_index_tuple_type(shape)
    if isinstance(shape_ty, TileTy):
        sizes = (shape,)
    else:
        tuple_val = shape.get_aggregate()
        assert isinstance(tuple_val, TupleValue)
        sizes = tuple_val.items

    dtype = require_dtype_spec(dtype)
    index_dtype = datatype.int32
    size_ty = TileTy(index_dtype, ())

    ty_strides = []
    ty_shape = []
    total_size = 1
    total_size_var = strictly_typed_const(total_size, size_ty)
    shape_vars = []
    stride_vars = []
    for size_var in reversed(sizes):
        if size_var.is_constant():
            size = size_var.get_constant()
            size_var = strictly_typed_const(size, size_ty)
        else:
            if size_var.get_type().dtype != index_dtype:
                # TODO: allow implicit cast?
                raise TileTypeError(f"Shared memory size must be {index_dtype},"
                                    f" got {size_var.get_type().dtype}")
            size = None
        ty_shape.append(size)
        shape_vars.append(size_var)
        ty_strides.append(total_size)
        stride_vars.append(total_size_var)

        if size is None or total_size is None:
            total_size = None
            total_size_var = raw_binary_arithmetic("mul", total_size_var, size_var)
        else:
            total_size *= size
            total_size_var = strictly_typed_const(total_size, size_ty)

    ty_shape.reverse()
    shape_vars.reverse()
    ty_strides.reverse()
    stride_vars.reverse()

    array_type = ArrayTy(
        TileTy(dtype, ()),
        shape=tuple(ty_shape),
        strides=tuple(ty_strides),
        index_dtype=index_dtype,
        memory_space=MemorySpace.SHARED,
    )

    if dynamic:
        base_ptr = add_operation(AllocDynSharedMemory,
                                 _array_base_pointer_type(array_type),
                                 shape=sizes)
    else:
        if total_size is None:
            raise TileTypeError("Shape must be constant when `dynamic` is False")

        base_ptr = add_operation(
            MemoryAllocation,
            _array_base_pointer_type(array_type),
            memory_space=MemorySpace.SHARED,
            shape=ty_shape,
            element_type=dtype,
        )

    array_value = ArrayValue(base_ptr=base_ptr, shape=tuple(shape_vars), strides=tuple(stride_vars))
    return make_aggregate(array_value, array_type)


@dataclass(eq=False)
class SyncThreads(Operation, opcode="syncthreads", memory_effect=MemoryEffect.STORE):
    pass


@impl(stub.syncthreads)
def syncthreads_impl() -> Operation:
    return add_operation(SyncThreads, None,)


impl(stub.printf)(printf_impl)


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    ptx_code: str = attribute()
    read_only_operands: tuple[Var, ...] = operand()
    write_only_operands: tuple[datatype.DType, ...] = attribute()
    read_write_operands: tuple[Var, ...] = operand()

    class RMWMode(Enum):
        READ_ONLY = auto()
        WRITE_ONLY = auto()
        READ_WRITE = auto()


@dataclass(eq=False, frozen=True)
class InlinePTXOperand:
    mode: InlinePTX.RMWMode
    type_code: str
    value: Var | datatype.DType


def require_inline_ptx_pair(var: Var) -> tuple[Var, Var]:
    pair_ty = var.get_type()
    if not isinstance(pair_ty, TupleTy) or len(pair_ty.value_types) != 2:
        raise TileTypeError(
            "Expected constraint arguments to be pairs of constraint strings and values"
        )
    pair_val = var.get_aggregate()
    assert isinstance(pair_val, TupleValue)
    return pair_val.as_tuple()


_INLINE_PTX_MODE_FROM_PREFIX = {
    "": InlinePTX.RMWMode.READ_ONLY,
    "=": InlinePTX.RMWMode.WRITE_ONLY,
    "+": InlinePTX.RMWMode.READ_WRITE,
}

_INLINE_PTX_TYPECODES = {
    "h",
    "r",
    "l",
    "f",
    "d",
    "C",
}

_INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE = {
    "h": datatype.int16,
    "r": datatype.int32,
    "l": datatype.int64,
    "f": datatype.float32,
    "d": datatype.float64,
}


def parse_inline_ptx_constraint(var: Var) -> tuple[str, InlinePTX.RMWMode, str]:
    constraint_str = require_constant_str(var)

    if len(constraint_str) not in (1, 2):
        raise TileTypeError(
            f"Invalid inline_ptx constraint {constraint_str}, expected length 1 or 2"
        )

    prefix = constraint_str[0:-1]
    type_char = constraint_str[-1]

    mode = _INLINE_PTX_MODE_FROM_PREFIX.get(prefix)
    if mode is None:
        raise TileTypeError(
            f"Unknown constraint rmw modifier {prefix!r}, expected "
            "'' (meaning readonly), '+' (meaning readwrite), or '=' (meaning writeonly)"
        )

    if type_char not in _INLINE_PTX_TYPECODES:
        expected = ", ".join(_INLINE_PTX_TYPECODES)
        raise TileTypeError(
            f"Unknown constraint dtype {type_char!r}, expected one of {expected}"
        )

    return constraint_str, mode, type_char


def validate_inline_ptx_operand(
    constraint_str: str, mode: InlinePTX.RMWMode, type_char: str, value: Var
) -> InlinePTXOperand:
    if mode is InlinePTX.RMWMode.WRITE_ONLY:
        if type_char == "C":
            # write-only arguments require specifying the output data type, but we don't
            # expose a dtype for pointers. Disallow this for now.
            raise TileTypeError("Write-only pointer outputs are not supported for inline_ptx")

        actual_dtype = require_dtype_spec(value)
        expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
        if actual_dtype != expected_dtype:
            raise TileTypeError(
                f"Expected dtype {expected_dtype} for constraint "
                f"{constraint_str}, got {actual_dtype}"
            )
        return InlinePTXOperand(mode=mode, type_code=type_char, value=actual_dtype)

    if type_char == "C":
        require_pointer_var(value)
        return InlinePTXOperand(mode=mode, type_code=type_char, value=value)

    value_ty = require_tile_type(value)
    if value_ty.shape != ():
        raise TileTypeError(
            f"Expected a scalar value for constraint {constraint_str!r}, but got {value_ty}"
        )

    actual_dtype = value_ty.dtype
    expected_dtype = _INLINE_PTX_SCALAR_DTYPE_FROM_TYPECODE[type_char]
    if actual_dtype != expected_dtype:
        raise TileTypeError(
            f"Expected value of type {expected_dtype} for "
            f"constraint {constraint_str}, got {actual_dtype}"
        )

    return InlinePTXOperand(mode=mode, type_code=type_char, value=value)


def require_constant_constraint_tuple(
    constraint_tuple: Var,
) -> InlinePTXOperand:
    constraint_var, value_var = require_inline_ptx_pair(constraint_tuple)
    constraint_str, mode, type_char = parse_inline_ptx_constraint(constraint_var)
    return validate_inline_ptx_operand(constraint_str, mode, type_char, value_var)


_INLINE_PTX_PLACEHOLDER_RE = re.compile(r"%(?P<index>[0-9]+)")


def require_inline_ptx_constraint_pairs(ptx_code: str, constraint_pairs: tuple) -> tuple:
    if not isinstance(constraint_pairs, tuple):
        raise TileTypeError(
            f"Expected a tuple of constraint pairs, but got {type(constraint_pairs)}"
        )

    ro_args, rw_args, wo_args = [], [], []
    # need to replace e.g. %0 with {$r0}, {$rw0}, or {$w0} for all ptx
    # interpolation directives.
    ptx_interpolation_replacements = []
    arg_specs = [require_constant_constraint_tuple(pair) for pair in constraint_pairs]

    for arg_spec in arg_specs:
        match arg_spec.mode:
            case InlinePTX.RMWMode.READ_ONLY:
                ptx_interpolation_replacements.append('{$r' + str(len(ro_args)) + '}')
                assert isinstance(arg_spec.value, Var)
                ro_args.append(arg_spec.value)
            case InlinePTX.RMWMode.READ_WRITE:
                ptx_interpolation_replacements.append('{$rw' + str(len(rw_args)) + '}')
                assert isinstance(arg_spec.value, Var)
                rw_args.append(arg_spec.value)
            case InlinePTX.RMWMode.WRITE_ONLY:
                ptx_interpolation_replacements.append('{$w' + str(len(wo_args)) + '}')
                assert isinstance(arg_spec.value, datatype.DType)
                wo_args.append(arg_spec.value)

    def rewrite(match: re.Match[str]) -> str:
        index = int(match.group("index"))
        if index >= len(ptx_interpolation_replacements):
            raise TileTypeError(
                f"inline_ptx placeholder %{index} is out of range "
                f"for {len(ptx_interpolation_replacements)} operands"
            )

        return ptx_interpolation_replacements[index]

    mlir_ptx_code = _INLINE_PTX_PLACEHOLDER_RE.sub(rewrite, ptx_code)
    return (
        mlir_ptx_code,
        tuple(ro_args),
        tuple(rw_args),
        tuple(wo_args),
    )


@impl(stub.inline_ptx)
def inline_ptx_impl(ptx_code: Var, constraint_pairs: tuple) -> tuple:
    ptx_code = require_constant_str(ptx_code)
    mlir_ptx_code, ro_args, rw_args, wo_args = require_inline_ptx_constraint_pairs(
        ptx_code, constraint_pairs)
    result_types = tuple(make_tile_ty(dtype, ()) for dtype in wo_args)
    results = add_operation(
        InlinePTX,
        result_types,
        ptx_code=mlir_ptx_code,
        read_only_operands=ro_args,
        write_only_operands=wo_args,
        read_write_operands=rw_args,
    )
    return build_tuple(results)


@dataclass(eq=False)
class RawNVVMIntrinsic(Operation, opcode="nvvm.call_intrinsic",
                       memory_effect=MemoryEffect.STORE):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


def require_scalar_tile_type(value: Var, valid_dtypes: tuple[datatype.DType, ...] = ()) -> TileTy:
    value_ty = require_tile_type(value)
    if value_ty.ndim != 0:
        raise TileTypeError(f"Expected scalar value, got {value_ty}")
    if valid_dtypes and value_ty.dtype not in valid_dtypes:
        raise TileTypeError(f"Expected type to be one of {valid_dtypes}, but got {value_ty.dtype}")
    return value_ty


def _require_nvvm_intrinsic_name(intrinsic: str) -> str:
    if not intrinsic.startswith("llvm."):
        raise TileTypeError(f"Expected intrinsic name to start with 'llvm.', but got {intrinsic!r}")
    return intrinsic


def shfl_sync_impl(mode: str, mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    """
    Implements the instructions as the psuedocode in the NVVM IR spec.
    https://docs.nvidia.com/cuda/archive/12.3.1/nvvm-ir-spec/index.html#data-movement

    See also Clang's lowering in __clang_cuda_intrinsics.h.
    """
    value_ty = require_scalar_tile_type(
        value,
        (datatype.int32, datatype.float32),
    )
    require_scalar_tile_type(mask, (datatype.int32,))
    require_scalar_tile_type(lane_mask, (datatype.int32,))
    width = require_constant_int(width)
    if width not in (1, 2, 4, 8, 16, 32):
        raise TileTypeError(f"Expected shuffle width to be a power of two in [1, 32], got {width}")

    WARP_SIZE = 32
    clamp = 0 if mode == 'up' else 0x1F
    mask_and_clamp = strictly_typed_const(
        ((WARP_SIZE - width) << 8) | clamp,
        make_tile_ty(datatype.int32, ()),
    )

    suffix = "i32" if datatype.is_integral(value_ty.dtype) else "f32"
    intrinsic = f"llvm.nvvm.shfl.sync.{mode}.{suffix}"
    return add_operation(
        RawNVVMIntrinsic,
        value_ty,
        intrinsic=intrinsic,
        operands_=(mask, value, lane_mask, mask_and_clamp),
    )


@impl(stub.shfl_sync)
def shfl_sync_idx_impl(mask: Var, value: Var, src_lane: Var, width: Var) -> Var:
    return shfl_sync_impl("idx", mask, value, src_lane, width)


@impl(stub.shfl_up_sync)
def shfl_sync_up_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("up", mask, value, delta, width)


@impl(stub.shfl_down_sync)
def shfl_sync_down_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("down", mask, value, delta, width)


@impl(stub.shfl_xor_sync)
def shfl_sync_xor_impl(mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    return shfl_sync_impl("bfly", mask, value, lane_mask, width)


def require_constant_result_dtype(dtype: Var) -> Type:
    if not dtype.is_constant():
        raise TileTypeError(f"Expected a dtype constructor but got {dtype}")

    const_dtype = dtype.get_constant()
    if isinstance(const_dtype, datatype.OpaquePointerSpec):
        if const_dtype == datatype.any_opaque_ptr:
            raise TileTypeError("Result type cannot have no memory space")
        memory_space = datatype.MemorySpace(const_dtype.value)
        return make_tile_ty(OpaquePointerTy(memory_space=memory_space), ())
    elif isinstance(const_dtype, datatype.VectorTy):
        return const_dtype
    elif isinstance(const_dtype, datatype.DType):
        return make_tile_ty(const_dtype, ())
    else:
        raise TileTypeError(f"Expected a type spec but got {dtype}")


def require_constant_result_dtypes(result_dtypes: Var) -> tuple[Type, ...]:
    require_tuple_type(result_dtypes)
    result_dtypes = result_dtypes.get_aggregate().items
    return tuple(require_constant_result_dtype(dtype) for dtype in result_dtypes)


@impl(stub.nvvm._raw_nvvm_intrinsic)
def _raw_nvvm_intrinsic_impl(intrinsic: Var, result_dtypes: Var, operands: Var) -> Operation:
    intrinsic = require_constant_str(intrinsic)
    intrinsic = _require_nvvm_intrinsic_name(intrinsic)
    require_tuple_type(operands)
    operand_items = operands.get_aggregate().items
    result_types = require_constant_result_dtypes(result_dtypes)

    results = add_operation(
        RawNVVMIntrinsic,
        result_types,
        intrinsic=intrinsic,
        operands_=operand_items,
    )

    match len(result_types):
        case 0:
            return None
        case 1:
            return results[0]
        case _:
            return build_tuple(results)


@dataclass(eq=False)
class ForeignFunction(Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@impl(stub.foreign_function._call_foreign_function)
def _call_foreign_function_impl(func: Var, return_type: Var, parameters: Var) -> Operation:
    function_name = require_constant_str(func)
    if return_type.is_constant() and return_type.get_constant() is None:
        result_type = tuple()
    else:
        result_type = require_constant_result_dtype(return_type)
    require_tuple_type(parameters)
    parameters = parameters.get_aggregate().items
    result = add_operation(
        ForeignFunction,
        result_type,
        function_name=function_name,
        operands_=parameters,
    )
    return result if result_type else None


__all__ = (
    "AddrSpaceCast",
    "ArrayGetItem",
    "ArraySetItem",
    "AtomicCAS",
    "AtomicExchange",
    "AtomicRMW",
    "AtomicRMWKind",
    "Assign",
    "AssumeBounded",
    "AssumeDivBy",
    "Branch",
    "CondBranch",
    "MakeTensorView",
    "Return",
    "RawBinaryArithmeticOperation",
    "RawBinaryBitwiseOperation",
    "RawBitwiseShiftOperation",
    "RawComparisonOperation",
    "TileAsType",
    "TypedConst",
    "branch",
    "cond_branch",
    "return_",
    "IfElse",
    "EndBranch",
    "Loop",
    "Continue",
    "Break",
    "TilePrintf",
    "PointerOffset",
    "LoadPointer",
    "ReinterpretPointer",
    "ReinterpretPointerAsArray",
    "StorePointer",
    "RawWhereOperation",
    "Unary",
)
