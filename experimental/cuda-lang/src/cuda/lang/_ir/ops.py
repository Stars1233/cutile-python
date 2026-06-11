# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import math
import re
import operator
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from cuda.tile._ir.ops_utils import promote_types
from cuda.tile._memory_model import MemoryOrder, MemoryScope
from cuda.tile._ir.op_impl import (
    require_optional_constant_int,
    require_tuple_type,
    require_constant_str,
    require_dtype_spec,
    require_constant_int_tuple,
    require_constant_int,
    require_constant_enum,
    require_optional_constant_enum,
    require_array_type,
    WILDCARD,
    require_constant_bool,
    require_constant_pointer_info,
    ImplRegistry,
)
from cuda.tile._ir.type import TensorLikeTy
from cuda.tile._ir.core_ops import (
    TypedConst, core_impl_registry,
)
from cuda.tile._ir.arithmetic_ops import (
    binary_arithmetic_tensorlike,
    binary_arithmetic_tensorlike_raw,
    RawBinaryArithmeticOperation,
    RawComparisonOperation,
    RawBinaryBitwiseOperation,
    RawBitwiseShiftOperation,
    TileAsType,
    TileBroadcast,
    TileReshape,
    RawWhereOperation,
    Unary,
    arithmetic_impl_registry,
    promote_and_broadcast_to,
)
from cuda.tile._ir.static_eval_ops import static_eval_impl_registry
from cuda.tile._ir.ops import (
    AssumeBounded,
    AssumeDivBy,
    MakeTensorView,
    PointerOffset,
    TilePrintf,
    array_impl_registry,
)
from cuda.tile._ir.arithmetic_ops import (
    astype,
)
from cuda.tile._ir.core_ops import (
    Assign,
    bind_method,
    build_tuple,
    loosely_typed_const,
    strictly_typed_const,
)
from cuda.tile._ir.cast_ops import (
    implicit_cast, address_space_cast, reinterpret_pointer, AddrSpaceCast, ReinterpretPointer,
)
from cuda.tile._ir.control_flow_ops import (
    control_flow_impl_registry,
    IfElse,
    EndBranch,
    Loop,
    Continue,
    Break,
    Return,
    return_,
    MakeDummy,
)
from cuda.tile._ir.ir import MemoryEffect, make_aggregate, add_operation_variadic
from cuda.lang._exception import TileCompilerError, TileTypeError
import cuda.lang._datatype as datatype
from cuda.tile._datatype import (
    is_pointer_dtype,
    pointer_dtype,
    PointerInfo,
    opaque_pointer_dtype, int32, bool_, uint64,
)
from cuda.tile._exception import TileValueError
import cuda.lang._mlir as mlir
from .atomics_support import (
    ATOMIC_CAS_DTYPES,
    ATOMIC_XCHG_DTYPES,
    AtomicRMWKind,
    atomic_rmw_op_name,
    require_atomic_dtype,
    require_atomic_memory_order_and_scope,
    require_atomic_rmw_value,
)
from .type_checking_helpers import (
    require_array_indices,
    require_scalar_or_vector_float_type,
    require_scalar_or_vector_type,
    require_scalar_type,
    require_pointer_type,
    require_signed_int_scalar_or_tuple,
    require_clusterlaunchcontrol_token_type,
)

from .type import (
    LocalArrayContextManagerTy, ContextManagerState, TensorMapTy,
    dtype_to_tensor_map_type, ArrayValue, PointerInfoTy,
    MemorySpace,
    Type,
    ArrayTy,
    ScalarTy,
    PointerTy,
    VectorTy,
    TupleTy,
    TupleValue
)

from .ir import (
    Operation,
    Block,
    attribute,
    operand,
    Var,
    add_operation,
    format_var,
    LocalArrayContextManagerValue,
)
from .._stub import math as cl_math
from .._stub.cluster_launch_control import clusterlaunchcontrol_try_cancel, \
    clusterlaunchcontrol_is_canceled, clusterlaunchcontrol_get_first_block_idx
from .._stub.tensor_map import TensorMapSwizzle
from .._stub.mbarrier import MbarrierScope
from .._stub import foreign_function, core_api, mbarrier as mbarrier_stub, tensor_map
from .._stub.types import Pointer
from .._stub.tcgen05 import CTAGroup, Tcgen05LdStShape
from .._stub import tcgen05 as tcgen05_stub
from .._stub.core_api import Array
from cuda.tile._ir import hir_stubs

cuda_lang_impl_registry = ImplRegistry()
cuda_lang_impl_registry.update(core_impl_registry())
cuda_lang_impl_registry.update(static_eval_impl_registry())
cuda_lang_impl_registry.update(arithmetic_impl_registry())
cuda_lang_impl_registry.update(control_flow_impl_registry())
cuda_lang_impl_registry.update(array_impl_registry)
impl = cuda_lang_impl_registry.impl


@impl(core_api.dtype_of)
def dtype_of_impl(value: Var):
    ty = value.get_type()
    if isinstance(ty, ScalarTy):
        dtype = ty.dtype
    elif isinstance(ty, PointerTy):
        dtype = ty.pointer_dtype
    else:
        raise TileTypeError(f"dtype_of() expects a scalar or a pointer as the argument, got {ty}")
    return loosely_typed_const(dtype)


# -------------------------------------------------------------------------------------
# Pointer dtype APIs
# -------------------------------------------------------------------------------------

@impl(is_pointer_dtype)
def is_pointer_dtype_impl(dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    return loosely_typed_const(is_pointer_dtype(dtype))


@impl(pointer_dtype)
def pointer_dtype_impl(pointee_dtype: Var, memory_space: Var) -> Var:
    pointee_dtype = require_dtype_spec(pointee_dtype)
    memory_space = require_constant_enum(memory_space, MemorySpace)
    res = pointer_dtype(pointee_dtype, memory_space)
    return loosely_typed_const(res)


@impl(opaque_pointer_dtype)
def opaque_pointer_dtype_impl(memory_space: Var) -> Var:
    memory_space = require_constant_enum(memory_space, MemorySpace)
    res = opaque_pointer_dtype(memory_space)
    return loosely_typed_const(res)


@impl(PointerInfo)
def pointer_info_impl(dtype: Var) -> Var:
    dtype = require_dtype_spec(dtype)
    try:
        res = PointerInfo(dtype)
    except TypeError as e:
        raise TileTypeError(str(e))

    return loosely_typed_const(res)


@impl(getattr, overload=(PointerInfoTy, "opaque"))
def pointer_info_opaque_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    return loosely_typed_const(info.opaque)


@impl(getattr, overload=(PointerInfoTy, "pointee_dtype"))
def pointer_info_pointee_dtype_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    try:
        pointee_dtype = info.pointee_dtype
    except ValueError as e:
        raise TileTypeError(str(e))

    return loosely_typed_const(pointee_dtype)


@impl(getattr, overload=(PointerInfoTy, "memory_space"))
def pointer_info_memory_space_impl(object: Var, name: Var) -> Var:
    info = require_constant_pointer_info(object)
    return loosely_typed_const(info.memory_space)


# -------------------------------------------------------------------------------------


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
class VectorGetItem(Operation, opcode="vector_getitem", memory_effect=MemoryEffect.LOAD):
    x: Var = operand()
    index: Var = operand()


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


def require_pointer_in_memory_space(ptr_value, spaces: tuple[MemorySpace, ...]) -> PointerTy:
    ptr_type = require_pointer_type(ptr_value)
    if ptr_type.memory_space not in spaces:
        expected = ' or '.join(map(str, spaces))
        raise TileTypeError(
            f"Expected pointer memory space to be {expected} "
            f"but got {ptr_type.memory_space}"
        )
    return ptr_type


def require_pointer_memory_order(
    operation: type[LoadPointer] | type[StorePointer],
    ordering_var: Var,
):
    ordering = require_optional_constant_enum(ordering_var, MemoryOrder)
    if ordering in operation.valid_orderings:
        return ordering

    formatted_expected = ", ".join(
        "None" if order is None else str(order) for order in operation.valid_orderings
    )
    operation_name = "load" if operation is LoadPointer else "store"
    raise TileTypeError(
        f"Invalid memory order for Pointer.{operation_name}. "
        f"Got {ordering}, expected one of {formatted_expected}"
    )


def _array_base_pointer_type(array_ty: ArrayTy) -> PointerTy:
    return PointerTy(pointer_dtype(array_ty.dtype, array_ty.memory_space))


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
    zero = strictly_typed_const(0, ScalarTy(uint64))
    offset = zero
    if len(indices) != len(array_val.strides):
        raise TileTypeError(
            f"Expected {len(array_val.strides)} indices but got {len(indices)}"
        )
    for index, stride in zip(indices, array_val.strides, strict=True):
        index = astype(index, datatype.uint64)
        stride = astype(stride, datatype.uint64)
        scaled = binary_arithmetic_tensorlike_raw("mul", index, stride)
        offset = binary_arithmetic_tensorlike_raw("add", offset, scaled)
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


@impl(getattr, overload=(ArrayTy, "get_base_pointer"))
@impl(getattr, overload=(ArrayTy, "get_element_pointer"))
def getattr_array_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(Array, name)
    return bind_method(object, unbound_func)


@impl(Array.get_base_pointer)
def _m_array_get_base_pointer_impl(self: Var) -> Var:
    return _get_array_base_pointer(self)


@impl(Array.get_element_pointer)
def _m_array_get_element_pointer_impl(self: Var, indices: Var) -> Var:
    return _array_get_element_pointer(self, require_array_indices(self, indices))


@impl(operator.setitem, overload=(VectorTy, WILDCARD, WILDCARD))
def vector_setitem(object: Var[VectorTy], key: Var, value: Var):
    raise TileTypeError("Vectors are immutable: item assignment is not supported")


@impl(operator.getitem, overload=(VectorTy, WILDCARD))
def vector_getitem(object: Var[VectorTy], key: Var) -> Var:
    result_dtype = object.get_type().element_dtype
    index = implicit_cast(key, datatype.int32, "vector getitem index")
    return add_operation(
        VectorGetItem,
        ScalarTy(result_dtype),
        x=object,
        index=index,
    )


@impl(operator.getitem, overload=(ArrayTy, WILDCARD))
def array_getitem(object: Var, key: Var) -> Var:
    array_ty = require_array_type(object)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    return add_operation(
        LoadPointer,
        PointerTy(array_ty.dtype) if is_pointer_dtype(array_ty.dtype) else ScalarTy(array_ty.dtype),
        pointer=pointer,
        alignment=None,
        volatile=False,
    )


@impl(operator.setitem, overload=(ArrayTy, WILDCARD, WILDCARD))
def array_setitem(object: Var, key: Var, value: Var):
    array_ty = require_array_type(object)
    require_scalar_type(value)
    value = astype(value, array_ty.dtype)
    indices = require_array_indices(object, key)
    pointer = _array_get_element_pointer(object, indices)
    add_operation_variadic(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        alignment=None,
        volatile=False,
        ordering=None,
    )


def _atomic_rmw_dispatch(
    kind: AtomicRMWKind,
    ptr: Var,
    val: Var,
    memory_order: Var,
    memory_scope: Var,
) -> Var:
    op_name = atomic_rmw_op_name(kind)
    ptr_ty = require_pointer_type(ptr)
    val, result_ty = require_atomic_rmw_value(kind, ptr_ty, val)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        op_name, memory_order, memory_scope
    )
    return add_operation(
        AtomicRMW,
        result_ty,
        kind=kind,
        pointer=ptr,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


@dataclass(eq=False)
class AtomicRMW(Operation, opcode="atomic_rmw", memory_effect=MemoryEffect.STORE):
    kind: AtomicRMWKind = attribute()
    pointer: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class AtomicExchange(Operation, opcode="atomic_xchg", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class AtomicCAS(Operation, opcode="atomic_cas", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    compare: Var = operand()
    value: Var = operand()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@impl(core_api.atomic_add, fixed_args=[AtomicRMWKind.ADD])
@impl(core_api.atomic_sub, fixed_args=[AtomicRMWKind.SUB])
@impl(core_api.atomic_and, fixed_args=[AtomicRMWKind.AND])
@impl(core_api.atomic_or, fixed_args=[AtomicRMWKind.OR])
@impl(core_api.atomic_xor, fixed_args=[AtomicRMWKind.XOR])
@impl(core_api.atomic_min, fixed_args=[AtomicRMWKind.MIN])
@impl(core_api.atomic_max, fixed_args=[AtomicRMWKind.MAX])
@impl(core_api.atomic_inc, fixed_args=[AtomicRMWKind.INC])
@impl(core_api.atomic_dec, fixed_args=[AtomicRMWKind.DEC])
def atomic_rmw_dispatch_impl(
    kind: AtomicRMWKind,
    ptr: Var,
    val: Var,
    memory_order: Var,
    memory_scope: Var,
) -> Var:
    return _atomic_rmw_dispatch(kind, ptr, val, memory_order, memory_scope)


@impl(core_api.atomic_xchg)
def atomic_xchg_impl(
    ptr: Var, val: Var, memory_order: Var, memory_scope: Var
) -> Var:
    ptr_ty = require_pointer_type(ptr)
    dtype = ptr_ty.pointee_dtype
    require_atomic_dtype("atomic_xchg", dtype, ATOMIC_XCHG_DTYPES)
    require_scalar_type(val)
    val = astype(val, dtype)
    result_ty = ScalarTy(dtype)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        "atomic_xchg", memory_order, memory_scope
    )
    return add_operation(
        AtomicExchange,
        result_ty,
        pointer=ptr,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


@impl(core_api.atomic_cas)
def atomic_cas_impl(
    ptr: Var, old: Var, val: Var, memory_order: Var, memory_scope: Var
) -> Var:
    ptr_ty = require_pointer_type(ptr)
    dtype = ptr_ty.pointee_dtype
    require_atomic_dtype("atomic_cas", dtype, ATOMIC_CAS_DTYPES)
    require_scalar_type(val)
    val = astype(val, dtype)
    compare_ty = require_scalar_type(old)
    if dtype != compare_ty.dtype:
        raise TileTypeError(
            f"Expected atomic compare value of type {dtype}, got {compare_ty.dtype}"
        )
    result_ty = ScalarTy(dtype)
    memory_order, memory_scope = require_atomic_memory_order_and_scope(
        "atomic_cas", memory_order, memory_scope
    )
    return add_operation(
        AtomicCAS,
        result_ty,
        pointer=ptr,
        compare=old,
        value=val,
        memory_order=memory_order,
        memory_scope=memory_scope,
    )


def _pointer_load(
    pointer: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> Var:
    pointee_dtype = require_pointer_type(pointer).pointee_dtype
    count = require_optional_constant_int(count)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    ordering = require_pointer_memory_order(LoadPointer, ordering)
    if ordering not in (None, MemoryOrder.WEAK) and alignment is None:
        raise TileTypeError("Expected explicit alignment on atomic load")
    if count is None or count == 1:
        result_ty = ScalarTy(pointee_dtype)
    else:
        result_ty = VectorTy(pointee_dtype, count)
    return add_operation(
        LoadPointer,
        result_ty,
        pointer=pointer,
        volatile=volatile,
        alignment=alignment,
        ordering=ordering,
    )


def _pointer_store(
    pointer: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> None:
    pointer_ty = require_pointer_type(pointer)
    volatile = require_constant_bool(volatile)
    alignment = require_optional_alignment(alignment)
    ordering = require_pointer_memory_order(StorePointer, ordering)
    if ordering not in (None, MemoryOrder.WEAK) and alignment is None:
        raise TileTypeError("Expected explicit alignment on atomic store")

    pointee_dtype = pointer_ty.pointee_dtype
    value = implicit_cast(value, pointee_dtype,
                          "Stored value type is incompatible with pointer type")

    add_operation_variadic(
        StorePointer,
        (),
        pointer=pointer,
        value=value,
        volatile=volatile,
        alignment=alignment,
        ordering=ordering,
    )


def _pointer_with_offset(pointer: Var, offset: Var) -> Var:
    require_pointer_type(pointer)
    offset = astype(offset, datatype.int64)
    return add_operation(
        PointerOffset,
        pointer.get_type(),
        pointer=pointer,
        offset=offset,
    )


@impl(operator.add, overload=(TensorLikeTy, TensorLikeTy))
async def add_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if isinstance(yty, PointerTy):
        xty, yty = yty, xty
    if isinstance(xty, PointerTy):
        offset_dtype = require_scalar_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TileTypeError(f"Expected integer pointer offset, got {offset_dtype}")
        return _pointer_with_offset(x, y)
    return binary_arithmetic_tensorlike("add", x, y)


@impl(operator.sub, overload=(TensorLikeTy, TensorLikeTy))
async def sub_impl(x: Var, y: Var) -> Var:
    xty, yty = x.get_type(), y.get_type()
    if isinstance(xty, PointerTy):
        offset_dtype = require_scalar_type(y).dtype
        if not datatype.is_integral(offset_dtype):
            raise TileTypeError(f"Expected integer pointer offset, got {offset_dtype}")
        y = astype(y, datatype.int64)
        c0 = loosely_typed_const(0)
        offset = binary_arithmetic_tensorlike('sub', c0, y)
        return _pointer_with_offset(x, offset)
    if isinstance(yty, PointerTy):
        raise TileTypeError('It is invalid to subtract a pointer from an integer')
    return binary_arithmetic_tensorlike("sub", x, y)


@impl(core_api.address_space_cast)
def address_space_cast_impl(value: Var, memory_space: Var) -> Var:
    memory_space = require_constant_enum(memory_space, MemorySpace)
    return address_space_cast(value, memory_space)


@impl(core_api.reinterpret_pointer_as_array)
def reinterpret_pointer_as_array_impl(pointer: Var, dtype: Var, shape: Var, strides: Var) -> Var:
    if not strides.is_constant() or strides.get_constant() is not None:
        raise TileTypeError(
            "Reinterpreting a pointer as an array with "
            "non-default strides is not yet implemented."
        )
    pointer_ty = require_pointer_type(pointer)
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    strides = _contiguous_strides(shape)
    memory_space = pointer_ty.memory_space

    typed_pointer_ty = PointerTy(pointer_dtype(dtype, memory_space))
    if pointer.get_type() != typed_pointer_ty:
        pointer = reinterpret_pointer(pointer, typed_pointer_ty.pointer_dtype)
    index_dtype = datatype.int32
    array_ty = ArrayTy(
        dtype,
        shape=shape,
        strides=strides,
        typing_hooks=pointer.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=memory_space,
    )
    result = add_operation(
        ReinterpretPointerAsArray,
        array_ty,
        pointer=pointer,
    )
    # FIXME: it seems that the index dtype should be derived from the dtype of shape/strides instead
    size_ty = ScalarTy(index_dtype)
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)
    result.set_aggregate(ArrayValue(pointer, shape_vars, stride_vars))
    return result


@impl(getattr, overload=(VectorTy, "element_count"))
def vector_element_count_impl(object: Var[VectorTy], name: Var):
    return loosely_typed_const(object.get_type().length)


@impl(getattr, overload=(VectorTy, "dtype"))
def vector_dtype_impl(object: Var[VectorTy], name: Var):
    return loosely_typed_const(object.get_type().element_dtype)


@impl(getattr, overload=(PointerTy, "opaque"))
def pointer_opaque_impl(object: Var[PointerTy], name: Var):
    return loosely_typed_const(object.get_type().opaque)


@impl(getattr, overload=(PointerTy, "pointee_dtype"))
def pointer_pointee_dtype_impl(object: Var[PointerTy], name: Var):
    ty = object.get_type()
    if ty.opaque:
        raise TileTypeError("Opaque pointers have no pointee_dtype")
    return loosely_typed_const(ty.pointee_dtype)


@impl(getattr, overload=(PointerTy, "memory_space"))
def pointer_memory_space_impl(object: Var[PointerTy], name: Var):
    return loosely_typed_const(object.get_type().memory_space)


@impl(getattr, overload=(PointerTy, "load"))
@impl(getattr, overload=(PointerTy, "store"))
def getattr_pointer_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(Pointer, name)
    return bind_method(object, unbound_func)


@impl(Pointer.load)
def pointer_load_impl(
    self: Var,
    count: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> Var:
    return _pointer_load(self, count, alignment, volatile, ordering)


@impl(Pointer.store)
def pointer_store_impl(
    self: Var,
    value: Var,
    alignment: Var,
    volatile: Var,
    ordering: Var,
) -> None:
    _pointer_store(self, value, alignment, volatile, ordering)


@dataclass(eq=False)
class Branch(Operation, opcode="br", terminator=True):
    target: Block = attribute()
    args: tuple[Var, ...] = operand()

    def _to_string_rhs(self) -> str:
        return f"{self.op} ^{self.target._name}({', '.join(format_var(arg) for arg in self.args)})"


def branch(target: Block, args: tuple[Var, ...]) -> None:
    add_operation_variadic(Branch, (), target=target, args=args)


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
) -> None:
    add_operation_variadic(
        CondBranch,
        (),
        cond=cond,
        true_args=true_args,
        false_args=false_args,
        true_target=true_target,
        false_target=false_target,
    )


@dataclass(eq=False)
class AllocLocalMemory(Operation, opcode="alloc_local_memory", memory_effect=MemoryEffect.STORE):
    count: int = attribute()
    alignment: int | None = attribute()


@dataclass(eq=False)
class DeallocLocalMemory(Operation,
                         opcode="dealloc_local_memory",
                         memory_effect=MemoryEffect.STORE):
    ptr: Var = operand()


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def _dtype_byte_width(dtype: datatype.DType) -> int:
    assert dtype.bitwidth % 8 == 0
    return dtype.bitwidth // 8


def require_optional_alignment(alignment: Var) -> int | None:
    alignment = require_optional_constant_int(alignment)

    if alignment is None:
        return None

    if alignment <= 0 or alignment & (alignment - 1):
        raise TileTypeError("alignment must be a positive power of two")

    return alignment


@impl(core_api.local_array)
def local_array_impl(shape: Var, dtype: Var, alignment: Var) -> Var:
    shape = require_constant_int_tuple(shape, allow_single_int=True)
    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TileTypeError(f"Requested {alignment=} is less than {dtype_byte_width}")

    state = ContextManagerState()
    agg_ty = LocalArrayContextManagerTy(dtype, shape, alignment, state)
    agg_val = LocalArrayContextManagerValue()
    return make_aggregate(agg_val, agg_ty)


@impl(hir_stubs.enter_context, overload=(LocalArrayContextManagerTy,))
def enter_context_local_array_impl(manager: Var):
    mgr_ty = manager.get_type()
    assert isinstance(mgr_ty, LocalArrayContextManagerTy)

    dtype_byte_width = _dtype_byte_width(mgr_ty.dtype)
    if mgr_ty.alignment is not None and mgr_ty.alignment < dtype_byte_width:
        raise TileTypeError(f"Requested alignment {mgr_ty.alignment}"
                            f" is less than item size {dtype_byte_width}")
    strides = _contiguous_strides(mgr_ty.shape)
    index_dtype = datatype.int32
    array_type = ArrayTy(
        mgr_ty.dtype,
        shape=mgr_ty.shape,
        strides=strides,
        typing_hooks=manager.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=MemorySpace.GENERIC,
    )
    size_ty = ScalarTy(index_dtype)
    shape_vars = tuple(strictly_typed_const(extent, size_ty) for extent in mgr_ty.shape)
    stride_vars = tuple(strictly_typed_const(extent, size_ty) for extent in strides)

    base_ptr = add_operation(
        AllocLocalMemory,
        _array_base_pointer_type(array_type),
        count=math.prod(mgr_ty.shape),
        alignment=mgr_ty.alignment,
    )

    def exit_callback():
        add_operation_variadic(DeallocLocalMemory, (), ptr=base_ptr)

    mgr_ty.state.exit_callback = exit_callback

    array_val = ArrayValue(base_ptr, shape_vars, stride_vars)
    return make_aggregate(array_val, array_type)


@dataclass(eq=False)
class GetDynSharedMemoryBasePtr(Operation, opcode="get_dyn_shared_memory_base_ptr"):
    initial_alignment = 1024


def get_dyn_shared_memory_base_ptr():
    result_ty = PointerTy(pointer_dtype(datatype.uint8, MemorySpace.SHARED))
    return add_operation(GetDynSharedMemoryBasePtr, result_ty)


@dataclass(eq=False)
class AllocStaticSharedMemory(Operation, opcode="alloc_static_shared_memory",
                              memory_effect=MemoryEffect.STORE):
    count: int = attribute()
    alignment: int | None = attribute()


@dataclass(eq=False)
class AllocDynSharedMemory(Operation, opcode="alloc_dyn_shared_memory",
                           memory_effect=MemoryEffect.STORE):
    shape: tuple[Var, ...] = operand()
    alignment: int | None = attribute()


@impl(core_api.shared_array)
def shared_array_impl(shape: Var, dtype: Var, dynamic: Var, alignment: Var) -> Operation:
    dynamic = require_constant_bool(dynamic)

    sizes = require_signed_int_scalar_or_tuple(shape)

    dtype = require_dtype_spec(dtype)
    alignment = require_optional_alignment(alignment)
    dtype_byte_width = _dtype_byte_width(dtype)
    if alignment is not None and alignment < dtype_byte_width:
        raise TileTypeError(f"Requested {alignment=} is less than {dtype_byte_width=}")
    index_dtype = datatype.int32
    size_ty = ScalarTy(index_dtype)

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
            total_size_var = binary_arithmetic_tensorlike_raw("mul", total_size_var, size_var)
        else:
            total_size *= size
            total_size_var = strictly_typed_const(total_size, size_ty)

    ty_shape.reverse()
    shape_vars.reverse()
    ty_strides.reverse()
    stride_vars.reverse()

    array_type = ArrayTy(
        dtype,
        shape=tuple(ty_shape),
        strides=tuple(ty_strides),
        typing_hooks=shape.ctx.typing_hooks,
        index_dtype=index_dtype,
        memory_space=MemorySpace.SHARED,
    )

    if dynamic:
        base_ptr = add_operation(AllocDynSharedMemory,
                                 _array_base_pointer_type(array_type),
                                 shape=sizes,
                                 alignment=alignment)
    else:
        if total_size is None:
            raise TileTypeError("Shape must be constant when `dynamic` is False")

        base_ptr = add_operation(AllocStaticSharedMemory,
                                 _array_base_pointer_type(array_type),
                                 count=total_size,
                                 alignment=alignment)

    array_value = ArrayValue(base_ptr=base_ptr, shape=tuple(shape_vars), strides=tuple(stride_vars))
    return make_aggregate(array_value, array_type)


@dataclass(eq=False)
class SyncThreads(Operation, opcode="syncthreads", memory_effect=MemoryEffect.STORE):
    pass


@impl(core_api.syncthreads)
def syncthreads_impl() -> None:
    add_operation_variadic(SyncThreads, (),)


@impl(core_api.elect_sync)
def elect_sync_impl(membermask) -> Var:
    mask = require_constant_int(membermask)
    mask = strictly_typed_const(mask & 0xffffffff, ScalarTy(int32))

    _, is_elected = add_operation_variadic(RawNVVMIntrinsic,
                                           (ScalarTy(int32), ScalarTy(bool_)),
                                           intrinsic="llvm.nvvm.elect.sync",
                                           operands_=(mask,))
    return is_elected


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
        require_pointer_type(value)
        return InlinePTXOperand(mode=mode, type_code=type_char, value=value)

    actual_dtype = require_scalar_type(value).dtype
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


@impl(core_api.inline_ptx)
def inline_ptx_impl(ptx_code: Var, constraint_pairs: tuple) -> Var[TupleTy]:
    ptx_code = require_constant_str(ptx_code)
    mlir_ptx_code, ro_args, rw_args, wo_args = require_inline_ptx_constraint_pairs(
        ptx_code, constraint_pairs)
    result_types = tuple(PointerTy(dtype) if is_pointer_dtype(dtype) else ScalarTy(dtype)
                         for dtype in wo_args)
    results = add_operation_variadic(
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


@dataclass(eq=False)
class RawMLIROperation(Operation, opcode="mlir.operation",
                       memory_effect=MemoryEffect.STORE):
    op_name: str = attribute()
    operands_: tuple[Var, ...] = operand()
    mlir_attributes: tuple[tuple[str, mlir.Attribute], ...] = attribute(default=())


def _get_dtype(ty: ScalarTy | VectorTy):
    match ty:
        case ScalarTy() as st:
            return st.dtype
        case VectorTy() as vt:
            return vt.element_dtype
        case _:
            assert False, "Match should have been exhaustive"


@impl(cl_math.ceil, fixed_args=["math.ceil"])
@impl(cl_math.exp, fixed_args=["math.exp"])
@impl(cl_math.exp2, fixed_args=["math.exp2"])
@impl(cl_math.sin, fixed_args=["math.sin"])
@impl(cl_math.cos, fixed_args=["math.cos"])
@impl(cl_math.tan, fixed_args=["math.tan"])
@impl(cl_math.sinh, fixed_args=["math.sinh"])
@impl(cl_math.cosh, fixed_args=["math.cosh"])
@impl(cl_math.tanh, fixed_args=["math.tanh"])
@impl(cl_math.sqrt, fixed_args=["math.sqrt"])
@impl(cl_math.rsqrt, fixed_args=["math.rsqrt"])
@impl(cl_math.floor, fixed_args=["math.floor"])
@impl(cl_math.log, fixed_args=["math.log"])
@impl(cl_math.log2, fixed_args=["math.log2"])
def math_float_unary_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )


@impl(cl_math.isnormal)
def math_isnormal_impl(x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_unrestricted_float)
    match x_ty:
        case ScalarTy():
            res_ty = ScalarTy(datatype.bool_)
        case VectorTy(length=length):
            res_ty = VectorTy(datatype.bool_, length=length)
        case _:
            assert False
    # see https://llvm.org/docs/LangRef.html#llvm-is-fpclass-intrinsic
    mask = strictly_typed_const((1 << 3) | (1 << 8), ScalarTy(datatype.int32))
    return add_operation(
        RawNVVMIntrinsic,
        res_ty,
        intrinsic="llvm.is.fpclass",
        operands_=(x, mask),
    )


@impl(cl_math.isnan, fixed_args=["math.isnan"])
@impl(cl_math.isinf, fixed_args=["math.isinf"])
@impl(cl_math.isfinite, fixed_args=["math.isfinite"])
def math_float_fpclass_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_float)
    match x_ty:
        case ScalarTy():
            res_ty = ScalarTy(datatype.bool_)
        case VectorTy(length=length):
            res_ty = VectorTy(datatype.bool_, length=length)
        case _:
            assert False
    return add_operation(
        RawMLIROperation,
        res_ty,
        op_name=op_name,
        operands_=(x,),
    )


def common_type(x: Var, y: Var):
    x_ty = x.get_loose_type()
    y_ty = y.get_loose_type()

    if not datatype.is_arithmetic(x_ty.tensor_dtype()):
        raise TileTypeError(f"Left-hand side has non-arithmetic dtype {x_ty.tensor_dtype()}")
    if not datatype.is_arithmetic(y_ty.tensor_dtype()):
        raise TileTypeError(f"Right-hand side has non-arithmetic dtype {y_ty.tensor_dtype()}")

    return promote_types(x_ty, y_ty, x.ctx.typing_hooks)


@impl(cl_math.atan2, fixed_args=["math.atan2"])
def math_float_binary_impl(op_name: str, x: Var, y: Var):
    require_scalar_or_vector_float_type(x)
    require_scalar_or_vector_float_type(y)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    return add_operation(
        RawMLIROperation,
        ty,
        op_name=op_name,
        operands_=(x, y),
    )


@impl(cl_math.abs)
def abs_impl(x: Var) -> Var:
    x_ty = require_scalar_or_vector_type(x)
    x_dtype = _get_dtype(x_ty)
    if datatype.is_float(x_dtype):
        op_name = "math.absf"
    elif datatype.is_integral(x_dtype):
        # If it's unsigned, then the absolute value is the identity
        if not datatype.is_signed(x_dtype):
            return x
        op_name = "math.absi"
    else:
        raise TileTypeError(f"abs() expects an arithmetic scalar, got {x_ty}")
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )


def _is_none(var: Var):
    return var.is_constant() and var.get_constant() is None


_TCGEN05_LD_VALID_COUNTS_BY_SHAPE = {
    Tcgen05LdStShape.SHAPE_16X64B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X128B: (1, 2, 4, 8, 16, 32, 64),
    Tcgen05LdStShape.SHAPE_16X256B: (1, 2, 4, 8, 16, 32),
    Tcgen05LdStShape.SHAPE_32X32B: (1, 2, 4, 8, 16, 32, 64, 128),
    Tcgen05LdStShape.SHAPE_16X32BX2: (1, 2, 4, 8, 16, 32, 64, 128),
}

_TCGEN05_LD_REGISTERS_PER_COUNT = {
    Tcgen05LdStShape.SHAPE_16X64B: 1,
    Tcgen05LdStShape.SHAPE_16X128B: 2,
    Tcgen05LdStShape.SHAPE_16X256B: 4,
    Tcgen05LdStShape.SHAPE_32X32B: 1,
    Tcgen05LdStShape.SHAPE_16X32BX2: 1,
}


@impl(tcgen05_stub.tcgen05_alloc)
def tcgen05_alloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(
        addr, (MemorySpace.SHARED_CLUSTER, MemorySpace.SHARED)
    )
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group = require_constant_enum(cta_group, CTAGroup)
    intrinsic = "llvm.nvvm.tcgen05.alloc.shared." + cta_group.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_dealloc)
def tcgen05_dealloc_impl(
    addr: Var,
    ncols: Var,
    cta_group: Var,
) -> None:
    require_pointer_in_memory_space(addr, (MemorySpace.TENSOR,))
    ncols = implicit_cast(ncols, datatype.int32, "cast num columns to int32")
    cta_group = require_constant_enum(cta_group, CTAGroup)
    intrinsic = "llvm.nvvm.tcgen05.dealloc." + cta_group.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, ncols),
    )


@impl(tcgen05_stub.tcgen05_commit)
def tcgen05_commit_impl(
    mbar: Var,
    multicast_mask: Var,
    cta_group: Var,
):
    require_mbarrier_ptr(mbar)
    operands = [mbar]
    cta_group = require_constant_enum(cta_group, CTAGroup)
    intrinsic = "llvm.nvvm.tcgen05.commit"
    if not _is_none(multicast_mask):
        intrinsic += '.mc'
        mask = implicit_cast(multicast_mask, datatype.int16, "multicast mask")
        operands.append(mask)
    intrinsic += '.shared.' + cta_group.value
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )


@impl(tcgen05_stub.tcgen05_ld)
def tcgen05_ld_impl(
    shape: Var,
    tmem_addr: Var,
    count: Var,
    pack: Var,
    offset: Var,
) -> Var:
    require_pointer_in_memory_space(tmem_addr, (MemorySpace.TENSOR,))
    shape = require_constant_enum(shape, Tcgen05LdStShape)
    count = require_constant_int(count)
    valid_counts = _TCGEN05_LD_VALID_COUNTS_BY_SHAPE[shape]
    if count not in valid_counts:
        valid = ", ".join(str(value) for value in valid_counts)
        raise TileValueError(
            f"Expected count for {shape.name} to be one of {valid}, got {count}"
        )

    has_offset = not _is_none(offset)
    uses_offset = shape is Tcgen05LdStShape.SHAPE_16X32BX2
    if uses_offset and not has_offset:
        raise TileTypeError("tcgen05_ld with SHAPE_16X32BX2 requires offset")
    if has_offset and not uses_offset:
        raise TileTypeError("tcgen05_ld offset is only valid with SHAPE_16X32BX2")

    operands = [tmem_addr]
    if has_offset:
        require_scalar_type(offset)
        operands.append(astype(offset, datatype.int64))

    if _is_none(pack):
        operands.append(strictly_typed_const(False, ScalarTy(datatype.bool_)))
    else:
        require_scalar_type(pack, (datatype.bool_,))
        operands.append(pack)

    intrinsic = f"llvm.nvvm.tcgen05.ld.{shape.value}.x{count}"
    total_registers = count * _TCGEN05_LD_REGISTERS_PER_COUNT[shape]
    result_type = (ScalarTy(datatype.int32)
                   if total_registers == 1 else VectorTy(datatype.int32, total_registers))

    [result] = add_operation_variadic(
        RawNVVMIntrinsic,
        (result_type,),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )
    return result


def shfl_sync_impl(mode: str, mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    """
    Implements the instructions as the psuedocode in the NVVM IR spec.
    https://docs.nvidia.com/cuda/archive/12.3.1/nvvm-ir-spec/index.html#data-movement

    See also Clang's lowering in __clang_cuda_intrinsics.h.
    """
    value_ty = require_scalar_type(value, (datatype.int32, datatype.uint32, datatype.float32))
    require_scalar_type(mask, (datatype.int32, datatype.uint32))
    require_scalar_type(lane_mask, (datatype.int32, datatype.uint32))
    width = require_constant_int(width)
    if width not in (1, 2, 4, 8, 16, 32):
        raise TileTypeError(f"Expected shuffle width to be a power of two in [1, 32], got {width}")

    WARP_SIZE = 32
    clamp = 0 if mode == 'up' else 0x1F
    mask_and_clamp = strictly_typed_const(
        ((WARP_SIZE - width) << 8) | clamp,
        ScalarTy(int32),
    )

    suffix = "i32" if datatype.is_integral(value_ty.dtype) else "f32"
    intrinsic = f"llvm.nvvm.shfl.sync.{mode}.{suffix}"
    return add_operation(
        RawNVVMIntrinsic,
        value_ty,
        intrinsic=intrinsic,
        operands_=(mask, value, lane_mask, mask_and_clamp),
    )


@impl(core_api.shfl_sync)
def shfl_sync_idx_impl(mask: Var, value: Var, src_lane: Var, width: Var) -> Var:
    return shfl_sync_impl("idx", mask, value, src_lane, width)


@impl(core_api.shfl_up_sync)
def shfl_sync_up_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("up", mask, value, delta, width)


@impl(core_api.shfl_down_sync)
def shfl_sync_down_impl(mask: Var, value: Var, delta: Var, width: Var) -> Var:
    return shfl_sync_impl("down", mask, value, delta, width)


@impl(core_api.shfl_xor_sync)
def shfl_sync_xor_impl(mask: Var, value: Var, lane_mask: Var, width: Var) -> Var:
    return shfl_sync_impl("bfly", mask, value, lane_mask, width)


@impl(getattr, overload=(TensorMapTy, "as_opaque_ptr"))
def getattr_tensor_map_method(object: Var, name: Var):
    name = require_constant_str(name)
    unbound_func = getattr(tensor_map.TensorMap, name)
    return bind_method(object, unbound_func)


@dataclass(eq=False)
class CreateTensorMap(Operation, opcode="create_tensor_map"):
    base_ptr: Var = operand()
    array_shape: tuple[Var, ...] = operand()
    array_strides: tuple[Var, ...] = operand()


@impl(tensor_map.tensor_map_tiled)
def tensor_map_tiled_impl(array: Var, tile_shape: Var, swizzle: Var) -> Var:
    array_ty = require_array_type(array)
    array_val = array.get_aggregate()
    assert isinstance(array_val, ArrayValue)

    tile_shape = require_constant_int_tuple(tile_shape, allow_single_int=True)
    swizzle = require_constant_enum(swizzle, TensorMapSwizzle)
    data_type = dtype_to_tensor_map_type(array_ty.dtype)
    map_ty = TensorMapTy(data_type=data_type,
                         tile_shape=tile_shape,
                         swizzle=swizzle)
    return add_operation(CreateTensorMap, map_ty,
                         base_ptr=array_val.base_ptr,
                         array_shape=array_val.shape,
                         array_strides=array_val.strides)


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()


def require_tensor_map_ty(var: Var) -> TensorMapTy:
    ty = var.get_type()
    if not isinstance(ty, TensorMapTy):
        raise TileTypeError(f"Expected a tensor map, got {ty}")
    return ty


@impl(tensor_map.TensorMap.as_opaque_ptr)
def tensor_map_as_opaque_ptr_impl(self: Var):
    require_tensor_map_ty(self)
    result_ty = PointerTy(opaque_pointer_dtype())
    return add_operation(TensorMapAsOpaquePtr, result_ty, tensor_map=self)


def require_constant_result_dtype(dtype: Var) -> Type:
    if not dtype.is_constant():
        raise TileTypeError(f"Expected a dtype constructor but got {dtype}")

    const_dtype = dtype.get_constant()
    if isinstance(const_dtype, datatype.OpaquePointerSpec):
        if const_dtype == datatype.any_opaque_ptr:
            raise TileTypeError("Result type cannot have no memory space")
        memory_space = datatype.MemorySpace(const_dtype.value)
        return PointerTy(opaque_pointer_dtype(memory_space=memory_space))
    elif isinstance(const_dtype, datatype.DType):
        return PointerTy(const_dtype) if is_pointer_dtype(const_dtype) else ScalarTy(const_dtype)
    else:
        raise TileTypeError(f"Expected a type spec but got {dtype}")


def require_constant_result_dtypes(result_dtypes: Var) -> tuple[Type, ...]:
    require_tuple_type(result_dtypes)
    result_dtypes = result_dtypes.get_aggregate().items
    return tuple(require_constant_result_dtype(dtype) for dtype in result_dtypes)


@impl(clusterlaunchcontrol_try_cancel)
def clusterlaunchcontrol_try_cancel_impl(addr: Var, mbar: Var, multicast: Var) -> None:
    addr_info = PointerInfo(require_pointer_type(addr).pointer_dtype)
    mbar_info = PointerInfo(require_pointer_type(mbar).pointer_dtype)
    multicast = require_constant_bool(multicast)

    if (
        addr_info.opaque
        or addr_info.pointee_dtype is not datatype.clusterlaunchcontrol_token
        or addr_info.memory_space is not MemorySpace.SHARED
    ):
        raise TileTypeError(
            "Expected a pointer to a cluster launch control "
            f"token in shared memory, got {addr.get_type()}"
        )

    if (
        mbar_info.opaque
        or mbar_info.pointee_dtype is not datatype.mbarrier
        or mbar_info.memory_space is not MemorySpace.SHARED
    ):
        raise TileTypeError(
            f"Expected a pointer to an mbarrier in shared memory, got {mbar.get_type()}"
        )

    intrinsic = "llvm.nvvm.clusterlaunchcontrol.try_cancel.async"
    if multicast:
        intrinsic += ".multicast"
    intrinsic += ".shared"

    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(addr, mbar),
    )


@impl(clusterlaunchcontrol_is_canceled)
def clusterlaunchcontrol_is_canceled_impl(token: Var) -> Var:
    require_clusterlaunchcontrol_token_type(token)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic="llvm.nvvm.clusterlaunchcontrol.query_cancel.is_canceled",
        operands_=(token,),
    )


@impl(clusterlaunchcontrol_get_first_block_idx)
def clusterlaunchcontrol_get_first_block_idx_impl(token: Var, axis: Var) -> Var:
    require_clusterlaunchcontrol_token_type(token)
    if not axis.is_constant():
        raise TileTypeError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    axis = axis.get_constant()
    if type(axis) not in (int, None):
        raise TileTypeError(
            f"Expected axis to be constant int or None, but got {axis=}"
        )
    cta_ids = tuple(
        add_operation(
            RawNVVMIntrinsic,
            ScalarTy(datatype.int32),
            intrinsic=f"llvm.nvvm.clusterlaunchcontrol.query_cancel.get_first_ctaid.{dim}",
            operands_=(token,),
        )
        for dim in ("x", "y", "z")
    )
    return build_tuple(cta_ids) if axis is None else cta_ids[axis]


@dataclass(eq=False)
class ForeignFunction(Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@impl(foreign_function._call_foreign_function)
def _call_foreign_function_impl(func: Var, return_type: Var, parameters: Var):
    function_name = require_constant_str(func)
    require_tuple_type(parameters)
    parameters = parameters.get_aggregate().items
    if return_type.is_constant() and return_type.get_constant() is None:
        add_operation_variadic(
            ForeignFunction,
            (),
            function_name=function_name,
            operands_=parameters,
        )
        return None
    else:
        result_type = require_constant_result_dtype(return_type)
        return add_operation(
            ForeignFunction,
            result_type,
            function_name=function_name,
            operands_=parameters,
        )


def require_mbarrier_ptr(
    mbar: Var,
    spaces: tuple[MemorySpace, ...] = (
        MemorySpace.SHARED,
        MemorySpace.SHARED_CLUSTER,
    ),
) -> PointerTy:
    mbar_ptr_type = require_pointer_in_memory_space(mbar, spaces)
    if mbar_ptr_type.opaque or mbar_ptr_type.pointee_dtype is not datatype.mbarrier:
        raise TileTypeError(f"Expected a pointer to an mbarrier, got {mbar}")
    return mbar_ptr_type


@impl(mbarrier_stub.mbarrier_init)
def mbarrier_init_impl(mbar: Var, participants: Var) -> Var:
    require_mbarrier_ptr(mbar)
    participants = astype(participants, datatype.int32)
    add_operation_variadic(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.init.shared",
        operands_=(mbar, participants),
    )


@impl(mbarrier_stub.mbarrier_invalidate)
def mbarrier_invalidate_impl(mbar: Var) -> Var:
    require_mbarrier_ptr(mbar)
    add_operation_variadic(
        RawNVVMIntrinsic,
        tuple(),
        intrinsic="llvm.nvvm.mbarrier.inval.shared",
        operands_=(mbar,),
    )


def _mbar_space_scope_suffix(scope: MbarrierScope, space: MemorySpace) -> str:
    match space:
        case MemorySpace.SHARED:
            space_str = 'cta'
        case MemorySpace.SHARED_CLUSTER:
            space_str = 'cluster'
        case _:
            raise TileCompilerError(f"Unexpected {space=}")
    return (
        ".scope."
        + scope.value
        + ".space."
        + space_str
    )


def require_mbarrier_ordering(
    ordering_var: Var,
    valid_orderings: tuple[MemoryOrder, ...],
) -> MemoryOrder:
    ordering = require_constant_enum(ordering_var, MemoryOrder)
    if ordering not in valid_orderings:
        formatted = ", ".join(str(o) for o in valid_orderings)
        raise TileTypeError(
            f"Invalid mbarrier memory order {ordering}, expected one of {formatted}"
        )
    return ordering


ARRIVE_ORDERINGS = (MemoryOrder.RELEASE, MemoryOrder.RELAXED)
WAIT_ORDERINGS = (MemoryOrder.ACQUIRE, MemoryOrder.RELAXED)


@impl(mbarrier_stub.mbarrier_arrive)
def mbarrier_arrive_impl(
    mbar: Var,
    count: Var,
    drop: Var,
    scope: Var,
    ordering: Var,
) -> Var | None:
    count = astype(count, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, ARRIVE_ORDERINGS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += '.drop'
    if ordering is MemoryOrder.RELAXED:
        intrinsic += '.relaxed'
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (ScalarTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation_variadic(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, count),
    )
    return results[0] if return_type else None


@impl(mbarrier_stub.mbarrier_arrive_expect_tx)
def mbarrier_arrive_expect_tx_impl(
    mbar: Var,
    bytes: Var,
    drop: Var,
    scope: Var,
    ordering: Var,
) -> Var | None:
    bytes = astype(bytes, datatype.int32)
    drop = require_constant_bool(drop)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, ARRIVE_ORDERINGS)
    space = require_mbarrier_ptr(mbar).memory_space
    intrinsic = "llvm.nvvm.mbarrier.arrive"
    if drop:
        intrinsic += '.drop'
    intrinsic += '.expect.tx'
    if ordering is MemoryOrder.RELAXED:
        intrinsic += '.relaxed'
    intrinsic += _mbar_space_scope_suffix(scope, space)

    return_type = (ScalarTy(datatype.uint64),) if space is MemorySpace.SHARED else ()
    results = add_operation_variadic(
        RawNVVMIntrinsic,
        return_type,
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )
    return results[0] if return_type else None


@impl(mbarrier_stub.mbarrier_expect_tx)
def mbarrier_expect_tx_impl(mbar: Var, bytes: Var, scope: Var):
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.expect.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(mbarrier_stub.mbarrier_complete_tx)
def mbarrier_complete_tx_impl(mbar: Var, bytes: Var, scope: Var) -> Var:
    space = require_mbarrier_ptr(mbar).memory_space
    bytes = astype(bytes, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    intrinsic = "llvm.nvvm.mbarrier.complete.tx"
    intrinsic += _mbar_space_scope_suffix(scope, space)
    add_operation_variadic(
        RawNVVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(mbar, bytes),
    )


@impl(mbarrier_stub.mbarrier_test_wait)
def mbarrier_test_wait_impl(
    mbar: Var, state: Var, scope: Var, ordering: Var
) -> Var:
    scope = require_constant_enum(scope, MbarrierScope)
    state = astype(state, datatype.int64)
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=(mbar, state),
    )


@impl(mbarrier_stub.mbarrier_test_wait_parity)
def mbarrier_test_wait_parity_impl(
    mbar: Var, parity: Var, scope: Var, ordering: Var
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.test.wait.parity"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=(mbar, parity),
    )


@impl(mbarrier_stub.mbarrier_try_wait)
def mbarrier_try_wait_impl(
    mbar: Var,
    state: Var,
    time_hint: Var,
    scope: Var,
    ordering: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    state = astype(state, datatype.int64)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait"
    args = (mbar, state)
    if not _is_none(time_hint):
        intrinsic += ".tl"
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=args,
    )


@impl(mbarrier_stub.mbarrier_try_wait_parity)
def mbarrier_try_wait_parity_impl(
    mbar: Var,
    parity: Var,
    time_hint: Var,
    scope: Var,
    ordering: Var,
) -> Var:
    require_mbarrier_ptr(mbar, (MemorySpace.SHARED,))
    parity = astype(parity, datatype.int32)
    scope = require_constant_enum(scope, MbarrierScope)
    ordering = require_mbarrier_ordering(ordering, WAIT_ORDERINGS)
    intrinsic = "llvm.nvvm.mbarrier.try.wait.parity"
    args = (mbar, parity)
    if not _is_none(time_hint):
        time_hint = astype(time_hint, datatype.int32)
        args = (*args, time_hint)
        intrinsic += ".tl"
    if ordering is MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    intrinsic += _mbar_space_scope_suffix(scope, MemorySpace.SHARED)
    return add_operation(
        RawNVVMIntrinsic,
        ScalarTy(datatype.bool_),
        intrinsic=intrinsic,
        operands_=args,
    )


@impl(core_api.map_shared_to_cluster)
def map_shared_to_cluster_impl(ptr: Var, rank: Var):
    ptr_ty = require_pointer_type(ptr)
    rank = astype(rank, datatype.int32)
    require_pointer_in_memory_space(ptr, (MemorySpace.SHARED,))
    if ptr_ty.opaque:
        result_dtype = opaque_pointer_dtype(MemorySpace.SHARED_CLUSTER)
    else:
        result_dtype = pointer_dtype(ptr_ty.pointee_dtype, MemorySpace.SHARED_CLUSTER)
    result_ty = PointerTy(result_dtype)
    return add_operation(
        RawNVVMIntrinsic,
        result_ty,
        intrinsic="llvm.nvvm.mapa.shared.cluster",
        operands_=(ptr, rank),
    )


__all__ = (
    "AddrSpaceCast",
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
    "MakeDummy",
    "Return",
    "RawBinaryArithmeticOperation",
    "RawBinaryBitwiseOperation",
    "RawBitwiseShiftOperation",
    "RawComparisonOperation",
    "TileAsType",
    "TileReshape",
    "TileBroadcast",
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
