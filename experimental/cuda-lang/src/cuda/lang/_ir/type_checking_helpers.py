# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._ir.type import MemorySpace, ScalarTy, VectorTy, PointerTy
from cuda.tile import TileTypeError, DType
from cuda.tile._ir.ir import Var
from cuda.tile._ir.op_impl import require_array_type
from cuda.tile._ir.ops import implicit_cast
from cuda.tile._ir.type import TupleTy, TupleValue
from cuda.tile._datatype import is_integral, is_signed
from cuda.lang._datatype import clusterlaunchcontrol_token, is_float, mbarrier


def is_none(var: Var):
    return var.is_constant() and var.get_constant() is None


def require_array_indices(array: Var, indices: Var) -> tuple[Var, ...]:
    array_ty = require_array_type(array)
    indices_ty = indices.get_type()
    if isinstance(indices_ty, TupleTy):
        tuple_value = indices.get_aggregate()
        assert isinstance(tuple_value, TupleValue)
        index_vars = tuple_value.items
    else:
        index_vars = (indices,)

    if len(index_vars) != array_ty.ndim:
        raise make_type_checking_error(f"Wrong number of indices ({len(index_vars)})"
                                       f" for an array of rank {array_ty.ndim}", indices)

    return tuple(implicit_cast(var, array_ty.index_dtype, "Invalid array index")
                 for var in index_vars)


def require_signed_int_scalar_or_tuple(var: Var) -> tuple[Var, ...]:
    ty = var.get_type()
    have_tuple = isinstance(ty, TupleTy)
    item_types = ty.value_types if have_tuple else (ty,)

    for pos, item_ty in enumerate(item_types):
        if (not isinstance(item_ty, ScalarTy)
                or not is_integral(item_ty.dtype) or not is_signed(item_ty.dtype)):
            what = f"item at position #{pos}" if have_tuple else "given value"
            raise make_type_checking_error(
                f"Expected a signed integer, but {what} has type {item_ty}")

    return var.get_aggregate().items if have_tuple else (var,)


def require_scalar_type(var: Var,
                        valid_dtypes: tuple[DType, ...] = ()) -> ScalarTy:
    ty = var.get_type()
    if not isinstance(ty, ScalarTy):
        raise make_type_checking_error(f"Expected a scalar, but given value has type {ty}", var)
    if valid_dtypes and ty.dtype not in valid_dtypes:
        if len(valid_dtypes) == 1:
            message = f"Expected {valid_dtypes[0]}, but given value has dtype {ty.dtype}"
        else:
            message = (f"Expected dtype to be one of {valid_dtypes},"
                       f" but given value has dtype {ty.dtype}")
        raise make_type_checking_error(message, var)

    return ty


def require_clusterlaunchcontrol_token_type(var: Var) -> ScalarTy:
    ty = var.get_type()
    if ty != ScalarTy(clusterlaunchcontrol_token):
        raise make_type_checking_error(f"Expected a clusterlaunchcontrol_token,"
                                       f" but given value has type {ty}")
    return ty


def require_pointer_type(var: Var) -> PointerTy:
    ty = var.get_type()
    if not isinstance(ty, PointerTy):
        raise make_type_checking_error(f"Expected a pointer, got {ty}", var)
    return ty


def require_pointer_in_memory_space(ptr_value, spaces: tuple[MemorySpace, ...]) -> PointerTy:
    ptr_type = require_pointer_type(ptr_value)
    if ptr_type.memory_space not in spaces:
        expected = ' or '.join(map(str, spaces))
        raise TileTypeError(
            f"Expected pointer memory space to be {expected} "
            f"but got {ptr_type.memory_space}"
        )
    return ptr_type


def require_mbarrier_ptr(
    mbar: Var,
    spaces: tuple[MemorySpace, ...] = (
        MemorySpace.SHARED,
        MemorySpace.SHARED_CLUSTER,
    ),
) -> PointerTy:
    mbar_ptr_type = require_pointer_in_memory_space(mbar, spaces)
    if mbar_ptr_type.opaque or mbar_ptr_type.pointee_dtype is not mbarrier:
        raise TileTypeError(f"Expected a pointer to an mbarrier, got {mbar}")
    return mbar_ptr_type


def require_vector_type(var: Var, length: int | None = None) -> VectorTy:
    ty = var.get_type()
    if not isinstance(ty, VectorTy):
        raise make_type_checking_error(f"Expected a vector, got {ty}", var)
    if length is not None and ty.length != length:
        raise make_type_checking_error(f"Expected a vector of length {length}, got {ty}", var)
    return ty


def require_scalar_or_vector_type(var: Var, dtype_predicate=None) -> VectorTy | ScalarTy:
    ty = var.get_type()

    match ty:
        case ScalarTy() as st:
            dtype = st.dtype
        case VectorTy() as vt:
            dtype = vt.element_dtype
        case _:
            raise make_type_checking_error(
                f"Expected a scalar or vector type but got {ty}", var
            )

    if dtype_predicate is not None and not dtype_predicate(dtype):
        return make_type_checking_error(
            "Expected scalar or vector to satisfy constraint "
            f"{dtype_predicate.__name__} but got {ty}"
        )

    return ty


def require_scalar_or_vector_float_type(var: Var) -> VectorTy | ScalarTy:
    return require_scalar_or_vector_type(var, is_float)


def make_type_checking_error(message: str, culprit: Var | None = None):
    # TODO: recover the context similarly to _make_type_error in cutile
    raise TileTypeError(message)
