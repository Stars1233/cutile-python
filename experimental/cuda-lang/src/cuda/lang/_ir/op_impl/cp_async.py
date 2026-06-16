# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.ops import loosely_typed_const
from cuda.lang._enums import MemorySpace
from cuda.lang._stub import cp_async
from cuda.lang._stub._nvvm_mlir_support import _raw_nvvm_mlir_operation_impl
from cuda.lang._stub import nvvm_mlir_interfaces
from ..type_checking_helpers import (
    make_type_checking_error,
    require_integral_scalar_type,
    require_mbarrier_ptr,
    require_none,
    require_optional,
    require_pointer_in_memory_space,
    require_uniform_int_tuple_type,
    tensor_map_descriptor_like,
)
from cuda.tile._ir.op_impl import require_constant_enum
import cuda.lang._mlir.nvvm as mlir


_registry = ImplRegistry()
impl = _registry.impl


def cp_async_impl_registry() -> ImplRegistry:
    return _registry


def validate_g2s_mode(mode: cp_async.TMALoadMode, im2col_count: int) -> None:
    match mode:
        case cp_async.TMALoadMode.TILE | cp_async.TMALoadMode.TILE_GATHER4:
            if im2col_count != 0:
                raise make_type_checking_error(
                    f"{mode.name} mode does not accept im2col_offsets"
                )

        case (
            cp_async.TMALoadMode.IM2COL
            | cp_async.TMALoadMode.IM2COL_W
            | cp_async.TMALoadMode.IM2COL_W_128
        ):
            if im2col_count == 0:
                raise make_type_checking_error(
                    f"{mode.name} mode requires im2col_offsets"
                )

        case _:
            raise make_type_checking_error(f"Unsupported TMA load mode {mode}")


@impl(cp_async.cp_async_bulk_tensor_global_to_shared)
def cp_async_bulk_tensor_global_to_shared_impl(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    im2col_offsets,
    multicast_mask,
    l2_cache_hint,
    mode,
    group,
    predicate,
):
    tensor_map = tensor_map_descriptor_like(src_tensor_map_descriptor)
    require_uniform_int_tuple_type(src_coordinates)
    im2col_offset_vars = require_uniform_int_tuple_type(im2col_offsets)
    require_mbarrier_ptr(mbarrier)
    mode = require_constant_enum(mode, cp_async.TMALoadMode)
    validate_g2s_mode(mode, len(im2col_offset_vars))
    mode = getattr(mlir.TMALoadMode, mode.name)
    dst_ty = require_pointer_in_memory_space(
        dst_memory,
        (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER),
    )
    is_cta_only = dst_ty.memory_space == MemorySpace.SHARED

    if is_cta_only:
        message = (
            "When the destination memory is in shared memory, the "
            "predicate, multicast mask, and group arguments are invalid."
        )
        require_none(predicate, message)
        require_none(multicast_mask, message)
        require_none(group, message)
    else:
        raise make_type_checking_error(
            "Copying from global to shared-cluster memory is not yet supported"
        )

    return _raw_nvvm_mlir_operation_impl(
        nvvm_mlir_interfaces.cp_async_bulk_tensor_shared_cluster_global,
        dst_memory,
        tensor_map,
        src_coordinates,
        mbarrier,
        im2col_offsets,
        multicast_mask,
        l2_cache_hint,
        loosely_typed_const(mode),
        loosely_typed_const(is_cta_only),
        group,
        predicate,
    )


@impl(cp_async.cp_async_bulk_tensor_shared_to_global)
def cp_async_bulk_tensor_shared_to_global_impl(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    l2_cache_hint,
    mode,
    predicate,
):
    require_pointer_in_memory_space(src_memory, (MemorySpace.SHARED,))
    tensor_map = tensor_map_descriptor_like(dst_tensor_map_descriptor)
    require_uniform_int_tuple_type(dst_coordinates)
    require_optional(l2_cache_hint, require_integral_scalar_type)
    mode = require_constant_enum(mode, cp_async.TMAStoreMode)
    mode = getattr(mlir.TMAStoreMode, mode.name)
    return _raw_nvvm_mlir_operation_impl(
        nvvm_mlir_interfaces.cp_async_bulk_tensor_global_shared_cta,
        tensor_map,
        src_memory,
        dst_coordinates,
        l2_cache_hint,
        loosely_typed_const(mode),
        predicate,
    )
