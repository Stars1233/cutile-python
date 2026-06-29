# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.op_impl import ImplRegistry
from cuda.tile._ir.ops import implicit_cast
import cuda.lang._datatype as datatype
from cuda.lang._ir.enum_to_mlir import cl_enum_to_mlir_attribute
from cuda.lang._mlir import BoolAttr
from cuda.lang._enums import MemorySpace
from cuda.lang._stub import copy_async
from cuda.lang._ir.type import TensorMapTy
from .raw_mlir_operation_utils import RawMLIROperationBuilder
from ..type_checking_helpers import (
    optional_cast,
    make_type_checking_error,
    require_boolean_scalar_type,
    require_mbarrier_ptr,
    require_none,
    require_optional,
    require_pointer_in_memory_space,
    require_uniform_int_tuple_type,
    tensor_map_descriptor_like,
    validate_tensor_map_load_mode,
)
from cuda.tile._ir.op_impl import require_constant_enum, require_optional_constant_enum


_registry = ImplRegistry()
impl = _registry.impl


def copy_async_impl_registry() -> ImplRegistry:
    return _registry


def validate_g2s_mode(mode: copy_async.TMALoadMode, im2col_count: int) -> None:
    match mode:
        case copy_async.TMALoadMode.TILE | copy_async.TMALoadMode.TILE_GATHER4:
            if im2col_count != 0:
                raise make_type_checking_error(
                    f"{mode.name} mode does not accept im2col_offsets"
                )

        case (
            copy_async.TMALoadMode.IM2COL
            | copy_async.TMALoadMode.IM2COL_W
            | copy_async.TMALoadMode.IM2COL_W_128
        ):
            if im2col_count == 0:
                raise make_type_checking_error(
                    f"{mode.name} mode requires im2col_offsets"
                )

        case _:
            raise make_type_checking_error(f"Unsupported TMA load mode {mode}")


@impl(copy_async.copy_async_bulk_tensor_global_to_shared)
def copy_async_bulk_tensor_global_to_shared_impl(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    im2col_offsets,
    multicast_mask,
    l2_cache_hint,
    mode,
    cta_group,
    predicate,
):
    src_tensor_map_ty = src_tensor_map_descriptor.get_type()
    src_coordinate_vars = require_uniform_int_tuple_type(src_coordinates)
    im2col_offset_vars = require_uniform_int_tuple_type(im2col_offsets)
    require_mbarrier_ptr(mbarrier, (MemorySpace.SHARED,))
    mode = require_constant_enum(mode, copy_async.TMALoadMode)
    validate_g2s_mode(mode, len(im2col_offset_vars))
    mode_attribute = cl_enum_to_mlir_attribute(mode)
    if isinstance(src_tensor_map_ty, TensorMapTy):
        validate_tensor_map_load_mode(src_tensor_map_ty, mode)
    tensor_map = tensor_map_descriptor_like(src_tensor_map_descriptor)
    dst_ty = require_pointer_in_memory_space(
        dst_memory,
        (MemorySpace.SHARED, MemorySpace.SHARED_CLUSTER),
    )
    is_cta_only = dst_ty.memory_space == MemorySpace.SHARED
    group_attr = None
    src_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in src_coordinate_vars
    )
    im2col_offsets = tuple(
        implicit_cast(offset, datatype.int16, "TMA im2col offsets")
        for offset in im2col_offset_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")

    if is_cta_only:
        message = (
            "When the destination memory is in shared memory, the "
            "predicate, multicast mask, and cta_group arguments are invalid."
        )
        require_none(predicate, message)
        require_none(multicast_mask, message)
        require_none(cta_group, message)
    else:
        multicast_mask = optional_cast(
            multicast_mask, datatype.int16, "TMA multicast mask"
        )
        require_optional(predicate, require_boolean_scalar_type)
        cta_group_value = require_optional_constant_enum(
            cta_group, copy_async.CTAGroup
        )
        group_attr = (
            None
            if cta_group_value is None
            else cl_enum_to_mlir_attribute(cta_group_value)
        )

    builder = (
        RawMLIROperationBuilder(
            name="nvvm.cp.async.bulk.tensor.shared.cluster.global"
        )
        .add_attribute("mode", mode_attribute)
        .add_attribute("isCTAOnly", BoolAttr(value=is_cta_only))
    )
    if not is_cta_only and group_attr is not None:
        builder = builder.add_attribute("group", group_attr)

    builder = (
        builder.add_operand(dst_memory)
        .add_operand(tensor_map)
        .add_variadic_operand(src_coordinates)
        .add_operand(mbarrier)
        .add_variadic_operand(im2col_offsets)
        .add_optional_operand(multicast_mask)
        .add_optional_operand(l2_cache_hint)
        .add_optional_operand(predicate)
    )
    builder.emit()


@impl(copy_async.copy_async_bulk_tensor_shared_to_global)
def copy_async_bulk_tensor_shared_to_global_impl(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    l2_cache_hint,
    mode,
    predicate,
):
    require_pointer_in_memory_space(src_memory, (MemorySpace.SHARED,))
    tensor_map = tensor_map_descriptor_like(dst_tensor_map_descriptor)
    dst_coordinate_vars = require_uniform_int_tuple_type(dst_coordinates)
    mode = require_constant_enum(mode, copy_async.TMAStoreMode)
    mode_attribute = cl_enum_to_mlir_attribute(mode)
    dst_coordinates = tuple(
        implicit_cast(coord, datatype.int32, "TMA coordinates")
        for coord in dst_coordinate_vars
    )
    l2_cache_hint = optional_cast(l2_cache_hint, datatype.int64, "TMA L2 cache hint")
    predicate = optional_cast(predicate, datatype.bool_, "TMA predicate")
    builder = (
        RawMLIROperationBuilder(name="nvvm.cp.async.bulk.tensor.global.shared.cta")
        .add_attribute("mode", mode_attribute)
        .add_operand(tensor_map)
        .add_operand(src_memory)
        .add_variadic_operand(dst_coordinates)
        .add_optional_operand(l2_cache_hint)
        .add_optional_operand(predicate)
    )
    builder.emit()
