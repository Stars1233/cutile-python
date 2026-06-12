# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import cuda.lang._datatype as datatype
from cuda.lang._stub import tcgen05 as tcgen05_stub
from cuda.lang._stub.tcgen05 import CTAGroup, Tcgen05LdStShape
from cuda.lang._ir.ir import Var
from cuda.lang._ir.ops import (
    MemorySpace,
    ScalarTy,
    TileTypeError,
    VectorTy,
    add_operation_variadic,
    astype,
    implicit_cast,
    require_scalar_type,
    strictly_typed_const,
)
from cuda.lang._ir.op_defs import RawNVVMIntrinsic
from cuda.lang._ir.type_checking_helpers import (
    is_none,
    require_mbarrier_ptr,
    require_pointer_in_memory_space,
)
from cuda.tile._exception import TileValueError
from cuda.tile._ir.op_impl import (
    ImplRegistry,
    require_constant_enum,
    require_constant_int,
)


_registry = ImplRegistry()
impl = _registry.impl


def tcgen05_impl_registry() -> ImplRegistry:
    return _registry


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
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.alloc.shared." + cta_group_value.value
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
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.dealloc." + cta_group_value.value
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
    cta_group_value = cast(CTAGroup, require_constant_enum(cta_group, CTAGroup))
    intrinsic = "llvm.nvvm.tcgen05.commit"
    if not is_none(multicast_mask):
        intrinsic += '.mc'
        mask = implicit_cast(multicast_mask, datatype.int16, "multicast mask")
        operands.append(mask)
    intrinsic += '.shared.' + cta_group_value.value
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
    shape_value = cast(Tcgen05LdStShape, require_constant_enum(shape, Tcgen05LdStShape))
    count_value = cast(int, require_constant_int(count))
    valid_counts = _TCGEN05_LD_VALID_COUNTS_BY_SHAPE[shape_value]
    if count_value not in valid_counts:
        valid = ", ".join(str(value) for value in valid_counts)
        raise TileValueError(
            f"Expected count for {shape_value.name} to be one of {valid}, got {count_value}"
        )

    has_offset = not is_none(offset)
    uses_offset = shape_value is Tcgen05LdStShape.SHAPE_16X32BX2
    if uses_offset and not has_offset:
        raise TileTypeError("tcgen05_ld with SHAPE_16X32BX2 requires offset")
    if has_offset and not uses_offset:
        raise TileTypeError("tcgen05_ld offset is only valid with SHAPE_16X32BX2")

    operands = [tmem_addr]
    if has_offset:
        require_scalar_type(offset)
        operands.append(astype(offset, datatype.int64))

    if is_none(pack):
        operands.append(strictly_typed_const(False, ScalarTy(datatype.bool_)))
    else:
        require_scalar_type(pack, (datatype.bool_,))
        operands.append(pack)

    intrinsic = f"llvm.nvvm.tcgen05.ld.{shape_value.value}.x{count_value}"
    total_registers = count_value * _TCGEN05_LD_REGISTERS_PER_COUNT[shape_value]
    result_type = (ScalarTy(datatype.int32)
                   if total_registers == 1 else VectorTy(datatype.int32, total_registers))

    [result] = add_operation_variadic(
        RawNVVMIntrinsic,
        (result_type,),
        intrinsic=intrinsic,
        operands_=tuple(operands),
    )
    return result
