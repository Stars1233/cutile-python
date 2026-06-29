# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import stub, function
from .._enums import TMALoadMode, TMAStoreMode, CTAGroup  # noqa: F401
from . import nvvm as _nvvm
from .static_requirements import require_constant_bool, require_constant_int


@stub
def copy_async_bulk_tensor_global_to_shared(
    src_tensor_map_descriptor,
    src_coordinates,
    dst_memory,
    mbarrier,
    /,
    *,
    im2col_offsets=(),
    multicast_mask=None,
    l2_cache_hint=None,
    mode=TMALoadMode.TILE,
    cta_group=None,
    predicate=None,
):
    """Initiate a multi-dimensional TMA copy from global to shared memory.

    See the `CUDA Programming Guide's multi-dimensional TMA alignment requirements
    <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#table-alignment-multi-dim-tma>`_
    for source and destination alignment requirements.

    Args:
        src_tensor_map_descriptor (TensorMap | P0):
        src_coordinates (tuple[int, ...]):
        dst_memory (P3 | P7):
        mbarrier:
        im2col_offsets (tuple[int, ...]):
        multicast_mask (int | None):
        l2_cache_hint (int | None):
        mode (TMALoadMode):
        cta_group (CTAGroup | None):
        predicate (bool | None):
    """


@stub
def copy_async_bulk_tensor_shared_to_global(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    /,
    *,
    l2_cache_hint=None,
    mode=TMAStoreMode.TILE,
    predicate=None,
):
    """Initiate a multi-dimensional TMA copy from shared to global memory.

    See the `CUDA Programming Guide's multi-dimensional TMA alignment requirements
    <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#table-alignment-multi-dim-tma>`_
    for source and destination alignment requirements.

    Args:
        src_memory (P3):
        dst_tensor_map_descriptor (TensorMap | P0):
        dst_coordinates (tuple[int, ...]):
        l2_cache_hint (int | None):
        mode (TMAStoreMode):
        predicate (bool | None):
    """


@function()
def copy_async_bulk_commit_group():
    """
    Commit all prior initiated but uncommitted cp.async.bulk instructions into
    a group of cp.async.bulk instructions.
    """
    _nvvm.cp_async_bulk_commit_group()


@function()
def copy_async_bulk_wait_group(number_of_groups, *, read=False):
    """Wait for completion of the most recent bulk async-groups.

    Args:
        number_of_groups (32-bit integer): How many of the prior async-bulk
            operation groups should be waited on.
        read (bool): Indicates that executing thread should wait until the bulk
            async operations in the specified bulk async-group must complete
            reading from the tensor map and reading from their source
            locations.
    """
    require_constant_int(number_of_groups)
    require_constant_bool(read)
    if read:
        _nvvm.cp_async_bulk_wait_group_read(number_of_groups)
    else:
        _nvvm.cp_async_bulk_wait_group(number_of_groups)
