# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._execution import stub
from .._enums import TMALoadMode, TMAStoreMode, CTAGroup


@stub
def cp_async_bulk_tensor_global_to_shared(
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
    group=None,
    predicate=None,
):
    """
    Args:
        src_tensor_map_descriptor (TensorMap | P0):
        src_coordinates (tuple[int, ...]):
        dst_memory (P3 | P7):
        mbarrier:
        im2col_offsets (tuple[int, ...]):
        multicast_mask (int | None):
        l2_cache_hint (int | None):
        mode (TMALoadMode):
        group (CTAGroup | None):
        predicate (bool | None):
    """


@stub
def cp_async_bulk_tensor_shared_to_global(
    src_memory,
    dst_tensor_map_descriptor,
    dst_coordinates,
    /,
    *,
    l2_cache_hint=None,
    mode=TMAStoreMode.TILE,
    predicate=None,
):
    """
    Args:
        src_memory (P3):
        dst_tensor_map_descriptor (TensorMap | P0):
        dst_coordinates (tuple[int, ...]):
        l2_cache_hint (int | None):
        mode (TMAStoreMode):
        predicate (bool | None):
    """


__all__ = (
    "TMALoadMode",
    "TMAStoreMode",
    "CTAGroup",
    "cp_async_bulk_tensor_global_to_shared",
    "cp_async_bulk_tensor_shared_to_global",
)
