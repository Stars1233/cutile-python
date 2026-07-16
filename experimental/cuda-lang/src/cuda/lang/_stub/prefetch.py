# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._execution import function

from .._enums import CachePolicy, PrefetchLevel
from . import nvvm_mlir_interfaces as _mlir
from .static_requirements import require_constant_enum


@function()
def prefetch(
    address,
    /,
    *,
    level: PrefetchLevel,
    eviction_priority: CachePolicy | None = None,
) -> None:
    """Prefetch the cache line containing ``address``.

    Args:
        address: Global, local, or generic pointer to the cache line.
        level: Cache level into which the line is prefetched.
        eviction_priority: Optional L2 eviction priority. Only
            :attr:`CachePolicy.L2_EVICT_NORMAL` and
            :attr:`CachePolicy.L2_EVICT_LAST` are supported.
    """
    require_constant_enum(level, PrefetchLevel)
    if eviction_priority is not None:
        require_constant_enum(eviction_priority, CachePolicy)
        cl.static_assert(
            level == PrefetchLevel.L2,
            "Prefetch eviction priority is supported only for L2",
        )
        cl.static_assert(
            eviction_priority
            in (CachePolicy.L2_EVICT_NORMAL, CachePolicy.L2_EVICT_LAST),
            "Prefetch eviction priority must be L2_EVICT_NORMAL or L2_EVICT_LAST",
        )

    if level == PrefetchLevel.L1:
        _mlir.prefetch(
            cache_level=_mlir.PrefetchCacheLevel.L1,
            addr=address,
        )
    elif eviction_priority is None:
        _mlir.prefetch(
            cache_level=_mlir.PrefetchCacheLevel.L2,
            addr=address,
        )
    elif eviction_priority == CachePolicy.L2_EVICT_NORMAL:
        _mlir.prefetch(
            cache_level=_mlir.PrefetchCacheLevel.L2,
            evict_priority=_mlir.CacheEvictionPriority.EvictNormal,
            addr=address,
        )
    elif eviction_priority == CachePolicy.L2_EVICT_LAST:
        _mlir.prefetch(
            cache_level=_mlir.PrefetchCacheLevel.L2,
            evict_priority=_mlir.CacheEvictionPriority.EvictLast,
            addr=address,
        )


@function()
def prefetch_uniform(address, /) -> None:
    """Prefetch the cache line containing a generic address into uniform L1.

    Args:
        address: Address contained by the cache line to be prefetched.
    """
    _mlir.prefetch(
        cache_level=_mlir.PrefetchCacheLevel.L1,
        addr=address,
        uniform=True,
    )


@function()
def prefetch_tensor_map(tensor_map, /, *, predicate=None) -> None:
    """Prefetch a tensor-map descriptor.

    Args:
        tensor_map: Tensor-map descriptor to prefetch.
        predicate: Optional run-time predicate guarding the PTX instruction.
    """
    _mlir.prefetch(
        addr=tensor_map.as_opaque_ptr(),
        predicate=predicate,
        tensormap=True,
    )


__all__ = (
    "PrefetchLevel",
    "prefetch",
    "prefetch_uniform",
    "prefetch_tensor_map",
)
