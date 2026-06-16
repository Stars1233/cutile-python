# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from cuda.tile import _cext
from cuda.tile._memory_model import MemorySpace


class TensorMapSwizzle(enum.Enum):
    """Swizzle modes for tiled tensor map descriptors."""

    SWIZZLE_NONE = _cext.CU_TENSOR_MAP_SWIZZLE_NONE
    SWIZZLE_32B = _cext.CU_TENSOR_MAP_SWIZZLE_32B
    SWIZZLE_64B = _cext.CU_TENSOR_MAP_SWIZZLE_64B
    SWIZZLE_128B = _cext.CU_TENSOR_MAP_SWIZZLE_128B
    SWIZZLE_128B_ATOM_32B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B
    SWIZZLE_128B_ATOM_32B_FLIP_8B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B
    SWIZZLE_128B_ATOM_64B = _cext.CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B


class MbarrierScope(enum.Enum):
    """Scope of the threads that observe an mbarrier operation."""

    BLOCK = "cta"
    CLUSTER = "cluster"


class TMALoadMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    IM2COL_W = 2
    IM2COL_W_128 = 3
    TILE_GATHER4 = 4


class TMAStoreMode(enum.Enum):
    TILE = 0
    IM2COL = 1
    TILE_SCATTER4 = 2


class CTAGroup(enum.Enum):
    """CTA group selection for tcgen05 tensor memory operations."""

    CTA_1 = "cg1"
    CTA_2 = "cg2"


__all__ = (
    "MemorySpace",
    "TensorMapSwizzle",
    "MbarrierScope",
    "TMALoadMode",
    "TMAStoreMode",
    "CTAGroup",
)
