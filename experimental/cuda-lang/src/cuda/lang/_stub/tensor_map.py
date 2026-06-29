# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from cuda.lang._enums import SwizzleMode
from cuda.lang._execution import stub


class TensorMap:
    """Descriptor for TMA access to a global array."""

    @stub
    def as_opaque_ptr(self):
        """Return this descriptor as an opaque pointer for low-level TMA intrinsics."""
        ...


@stub
def tensor_map_tiled(array,
                     tile_shape: int | tuple[int, ...],
                     *,
                     order: tuple[int, ...] | Literal["C", "F"] = "C",
                     swizzle: SwizzleMode = SwizzleMode.SWIZZLE_NONE) -> TensorMap:
    """Create a tiled tensor map descriptor for a global array."""
    ...
