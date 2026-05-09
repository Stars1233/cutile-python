# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._enums import TensorMapSwizzle
from cuda.lang._execution import stub


class TensorMap:
    @stub
    def as_opaque_ptr(self):
        ...


@stub
def tensor_map_tiled(array,
                     tile_shape: int | tuple[int, ...],
                     *,
                     swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_NONE) -> TensorMap:
    ...
