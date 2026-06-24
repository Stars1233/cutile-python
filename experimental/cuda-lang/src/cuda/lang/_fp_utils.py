# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

_FLOAT_SMALLEST_NORMAL = {
    16: 2**-14,
    32: 2**-126,
    64: 2**-1022,
}


def isnormal(x, for_precision: int = 64):
    # TODO: use math.isnormal in python >=3.15
    normal_min = _FLOAT_SMALLEST_NORMAL[for_precision]
    return math.isfinite(x) and x != 0.0 and math.fabs(x) >= normal_min


__all__ = ("isnormal",)
