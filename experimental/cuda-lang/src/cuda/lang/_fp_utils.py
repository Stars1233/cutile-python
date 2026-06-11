# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

# These numbers were recorded with numpy:
# >>> [print(np.finfo(dt).smallest_normal) for dt in (np.float16, np.float32, np.float64)]
# 6.104e-05
# 1.1754944e-38
# 2.2250738585072014e-308
_FLOAT_SMALLEST_NORMAL = {
    16: 6.104e-05,
    32: 1.1754944e-38,
    64: 2.2250738585072014e-308,
}


def isnormal(x, for_precision: int = 64):
    # TODO: use math.isnormal in python >=3.15
    normal_min = _FLOAT_SMALLEST_NORMAL[for_precision]
    return math.isfinite(x) and x != 0.0 and math.fabs(x) >= normal_min


__all__ = ("isnormal",)
