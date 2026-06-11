# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.lang._execution import stub


@stub
def ceil(x, /): ...
@stub
def exp(x, /): ...
@stub
def exp2(x, /): ...
@stub
def sin(x, /): ...
@stub
def cos(x, /): ...
@stub
def tan(x, /): ...
@stub
def sinh(x, /): ...
@stub
def cosh(x, /): ...
@stub
def tanh(x, /): ...
@stub
def sqrt(x, /): ...
@stub
def rsqrt(x, /): ...
@stub
def floor(x, /): ...
@stub
def log(x, /): ...
@stub
def log2(x, /): ...
@stub
def abs(x, /): ...
@stub
def atan2(x, y, /): ...
@stub
def isnan(x, /): ...
@stub
def isinf(x, /): ...
@stub
def isfinite(x, /): ...
@stub
def isnormal(x, /): ...


__all__ = (
    "ceil",
    "exp",
    "exp2",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "sqrt",
    "rsqrt",
    "floor",
    "log",
    "log2",
    "abs",
    "atan2",
)
