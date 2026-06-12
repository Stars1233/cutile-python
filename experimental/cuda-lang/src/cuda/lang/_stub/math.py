# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


from cuda.lang._execution import stub


@stub
def ceil(x, /):
    """Round ``x`` toward positive infinity."""
    ...


@stub
def exp(x, /):
    """Compute ``e`` raised to the power ``x``."""
    ...


@stub
def exp2(x, /):
    """Compute ``2`` raised to the power ``x``."""
    ...


@stub
def sin(x, /):
    """Compute the sine of ``x``."""
    ...


@stub
def cos(x, /):
    """Compute the cosine of ``x``."""
    ...


@stub
def tan(x, /):
    """Compute the tangent of ``x``."""
    ...


@stub
def sinh(x, /):
    """Compute the hyperbolic sine of ``x``."""
    ...


@stub
def cosh(x, /):
    """Compute the hyperbolic cosine of ``x``."""
    ...


@stub
def tanh(x, /):
    """Compute the hyperbolic tangent of ``x``."""
    ...


@stub
def sqrt(x, /):
    """Compute the square root of ``x``."""
    ...


@stub
def rsqrt(x, /):
    """Compute the reciprocal square root of ``x``."""
    ...


@stub
def floor(x, /):
    """Round ``x`` toward negative infinity."""
    ...


@stub
def log(x, /):
    """Compute the natural logarithm of ``x``."""
    ...


@stub
def log2(x, /):
    """Compute the base-2 logarithm of ``x``."""
    ...


@stub
def abs(x, /):
    """Compute the absolute value of ``x``."""
    ...


@stub
def atan2(x, y, /):
    """Compute the angle whose tangent is ``x / y``."""
    ...


@stub
def isnan(x, /):
    """Return whether ``x`` is NaN."""
    ...


@stub
def isinf(x, /):
    """Return whether ``x`` is positive or negative infinity."""
    ...


@stub
def isfinite(x, /):
    """Return whether ``x`` is finite."""
    ...


@stub
def isnormal(x, /):
    """Return whether ``x`` is a normal floating-point value."""
    ...


@stub
def pow(x, y, /):
    """Calculate the value of ``x`` to the power of ``y``."""
    ...


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
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
    "isnan",
    "isinf",
    "isfinite",
    "isnormal",
    "pow",
)
