
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._stub.foreign_function import _call_foreign_function
from cuda.lang._ir.type import MemorySpace
from cuda.lang._stub.core_api import address_space_cast, static_assert
from cuda.lang._execution import function
from cuda.lang._datatype import (
    float32,
    float64,
    int16,
    int32,
    int64,
    opaque_ptr,
    is_literal_or_exact_dtype,
    satisfies_pointer_constraint,
)


@function
def abs(x_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_abs.html#__nv_abs>`__.

    Args:
        x_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_abs expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_abs", int32, (x_,))

__nv_abs = abs

@function
def acos(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acos.html#__nv_acos>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_acos expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_acos", float64, (x_,))

__nv_acos = acos

@function
def acosf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosf.html#__nv_acosf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_acosf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_acosf", float32, (x_,))

__nv_acosf = acosf

@function
def acosh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acosh.html#__nv_acosh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_acosh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_acosh", float64, (x_,))

__nv_acosh = acosh

@function
def acoshf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_acoshf.html#__nv_acoshf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_acoshf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_acoshf", float32, (x_,))

__nv_acoshf = acoshf

@function
def asin(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asin.html#__nv_asin>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_asin expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_asin", float64, (x_,))

__nv_asin = asin

@function
def asinf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinf.html#__nv_asinf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_asinf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_asinf", float32, (x_,))

__nv_asinf = asinf

@function
def asinh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinh.html#__nv_asinh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_asinh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_asinh", float64, (x_,))

__nv_asinh = asinh

@function
def asinhf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_asinhf.html#__nv_asinhf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_asinhf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_asinhf", float32, (x_,))

__nv_asinhf = asinhf

@function
def atan(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan.html#__nv_atan>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_atan expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_atan", float64, (x_,))

__nv_atan = atan

@function
def atan2(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2.html#__nv_atan2>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_atan2 expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_atan2 expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_atan2", float64, (x_, y_))

__nv_atan2 = atan2

@function
def atan2f(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atan2f.html#__nv_atan2f>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_atan2f expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_atan2f expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_atan2f", float32, (x_, y_))

__nv_atan2f = atan2f

@function
def atanf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanf.html#__nv_atanf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_atanf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_atanf", float32, (x_,))

__nv_atanf = atanf

@function
def atanh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanh.html#__nv_atanh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_atanh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_atanh", float64, (x_,))

__nv_atanh = atanh

@function
def atanhf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_atanhf.html#__nv_atanhf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_atanhf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_atanhf", float32, (x_,))

__nv_atanhf = atanhf

@function
def brev(x_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brev.html#__nv_brev>`__.

    Args:
        x_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_brev expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_brev", int32, (x_,))

__nv_brev = brev

@function
def brevll(x_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_brevll.html#__nv_brevll>`__.

    Args:
        x_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_brevll expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_brevll", int64, (x_,))

__nv_brevll = brevll

@function
def byte_perm(x_: int32, y_: int32, z_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_byte_perm.html#__nv_byte_perm>`__.

    Args:
        x_: ``int32``
        y_: ``int32``
        z_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_byte_perm expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_byte_perm expects argument 1 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, int32),
        '__nv_byte_perm expects argument 2 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)
    z_ = int32(z_)

    return _call_foreign_function("__nv_byte_perm", int32, (x_, y_, z_))

__nv_byte_perm = byte_perm

@function
def cbrt(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrt.html#__nv_cbrt>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_cbrt expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_cbrt", float64, (x_,))

__nv_cbrt = cbrt

@function
def cbrtf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cbrtf.html#__nv_cbrtf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_cbrtf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_cbrtf", float32, (x_,))

__nv_cbrtf = cbrtf

@function
def ceil(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceil.html#__nv_ceil>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ceil expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_ceil", float64, (x_,))

__nv_ceil = ceil

@function
def ceilf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ceilf.html#__nv_ceilf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_ceilf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_ceilf", float32, (x_,))

__nv_ceilf = ceilf

@function
def clz(x_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clz.html#__nv_clz>`__.

    Args:
        x_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_clz expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_clz", int32, (x_,))

__nv_clz = clz

@function
def clzll(x_: int64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_clzll.html#__nv_clzll>`__.

    Args:
        x_: ``int64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_clzll expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_clzll", int32, (x_,))

__nv_clzll = clzll

@function
def copysign(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysign.html#__nv_copysign>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_copysign expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_copysign expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_copysign", float64, (x_, y_))

__nv_copysign = copysign

@function
def copysignf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_copysignf.html#__nv_copysignf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_copysignf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_copysignf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_copysignf", float32, (x_, y_))

__nv_copysignf = copysignf

@function
def cos(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cos.html#__nv_cos>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_cos expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_cos", float64, (x_,))

__nv_cos = cos

@function
def cosf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosf.html#__nv_cosf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_cosf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_cosf", float32, (x_,))

__nv_cosf = cosf

@function
def cosh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cosh.html#__nv_cosh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_cosh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_cosh", float64, (x_,))

__nv_cosh = cosh

@function
def coshf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_coshf.html#__nv_coshf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_coshf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_coshf", float32, (x_,))

__nv_coshf = coshf

@function
def cospi(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospi.html#__nv_cospi>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_cospi expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_cospi", float64, (x_,))

__nv_cospi = cospi

@function
def cospif(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_cospif.html#__nv_cospif>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_cospif expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_cospif", float32, (x_,))

__nv_cospif = cospif

@function
def dadd_rd(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rd.html#__nv_dadd_rd>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dadd_rd expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dadd_rd expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dadd_rd", float64, (x_, y_))

__nv_dadd_rd = dadd_rd

@function
def dadd_rn(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rn.html#__nv_dadd_rn>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dadd_rn expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dadd_rn expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dadd_rn", float64, (x_, y_))

__nv_dadd_rn = dadd_rn

@function
def dadd_ru(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_ru.html#__nv_dadd_ru>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dadd_ru expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dadd_ru expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dadd_ru", float64, (x_, y_))

__nv_dadd_ru = dadd_ru

@function
def dadd_rz(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dadd_rz.html#__nv_dadd_rz>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dadd_rz expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dadd_rz expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dadd_rz", float64, (x_, y_))

__nv_dadd_rz = dadd_rz

@function
def ddiv_rd(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rd.html#__nv_ddiv_rd>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ddiv_rd expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_ddiv_rd expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_ddiv_rd", float64, (x_, y_))

__nv_ddiv_rd = ddiv_rd

@function
def ddiv_rn(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rn.html#__nv_ddiv_rn>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ddiv_rn expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_ddiv_rn expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_ddiv_rn", float64, (x_, y_))

__nv_ddiv_rn = ddiv_rn

@function
def ddiv_ru(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_ru.html#__nv_ddiv_ru>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ddiv_ru expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_ddiv_ru expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_ddiv_ru", float64, (x_, y_))

__nv_ddiv_ru = ddiv_ru

@function
def ddiv_rz(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ddiv_rz.html#__nv_ddiv_rz>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ddiv_rz expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_ddiv_rz expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_ddiv_rz", float64, (x_, y_))

__nv_ddiv_rz = ddiv_rz

@function
def dmul_rd(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rd.html#__nv_dmul_rd>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dmul_rd expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dmul_rd expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dmul_rd", float64, (x_, y_))

__nv_dmul_rd = dmul_rd

@function
def dmul_rn(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rn.html#__nv_dmul_rn>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dmul_rn expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dmul_rn expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dmul_rn", float64, (x_, y_))

__nv_dmul_rn = dmul_rn

@function
def dmul_ru(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_ru.html#__nv_dmul_ru>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dmul_ru expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dmul_ru expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dmul_ru", float64, (x_, y_))

__nv_dmul_ru = dmul_ru

@function
def dmul_rz(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dmul_rz.html#__nv_dmul_rz>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dmul_rz expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_dmul_rz expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_dmul_rz", float64, (x_, y_))

__nv_dmul_rz = dmul_rz

@function
def double2float_rd(d_: float64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rd.html#__nv_double2float_rd>`__.

    Args:
        d_: ``float64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2float_rd expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2float_rd", float32, (d_,))

__nv_double2float_rd = double2float_rd

@function
def double2float_rn(d_: float64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rn.html#__nv_double2float_rn>`__.

    Args:
        d_: ``float64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2float_rn expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2float_rn", float32, (d_,))

__nv_double2float_rn = double2float_rn

@function
def double2float_ru(d_: float64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_ru.html#__nv_double2float_ru>`__.

    Args:
        d_: ``float64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2float_ru expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2float_ru", float32, (d_,))

__nv_double2float_ru = double2float_ru

@function
def double2float_rz(d_: float64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2float_rz.html#__nv_double2float_rz>`__.

    Args:
        d_: ``float64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2float_rz expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2float_rz", float32, (d_,))

__nv_double2float_rz = double2float_rz

@function
def double2hiint(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2hiint.html#__nv_double2hiint>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2hiint expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2hiint", int32, (d_,))

__nv_double2hiint = double2hiint

@function
def double2int_rd(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rd.html#__nv_double2int_rd>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2int_rd expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2int_rd", int32, (d_,))

__nv_double2int_rd = double2int_rd

@function
def double2int_rn(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rn.html#__nv_double2int_rn>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2int_rn expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2int_rn", int32, (d_,))

__nv_double2int_rn = double2int_rn

@function
def double2int_ru(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_ru.html#__nv_double2int_ru>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2int_ru expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2int_ru", int32, (d_,))

__nv_double2int_ru = double2int_ru

@function
def double2int_rz(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2int_rz.html#__nv_double2int_rz>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2int_rz expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2int_rz", int32, (d_,))

__nv_double2int_rz = double2int_rz

@function
def double2ll_rd(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rd.html#__nv_double2ll_rd>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ll_rd expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ll_rd", int64, (f_,))

__nv_double2ll_rd = double2ll_rd

@function
def double2ll_rn(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rn.html#__nv_double2ll_rn>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ll_rn expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ll_rn", int64, (f_,))

__nv_double2ll_rn = double2ll_rn

@function
def double2ll_ru(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_ru.html#__nv_double2ll_ru>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ll_ru expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ll_ru", int64, (f_,))

__nv_double2ll_ru = double2ll_ru

@function
def double2ll_rz(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ll_rz.html#__nv_double2ll_rz>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ll_rz expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ll_rz", int64, (f_,))

__nv_double2ll_rz = double2ll_rz

@function
def double2loint(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2loint.html#__nv_double2loint>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2loint expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2loint", int32, (d_,))

__nv_double2loint = double2loint

@function
def double2uint_rd(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rd.html#__nv_double2uint_rd>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2uint_rd expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2uint_rd", int32, (d_,))

__nv_double2uint_rd = double2uint_rd

@function
def double2uint_rn(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rn.html#__nv_double2uint_rn>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2uint_rn expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2uint_rn", int32, (d_,))

__nv_double2uint_rn = double2uint_rn

@function
def double2uint_ru(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_ru.html#__nv_double2uint_ru>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2uint_ru expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2uint_ru", int32, (d_,))

__nv_double2uint_ru = double2uint_ru

@function
def double2uint_rz(d_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2uint_rz.html#__nv_double2uint_rz>`__.

    Args:
        d_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(d_, float64),
        '__nv_double2uint_rz expects argument 0 to have type float64'
    )

    d_ = float64(d_)

    return _call_foreign_function("__nv_double2uint_rz", int32, (d_,))

__nv_double2uint_rz = double2uint_rz

@function
def double2ull_rd(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rd.html#__nv_double2ull_rd>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ull_rd expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ull_rd", int64, (f_,))

__nv_double2ull_rd = double2ull_rd

@function
def double2ull_rn(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rn.html#__nv_double2ull_rn>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ull_rn expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ull_rn", int64, (f_,))

__nv_double2ull_rn = double2ull_rn

@function
def double2ull_ru(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_ru.html#__nv_double2ull_ru>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ull_ru expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ull_ru", int64, (f_,))

__nv_double2ull_ru = double2ull_ru

@function
def double2ull_rz(f_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double2ull_rz.html#__nv_double2ull_rz>`__.

    Args:
        f_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_double2ull_rz expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_double2ull_rz", int64, (f_,))

__nv_double2ull_rz = double2ull_rz

@function
def double_as_longlong(x_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_double_as_longlong.html#__nv_double_as_longlong>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_double_as_longlong expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_double_as_longlong", int64, (x_,))

__nv_double_as_longlong = double_as_longlong

@function
def drcp_rd(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rd.html#__nv_drcp_rd>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_drcp_rd expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_drcp_rd", float64, (x_,))

__nv_drcp_rd = drcp_rd

@function
def drcp_rn(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rn.html#__nv_drcp_rn>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_drcp_rn expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_drcp_rn", float64, (x_,))

__nv_drcp_rn = drcp_rn

@function
def drcp_ru(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_ru.html#__nv_drcp_ru>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_drcp_ru expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_drcp_ru", float64, (x_,))

__nv_drcp_ru = drcp_ru

@function
def drcp_rz(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_drcp_rz.html#__nv_drcp_rz>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_drcp_rz expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_drcp_rz", float64, (x_,))

__nv_drcp_rz = drcp_rz

@function
def dsqrt_rd(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rd.html#__nv_dsqrt_rd>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dsqrt_rd expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_dsqrt_rd", float64, (x_,))

__nv_dsqrt_rd = dsqrt_rd

@function
def dsqrt_rn(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rn.html#__nv_dsqrt_rn>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dsqrt_rn expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_dsqrt_rn", float64, (x_,))

__nv_dsqrt_rn = dsqrt_rn

@function
def dsqrt_ru(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_ru.html#__nv_dsqrt_ru>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dsqrt_ru expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_dsqrt_ru", float64, (x_,))

__nv_dsqrt_ru = dsqrt_ru

@function
def dsqrt_rz(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_dsqrt_rz.html#__nv_dsqrt_rz>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_dsqrt_rz expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_dsqrt_rz", float64, (x_,))

__nv_dsqrt_rz = dsqrt_rz

@function
def erf(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erf.html#__nv_erf>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_erf expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_erf", float64, (x_,))

__nv_erf = erf

@function
def erfc(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfc.html#__nv_erfc>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_erfc expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_erfc", float64, (x_,))

__nv_erfc = erfc

@function
def erfcf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcf.html#__nv_erfcf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_erfcf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_erfcf", float32, (x_,))

__nv_erfcf = erfcf

@function
def erfcinv(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinv.html#__nv_erfcinv>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_erfcinv expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_erfcinv", float64, (x_,))

__nv_erfcinv = erfcinv

@function
def erfcinvf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcinvf.html#__nv_erfcinvf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_erfcinvf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_erfcinvf", float32, (x_,))

__nv_erfcinvf = erfcinvf

@function
def erfcx(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcx.html#__nv_erfcx>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_erfcx expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_erfcx", float64, (x_,))

__nv_erfcx = erfcx

@function
def erfcxf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfcxf.html#__nv_erfcxf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_erfcxf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_erfcxf", float32, (x_,))

__nv_erfcxf = erfcxf

@function
def erff(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erff.html#__nv_erff>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_erff expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_erff", float32, (x_,))

__nv_erff = erff

@function
def erfinv(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinv.html#__nv_erfinv>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_erfinv expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_erfinv", float64, (x_,))

__nv_erfinv = erfinv

@function
def erfinvf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erfinvf.html#__nv_erfinvf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_erfinvf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_erfinvf", float32, (x_,))

__nv_erfinvf = erfinvf

@function
def exp(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp.html#__nv_exp>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_exp expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_exp", float64, (x_,))

__nv_exp = exp

@function
def exp10(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10.html#__nv_exp10>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_exp10 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_exp10", float64, (x_,))

__nv_exp10 = exp10

@function
def exp10f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp10f.html#__nv_exp10f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_exp10f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_exp10f", float32, (x_,))

__nv_exp10f = exp10f

@function
def exp2(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2.html#__nv_exp2>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_exp2 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_exp2", float64, (x_,))

__nv_exp2 = exp2

@function
def exp2f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_exp2f.html#__nv_exp2f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_exp2f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_exp2f", float32, (x_,))

__nv_exp2f = exp2f

@function
def expf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expf.html#__nv_expf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_expf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_expf", float32, (x_,))

__nv_expf = expf

@function
def expm1(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1.html#__nv_expm1>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_expm1 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_expm1", float64, (x_,))

__nv_expm1 = expm1

@function
def expm1f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_expm1f.html#__nv_expm1f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_expm1f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_expm1f", float32, (x_,))

__nv_expm1f = expm1f

@function
def fabs(f_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabs.html#__nv_fabs>`__.

    Args:
        f_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_fabs expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_fabs", float64, (f_,))

__nv_fabs = fabs

@function
def fabsf(f_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fabsf.html#__nv_fabsf>`__.

    Args:
        f_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_fabsf expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_fabsf", float32, (f_,))

__nv_fabsf = fabsf

@function
def fadd_rd(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rd.html#__nv_fadd_rd>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fadd_rd expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fadd_rd expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fadd_rd", float32, (x_, y_))

__nv_fadd_rd = fadd_rd

@function
def fadd_rn(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rn.html#__nv_fadd_rn>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fadd_rn expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fadd_rn expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fadd_rn", float32, (x_, y_))

__nv_fadd_rn = fadd_rn

@function
def fadd_ru(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_ru.html#__nv_fadd_ru>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fadd_ru expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fadd_ru expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fadd_ru", float32, (x_, y_))

__nv_fadd_ru = fadd_ru

@function
def fadd_rz(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fadd_rz.html#__nv_fadd_rz>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fadd_rz expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fadd_rz expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fadd_rz", float32, (x_, y_))

__nv_fadd_rz = fadd_rz

@function
def fast_cosf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_cosf.html#__nv_fast_cosf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_cosf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_cosf", float32, (x_,))

__nv_fast_cosf = fast_cosf

@function
def fast_exp10f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_exp10f.html#__nv_fast_exp10f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_exp10f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_exp10f", float32, (x_,))

__nv_fast_exp10f = fast_exp10f

@function
def fast_expf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_expf.html#__nv_fast_expf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_expf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_expf", float32, (x_,))

__nv_fast_expf = fast_expf

@function
def fast_fdividef(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_fdividef.html#__nv_fast_fdividef>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_fdividef expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fast_fdividef expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fast_fdividef", float32, (x_, y_))

__nv_fast_fdividef = fast_fdividef

@function
def fast_log10f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log10f.html#__nv_fast_log10f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_log10f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_log10f", float32, (x_,))

__nv_fast_log10f = fast_log10f

@function
def fast_log2f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_log2f.html#__nv_fast_log2f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_log2f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_log2f", float32, (x_,))

__nv_fast_log2f = fast_log2f

@function
def fast_logf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_logf.html#__nv_fast_logf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_logf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_logf", float32, (x_,))

__nv_fast_logf = fast_logf

@function
def fast_powf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_powf.html#__nv_fast_powf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_powf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fast_powf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fast_powf", float32, (x_, y_))

__nv_fast_powf = fast_powf

@function
def __nv_fast_sincosf(x_: float32, sptr_: opaque_ptr, cptr_: opaque_ptr) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sincosf.html#__nv_fast_sincosf>`__.

    Args:
        x_: ``float32``
        sptr_: ``opaque_ptr``
        cptr_: ``opaque_ptr``

    Returns:
        ``None``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_sincosf expects argument 0 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(sptr_, opaque_ptr),
        '__nv_fast_sincosf expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )
    static_assert(
        satisfies_pointer_constraint(cptr_, opaque_ptr),
        '__nv_fast_sincosf expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    sptr_ = address_space_cast(sptr_, MemorySpace.GENERIC)
    cptr_ = address_space_cast(cptr_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_fast_sincosf", None, (x_, sptr_, cptr_))

@function
def fast_sinf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sinf.html#__nv_fast_sinf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_sinf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_sinf", float32, (x_,))

__nv_fast_sinf = fast_sinf

@function
def fast_tanf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_tanf.html#__nv_fast_tanf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fast_tanf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fast_tanf", float32, (x_,))

__nv_fast_tanf = fast_tanf

@function
def fdim(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdim.html#__nv_fdim>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fdim expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fdim expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_fdim", float64, (x_, y_))

__nv_fdim = fdim

@function
def fdimf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdimf.html#__nv_fdimf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fdimf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fdimf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fdimf", float32, (x_, y_))

__nv_fdimf = fdimf

@function
def fdiv_rd(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rd.html#__nv_fdiv_rd>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fdiv_rd expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fdiv_rd expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fdiv_rd", float32, (x_, y_))

__nv_fdiv_rd = fdiv_rd

@function
def fdiv_rn(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rn.html#__nv_fdiv_rn>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fdiv_rn expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fdiv_rn expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fdiv_rn", float32, (x_, y_))

__nv_fdiv_rn = fdiv_rn

@function
def fdiv_ru(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_ru.html#__nv_fdiv_ru>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fdiv_ru expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fdiv_ru expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fdiv_ru", float32, (x_, y_))

__nv_fdiv_ru = fdiv_ru

@function
def fdiv_rz(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fdiv_rz.html#__nv_fdiv_rz>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fdiv_rz expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fdiv_rz expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fdiv_rz", float32, (x_, y_))

__nv_fdiv_rz = fdiv_rz

@function
def ffs(x_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffs.html#__nv_ffs>`__.

    Args:
        x_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_ffs expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_ffs", int32, (x_,))

__nv_ffs = ffs

@function
def ffsll(x_: int64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ffsll.html#__nv_ffsll>`__.

    Args:
        x_: ``int64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_ffsll expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_ffsll", int32, (x_,))

__nv_ffsll = ffsll

@function
def finitef(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_finitef.html#__nv_finitef>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_finitef expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_finitef", int32, (x_,))

__nv_finitef = finitef

@function
def float2half_rn(f_: float32) -> int16:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2half_rn.html#__nv_float2half_rn>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int16``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2half_rn expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2half_rn", int16, (f_,))

__nv_float2half_rn = float2half_rn

@function
def float2int_rd(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rd.html#__nv_float2int_rd>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2int_rd expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2int_rd", int32, (in_,))

__nv_float2int_rd = float2int_rd

@function
def float2int_rn(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rn.html#__nv_float2int_rn>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2int_rn expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2int_rn", int32, (in_,))

__nv_float2int_rn = float2int_rn

@function
def float2int_ru(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_ru.html#__nv_float2int_ru>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2int_ru expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2int_ru", int32, (in_,))

__nv_float2int_ru = float2int_ru

@function
def float2int_rz(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2int_rz.html#__nv_float2int_rz>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2int_rz expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2int_rz", int32, (in_,))

__nv_float2int_rz = float2int_rz

@function
def float2ll_rd(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rd.html#__nv_float2ll_rd>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ll_rd expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ll_rd", int64, (f_,))

__nv_float2ll_rd = float2ll_rd

@function
def float2ll_rn(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rn.html#__nv_float2ll_rn>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ll_rn expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ll_rn", int64, (f_,))

__nv_float2ll_rn = float2ll_rn

@function
def float2ll_ru(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_ru.html#__nv_float2ll_ru>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ll_ru expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ll_ru", int64, (f_,))

__nv_float2ll_ru = float2ll_ru

@function
def float2ll_rz(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ll_rz.html#__nv_float2ll_rz>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ll_rz expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ll_rz", int64, (f_,))

__nv_float2ll_rz = float2ll_rz

@function
def float2uint_rd(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rd.html#__nv_float2uint_rd>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2uint_rd expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2uint_rd", int32, (in_,))

__nv_float2uint_rd = float2uint_rd

@function
def float2uint_rn(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rn.html#__nv_float2uint_rn>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2uint_rn expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2uint_rn", int32, (in_,))

__nv_float2uint_rn = float2uint_rn

@function
def float2uint_ru(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_ru.html#__nv_float2uint_ru>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2uint_ru expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2uint_ru", int32, (in_,))

__nv_float2uint_ru = float2uint_ru

@function
def float2uint_rz(in_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2uint_rz.html#__nv_float2uint_rz>`__.

    Args:
        in_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, float32),
        '__nv_float2uint_rz expects argument 0 to have type float32'
    )

    in_ = float32(in_)

    return _call_foreign_function("__nv_float2uint_rz", int32, (in_,))

__nv_float2uint_rz = float2uint_rz

@function
def float2ull_rd(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rd.html#__nv_float2ull_rd>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ull_rd expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ull_rd", int64, (f_,))

__nv_float2ull_rd = float2ull_rd

@function
def float2ull_rn(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rn.html#__nv_float2ull_rn>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ull_rn expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ull_rn", int64, (f_,))

__nv_float2ull_rn = float2ull_rn

@function
def float2ull_ru(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_ru.html#__nv_float2ull_ru>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ull_ru expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ull_ru", int64, (f_,))

__nv_float2ull_ru = float2ull_ru

@function
def float2ull_rz(f_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float2ull_rz.html#__nv_float2ull_rz>`__.

    Args:
        f_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_float2ull_rz expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_float2ull_rz", int64, (f_,))

__nv_float2ull_rz = float2ull_rz

@function
def float_as_int(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_float_as_int.html#__nv_float_as_int>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_float_as_int expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_float_as_int", int32, (x_,))

__nv_float_as_int = float_as_int

@function
def floor(f_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floor.html#__nv_floor>`__.

    Args:
        f_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float64),
        '__nv_floor expects argument 0 to have type float64'
    )

    f_ = float64(f_)

    return _call_foreign_function("__nv_floor", float64, (f_,))

__nv_floor = floor

@function
def floorf(f_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_floorf.html#__nv_floorf>`__.

    Args:
        f_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(f_, float32),
        '__nv_floorf expects argument 0 to have type float32'
    )

    f_ = float32(f_)

    return _call_foreign_function("__nv_floorf", float32, (f_,))

__nv_floorf = floorf

@function
def fma(x_: float64, y_: float64, z_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma.html#__nv_fma>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        z_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fma expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fma expects argument 1 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float64),
        '__nv_fma expects argument 2 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    z_ = float64(z_)

    return _call_foreign_function("__nv_fma", float64, (x_, y_, z_))

__nv_fma = fma

@function
def fma_rd(x_: float64, y_: float64, z_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rd.html#__nv_fma_rd>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        z_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fma_rd expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fma_rd expects argument 1 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float64),
        '__nv_fma_rd expects argument 2 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    z_ = float64(z_)

    return _call_foreign_function("__nv_fma_rd", float64, (x_, y_, z_))

__nv_fma_rd = fma_rd

@function
def fma_rn(x_: float64, y_: float64, z_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rn.html#__nv_fma_rn>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        z_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fma_rn expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fma_rn expects argument 1 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float64),
        '__nv_fma_rn expects argument 2 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    z_ = float64(z_)

    return _call_foreign_function("__nv_fma_rn", float64, (x_, y_, z_))

__nv_fma_rn = fma_rn

@function
def fma_ru(x_: float64, y_: float64, z_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_ru.html#__nv_fma_ru>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        z_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fma_ru expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fma_ru expects argument 1 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float64),
        '__nv_fma_ru expects argument 2 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    z_ = float64(z_)

    return _call_foreign_function("__nv_fma_ru", float64, (x_, y_, z_))

__nv_fma_ru = fma_ru

@function
def fma_rz(x_: float64, y_: float64, z_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fma_rz.html#__nv_fma_rz>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        z_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fma_rz expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fma_rz expects argument 1 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float64),
        '__nv_fma_rz expects argument 2 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    z_ = float64(z_)

    return _call_foreign_function("__nv_fma_rz", float64, (x_, y_, z_))

__nv_fma_rz = fma_rz

@function
def fmaf(x_: float32, y_: float32, z_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf.html#__nv_fmaf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        z_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaf expects argument 1 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float32),
        '__nv_fmaf expects argument 2 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    z_ = float32(z_)

    return _call_foreign_function("__nv_fmaf", float32, (x_, y_, z_))

__nv_fmaf = fmaf

@function
def fmaf_rd(x_: float32, y_: float32, z_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rd.html#__nv_fmaf_rd>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        z_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaf_rd expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaf_rd expects argument 1 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float32),
        '__nv_fmaf_rd expects argument 2 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    z_ = float32(z_)

    return _call_foreign_function("__nv_fmaf_rd", float32, (x_, y_, z_))

__nv_fmaf_rd = fmaf_rd

@function
def fmaf_rn(x_: float32, y_: float32, z_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rn.html#__nv_fmaf_rn>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        z_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaf_rn expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaf_rn expects argument 1 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float32),
        '__nv_fmaf_rn expects argument 2 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    z_ = float32(z_)

    return _call_foreign_function("__nv_fmaf_rn", float32, (x_, y_, z_))

__nv_fmaf_rn = fmaf_rn

@function
def fmaf_ru(x_: float32, y_: float32, z_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_ru.html#__nv_fmaf_ru>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        z_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaf_ru expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaf_ru expects argument 1 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float32),
        '__nv_fmaf_ru expects argument 2 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    z_ = float32(z_)

    return _call_foreign_function("__nv_fmaf_ru", float32, (x_, y_, z_))

__nv_fmaf_ru = fmaf_ru

@function
def fmaf_rz(x_: float32, y_: float32, z_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaf_rz.html#__nv_fmaf_rz>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        z_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaf_rz expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaf_rz expects argument 1 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, float32),
        '__nv_fmaf_rz expects argument 2 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    z_ = float32(z_)

    return _call_foreign_function("__nv_fmaf_rz", float32, (x_, y_, z_))

__nv_fmaf_rz = fmaf_rz

@function
def fmax(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmax.html#__nv_fmax>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fmax expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fmax expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_fmax", float64, (x_, y_))

__nv_fmax = fmax

@function
def fmaxf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmaxf.html#__nv_fmaxf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmaxf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmaxf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmaxf", float32, (x_, y_))

__nv_fmaxf = fmaxf

@function
def fmin(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmin.html#__nv_fmin>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fmin expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fmin expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_fmin", float64, (x_, y_))

__nv_fmin = fmin

@function
def fminf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fminf.html#__nv_fminf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fminf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fminf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fminf", float32, (x_, y_))

__nv_fminf = fminf

@function
def fmod(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmod.html#__nv_fmod>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_fmod expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_fmod expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_fmod", float64, (x_, y_))

__nv_fmod = fmod

@function
def fmodf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmodf.html#__nv_fmodf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmodf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmodf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmodf", float32, (x_, y_))

__nv_fmodf = fmodf

@function
def fmul_rd(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rd.html#__nv_fmul_rd>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmul_rd expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmul_rd expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmul_rd", float32, (x_, y_))

__nv_fmul_rd = fmul_rd

@function
def fmul_rn(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rn.html#__nv_fmul_rn>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmul_rn expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmul_rn expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmul_rn", float32, (x_, y_))

__nv_fmul_rn = fmul_rn

@function
def fmul_ru(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_ru.html#__nv_fmul_ru>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmul_ru expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmul_ru expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmul_ru", float32, (x_, y_))

__nv_fmul_ru = fmul_ru

@function
def fmul_rz(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fmul_rz.html#__nv_fmul_rz>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fmul_rz expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fmul_rz expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fmul_rz", float32, (x_, y_))

__nv_fmul_rz = fmul_rz

@function
def frcp_rd(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rd.html#__nv_frcp_rd>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frcp_rd expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_frcp_rd", float32, (x_,))

__nv_frcp_rd = frcp_rd

@function
def frcp_rn(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rn.html#__nv_frcp_rn>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frcp_rn expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_frcp_rn", float32, (x_,))

__nv_frcp_rn = frcp_rn

@function
def frcp_ru(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_ru.html#__nv_frcp_ru>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frcp_ru expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_frcp_ru", float32, (x_,))

__nv_frcp_ru = frcp_ru

@function
def frcp_rz(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frcp_rz.html#__nv_frcp_rz>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frcp_rz expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_frcp_rz", float32, (x_,))

__nv_frcp_rz = frcp_rz

@function
def __nv_frexp(x_: float64, b_: opaque_ptr) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexp.html#__nv_frexp>`__.

    Args:
        x_: ``float64``
        b_: ``opaque_ptr``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_frexp expects argument 0 to have type float64'
    )
    static_assert(
        satisfies_pointer_constraint(b_, opaque_ptr),
        '__nv_frexp expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float64(x_)
    b_ = address_space_cast(b_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_frexp", float64, (x_, b_))

@function
def __nv_frexpf(x_: float32, b_: opaque_ptr) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexpf.html#__nv_frexpf>`__.

    Args:
        x_: ``float32``
        b_: ``opaque_ptr``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frexpf expects argument 0 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(b_, opaque_ptr),
        '__nv_frexpf expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    b_ = address_space_cast(b_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_frexpf", float32, (x_, b_))

@function
def frsqrt_rn(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frsqrt_rn.html#__nv_frsqrt_rn>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_frsqrt_rn expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_frsqrt_rn", float32, (x_,))

__nv_frsqrt_rn = frsqrt_rn

@function
def fsqrt_rd(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rd.html#__nv_fsqrt_rd>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsqrt_rd expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fsqrt_rd", float32, (x_,))

__nv_fsqrt_rd = fsqrt_rd

@function
def fsqrt_rn(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rn.html#__nv_fsqrt_rn>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsqrt_rn expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fsqrt_rn", float32, (x_,))

__nv_fsqrt_rn = fsqrt_rn

@function
def fsqrt_ru(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_ru.html#__nv_fsqrt_ru>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsqrt_ru expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fsqrt_ru", float32, (x_,))

__nv_fsqrt_ru = fsqrt_ru

@function
def fsqrt_rz(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsqrt_rz.html#__nv_fsqrt_rz>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsqrt_rz expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_fsqrt_rz", float32, (x_,))

__nv_fsqrt_rz = fsqrt_rz

@function
def fsub_rd(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rd.html#__nv_fsub_rd>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsub_rd expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fsub_rd expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fsub_rd", float32, (x_, y_))

__nv_fsub_rd = fsub_rd

@function
def fsub_rn(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rn.html#__nv_fsub_rn>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsub_rn expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fsub_rn expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fsub_rn", float32, (x_, y_))

__nv_fsub_rn = fsub_rn

@function
def fsub_ru(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_ru.html#__nv_fsub_ru>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsub_ru expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fsub_ru expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fsub_ru", float32, (x_, y_))

__nv_fsub_ru = fsub_ru

@function
def fsub_rz(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fsub_rz.html#__nv_fsub_rz>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_fsub_rz expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_fsub_rz expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_fsub_rz", float32, (x_, y_))

__nv_fsub_rz = fsub_rz

@function
def hadd(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hadd.html#__nv_hadd>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_hadd expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_hadd expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_hadd", int32, (x_, y_))

__nv_hadd = hadd

@function
def half2float(h_: int16) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_half2float.html#__nv_half2float>`__.

    Args:
        h_: ``int16``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(h_, int16),
        '__nv_half2float expects argument 0 to have type int16'
    )

    h_ = int16(h_)

    return _call_foreign_function("__nv_half2float", float32, (h_,))

__nv_half2float = half2float

@function
def hiloint2double(x_: int32, y_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hiloint2double.html#__nv_hiloint2double>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_hiloint2double expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_hiloint2double expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_hiloint2double", float64, (x_, y_))

__nv_hiloint2double = hiloint2double

@function
def hypot(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypot.html#__nv_hypot>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_hypot expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_hypot expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_hypot", float64, (x_, y_))

__nv_hypot = hypot

@function
def hypotf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_hypotf.html#__nv_hypotf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_hypotf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_hypotf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_hypotf", float32, (x_, y_))

__nv_hypotf = hypotf

@function
def ilogb(x_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogb.html#__nv_ilogb>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ilogb expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_ilogb", int32, (x_,))

__nv_ilogb = ilogb

@function
def ilogbf(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ilogbf.html#__nv_ilogbf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_ilogbf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_ilogbf", int32, (x_,))

__nv_ilogbf = ilogbf

@function
def int2double_rn(i_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2double_rn.html#__nv_int2double_rn>`__.

    Args:
        i_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(i_, int32),
        '__nv_int2double_rn expects argument 0 to have type int32'
    )

    i_ = int32(i_)

    return _call_foreign_function("__nv_int2double_rn", float64, (i_,))

__nv_int2double_rn = int2double_rn

@function
def int2float_rd(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rd.html#__nv_int2float_rd>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_int2float_rd expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_int2float_rd", float32, (in_,))

__nv_int2float_rd = int2float_rd

@function
def int2float_rn(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rn.html#__nv_int2float_rn>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_int2float_rn expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_int2float_rn", float32, (in_,))

__nv_int2float_rn = int2float_rn

@function
def int2float_ru(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_ru.html#__nv_int2float_ru>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_int2float_ru expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_int2float_ru", float32, (in_,))

__nv_int2float_ru = int2float_ru

@function
def int2float_rz(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int2float_rz.html#__nv_int2float_rz>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_int2float_rz expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_int2float_rz", float32, (in_,))

__nv_int2float_rz = int2float_rz

@function
def int_as_float(x_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_int_as_float.html#__nv_int_as_float>`__.

    Args:
        x_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_int_as_float expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_int_as_float", float32, (x_,))

__nv_int_as_float = int_as_float

@function
def isfinited(x_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isfinited.html#__nv_isfinited>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_isfinited expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_isfinited", int32, (x_,))

__nv_isfinited = isfinited

@function
def isinfd(x_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinfd.html#__nv_isinfd>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_isinfd expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_isinfd", int32, (x_,))

__nv_isinfd = isinfd

@function
def isinff(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isinff.html#__nv_isinff>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_isinff expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_isinff", int32, (x_,))

__nv_isinff = isinff

@function
def isnand(x_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnand.html#__nv_isnand>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_isnand expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_isnand", int32, (x_,))

__nv_isnand = isnand

@function
def isnanf(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_isnanf.html#__nv_isnanf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_isnanf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_isnanf", int32, (x_,))

__nv_isnanf = isnanf

@function
def j0(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0.html#__nv_j0>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_j0 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_j0", float64, (x_,))

__nv_j0 = j0

@function
def j0f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j0f.html#__nv_j0f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_j0f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_j0f", float32, (x_,))

__nv_j0f = j0f

@function
def j1(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1.html#__nv_j1>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_j1 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_j1", float64, (x_,))

__nv_j1 = j1

@function
def j1f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_j1f.html#__nv_j1f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_j1f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_j1f", float32, (x_,))

__nv_j1f = j1f

@function
def jn(n_: int32, x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jn.html#__nv_jn>`__.

    Args:
        n_: ``int32``
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(n_, int32),
        '__nv_jn expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_jn expects argument 1 to have type float64'
    )

    n_ = int32(n_)
    x_ = float64(x_)

    return _call_foreign_function("__nv_jn", float64, (n_, x_))

__nv_jn = jn

@function
def jnf(n_: int32, x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_jnf.html#__nv_jnf>`__.

    Args:
        n_: ``int32``
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(n_, int32),
        '__nv_jnf expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_jnf expects argument 1 to have type float32'
    )

    n_ = int32(n_)
    x_ = float32(x_)

    return _call_foreign_function("__nv_jnf", float32, (n_, x_))

__nv_jnf = jnf

@function
def ldexp(x_: float64, y_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexp.html#__nv_ldexp>`__.

    Args:
        x_: ``float64``
        y_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_ldexp expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_ldexp expects argument 1 to have type int32'
    )

    x_ = float64(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_ldexp", float64, (x_, y_))

__nv_ldexp = ldexp

@function
def ldexpf(x_: float32, y_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ldexpf.html#__nv_ldexpf>`__.

    Args:
        x_: ``float32``
        y_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_ldexpf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_ldexpf expects argument 1 to have type int32'
    )

    x_ = float32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_ldexpf", float32, (x_, y_))

__nv_ldexpf = ldexpf

@function
def lgamma(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgamma.html#__nv_lgamma>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_lgamma expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_lgamma", float64, (x_,))

__nv_lgamma = lgamma

@function
def lgammaf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_lgammaf.html#__nv_lgammaf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_lgammaf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_lgammaf", float32, (x_,))

__nv_lgammaf = lgammaf

@function
def ll2double_rd(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rd.html#__nv_ll2double_rd>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2double_rd expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2double_rd", float64, (l_,))

__nv_ll2double_rd = ll2double_rd

@function
def ll2double_rn(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rn.html#__nv_ll2double_rn>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2double_rn expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2double_rn", float64, (l_,))

__nv_ll2double_rn = ll2double_rn

@function
def ll2double_ru(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_ru.html#__nv_ll2double_ru>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2double_ru expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2double_ru", float64, (l_,))

__nv_ll2double_ru = ll2double_ru

@function
def ll2double_rz(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2double_rz.html#__nv_ll2double_rz>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2double_rz expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2double_rz", float64, (l_,))

__nv_ll2double_rz = ll2double_rz

@function
def ll2float_rd(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rd.html#__nv_ll2float_rd>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2float_rd expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2float_rd", float32, (l_,))

__nv_ll2float_rd = ll2float_rd

@function
def ll2float_rn(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rn.html#__nv_ll2float_rn>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2float_rn expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2float_rn", float32, (l_,))

__nv_ll2float_rn = ll2float_rn

@function
def ll2float_ru(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_ru.html#__nv_ll2float_ru>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2float_ru expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2float_ru", float32, (l_,))

__nv_ll2float_ru = ll2float_ru

@function
def ll2float_rz(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ll2float_rz.html#__nv_ll2float_rz>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ll2float_rz expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ll2float_rz", float32, (l_,))

__nv_ll2float_rz = ll2float_rz

@function
def llabs(x_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llabs.html#__nv_llabs>`__.

    Args:
        x_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_llabs expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_llabs", int64, (x_,))

__nv_llabs = llabs

@function
def llmax(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmax.html#__nv_llmax>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_llmax expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_llmax expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_llmax", int64, (x_, y_))

__nv_llmax = llmax

@function
def llmin(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llmin.html#__nv_llmin>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_llmin expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_llmin expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_llmin", int64, (x_, y_))

__nv_llmin = llmin

@function
def llrint(x_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrint.html#__nv_llrint>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_llrint expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_llrint", int64, (x_,))

__nv_llrint = llrint

@function
def llrintf(x_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llrintf.html#__nv_llrintf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_llrintf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_llrintf", int64, (x_,))

__nv_llrintf = llrintf

@function
def llround(x_: float64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llround.html#__nv_llround>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_llround expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_llround", int64, (x_,))

__nv_llround = llround

@function
def llroundf(x_: float32) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_llroundf.html#__nv_llroundf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_llroundf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_llroundf", int64, (x_,))

__nv_llroundf = llroundf

@function
def log(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log.html#__nv_log>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_log expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_log", float64, (x_,))

__nv_log = log

@function
def log10(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10.html#__nv_log10>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_log10 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_log10", float64, (x_,))

__nv_log10 = log10

@function
def log10f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log10f.html#__nv_log10f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_log10f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_log10f", float32, (x_,))

__nv_log10f = log10f

@function
def log1p(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1p.html#__nv_log1p>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_log1p expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_log1p", float64, (x_,))

__nv_log1p = log1p

@function
def log1pf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log1pf.html#__nv_log1pf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_log1pf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_log1pf", float32, (x_,))

__nv_log1pf = log1pf

@function
def log2(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2.html#__nv_log2>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_log2 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_log2", float64, (x_,))

__nv_log2 = log2

@function
def log2f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_log2f.html#__nv_log2f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_log2f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_log2f", float32, (x_,))

__nv_log2f = log2f

@function
def logb(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logb.html#__nv_logb>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_logb expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_logb", float64, (x_,))

__nv_logb = logb

@function
def logbf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logbf.html#__nv_logbf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_logbf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_logbf", float32, (x_,))

__nv_logbf = logbf

@function
def logf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_logf.html#__nv_logf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_logf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_logf", float32, (x_,))

__nv_logf = logf

@function
def longlong_as_double(x_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_longlong_as_double.html#__nv_longlong_as_double>`__.

    Args:
        x_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_longlong_as_double expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_longlong_as_double", float64, (x_,))

__nv_longlong_as_double = longlong_as_double

@function
def max(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_max.html#__nv_max>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_max expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_max expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_max", int32, (x_, y_))

__nv_max = max

@function
def min(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_min.html#__nv_min>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_min expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_min expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_min", int32, (x_, y_))

__nv_min = min

@function
def __nv_modf(x_: float64, b_: opaque_ptr) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modf.html#__nv_modf>`__.

    Args:
        x_: ``float64``
        b_: ``opaque_ptr``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_modf expects argument 0 to have type float64'
    )
    static_assert(
        satisfies_pointer_constraint(b_, opaque_ptr),
        '__nv_modf expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float64(x_)
    b_ = address_space_cast(b_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_modf", float64, (x_, b_))

@function
def __nv_modff(x_: float32, b_: opaque_ptr) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modff.html#__nv_modff>`__.

    Args:
        x_: ``float32``
        b_: ``opaque_ptr``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_modff expects argument 0 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(b_, opaque_ptr),
        '__nv_modff expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    b_ = address_space_cast(b_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_modff", float32, (x_, b_))

@function
def mul24(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul24.html#__nv_mul24>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_mul24 expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_mul24 expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_mul24", int32, (x_, y_))

__nv_mul24 = mul24

@function
def mul64hi(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mul64hi.html#__nv_mul64hi>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_mul64hi expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_mul64hi expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_mul64hi", int64, (x_, y_))

__nv_mul64hi = mul64hi

@function
def mulhi(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_mulhi.html#__nv_mulhi>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_mulhi expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_mulhi expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_mulhi", int32, (x_, y_))

__nv_mulhi = mulhi

@function
def nearbyint(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyint.html#__nv_nearbyint>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_nearbyint expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_nearbyint", float64, (x_,))

__nv_nearbyint = nearbyint

@function
def nearbyintf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nearbyintf.html#__nv_nearbyintf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_nearbyintf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_nearbyintf", float32, (x_,))

__nv_nearbyintf = nearbyintf

@function
def nextafter(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafter.html#__nv_nextafter>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_nextafter expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_nextafter expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_nextafter", float64, (x_, y_))

__nv_nextafter = nextafter

@function
def nextafterf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_nextafterf.html#__nv_nextafterf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_nextafterf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_nextafterf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_nextafterf", float32, (x_, y_))

__nv_nextafterf = nextafterf

@function
def normcdf(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdf.html#__nv_normcdf>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_normcdf expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_normcdf", float64, (x_,))

__nv_normcdf = normcdf

@function
def normcdff(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdff.html#__nv_normcdff>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_normcdff expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_normcdff", float32, (x_,))

__nv_normcdff = normcdff

@function
def normcdfinv(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinv.html#__nv_normcdfinv>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_normcdfinv expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_normcdfinv", float64, (x_,))

__nv_normcdfinv = normcdfinv

@function
def normcdfinvf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_normcdfinvf.html#__nv_normcdfinvf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_normcdfinvf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_normcdfinvf", float32, (x_,))

__nv_normcdfinvf = normcdfinvf

@function
def popc(x_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popc.html#__nv_popc>`__.

    Args:
        x_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_popc expects argument 0 to have type int32'
    )

    x_ = int32(x_)

    return _call_foreign_function("__nv_popc", int32, (x_,))

__nv_popc = popc

@function
def popcll(x_: int64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_popcll.html#__nv_popcll>`__.

    Args:
        x_: ``int64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_popcll expects argument 0 to have type int64'
    )

    x_ = int64(x_)

    return _call_foreign_function("__nv_popcll", int32, (x_,))

__nv_popcll = popcll

@function
def pow(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_pow.html#__nv_pow>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_pow expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_pow expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_pow", float64, (x_, y_))

__nv_pow = pow

@function
def powf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powf.html#__nv_powf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_powf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_powf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_powf", float32, (x_, y_))

__nv_powf = powf

@function
def powi(x_: float64, y_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powi.html#__nv_powi>`__.

    Args:
        x_: ``float64``
        y_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_powi expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_powi expects argument 1 to have type int32'
    )

    x_ = float64(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_powi", float64, (x_, y_))

__nv_powi = powi

@function
def powif(x_: float32, y_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_powif.html#__nv_powif>`__.

    Args:
        x_: ``float32``
        y_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_powif expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_powif expects argument 1 to have type int32'
    )

    x_ = float32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_powif", float32, (x_, y_))

__nv_powif = powif

@function
def rcbrt(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrt.html#__nv_rcbrt>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_rcbrt expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_rcbrt", float64, (x_,))

__nv_rcbrt = rcbrt

@function
def rcbrtf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rcbrtf.html#__nv_rcbrtf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_rcbrtf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_rcbrtf", float32, (x_,))

__nv_rcbrtf = rcbrtf

@function
def remainder(x_: float64, y_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainder.html#__nv_remainder>`__.

    Args:
        x_: ``float64``
        y_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_remainder expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_remainder expects argument 1 to have type float64'
    )

    x_ = float64(x_)
    y_ = float64(y_)

    return _call_foreign_function("__nv_remainder", float64, (x_, y_))

__nv_remainder = remainder

@function
def remainderf(x_: float32, y_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remainderf.html#__nv_remainderf>`__.

    Args:
        x_: ``float32``
        y_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_remainderf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_remainderf expects argument 1 to have type float32'
    )

    x_ = float32(x_)
    y_ = float32(y_)

    return _call_foreign_function("__nv_remainderf", float32, (x_, y_))

__nv_remainderf = remainderf

@function
def __nv_remquo(x_: float64, y_: float64, c_: opaque_ptr) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquo.html#__nv_remquo>`__.

    Args:
        x_: ``float64``
        y_: ``float64``
        c_: ``opaque_ptr``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_remquo expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float64),
        '__nv_remquo expects argument 1 to have type float64'
    )
    static_assert(
        satisfies_pointer_constraint(c_, opaque_ptr),
        '__nv_remquo expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float64(x_)
    y_ = float64(y_)
    c_ = address_space_cast(c_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_remquo", float64, (x_, y_, c_))

@function
def __nv_remquof(x_: float32, y_: float32, quo_: opaque_ptr) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquof.html#__nv_remquof>`__.

    Args:
        x_: ``float32``
        y_: ``float32``
        quo_: ``opaque_ptr``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_remquof expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, float32),
        '__nv_remquof expects argument 1 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(quo_, opaque_ptr),
        '__nv_remquof expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    y_ = float32(y_)
    quo_ = address_space_cast(quo_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_remquof", float32, (x_, y_, quo_))

@function
def rhadd(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rhadd.html#__nv_rhadd>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_rhadd expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_rhadd expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_rhadd", int32, (x_, y_))

__nv_rhadd = rhadd

@function
def rint(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rint.html#__nv_rint>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_rint expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_rint", float64, (x_,))

__nv_rint = rint

@function
def rintf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rintf.html#__nv_rintf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_rintf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_rintf", float32, (x_,))

__nv_rintf = rintf

@function
def round(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_round.html#__nv_round>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_round expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_round", float64, (x_,))

__nv_round = round

@function
def roundf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_roundf.html#__nv_roundf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_roundf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_roundf", float32, (x_,))

__nv_roundf = roundf

@function
def rsqrt(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrt.html#__nv_rsqrt>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_rsqrt expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_rsqrt", float64, (x_,))

__nv_rsqrt = rsqrt

@function
def rsqrtf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_rsqrtf.html#__nv_rsqrtf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_rsqrtf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_rsqrtf", float32, (x_,))

__nv_rsqrtf = rsqrtf

@function
def sad(x_: int32, y_: int32, z_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sad.html#__nv_sad>`__.

    Args:
        x_: ``int32``
        y_: ``int32``
        z_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_sad expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_sad expects argument 1 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, int32),
        '__nv_sad expects argument 2 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)
    z_ = int32(z_)

    return _call_foreign_function("__nv_sad", int32, (x_, y_, z_))

__nv_sad = sad

@function
def saturatef(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_saturatef.html#__nv_saturatef>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_saturatef expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_saturatef", float32, (x_,))

__nv_saturatef = saturatef

@function
def scalbn(x_: float64, y_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbn.html#__nv_scalbn>`__.

    Args:
        x_: ``float64``
        y_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_scalbn expects argument 0 to have type float64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_scalbn expects argument 1 to have type int32'
    )

    x_ = float64(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_scalbn", float64, (x_, y_))

__nv_scalbn = scalbn

@function
def scalbnf(x_: float32, y_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_scalbnf.html#__nv_scalbnf>`__.

    Args:
        x_: ``float32``
        y_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_scalbnf expects argument 0 to have type float32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_scalbnf expects argument 1 to have type int32'
    )

    x_ = float32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_scalbnf", float32, (x_, y_))

__nv_scalbnf = scalbnf

@function
def signbitd(x_: float64) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitd.html#__nv_signbitd>`__.

    Args:
        x_: ``float64``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_signbitd expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_signbitd", int32, (x_,))

__nv_signbitd = signbitd

@function
def signbitf(x_: float32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_signbitf.html#__nv_signbitf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_signbitf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_signbitf", int32, (x_,))

__nv_signbitf = signbitf

@function
def sin(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sin.html#__nv_sin>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sin expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_sin", float64, (x_,))

__nv_sin = sin

@function
def __nv_sincos(x_: float64, sptr_: opaque_ptr, cptr_: opaque_ptr) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincos.html#__nv_sincos>`__.

    Args:
        x_: ``float64``
        sptr_: ``opaque_ptr``
        cptr_: ``opaque_ptr``

    Returns:
        ``None``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sincos expects argument 0 to have type float64'
    )
    static_assert(
        satisfies_pointer_constraint(sptr_, opaque_ptr),
        '__nv_sincos expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )
    static_assert(
        satisfies_pointer_constraint(cptr_, opaque_ptr),
        '__nv_sincos expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float64(x_)
    sptr_ = address_space_cast(sptr_, MemorySpace.GENERIC)
    cptr_ = address_space_cast(cptr_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_sincos", None, (x_, sptr_, cptr_))

@function
def __nv_sincosf(x_: float32, sptr_: opaque_ptr, cptr_: opaque_ptr) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincosf.html#__nv_sincosf>`__.

    Args:
        x_: ``float32``
        sptr_: ``opaque_ptr``
        cptr_: ``opaque_ptr``

    Returns:
        ``None``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sincosf expects argument 0 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(sptr_, opaque_ptr),
        '__nv_sincosf expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )
    static_assert(
        satisfies_pointer_constraint(cptr_, opaque_ptr),
        '__nv_sincosf expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    sptr_ = address_space_cast(sptr_, MemorySpace.GENERIC)
    cptr_ = address_space_cast(cptr_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_sincosf", None, (x_, sptr_, cptr_))

@function
def __nv_sincospi(x_: float64, sptr_: opaque_ptr, cptr_: opaque_ptr) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospi.html#__nv_sincospi>`__.

    Args:
        x_: ``float64``
        sptr_: ``opaque_ptr``
        cptr_: ``opaque_ptr``

    Returns:
        ``None``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sincospi expects argument 0 to have type float64'
    )
    static_assert(
        satisfies_pointer_constraint(sptr_, opaque_ptr),
        '__nv_sincospi expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )
    static_assert(
        satisfies_pointer_constraint(cptr_, opaque_ptr),
        '__nv_sincospi expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float64(x_)
    sptr_ = address_space_cast(sptr_, MemorySpace.GENERIC)
    cptr_ = address_space_cast(cptr_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_sincospi", None, (x_, sptr_, cptr_))

@function
def __nv_sincospif(x_: float32, sptr_: opaque_ptr, cptr_: opaque_ptr) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospif.html#__nv_sincospif>`__.

    Args:
        x_: ``float32``
        sptr_: ``opaque_ptr``
        cptr_: ``opaque_ptr``

    Returns:
        ``None``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sincospif expects argument 0 to have type float32'
    )
    static_assert(
        satisfies_pointer_constraint(sptr_, opaque_ptr),
        '__nv_sincospif expects argument 1 to satisfy '
        'pointer constraint opaque_ptr'
    )
    static_assert(
        satisfies_pointer_constraint(cptr_, opaque_ptr),
        '__nv_sincospif expects argument 2 to satisfy '
        'pointer constraint opaque_ptr'
    )

    x_ = float32(x_)
    sptr_ = address_space_cast(sptr_, MemorySpace.GENERIC)
    cptr_ = address_space_cast(cptr_, MemorySpace.GENERIC)

    return _call_foreign_function("__nv_sincospif", None, (x_, sptr_, cptr_))

@function
def sinf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinf.html#__nv_sinf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sinf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_sinf", float32, (x_,))

__nv_sinf = sinf

@function
def sinh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinh.html#__nv_sinh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sinh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_sinh", float64, (x_,))

__nv_sinh = sinh

@function
def sinhf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinhf.html#__nv_sinhf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sinhf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_sinhf", float32, (x_,))

__nv_sinhf = sinhf

@function
def sinpi(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpi.html#__nv_sinpi>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sinpi expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_sinpi", float64, (x_,))

__nv_sinpi = sinpi

@function
def sinpif(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sinpif.html#__nv_sinpif>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sinpif expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_sinpif", float32, (x_,))

__nv_sinpif = sinpif

@function
def sqrt(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrt.html#__nv_sqrt>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_sqrt expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_sqrt", float64, (x_,))

__nv_sqrt = sqrt

@function
def sqrtf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sqrtf.html#__nv_sqrtf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_sqrtf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_sqrtf", float32, (x_,))

__nv_sqrtf = sqrtf

@function
def tan(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tan.html#__nv_tan>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_tan expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_tan", float64, (x_,))

__nv_tan = tan

@function
def tanf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanf.html#__nv_tanf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_tanf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_tanf", float32, (x_,))

__nv_tanf = tanf

@function
def tanh(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanh.html#__nv_tanh>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_tanh expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_tanh", float64, (x_,))

__nv_tanh = tanh

@function
def tanhf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tanhf.html#__nv_tanhf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_tanhf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_tanhf", float32, (x_,))

__nv_tanhf = tanhf

@function
def tgamma(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgamma.html#__nv_tgamma>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_tgamma expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_tgamma", float64, (x_,))

__nv_tgamma = tgamma

@function
def tgammaf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_tgammaf.html#__nv_tgammaf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_tgammaf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_tgammaf", float32, (x_,))

__nv_tgammaf = tgammaf

@function
def trunc(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_trunc.html#__nv_trunc>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_trunc expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_trunc", float64, (x_,))

__nv_trunc = trunc

@function
def truncf(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_truncf.html#__nv_truncf>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_truncf expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_truncf", float32, (x_,))

__nv_truncf = truncf

@function
def uhadd(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uhadd.html#__nv_uhadd>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_uhadd expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_uhadd expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_uhadd", int32, (x_, y_))

__nv_uhadd = uhadd

@function
def uint2double_rn(i_: int32) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2double_rn.html#__nv_uint2double_rn>`__.

    Args:
        i_: ``int32``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(i_, int32),
        '__nv_uint2double_rn expects argument 0 to have type int32'
    )

    i_ = int32(i_)

    return _call_foreign_function("__nv_uint2double_rn", float64, (i_,))

__nv_uint2double_rn = uint2double_rn

@function
def uint2float_rd(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rd.html#__nv_uint2float_rd>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_uint2float_rd expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_uint2float_rd", float32, (in_,))

__nv_uint2float_rd = uint2float_rd

@function
def uint2float_rn(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rn.html#__nv_uint2float_rn>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_uint2float_rn expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_uint2float_rn", float32, (in_,))

__nv_uint2float_rn = uint2float_rn

@function
def uint2float_ru(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_ru.html#__nv_uint2float_ru>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_uint2float_ru expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_uint2float_ru", float32, (in_,))

__nv_uint2float_ru = uint2float_ru

@function
def uint2float_rz(in_: int32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_uint2float_rz.html#__nv_uint2float_rz>`__.

    Args:
        in_: ``int32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(in_, int32),
        '__nv_uint2float_rz expects argument 0 to have type int32'
    )

    in_ = int32(in_)

    return _call_foreign_function("__nv_uint2float_rz", float32, (in_,))

__nv_uint2float_rz = uint2float_rz

@function
def ull2double_rd(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rd.html#__nv_ull2double_rd>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2double_rd expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2double_rd", float64, (l_,))

__nv_ull2double_rd = ull2double_rd

@function
def ull2double_rn(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rn.html#__nv_ull2double_rn>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2double_rn expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2double_rn", float64, (l_,))

__nv_ull2double_rn = ull2double_rn

@function
def ull2double_ru(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_ru.html#__nv_ull2double_ru>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2double_ru expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2double_ru", float64, (l_,))

__nv_ull2double_ru = ull2double_ru

@function
def ull2double_rz(l_: int64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2double_rz.html#__nv_ull2double_rz>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2double_rz expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2double_rz", float64, (l_,))

__nv_ull2double_rz = ull2double_rz

@function
def ull2float_rd(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rd.html#__nv_ull2float_rd>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2float_rd expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2float_rd", float32, (l_,))

__nv_ull2float_rd = ull2float_rd

@function
def ull2float_rn(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rn.html#__nv_ull2float_rn>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2float_rn expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2float_rn", float32, (l_,))

__nv_ull2float_rn = ull2float_rn

@function
def ull2float_ru(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_ru.html#__nv_ull2float_ru>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2float_ru expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2float_ru", float32, (l_,))

__nv_ull2float_ru = ull2float_ru

@function
def ull2float_rz(l_: int64) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ull2float_rz.html#__nv_ull2float_rz>`__.

    Args:
        l_: ``int64``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(l_, int64),
        '__nv_ull2float_rz expects argument 0 to have type int64'
    )

    l_ = int64(l_)

    return _call_foreign_function("__nv_ull2float_rz", float32, (l_,))

__nv_ull2float_rz = ull2float_rz

@function
def ullmax(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmax.html#__nv_ullmax>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_ullmax expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_ullmax expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_ullmax", int64, (x_, y_))

__nv_ullmax = ullmax

@function
def ullmin(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ullmin.html#__nv_ullmin>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_ullmin expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_ullmin expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_ullmin", int64, (x_, y_))

__nv_ullmin = ullmin

@function
def umax(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umax.html#__nv_umax>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_umax expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_umax expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_umax", int32, (x_, y_))

__nv_umax = umax

@function
def umin(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umin.html#__nv_umin>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_umin expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_umin expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_umin", int32, (x_, y_))

__nv_umin = umin

@function
def umul24(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul24.html#__nv_umul24>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_umul24 expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_umul24 expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_umul24", int32, (x_, y_))

__nv_umul24 = umul24

@function
def umul64hi(x_: int64, y_: int64) -> int64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umul64hi.html#__nv_umul64hi>`__.

    Args:
        x_: ``int64``
        y_: ``int64``

    Returns:
        ``int64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int64),
        '__nv_umul64hi expects argument 0 to have type int64'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int64),
        '__nv_umul64hi expects argument 1 to have type int64'
    )

    x_ = int64(x_)
    y_ = int64(y_)

    return _call_foreign_function("__nv_umul64hi", int64, (x_, y_))

__nv_umul64hi = umul64hi

@function
def umulhi(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_umulhi.html#__nv_umulhi>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_umulhi expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_umulhi expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_umulhi", int32, (x_, y_))

__nv_umulhi = umulhi

@function
def urhadd(x_: int32, y_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_urhadd.html#__nv_urhadd>`__.

    Args:
        x_: ``int32``
        y_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_urhadd expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_urhadd expects argument 1 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)

    return _call_foreign_function("__nv_urhadd", int32, (x_, y_))

__nv_urhadd = urhadd

@function
def usad(x_: int32, y_: int32, z_: int32) -> int32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_usad.html#__nv_usad>`__.

    Args:
        x_: ``int32``
        y_: ``int32``
        z_: ``int32``

    Returns:
        ``int32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, int32),
        '__nv_usad expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(y_, int32),
        '__nv_usad expects argument 1 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(z_, int32),
        '__nv_usad expects argument 2 to have type int32'
    )

    x_ = int32(x_)
    y_ = int32(y_)
    z_ = int32(z_)

    return _call_foreign_function("__nv_usad", int32, (x_, y_, z_))

__nv_usad = usad

@function
def y0(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0.html#__nv_y0>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_y0 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_y0", float64, (x_,))

__nv_y0 = y0

@function
def y0f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y0f.html#__nv_y0f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_y0f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_y0f", float32, (x_,))

__nv_y0f = y0f

@function
def y1(x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1.html#__nv_y1>`__.

    Args:
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_y1 expects argument 0 to have type float64'
    )

    x_ = float64(x_)

    return _call_foreign_function("__nv_y1", float64, (x_,))

__nv_y1 = y1

@function
def y1f(x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_y1f.html#__nv_y1f>`__.

    Args:
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_y1f expects argument 0 to have type float32'
    )

    x_ = float32(x_)

    return _call_foreign_function("__nv_y1f", float32, (x_,))

__nv_y1f = y1f

@function
def yn(n_: int32, x_: float64) -> float64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_yn.html#__nv_yn>`__.

    Args:
        n_: ``int32``
        x_: ``float64``

    Returns:
        ``float64``
    '''
    static_assert(
        is_literal_or_exact_dtype(n_, int32),
        '__nv_yn expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(x_, float64),
        '__nv_yn expects argument 1 to have type float64'
    )

    n_ = int32(n_)
    x_ = float64(x_)

    return _call_foreign_function("__nv_yn", float64, (n_, x_))

__nv_yn = yn

@function
def ynf(n_: int32, x_: float32) -> float32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_ynf.html#__nv_ynf>`__.

    Args:
        n_: ``int32``
        x_: ``float32``

    Returns:
        ``float32``
    '''
    static_assert(
        is_literal_or_exact_dtype(n_, int32),
        '__nv_ynf expects argument 0 to have type int32'
    )
    static_assert(
        is_literal_or_exact_dtype(x_, float32),
        '__nv_ynf expects argument 1 to have type float32'
    )

    n_ = int32(n_)
    x_ = float32(x_)

    return _call_foreign_function("__nv_ynf", float32, (n_, x_))

__nv_ynf = ynf
__all__ = (
    "abs",
    "__nv_abs",
    "acos",
    "__nv_acos",
    "acosf",
    "__nv_acosf",
    "acosh",
    "__nv_acosh",
    "acoshf",
    "__nv_acoshf",
    "asin",
    "__nv_asin",
    "asinf",
    "__nv_asinf",
    "asinh",
    "__nv_asinh",
    "asinhf",
    "__nv_asinhf",
    "atan",
    "__nv_atan",
    "atan2",
    "__nv_atan2",
    "atan2f",
    "__nv_atan2f",
    "atanf",
    "__nv_atanf",
    "atanh",
    "__nv_atanh",
    "atanhf",
    "__nv_atanhf",
    "brev",
    "__nv_brev",
    "brevll",
    "__nv_brevll",
    "byte_perm",
    "__nv_byte_perm",
    "cbrt",
    "__nv_cbrt",
    "cbrtf",
    "__nv_cbrtf",
    "ceil",
    "__nv_ceil",
    "ceilf",
    "__nv_ceilf",
    "clz",
    "__nv_clz",
    "clzll",
    "__nv_clzll",
    "copysign",
    "__nv_copysign",
    "copysignf",
    "__nv_copysignf",
    "cos",
    "__nv_cos",
    "cosf",
    "__nv_cosf",
    "cosh",
    "__nv_cosh",
    "coshf",
    "__nv_coshf",
    "cospi",
    "__nv_cospi",
    "cospif",
    "__nv_cospif",
    "dadd_rd",
    "__nv_dadd_rd",
    "dadd_rn",
    "__nv_dadd_rn",
    "dadd_ru",
    "__nv_dadd_ru",
    "dadd_rz",
    "__nv_dadd_rz",
    "ddiv_rd",
    "__nv_ddiv_rd",
    "ddiv_rn",
    "__nv_ddiv_rn",
    "ddiv_ru",
    "__nv_ddiv_ru",
    "ddiv_rz",
    "__nv_ddiv_rz",
    "dmul_rd",
    "__nv_dmul_rd",
    "dmul_rn",
    "__nv_dmul_rn",
    "dmul_ru",
    "__nv_dmul_ru",
    "dmul_rz",
    "__nv_dmul_rz",
    "double2float_rd",
    "__nv_double2float_rd",
    "double2float_rn",
    "__nv_double2float_rn",
    "double2float_ru",
    "__nv_double2float_ru",
    "double2float_rz",
    "__nv_double2float_rz",
    "double2hiint",
    "__nv_double2hiint",
    "double2int_rd",
    "__nv_double2int_rd",
    "double2int_rn",
    "__nv_double2int_rn",
    "double2int_ru",
    "__nv_double2int_ru",
    "double2int_rz",
    "__nv_double2int_rz",
    "double2ll_rd",
    "__nv_double2ll_rd",
    "double2ll_rn",
    "__nv_double2ll_rn",
    "double2ll_ru",
    "__nv_double2ll_ru",
    "double2ll_rz",
    "__nv_double2ll_rz",
    "double2loint",
    "__nv_double2loint",
    "double2uint_rd",
    "__nv_double2uint_rd",
    "double2uint_rn",
    "__nv_double2uint_rn",
    "double2uint_ru",
    "__nv_double2uint_ru",
    "double2uint_rz",
    "__nv_double2uint_rz",
    "double2ull_rd",
    "__nv_double2ull_rd",
    "double2ull_rn",
    "__nv_double2ull_rn",
    "double2ull_ru",
    "__nv_double2ull_ru",
    "double2ull_rz",
    "__nv_double2ull_rz",
    "double_as_longlong",
    "__nv_double_as_longlong",
    "drcp_rd",
    "__nv_drcp_rd",
    "drcp_rn",
    "__nv_drcp_rn",
    "drcp_ru",
    "__nv_drcp_ru",
    "drcp_rz",
    "__nv_drcp_rz",
    "dsqrt_rd",
    "__nv_dsqrt_rd",
    "dsqrt_rn",
    "__nv_dsqrt_rn",
    "dsqrt_ru",
    "__nv_dsqrt_ru",
    "dsqrt_rz",
    "__nv_dsqrt_rz",
    "erf",
    "__nv_erf",
    "erfc",
    "__nv_erfc",
    "erfcf",
    "__nv_erfcf",
    "erfcinv",
    "__nv_erfcinv",
    "erfcinvf",
    "__nv_erfcinvf",
    "erfcx",
    "__nv_erfcx",
    "erfcxf",
    "__nv_erfcxf",
    "erff",
    "__nv_erff",
    "erfinv",
    "__nv_erfinv",
    "erfinvf",
    "__nv_erfinvf",
    "exp",
    "__nv_exp",
    "exp10",
    "__nv_exp10",
    "exp10f",
    "__nv_exp10f",
    "exp2",
    "__nv_exp2",
    "exp2f",
    "__nv_exp2f",
    "expf",
    "__nv_expf",
    "expm1",
    "__nv_expm1",
    "expm1f",
    "__nv_expm1f",
    "fabs",
    "__nv_fabs",
    "fabsf",
    "__nv_fabsf",
    "fadd_rd",
    "__nv_fadd_rd",
    "fadd_rn",
    "__nv_fadd_rn",
    "fadd_ru",
    "__nv_fadd_ru",
    "fadd_rz",
    "__nv_fadd_rz",
    "fast_cosf",
    "__nv_fast_cosf",
    "fast_exp10f",
    "__nv_fast_exp10f",
    "fast_expf",
    "__nv_fast_expf",
    "fast_fdividef",
    "__nv_fast_fdividef",
    "fast_log10f",
    "__nv_fast_log10f",
    "fast_log2f",
    "__nv_fast_log2f",
    "fast_logf",
    "__nv_fast_logf",
    "fast_powf",
    "__nv_fast_powf",
    "__nv_fast_sincosf",
    "fast_sinf",
    "__nv_fast_sinf",
    "fast_tanf",
    "__nv_fast_tanf",
    "fdim",
    "__nv_fdim",
    "fdimf",
    "__nv_fdimf",
    "fdiv_rd",
    "__nv_fdiv_rd",
    "fdiv_rn",
    "__nv_fdiv_rn",
    "fdiv_ru",
    "__nv_fdiv_ru",
    "fdiv_rz",
    "__nv_fdiv_rz",
    "ffs",
    "__nv_ffs",
    "ffsll",
    "__nv_ffsll",
    "finitef",
    "__nv_finitef",
    "float2half_rn",
    "__nv_float2half_rn",
    "float2int_rd",
    "__nv_float2int_rd",
    "float2int_rn",
    "__nv_float2int_rn",
    "float2int_ru",
    "__nv_float2int_ru",
    "float2int_rz",
    "__nv_float2int_rz",
    "float2ll_rd",
    "__nv_float2ll_rd",
    "float2ll_rn",
    "__nv_float2ll_rn",
    "float2ll_ru",
    "__nv_float2ll_ru",
    "float2ll_rz",
    "__nv_float2ll_rz",
    "float2uint_rd",
    "__nv_float2uint_rd",
    "float2uint_rn",
    "__nv_float2uint_rn",
    "float2uint_ru",
    "__nv_float2uint_ru",
    "float2uint_rz",
    "__nv_float2uint_rz",
    "float2ull_rd",
    "__nv_float2ull_rd",
    "float2ull_rn",
    "__nv_float2ull_rn",
    "float2ull_ru",
    "__nv_float2ull_ru",
    "float2ull_rz",
    "__nv_float2ull_rz",
    "float_as_int",
    "__nv_float_as_int",
    "floor",
    "__nv_floor",
    "floorf",
    "__nv_floorf",
    "fma",
    "__nv_fma",
    "fma_rd",
    "__nv_fma_rd",
    "fma_rn",
    "__nv_fma_rn",
    "fma_ru",
    "__nv_fma_ru",
    "fma_rz",
    "__nv_fma_rz",
    "fmaf",
    "__nv_fmaf",
    "fmaf_rd",
    "__nv_fmaf_rd",
    "fmaf_rn",
    "__nv_fmaf_rn",
    "fmaf_ru",
    "__nv_fmaf_ru",
    "fmaf_rz",
    "__nv_fmaf_rz",
    "fmax",
    "__nv_fmax",
    "fmaxf",
    "__nv_fmaxf",
    "fmin",
    "__nv_fmin",
    "fminf",
    "__nv_fminf",
    "fmod",
    "__nv_fmod",
    "fmodf",
    "__nv_fmodf",
    "fmul_rd",
    "__nv_fmul_rd",
    "fmul_rn",
    "__nv_fmul_rn",
    "fmul_ru",
    "__nv_fmul_ru",
    "fmul_rz",
    "__nv_fmul_rz",
    "frcp_rd",
    "__nv_frcp_rd",
    "frcp_rn",
    "__nv_frcp_rn",
    "frcp_ru",
    "__nv_frcp_ru",
    "frcp_rz",
    "__nv_frcp_rz",
    "__nv_frexp",
    "__nv_frexpf",
    "frsqrt_rn",
    "__nv_frsqrt_rn",
    "fsqrt_rd",
    "__nv_fsqrt_rd",
    "fsqrt_rn",
    "__nv_fsqrt_rn",
    "fsqrt_ru",
    "__nv_fsqrt_ru",
    "fsqrt_rz",
    "__nv_fsqrt_rz",
    "fsub_rd",
    "__nv_fsub_rd",
    "fsub_rn",
    "__nv_fsub_rn",
    "fsub_ru",
    "__nv_fsub_ru",
    "fsub_rz",
    "__nv_fsub_rz",
    "hadd",
    "__nv_hadd",
    "half2float",
    "__nv_half2float",
    "hiloint2double",
    "__nv_hiloint2double",
    "hypot",
    "__nv_hypot",
    "hypotf",
    "__nv_hypotf",
    "ilogb",
    "__nv_ilogb",
    "ilogbf",
    "__nv_ilogbf",
    "int2double_rn",
    "__nv_int2double_rn",
    "int2float_rd",
    "__nv_int2float_rd",
    "int2float_rn",
    "__nv_int2float_rn",
    "int2float_ru",
    "__nv_int2float_ru",
    "int2float_rz",
    "__nv_int2float_rz",
    "int_as_float",
    "__nv_int_as_float",
    "isfinited",
    "__nv_isfinited",
    "isinfd",
    "__nv_isinfd",
    "isinff",
    "__nv_isinff",
    "isnand",
    "__nv_isnand",
    "isnanf",
    "__nv_isnanf",
    "j0",
    "__nv_j0",
    "j0f",
    "__nv_j0f",
    "j1",
    "__nv_j1",
    "j1f",
    "__nv_j1f",
    "jn",
    "__nv_jn",
    "jnf",
    "__nv_jnf",
    "ldexp",
    "__nv_ldexp",
    "ldexpf",
    "__nv_ldexpf",
    "lgamma",
    "__nv_lgamma",
    "lgammaf",
    "__nv_lgammaf",
    "ll2double_rd",
    "__nv_ll2double_rd",
    "ll2double_rn",
    "__nv_ll2double_rn",
    "ll2double_ru",
    "__nv_ll2double_ru",
    "ll2double_rz",
    "__nv_ll2double_rz",
    "ll2float_rd",
    "__nv_ll2float_rd",
    "ll2float_rn",
    "__nv_ll2float_rn",
    "ll2float_ru",
    "__nv_ll2float_ru",
    "ll2float_rz",
    "__nv_ll2float_rz",
    "llabs",
    "__nv_llabs",
    "llmax",
    "__nv_llmax",
    "llmin",
    "__nv_llmin",
    "llrint",
    "__nv_llrint",
    "llrintf",
    "__nv_llrintf",
    "llround",
    "__nv_llround",
    "llroundf",
    "__nv_llroundf",
    "log",
    "__nv_log",
    "log10",
    "__nv_log10",
    "log10f",
    "__nv_log10f",
    "log1p",
    "__nv_log1p",
    "log1pf",
    "__nv_log1pf",
    "log2",
    "__nv_log2",
    "log2f",
    "__nv_log2f",
    "logb",
    "__nv_logb",
    "logbf",
    "__nv_logbf",
    "logf",
    "__nv_logf",
    "longlong_as_double",
    "__nv_longlong_as_double",
    "max",
    "__nv_max",
    "min",
    "__nv_min",
    "__nv_modf",
    "__nv_modff",
    "mul24",
    "__nv_mul24",
    "mul64hi",
    "__nv_mul64hi",
    "mulhi",
    "__nv_mulhi",
    "nearbyint",
    "__nv_nearbyint",
    "nearbyintf",
    "__nv_nearbyintf",
    "nextafter",
    "__nv_nextafter",
    "nextafterf",
    "__nv_nextafterf",
    "normcdf",
    "__nv_normcdf",
    "normcdff",
    "__nv_normcdff",
    "normcdfinv",
    "__nv_normcdfinv",
    "normcdfinvf",
    "__nv_normcdfinvf",
    "popc",
    "__nv_popc",
    "popcll",
    "__nv_popcll",
    "pow",
    "__nv_pow",
    "powf",
    "__nv_powf",
    "powi",
    "__nv_powi",
    "powif",
    "__nv_powif",
    "rcbrt",
    "__nv_rcbrt",
    "rcbrtf",
    "__nv_rcbrtf",
    "remainder",
    "__nv_remainder",
    "remainderf",
    "__nv_remainderf",
    "__nv_remquo",
    "__nv_remquof",
    "rhadd",
    "__nv_rhadd",
    "rint",
    "__nv_rint",
    "rintf",
    "__nv_rintf",
    "round",
    "__nv_round",
    "roundf",
    "__nv_roundf",
    "rsqrt",
    "__nv_rsqrt",
    "rsqrtf",
    "__nv_rsqrtf",
    "sad",
    "__nv_sad",
    "saturatef",
    "__nv_saturatef",
    "scalbn",
    "__nv_scalbn",
    "scalbnf",
    "__nv_scalbnf",
    "signbitd",
    "__nv_signbitd",
    "signbitf",
    "__nv_signbitf",
    "sin",
    "__nv_sin",
    "__nv_sincos",
    "__nv_sincosf",
    "__nv_sincospi",
    "__nv_sincospif",
    "sinf",
    "__nv_sinf",
    "sinh",
    "__nv_sinh",
    "sinhf",
    "__nv_sinhf",
    "sinpi",
    "__nv_sinpi",
    "sinpif",
    "__nv_sinpif",
    "sqrt",
    "__nv_sqrt",
    "sqrtf",
    "__nv_sqrtf",
    "tan",
    "__nv_tan",
    "tanf",
    "__nv_tanf",
    "tanh",
    "__nv_tanh",
    "tanhf",
    "__nv_tanhf",
    "tgamma",
    "__nv_tgamma",
    "tgammaf",
    "__nv_tgammaf",
    "trunc",
    "__nv_trunc",
    "truncf",
    "__nv_truncf",
    "uhadd",
    "__nv_uhadd",
    "uint2double_rn",
    "__nv_uint2double_rn",
    "uint2float_rd",
    "__nv_uint2float_rd",
    "uint2float_rn",
    "__nv_uint2float_rn",
    "uint2float_ru",
    "__nv_uint2float_ru",
    "uint2float_rz",
    "__nv_uint2float_rz",
    "ull2double_rd",
    "__nv_ull2double_rd",
    "ull2double_rn",
    "__nv_ull2double_rn",
    "ull2double_ru",
    "__nv_ull2double_ru",
    "ull2double_rz",
    "__nv_ull2double_rz",
    "ull2float_rd",
    "__nv_ull2float_rd",
    "ull2float_rn",
    "__nv_ull2float_rn",
    "ull2float_ru",
    "__nv_ull2float_ru",
    "ull2float_rz",
    "__nv_ull2float_rz",
    "ullmax",
    "__nv_ullmax",
    "ullmin",
    "__nv_ullmin",
    "umax",
    "__nv_umax",
    "umin",
    "__nv_umin",
    "umul24",
    "__nv_umul24",
    "umul64hi",
    "__nv_umul64hi",
    "umulhi",
    "__nv_umulhi",
    "urhadd",
    "__nv_urhadd",
    "usad",
    "__nv_usad",
    "y0",
    "__nv_y0",
    "y0f",
    "__nv_y0f",
    "y1",
    "__nv_y1",
    "y1f",
    "__nv_y1f",
    "yn",
    "__nv_yn",
    "ynf",
    "__nv_ynf",
)
