# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import operator
import cuda.lang as cl
from cuda.lang._datatype import is_integral, is_signed


_ALL_ARITHMETIC_DTYPES = [
    (torch.bool, cl.bool_),
    (torch.int8, cl.int8),
    (torch.uint8, cl.uint8),
    (torch.int16, cl.int16),
    (torch.uint16, cl.uint16),
    (torch.int32, cl.int32),
    (torch.uint32, cl.uint32),
    (torch.int64, cl.int64),
    (torch.uint64, cl.uint64),
    (torch.float16, cl.float16),
    (torch.float32, cl.float32),
    (torch.float64, cl.float64),
]


def _is_non_bool_arithmetic(dtype):
    _, cl_dtype = dtype
    return cl_dtype != cl.bool_


def _is_integral_dtype(dtype):
    _, cl_dtype = dtype
    return is_integral(cl_dtype)


def _is_signed_integral_dtype(dtype):
    _, cl_dtype = dtype
    return is_integral(cl_dtype) and is_signed(cl_dtype)


_INTEGRAL_DTYPES = list(filter(_is_integral_dtype, _ALL_ARITHMETIC_DTYPES))
_SIGNED_INTEGRAL_DTYPES = list(
    filter(_is_signed_integral_dtype, _ALL_ARITHMETIC_DTYPES)
)


def _dtype_to_str(dtype):
    return str(dtype[1])


@pytest.mark.parametrize("from_dtype", _ALL_ARITHMETIC_DTYPES, ids=_dtype_to_str)
@pytest.mark.parametrize("to_dtype", _ALL_ARITHMETIC_DTYPES, ids=_dtype_to_str)
def test_type_conversions(from_dtype, to_dtype):
    to_torch_dtype, to_cl_dtype = to_dtype
    from_torch_dtype, from_cl_dtype = from_dtype

    @cl.kernel
    def kernel(a, b):
        casted = to_cl_dtype(b[0])
        a[0] = casted

    a = torch.zeros(1, dtype=to_torch_dtype, device="cuda")
    b = torch.tensor([2], dtype=from_torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b))
    assert a[0] == to_cl_dtype._py_type(b[0])


@pytest.mark.parametrize(
    "dtype",
    filter(_is_non_bool_arithmetic, _ALL_ARITHMETIC_DTYPES),
    ids=_dtype_to_str,
)
@pytest.mark.parametrize(
    "operation",
    [operator.add, operator.sub, operator.mul, operator.truediv],
)
def test_arithmetic(dtype, operation):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, b, c):
        x = operation(a[0], b[0])
        c[0] = cl_dtype(x)

    a = torch.tensor([10], dtype=torch_dtype, device="cuda")
    b = torch.tensor([2], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c[0] == cl_dtype._py_type(operation(10, 2))


@pytest.mark.parametrize(
    "dtype",
    filter(_is_non_bool_arithmetic, _ALL_ARITHMETIC_DTYPES),
    ids=_dtype_to_str,
)
@pytest.mark.parametrize(
    "operation",
    [operator.pos, operator.neg],
)
def test_unary_arithmetic(dtype, operation):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, c):
        c[0] = cl_dtype(operation(a[0]))

    a = torch.tensor([10], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, c))
    expected = cl_dtype._py_type(operation(10))
    assert c[0] == expected


@pytest.mark.parametrize("dtype", _INTEGRAL_DTYPES, ids=_dtype_to_str)
@pytest.mark.parametrize("operation", [operator.truediv, operator.floordiv])
def test_integer_division(dtype, operation):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, b, c):
        x = operation(a[0], b[0])
        c[0] = cl_dtype(x)

    a = torch.tensor([10], dtype=torch_dtype, device="cuda")
    b = torch.tensor([3], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c[0] == cl_dtype._py_type(operation(10, 3))


@pytest.mark.parametrize("dtype", _SIGNED_INTEGRAL_DTYPES, ids=_dtype_to_str)
def test_integer_floordiv_signed_rounds_down(dtype):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, b, c):
        c[0] = cl_dtype(a[0] // b[0])

    a = torch.tensor([-3], dtype=torch_dtype, device="cuda")
    b = torch.tensor([2], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c[0] == cl_dtype._py_type(-3 // 2)


@pytest.mark.parametrize("dtype", _INTEGRAL_DTYPES, ids=_dtype_to_str)
@pytest.mark.parametrize(
    "operation,lhs,rhs",
    [
        (operator.and_, 0b1100, 0b1010),
        (operator.or_, 0b1100, 0b1010),
    ],
)
def test_integer_bitwise(dtype, operation, lhs, rhs):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, b, c):
        c[0] = operation(a[0], b[0])

    a = torch.tensor([lhs], dtype=torch_dtype, device="cuda")
    b = torch.tensor([rhs], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c[0] == cl_dtype._py_type(operation(lhs, rhs))


@pytest.mark.parametrize("dtype", _INTEGRAL_DTYPES, ids=_dtype_to_str)
def test_integer_bitwise_multiple_outputs(dtype):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(a, b, out_and, out_or):
        out_and[0] = a[0] & b[0]
        out_or[0] = a[0] | b[0]

    lhs = 0b1100
    rhs = 0b1010
    a = torch.tensor([lhs], dtype=torch_dtype, device="cuda")
    b = torch.tensor([rhs], dtype=torch_dtype, device="cuda")
    out_and = torch.tensor([0], dtype=torch_dtype, device="cuda")
    out_or = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, out_and, out_or))
    assert out_and[0] == cl_dtype._py_type(lhs & rhs)
    assert out_or[0] == cl_dtype._py_type(lhs | rhs)


@pytest.mark.parametrize("dtype", _INTEGRAL_DTYPES, ids=_dtype_to_str)
@pytest.mark.parametrize(
    "operation,lhs,rhs,signed_only",
    [
        (operator.lshift, 3, 2, False),
        (operator.rshift, 48, 3, False),
        (operator.rshift, -16, 2, True),
    ],
)
def test_integer_bitshift(dtype, operation, lhs, rhs, signed_only):
    torch_dtype, cl_dtype = dtype
    if signed_only and not is_signed(cl_dtype):
        # signed right shift case only applies to signed integer dtypes
        return

    @cl.kernel
    def kernel(a, b, c):
        c[0] = operation(a[0], b[0])

    a = torch.tensor([lhs], dtype=torch_dtype, device="cuda")
    b = torch.tensor([rhs], dtype=torch_dtype, device="cuda")
    c = torch.tensor([0], dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (a, b, c))
    assert c[0] == cl_dtype._py_type(operation(lhs, rhs))


@pytest.mark.parametrize(
    "dtype",
    filter(_is_non_bool_arithmetic, _ALL_ARITHMETIC_DTYPES),
    ids=_dtype_to_str,
)
@pytest.mark.parametrize(
    "operation",
    [operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne],
)
def test_comparison(dtype, operation):
    torch_dtype, cl_dtype = dtype

    @cl.kernel
    def kernel(res, x):
        cmp = operation(x[0], x[1])
        res[0] = cmp

    x = [1, 2]
    dx = torch.tensor(x, dtype=torch_dtype, device="cuda")
    res = torch.tensor([0], dtype=torch.bool, device="cuda")
    cl.launch(torch.cuda.current_stream(), (2,), (2,), kernel, (res, dx))
    assert res.cpu()[0] == operation(x[0], x[1])


def test_bool_comparison():
    @cl.kernel
    def kernel(res, x):
        res[0] = x[0] == x[1]
        res[1] = x[0] != x[1]

    x = [True, False]
    dx = torch.tensor(x, dtype=torch.bool, device="cuda")
    res = torch.zeros(2, dtype=torch.bool, device="cuda")
    cl.launch(torch.cuda.current_stream(), (2,), (2,), kernel, (res, dx))
    assert res.cpu()[0] == False  # noqa: E712
    assert res.cpu()[1] == True  # noqa: E712
