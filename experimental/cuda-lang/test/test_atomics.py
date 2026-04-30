# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import cuda.lang as cl
from cuda.lang._exception import TileTypeError

from .util import compile_for_arguments, make_symbolic_tensor

ALL_INT_DTYPES = ["int32", "int64"]
ALL_UINT_DTYPES = ["uint32", "uint64"]
ALL_FLOAT_DTYPES = ["float32", "float64"]
ALL_REAL_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES + ALL_FLOAT_DTYPES
ALL_INTEGER_DTYPES = ALL_INT_DTYPES + ALL_UINT_DTYPES


def _torch_dtype(dtype):
    return getattr(torch, dtype)


def _cl_dtype(dtype):
    return getattr(cl, dtype)


def _scalar(dtype, value):
    return _cl_dtype(dtype)(value)


RMW_CASES = [
    ("atomic_add", ALL_REAL_DTYPES, 7, 3, 10),
    ("atomic_sub", ALL_REAL_DTYPES, 7, 3, 4),
    ("atomic_and", ALL_INTEGER_DTYPES, 0b1110, 0b1011, 0b1010),
    ("atomic_or", ALL_INTEGER_DTYPES, 0b1100, 0b0011, 0b1111),
    ("atomic_xor", ALL_INTEGER_DTYPES, 0b1100, 0b1010, 0b0110),
    ("atomic_min", ALL_REAL_DTYPES, 7, 3, 3),
    ("atomic_max", ALL_REAL_DTYPES, 7, 11, 11),
    ("atomic_inc", ["uint32"], 7, 11, 8),
    ("atomic_dec", ["uint32"], 7, 11, 6),
]

RMW_VARIANTS = [
    (op, dtype, initial, update, expected_new)
    for op, dtypes, initial, update, expected_new in RMW_CASES
    for dtype in dtypes
]


@pytest.mark.parametrize("op,dtype,initial,update,expected_new", RMW_VARIANTS)
def test_atomic_rmw_supported_types(op, dtype, initial, update, expected_new):
    atomic = getattr(cl, op)
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        out[0] = atomic(A, 0, _scalar(dtype, update))

    A = torch.tensor([initial], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([initial], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([expected_new], dtype=torch_dtype))


@pytest.mark.parametrize("dtype", ALL_REAL_DTYPES)
def test_atomic_exch_supported_types(dtype):
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_exch(A, 0, _scalar(dtype, 11))

    A = torch.tensor([7], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([7], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([11], dtype=torch_dtype))


@pytest.mark.parametrize("dtype", ALL_INTEGER_DTYPES)
def test_atomic_cas_supported_types(dtype):
    torch_dtype = _torch_dtype(dtype)

    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_cas(A, 0, _scalar(dtype, 7), _scalar(dtype, 11))

    A = torch.tensor([7], dtype=torch_dtype, device="cuda")
    out = torch.zeros(1, dtype=torch_dtype, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert torch.allclose(out.cpu(), torch.tensor([7], dtype=torch_dtype))
    assert torch.allclose(A.cpu(), torch.tensor([11], dtype=torch_dtype))


def test_atomic_cas_failure():
    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_cas(A, 0, cl.int32(8), cl.int32(11))

    A = torch.tensor([7], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 7
    assert A.cpu()[0].item() == 7


def test_atomic_inc_wrap():
    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_inc(A, 0, cl.uint32(7))

    A = torch.tensor([7], dtype=torch.uint32, device="cuda")
    out = torch.zeros(1, dtype=torch.uint32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 7
    assert A.cpu()[0].item() == 0


def test_atomic_dec_wrap():
    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_dec(A, 0, cl.uint32(7))

    A = torch.tensor([0], dtype=torch.uint32, device="cuda")
    out = torch.zeros(1, dtype=torch.uint32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 0
    assert A.cpu()[0].item() == 7


def test_atomic_tuple_index():
    @cl.kernel
    def kernel(A, out):
        out[0] = cl.atomic_add(A, (0, 1), cl.int32(5))

    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A, out))
    assert out.cpu()[0].item() == 2
    assert A.cpu()[0, 1].item() == 7


def test_atomic_add_requires_matching_dtype():
    def kernel(A):
        cl.atomic_add(A, 0, cl.float32(1))

    with pytest.raises(TileTypeError):
        compile_for_arguments(kernel, [make_symbolic_tensor((1,), cl.int32)])


def test_atomic_cas_requires_matching_dtype():
    def kernel(A):
        cl.atomic_cas(A, 0, cl.int32(1), cl.int64(2))

    with pytest.raises(TileTypeError):
        compile_for_arguments(kernel, [make_symbolic_tensor((1,), cl.int32)])
