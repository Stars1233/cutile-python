# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import cuda.lang._datatype as datatype
import builtins
import math as host_math
import sys
import torch
import pytest
from cuda.lang import compile_simt
from cuda.lang._stub import math as device_math
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import TileTypeError
from cuda.lang._fp_utils import _FLOAT_SMALLEST_NORMAL, isnormal


rng = torch.Generator().manual_seed(0)
FLOAT_TYPES = (
    cl.float16,
    cl.float32,
    cl.float64,
)
FLOAT_TOLERANCES = {
    cl.float16: dict(rel=1e-2, abs=1e-2),
    cl.float32: dict(rel=1e-5, abs=1e-5),
    cl.float64: dict(rel=1e-10, abs=1e-10),
}


def assert_close_float(actual, expected, dtype):
    tol = FLOAT_TOLERANCES[dtype]
    torch.testing.assert_close(actual, expected, rtol=tol["rel"], atol=tol["abs"])


SIGNED_INT_TYPES = datatype.signed_integral_dtypes
UNSIGNED_INT_TYPES = datatype.unsigned_integral_dtypes

UNARY_FLOAT_OPS = (
    (device_math.ceil, host_math.ceil),
    (device_math.exp, host_math.exp),
    (device_math.sin, host_math.sin),
    (device_math.cos, host_math.cos),
    (device_math.tan, host_math.tan),
    (device_math.sinh, host_math.sinh),
    (device_math.cosh, host_math.cosh),
    (device_math.tanh, host_math.tanh),
    (device_math.sqrt, host_math.sqrt),
    (device_math.rsqrt, lambda x: 1 / host_math.sqrt(x)),
    (device_math.floor, host_math.floor),
    (device_math.log, host_math.log),
    (device_math.log2, host_math.log2),
    (device_math.abs, builtins.abs),
)

BINARY_FLOAT_OPS = ((device_math.atan2, host_math.atan2),)

FPCLASS_OPS = (
    (device_math.isinf, host_math.isinf),
    (device_math.isnan, host_math.isnan),
    (device_math.isfinite, host_math.isfinite),
    (device_math.isnormal, isnormal),
)


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", FPCLASS_OPS)
@pytest.mark.parametrize(
    "input",
    (
        float("-0.0"),
        float("0.0"),
        float("inf"),
        float("-inf"),
        float("nan"),
        "subnormal",
    ),
)
@pytest.mark.parametrize("vector", (True, False))
def test_math_fpclass(dtype, device_op, host_op, input, vector):
    subnormal = input == "subnormal"
    if subnormal:
        smallest = _FLOAT_SMALLEST_NORMAL[dtype.bitwidth]
        input = smallest / 2

    @cl.kernel
    def kernel(out, inp):
        if vector:
            v = device_op(inp.get_base_pointer().load(count=2))
            out[0] = v[0]
        else:
            out[0] = device_op(inp[0])

    out = torch.zeros(1, dtype=torch.bool).cuda()
    inp = torch.tensor([input, input], dtype=datatype.to_torch_dtype(dtype)).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out, inp))
    if host_op == isnormal:
        expect = host_op(input, dtype.bitwidth)
    else:
        expect = host_op(input)
    got = out.cpu().item()
    assert got == expect, f"{host_op}({input}) {expect=} {got=}"


def test_isnormal_non_arithmetic_float():
    @cl.kernel
    def kernel():
        device_math.isnormal(cl.float8_e4m3fn(float("inf")))

    with pytest.raises(
        TileTypeError, match="Expected scalar or vector satisfying constraint"
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", UNARY_FLOAT_OPS)
def test_math_unary_float(dtype, device_op, host_op):
    @cl.kernel
    def kernel(inp, out):
        out[0] = device_op(inp[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    expected = host_op(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == pytest.approx(expected, **FLOAT_TOLERANCES[dtype])


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="math.exp2 requires Python 3.11+"
)
@pytest.mark.parametrize("dtype", FLOAT_TYPES)
def test_math_exp2(dtype):
    from math import exp2

    @cl.kernel
    def kernel(inp, out):
        out[0] = device_math.exp2(inp[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    expected = exp2(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == pytest.approx(expected, **FLOAT_TOLERANCES[dtype])


def test_math_vector_splat():
    vector_dtype = cl.float32
    scalar_dtype = cl.float64

    @cl.kernel
    def kernel(inp, out):
        with cl.local_array(4, vector_dtype) as arr:
            arr[0] = 0.5
            arr[1] = 1.5
            arr[2] = 2.5
            arr[3] = 3.5
            v = arr.get_base_pointer().load(count=4)
            v = device_math.atan2(v, scalar_dtype(inp[0]))
            out.get_base_pointer().store(v)

    scalar_torch_dt = datatype.to_torch_dtype(scalar_dtype)
    out_torch_dt = datatype.to_torch_dtype(scalar_dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    inp = torch.tensor([host_inp], dtype=scalar_torch_dt, device="cuda")
    out = torch.zeros(4, dtype=out_torch_dt, device="cuda")
    scalar = inp.cpu().item()
    expected = [host_math.atan2(x, scalar) for x in (0.5, 1.5, 2.5, 3.5)]
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert_close_float(
        out.cpu(), torch.tensor(expected, dtype=out_torch_dt), scalar_dtype
    )


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", BINARY_FLOAT_OPS)
def test_math_binary_float(dtype, device_op, host_op):

    @cl.kernel
    def kernel(lhs, rhs, out):
        out[0] = device_op(lhs[0], rhs[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_lhs = torch.rand((), generator=rng).item() + 0.5
    host_rhs = torch.rand((), generator=rng).item() + 0.5
    expected = host_op(host_lhs, host_rhs)
    lhs = torch.tensor([host_lhs], dtype=torch_dt, device="cuda")
    rhs = torch.tensor([host_rhs], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    assert out[0].item() == pytest.approx(expected, **FLOAT_TOLERANCES[dtype])


def test_math_binary_float_promotion():
    dt1, dt2 = cl.float16, cl.float64

    @cl.kernel
    def kernel(lhs, rhs, out):
        out[0] = device_math.atan2(dt1(lhs[0]), dt2(rhs[0]))

    tdt1 = datatype.to_torch_dtype(dt1)
    tdt2 = datatype.to_torch_dtype(dt2)
    host_lhs = torch.rand((), generator=rng).item() + 0.5
    host_rhs = torch.rand((), generator=rng).item() + 0.5
    lhs = torch.tensor([host_lhs], dtype=tdt1, device="cuda")
    rhs = torch.tensor([host_rhs], dtype=tdt2, device="cuda")
    out = torch.tensor([0.0], dtype=tdt2, device="cuda")
    expected = host_math.atan2(lhs.cpu().item(), rhs.cpu().item())
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (lhs, rhs, out))
    assert out[0].item() == pytest.approx(expected, **FLOAT_TOLERANCES[dt2])


@pytest.mark.parametrize("dtype", SIGNED_INT_TYPES)
@pytest.mark.parametrize("host_inp", (-5, 0, 5))
def test_math_abs_signed_int(dtype, host_inp):
    @cl.kernel
    def kernel(inp, out):
        out[0] = device_math.abs(dtype(inp[0]))

    torch_dt = datatype.to_torch_dtype(dtype)
    expected = builtins.abs(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == expected


def test_math_abs_unsigned_int():
    # absolute value of unsigned number should be identity
    @cl.kernel
    def kernel():
        device_math.abs(cl.uint32(5.0))

    result = compile_simt(kernel, [KernelSignature([])])
    assert "math.abs" not in result.mlir


def test_vector():
    @cl.kernel
    def kernel(out):
        with cl.local_array(4, cl.float32) as arr:
            arr[0] = 0.5
            arr[1] = 1.5
            arr[2] = 2.5
            arr[3] = 3.5
            v = arr.get_base_pointer().load(count=4)
            v = device_math.floor(v)
            out.get_base_pointer().store(v)

    out = torch.zeros(4, dtype=torch.float32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    print(out.cpu().tolist())
    torch.testing.assert_close(out.cpu().tolist(), [0.0, 1.0, 2.0, 3.0])


def test_type_error():
    @cl.kernel
    def kernel():
        device_math.sin(cl.int32(5.0))

    with pytest.raises(
        TileTypeError, match="Expected a scalar or vector float type, but got int32"
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())
