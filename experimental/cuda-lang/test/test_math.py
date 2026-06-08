# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import cuda.lang._datatype as datatype
import builtins
import math as host_math
import torch
import pytest
from cuda.lang import compile_simt
from cuda.lang._stub import math as device_math
from cuda.lang.compilation import KernelSignature
from cuda.lang._exception import TileTypeError


FLOAT_TYPES = (
    cl.float16,
    cl.float32,
    cl.float64,
)
SIGNED_INT_TYPES = datatype.signed_integral_dtypes
UNSIGNED_INT_TYPES = datatype.unsigned_integral_dtypes

UNARY_FLOAT_OPS = (
    (device_math.ceil, host_math.ceil),
    (device_math.sin, host_math.sin),
    (device_math.cos, host_math.cos),
    (device_math.tan, host_math.tan),
    (device_math.sinh, host_math.sinh),
    (device_math.cosh, host_math.cosh),
    (device_math.tanh, host_math.tanh),
    (device_math.sqrt, host_math.sqrt),
    (device_math.floor, host_math.floor),
    (device_math.log, host_math.log),
    (device_math.log2, host_math.log2),
    (device_math.abs, builtins.abs),
)


@pytest.mark.parametrize("dtype", FLOAT_TYPES)
@pytest.mark.parametrize("device_op, host_op", UNARY_FLOAT_OPS)
def test_math_unary_float(dtype, device_op, host_op):
    rng = torch.Generator().manual_seed(0)

    @cl.kernel
    def kernel(inp, out):
        out[0] = device_op(inp[0])

    torch_dt = datatype.to_torch_dtype(dtype)
    host_inp = torch.rand((), generator=rng).item() + 0.5
    expected = host_op(host_inp)
    inp = torch.tensor([host_inp], dtype=torch_dt, device="cuda")
    out = torch.tensor([0.0], dtype=torch_dt, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (inp, out))
    assert out[0].item() == pytest.approx(expected, rel=1e-3, abs=1e-3)


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
