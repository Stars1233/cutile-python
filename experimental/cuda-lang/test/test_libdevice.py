# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch
import inspect

ENTRYPOINTS = tuple(
    filter(
        lambda x: not x.startswith("__nv_"),
        cl.libdevice.__all__,
    )
)


def make_kernel(func, dtypes: tuple) -> cl.kernel:
    '''
    Create a kernel that calls ``func`` with the given dtypes.
    ast2hir needs the source of the program and I don't know
    of a better way than this.
    '''
    match dtypes:
        case (cl.int16,):
            def k(f32, f64, i16, i32, i64):
                func(i16[0])
            return k
        case (cl.int32,):
            def k(f32, f64, i16, i32, i64):
                func(i32[0])
            return k
        case (cl.int32, cl.int32):
            def k(f32, f64, i16, i32, i64):
                func(i32[0], i32[0])
            return k
        case (cl.int32, cl.int32, cl.int32):
            def k(f32, f64, i16, i32, i64):
                func(i32[0], i32[0], i32[0])
            return k
        case (cl.int64,):
            def k(f32, f64, i16, i32, i64):
                func(i64[0])
            return k
        case (cl.int64, cl.int64):
            def k(f32, f64, i16, i32, i64):
                func(i64[0], i64[0])
            return k
        case (cl.int64, cl.int64, cl.int64):
            def k(f32, f64, i16, i32, i64):
                func(i64[0], i64[0], i64[0])
            return k
        case (cl.float32,):
            def k(f32, f64, i16, i32, i64):
                func(f32[0])
            return k
        case (cl.float32, cl.float32):
            def k(f32, f64, i16, i32, i64):
                func(f32[0], f32[0])
            return k
        case (cl.float32, cl.float32, cl.float32):
            def k(f32, f64, i16, i32, i64):
                func(f32[0], f32[0], f32[0])
            return k
        case (cl.float64,):
            def k(f32, f64, i16, i32, i64):
                func(f64[0])
            return k
        case (cl.float64, cl.float64):
            def k(f32, f64, i16, i32, i64):
                func(f64[0], f64[0])
            return k
        case (cl.float64, cl.float64, cl.float64):
            def k(f32, f64, i16, i32, i64):
                func(f64[0], f64[0], f64[0])
            return k
        case (cl.float32, cl.int32):
            def k(f32, f64, i16, i32, i64):
                func(f32[0], i32[0])
            return k
        case (cl.float64, cl.int32):
            def k(f32, f64, i16, i32, i64):
                func(f64[0], i32[0])
            return k
        case (cl.int32, cl.float32):
            def k(f32, f64, i16, i32, i64):
                func(i32[0], f32[0])
            return k
        case (cl.int32, cl.float64):
            def k(f32, f64, i16, i32, i64):
                func(i32[0], f64[0])
            return k
        case _:
            raise ValueError(f"Unsupported dtypes: {dtypes}")


@pytest.mark.parametrize("function_name", ENTRYPOINTS)
def test_libdevice_functions(function_name):
    func = getattr(cl.libdevice, function_name)
    params = inspect.signature(func).parameters
    dtypes = tuple(p.annotation for p in params.values())
    kernel = make_kernel(func, dtypes)
    kernel = cl.kernel(kernel)
    f32 = torch.randn(1, dtype=torch.float32, device="cuda")
    f64 = torch.randn(1, dtype=torch.float64, device="cuda")
    i16 = torch.randint(0, 10, (1,), dtype=torch.int16, device="cuda")
    i32 = torch.randint(0, 10, (1,), dtype=torch.int32, device="cuda")
    i64 = torch.randint(0, 10, (1,), dtype=torch.int64, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (f32, f64, i16, i32, i64),
    )
