# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from conftest import dtype_id, shape_id, requires_tileiras
import torch
import pytest
import cuda.tile as ct
from util import (
    estimate_bench_iter, require_hopper_or_newer, torch_use_tf32_matmul, require_blackwell_or_newer)

from itertools import product
from functools import cache
from math import ceil
from cuda.tile.tune import exhaustive_search

import benchmark_tuning
from kernels.kernel_utils import block_quantize, swizzle_32_4_4
from kernels.matmul import (
    matmul_kernel, matmul_split_k_kernel, batch_matmul_kernel, persistent_matmul_kernel)
from kernels.scaled_matmul import block_scaled_matmul_kernel
from cuda.tile._bytecode.version import BytecodeVersion


def _run_matmul_benchmark(shape, dtype, backend, benchmark,
                          extra_args=(), atol=1e-3, rtol=1e-3):
    m, n, k = shape
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")

    args = (A, B, C) + extra_args

    with torch_use_tf32_matmul():
        backend(*args)
        torch.testing.assert_close(C, A @ B, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args, cudagraph=True)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True
    )

    flop_count = 2 * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def _run_batch_matmul_benchmark(shape, dtype, backend, benchmark,
                                extra_args=(), atol=1e-3, rtol=1e-3):
    b, m, n, k = shape
    A = torch.rand((b, m, k), dtype=torch.float32, device="cuda").to(dtype)
    B = torch.rand((b, k, n), dtype=torch.float32, device="cuda").to(dtype)
    C = torch.zeros((b, m, n), dtype=torch.float32, device="cuda")

    args = (b, A, B, C) + extra_args

    with torch_use_tf32_matmul():
        backend(*args)
        if dtype != torch.float8_e5m2:
            ref = ref_batch_matmul(b, A, B)
            torch.testing.assert_close(C, ref, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args, cudagraph=True)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True,
    )
    flop_count = 2 * b * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, B, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def _make_scaled_matmul_inputs(shape, dtype, scaling_block_size):
    m, n, k = shape

    A = torch.rand((m, k), device='cuda')
    B = torch.rand((n, k), device='cuda')

    A, A_s = block_quantize(A, scaling_block_size, dtype)
    B, B_s = block_quantize(B, scaling_block_size, dtype)

    B = B.T
    B_s = B_s.T

    k = A.shape[-1]

    C = torch.zeros((m, n), dtype=torch.float32, device="cuda")
    return A, B, A_s, B_s, C


def _run_swizzled_scaled_matmul_benchmark(shape, dtype, backend,
                                          benchmark, extra_args=(), atol=1e-3, rtol=1e-3):
    m, n, k = shape
    scaling_block_size = 32  # this must be 32 because of hardware limitations
    A, B, A_s, B_s, C = _make_scaled_matmul_inputs(shape, dtype, scaling_block_size)
    A_s_swizzled = swizzle_32_4_4(A_s)
    B_s_swizzled = swizzle_32_4_4(B_s.T.contiguous())
    args = (A, A_s_swizzled, B, B_s_swizzled, C) + extra_args

    ref_A_s = torch.repeat_interleave(A_s, scaling_block_size, dim=1).to(torch.float32)
    ref_B_s = torch.repeat_interleave(B_s, scaling_block_size, dim=0).to(torch.float32)
    ref = (A.to(torch.float32) * ref_A_s) @ (B.to(torch.float32) * ref_B_s)

    res = backend(*args)
    torch.testing.assert_close(res, ref, atol=atol, rtol=rtol)

    torch.cuda.synchronize()
    warmup_rounds, iterations, rounds = estimate_bench_iter(backend, args, cudagraph=True)
    benchmark.pedantic(
        backend, args,
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations, cudagraph=True
    )

    flop_count = 2 * m * n * k
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (A, A_s, B, B_s, C)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


# =============================== Matmul =============================

@pytest.fixture(params=[
    (1024, 1024, 1024),
    (8192, 8192, 8192),
    (12288, 4096, 2560),
], ids=shape_id)
def matmul_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def matmul_dtype(request):
    return request.param


@pytest.mark.benchmark(group='matmul')
def bench_matmul(matmul_shape, matmul_dtype, backend, benchmark):
    _run_matmul_benchmark(matmul_shape, matmul_dtype, backend, benchmark)


def _matmul_search_space():
    return [
        {"tm": tm, "tn": tn, "tk": tk, "num_ctas": num_ctas}
        for tm, tn, tk, num_ctas in product(
            (128, 256),
            (128, 256),
            (32, 64, 128),
            (1, 2),
        )
    ]


@cache
def get_kernel(kernel, num_ctas):
    return kernel.replace_hints(num_ctas=num_ctas)


def tune_matmul():
    m, n, k = (4096, 4096, 4096)
    dtype = torch.float16
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")
    return exhaustive_search(
        _matmul_search_space(),
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (ceil(m / cfg["tm"]) * ceil(n / cfg["tn"]), 1, 1),
        kernel=matmul_kernel,
        args_fn=lambda cfg: (A, B, C, cfg["tm"], cfg["tn"], cfg["tk"]),
        hints_fn=lambda cfg: {"num_ctas": cfg["num_ctas"]},
    )


def cutile_matmul(A, B, C):
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    cfg = benchmark_tuning.get_tuned_config(tune_matmul)
    tm, tn, tk = cfg["tm"], cfg["tn"], cfg["tk"]
    kernel = get_kernel(matmul_kernel, cfg["num_ctas"])
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))


def torch_matmul(A, B, C):
    with torch_use_tf32_matmul():
        torch.matmul(A, B, out=C)


# =============================== Matmul Split K =============================


@pytest.fixture(params=[
    (256, 256, 4096),
    (128, 128, 8192)
], ids=shape_id)
def split_k_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def split_k_dtype(request):
    return request.param


def _matmul_split_k_search_space():
    return [
        {"tm": tm, "tn": tn, "tk": tk, "split_k": split_k}
        for tm, tn, tk, split_k in product(
            (32, 64),
            (64, 128),
            (128, 256),
            (2, 4, 8),
        )
    ]


def _matmul_split_k_lock_count(m, n):
    return max(
        ceil(m / cfg["tm"]) * ceil(n / cfg["tn"])
        for cfg in _matmul_split_k_search_space()
    )


@pytest.mark.benchmark(group='matmul_split_k')
def bench_matmul_split_k(split_k_shape, split_k_dtype, backend, benchmark):
    m, n, _ = split_k_shape
    LOCKS = torch.zeros(_matmul_split_k_lock_count(m, n), dtype=torch.int32, device="cuda")
    COUNTS = torch.zeros_like(LOCKS)
    extra_args = (LOCKS, COUNTS)
    _run_matmul_benchmark(split_k_shape, split_k_dtype, backend, benchmark, extra_args)


def tune_matmul_split_k():
    m, n, k = (256, 256, 4096)
    dtype = torch.float16
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")
    LOCKS = torch.zeros(_matmul_split_k_lock_count(m, n), dtype=torch.int32, device="cuda")
    COUNTS = torch.zeros_like(LOCKS)
    return exhaustive_search(
        _matmul_split_k_search_space(),
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (
            ceil(m / cfg["tm"]) * ceil(n / cfg["tn"]),
            cfg["split_k"],
            1,
        ),
        kernel=matmul_split_k_kernel,
        args_fn=lambda cfg: (
            A, B, C, LOCKS, COUNTS,
            cfg["tm"], cfg["tn"], cfg["tk"], cfg["split_k"],
        ),
    )


def cutile_matmul_split_k(A, B, C, LOCKS, COUNTS):
    cfg = benchmark_tuning.get_tuned_config(tune_matmul_split_k)
    tm, tn, tk = cfg["tm"], cfg["tn"], cfg["tk"]
    split_k = cfg["split_k"]
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), split_k, 1)
    ct.launch(torch.cuda.current_stream(), grid, matmul_split_k_kernel,
              (A, B, C, LOCKS, COUNTS, tm, tn, tk, split_k))


def torch_matmul_split_k(A, B, C, *args):
    torch_matmul(A, B, C)


# =============================== Batch Matmul in FP8 =============================

@pytest.fixture(params=[
    (2, 1024, 1024, 1024),
    (4, 8192, 8192, 2000),
    (8, 12288, 4096, 2560),
], ids=shape_id)
def batch_matmul_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float8_e4m3fn, torch.float8_e5m2
], ids=dtype_id)
def batch_matmul_dtype(request):
    return request.param


@require_hopper_or_newer()
@pytest.mark.benchmark(group='batch_matmul')
def bench_batch_matmul(batch_matmul_shape, batch_matmul_dtype, backend, benchmark):
    _run_batch_matmul_benchmark(batch_matmul_shape, batch_matmul_dtype, backend, benchmark)


def tune_batch_matmul():
    b, m, n, k = (4, 8192, 8192, 2000)
    fp8_dtype = torch.float8_e4m3fn
    A = torch.rand((b, m, k), dtype=torch.float32, device="cuda").to(fp8_dtype)
    B = torch.rand((b, k, n), dtype=torch.float32, device="cuda").to(fp8_dtype)
    C = torch.zeros((b, m, n), dtype=torch.float32, device="cuda")
    return exhaustive_search(
        _matmul_search_space(),
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (b, ceil(m / cfg["tm"]), ceil(n / cfg["tn"])),
        kernel=batch_matmul_kernel,
        args_fn=lambda cfg: (A, B, C, cfg["tm"], cfg["tn"], cfg["tk"]),
        hints_fn=lambda cfg: {"num_ctas": cfg["num_ctas"]},
    )


def cutile_batch_matmul(bs, A, B, C):
    m, n = A.shape[1], B.shape[2]
    cfg = benchmark_tuning.get_tuned_config(tune_batch_matmul)
    tm, tn, tk = cfg["tm"], cfg["tn"], cfg["tk"]
    kernel = get_kernel(batch_matmul_kernel, cfg["num_ctas"])
    grid = (bs, ct.cdiv(m, tm), ct.cdiv(n, tn))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))


def torch_batch_matmul(bs, A, B, C):
    if A.dtype == torch.float8_e5m2:
        pytest.skip("float8_e5m2 matmul on torch is not supported")
    inv_sa = torch.full((), 1.0, device=A.device, dtype=torch.float32)
    inv_sb = torch.full((), 1.0, device=B.device, dtype=torch.float32)
    with torch_use_tf32_matmul():
        for i in range(bs):
            # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
            # So we need to transpose B to column-major view
            A_row = A[i].contiguous()
            B_col = B[i].transpose(-2, -1).contiguous().transpose(-2, -1)
            C[i] = torch._scaled_mm(
                A_row, B_col, scale_a=inv_sa, scale_b=inv_sb, out_dtype=torch.float32
            )


def ref_batch_matmul(bs, A, B):
    ref = torch.zeros((bs, A.shape[1], B.shape[2]), dtype=torch.float32, device="cuda")
    torch_batch_matmul(bs, A, B, ref)
    return ref


# =============================== Persistent Matmul =============================

@pytest.fixture(params=[
    (1024, 1024, 1024),
    (8192, 8192, 8192),
    (12288, 4096, 2560),
], ids=shape_id)
def persistent_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float16, torch.float32
], ids=dtype_id)
def persistent_dtype(request):
    return request.param


@pytest.mark.benchmark(group='persistent_matmul')
def bench_persistent_matmul(persistent_shape, persistent_dtype, backend, benchmark):
    _run_matmul_benchmark(persistent_shape, persistent_dtype, backend, benchmark)


def tune_persistent_matmul():
    m, n, k = (4096, 4096, 4096)
    dtype = torch.float16
    A = torch.rand((m, k), dtype=dtype, device="cuda")
    B = torch.rand((k, n), dtype=dtype, device="cuda")
    C = torch.zeros((m, n), dtype=dtype, device="cuda")
    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
    return exhaustive_search(
        _matmul_search_space(),
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (
            min(NUM_SMS, ceil(m / cfg["tm"]) * ceil(n / cfg["tn"])),
        ),
        kernel=persistent_matmul_kernel,
        args_fn=lambda cfg: (A, B, C, cfg["tm"], cfg["tn"], cfg["tk"]),
        hints_fn=lambda cfg: {"num_ctas": cfg["num_ctas"]},
    )


def cutile_persistent_matmul(A, B, C):
    NUM_SMS = torch.cuda.get_device_properties(
            "cuda"
        ).multi_processor_count
    M, N = A.shape[0], B.shape[1]
    cfg = benchmark_tuning.get_tuned_config(tune_persistent_matmul)
    tm, tn, tk = cfg["tm"], cfg["tn"], cfg["tk"]
    kernel = get_kernel(persistent_matmul_kernel, cfg["num_ctas"])

    grid_size = min(
        NUM_SMS,
        ct.cdiv(M, tm) * ct.cdiv(N, tn),
    )
    grid = (grid_size,)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (A, B, C, tm, tn, tk))


def torch_persistent_matmul(A, B, C, *args):
    torch_matmul(A, B, C)


# =============================== Scaled Swizzled Matmul =============================

@pytest.fixture(params=[
    (1024, 1024, 1024),
    (8192, 8192, 8192),
    (12288, 4096, 2560),
], ids=shape_id)
def scaled_matmul_shape(request):
    return request.param


@pytest.fixture(params=[
    torch.float8_e4m3fn
], ids=dtype_id)
def scaled_matmul_dtype(request):
    return request.param


@require_blackwell_or_newer()
@requires_tileiras(BytecodeVersion.V_13_3)
@pytest.mark.benchmark(group='mma_scaled_swizzled')
def bench_swizzled_scaled_matmul(scaled_matmul_shape, scaled_matmul_dtype, backend, benchmark):
    _run_swizzled_scaled_matmul_benchmark(scaled_matmul_shape,
                                          scaled_matmul_dtype, backend, benchmark)


def cutile_swizzled_scaled_matmul(A, A_s_swizzled, B, B_s_swizzled, C):
    scaling_block_size = 32
    cfg = benchmark_tuning.get_tuned_config(tune_swizzled_scaled_matmul)
    kernel = get_kernel(block_scaled_matmul_kernel, num_ctas=cfg['num_ctas'])
    tm, tn, tk = cfg['tm'], cfg['tn'], cfg['tk']
    m, n, _ = A.shape[0], B.shape[1], A.shape[1]
    grid = (ct.cdiv(m, tm) * ct.cdiv(n, tn), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (A, A_s_swizzled, B, B_s_swizzled, C, tm, tn, tk, scaling_block_size))
    return C


def torch_swizzled_scaled_matmul(A, A_s_swizzled, B, B_s_swizzled, C):

    return torch.nn.functional.scaled_mm(
                                    A, B,
                                    scale_a=A_s_swizzled, scale_b=B_s_swizzled,
                                    scale_recipe_a=torch.nn.functional.ScalingType.BlockWise1x32,
                                    scale_recipe_b=torch.nn.functional.ScalingType.BlockWise1x32,
                                    swizzle_a=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
                                    swizzle_b=torch.nn.functional.SwizzleType.SWIZZLE_32_4_4,
                                    output_dtype=torch.float32)


def _swizzled_scaled_matmul_search_space():
    return [
        {"tm": tm, "tn": tn, "tk": tk, "num_ctas": num_ctas}
        for tm, tn, tk, num_ctas in product(
            (128, 256),
            (128, 256),
            (128, 256),
            (1, 2),
        )
    ]


def tune_swizzled_scaled_matmul():
    m, n, k = (4096, 4096, 4096)
    dtype = torch.float8_e4m3fn
    scaling_block_size = 32
    A, B, A_s, B_s, C = _make_scaled_matmul_inputs((m, n, k), dtype, scaling_block_size)
    A_s_swizzled = swizzle_32_4_4(A_s)
    B_s_swizzled = swizzle_32_4_4(B_s.T.contiguous())
    with ct.compiler_timeout(5):
        return exhaustive_search(
            _swizzled_scaled_matmul_search_space(),
            torch.cuda.current_stream(),
            grid_fn=lambda cfg: (ct.cdiv(m, cfg["tm"]) * ct.cdiv(n, cfg["tn"]), ),
            kernel=block_scaled_matmul_kernel,
            args_fn=lambda cfg: (A, A_s_swizzled, B, B_s_swizzled, C,
                                 cfg["tm"], cfg["tn"], cfg["tk"], scaling_block_size),
            hints_fn=lambda cfg: {"num_ctas": cfg["num_ctas"]},
        )
