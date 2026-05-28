# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import benchmark_tuning
from conftest import dtype_id, shape_id, get_tileiras_version

import pytest
import torch
import cuda.tile as ct
from cuda.tile.tune import exhaustive_search
from cuda.tile._bytecode import BytecodeVersion
import itertools
from math import ceil
from util import estimate_bench_iter
from kernels.rms_norm import (
    rms_norm_kernel, rms_norm_kernel_gather, rms_norm_kernel_static_persistent
)
from functools import cache


timeout = 2  # sec


def get_shape_params():
    return [(65536, 1024),
            (65536, 2048),
            (65536, 4096)]


@pytest.fixture(params=get_shape_params(), ids=shape_id)
def shape(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.float32, torch.bfloat16], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.fixture(params=['persistent', 'gather', 'regular'])
def algo(request):
    return request.param


@pytest.mark.benchmark(group='rms_norm')
def bench_rms_norm(shape, dtype, algo, backend, benchmark):
    x_shape = shape
    w_shape = (shape[1], )
    x = torch.rand(x_shape, dtype=dtype, device="cuda")
    weight = torch.randn(w_shape, dtype=dtype, device="cuda")

    eps = 1e-5

    if algo == 'persistent':
        static_persistent, gather = True, False
    elif algo == 'gather':
        static_persistent, gather = False, True
    else:
        static_persistent, gather = False, False

    if algo == 'persistent' and get_tileiras_version() < BytecodeVersion.V_13_3:
        pytest.skip("earlier version of tileiras has bug compiling this kernel")

    with ct.compiler_timeout(5):
        o = backend(x, weight, eps, static_persistent, gather)
        ref = ref_rms_norm(x, weight, eps)
        torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
        torch.cuda.synchronize()

        warmup_rounds, iterations, rounds = estimate_bench_iter(
            backend, (x, weight, eps, static_persistent, gather),
            cudagraph=True
        )

        benchmark.pedantic(
            backend, (x, weight, eps, static_persistent, gather),
            rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
            cudagraph=True
        )

    M, N = x.shape
    flop_count = M * (4 * N + 2)
    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (x, weight, o)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def _static_persistent_autotune_grid(x, cfg):
    """Grid function for static persistent RMS Norm autotuning"""
    NUM_SMS = torch.cuda.get_device_properties(
        "cuda"
    ).multi_processor_count
    M = x.shape[0]
    grid_size = min(NUM_SMS, ceil(M / cfg["tile_size_m"]))
    return (grid_size,)


def _static_persistent_autotune_configs(x_shape):
    """Iterator of autotune configurations for RMS Norm kernel."""
    ts_m_vals = [2, 4, 8, 16]
    ts_n_vals = [2**9, 2**10, 2**11, 2**12, 2**13, 2**14]
    for ts_m, ts_n, in itertools.product(
        ts_m_vals, ts_n_vals,
    ):
        if ts_n <= x_shape[1] and x_shape[1] % ts_n == 0:
            yield {
                "tile_size_m": ts_m,
                "tile_size_n": ts_n,
            }


def _standard_autotune_configs():
    """Get autotune configurations for RMS Norm kernel"""
    ts_vals = [2**7, 2**8, 2**9, 2**10, 2**11, 2**12]
    num_worker_warps = [4, 8]
    for ts, w in itertools.product(ts_vals, num_worker_warps):
        yield {
            "tile_size": ts,
            "num_worker_warps": w,
        }


@cache
def _rms_norm_gather_kernel(num_worker_warps):
    return rms_norm_kernel_gather.replace_hints(
        num_worker_warps=num_worker_warps
    )


@cache
def _rms_norm_regular_kernel(num_worker_warps):
    return rms_norm_kernel.replace_hints(
        num_worker_warps=num_worker_warps,
    )


def tune_rms_norm(algo, shape, dtype):
    x = torch.rand(shape, dtype=dtype, device="cuda")
    weight = torch.randn((shape[1],), dtype=dtype, device="cuda")
    y = torch.empty_like(x)
    eps = 1e-5

    def hints_fn(cfg):
        return {"num_worker_warps": cfg.get("num_worker_warps", None)}

    with ct.compiler_timeout(timeout):
        if algo == 'persistent':
            search_space = list(_static_persistent_autotune_configs(shape))

            def grid_fn(cfg):
                return _static_persistent_autotune_grid(x, cfg)

            def args_fn(cfg):
                return (x, y.clone(), weight,
                        cfg["tile_size_m"],
                        cfg["tile_size_n"],
                        shape[1], eps)

            kernel = rms_norm_kernel_static_persistent
        else:
            rstd = torch.empty((shape[0],), dtype=torch.float32, device='cuda')
            search_space = list(_standard_autotune_configs())
            kernel = rms_norm_kernel_gather if algo == 'gather' else rms_norm_kernel

            def grid_fn(_cfg):
                return (shape[0], )

            def args_fn(cfg):
                return (x, weight, y.clone(),
                        rstd.clone(),
                        shape[1],
                        eps, cfg["tile_size"])

        return exhaustive_search(
                search_space,
                torch.cuda.current_stream(),
                grid_fn=grid_fn,
                kernel=kernel,
                args_fn=args_fn,
                hints_fn=hints_fn)


def _rms_norm_static_persistent_base(stream, x, y, weight, eps):
    cfg = benchmark_tuning.get_tuned_config(
        tune_rms_norm,
        algo='persistent',
        shape=x.shape,
        dtype=x.dtype
    )
    kernel = rms_norm_kernel_static_persistent
    grid = _static_persistent_autotune_grid(x, cfg)
    ct.launch(
        stream, grid,
        kernel,
        (x, y, weight, cfg["tile_size_m"], cfg["tile_size_n"], x.shape[1], eps),
    )
    return y


def _rms_norm_standard_gather_base(stream, x, weight, y, rstd, N, eps):
    cfg = benchmark_tuning.get_tuned_config(
        tune_rms_norm,
        algo='gather',
        shape=x.shape,
        dtype=x.dtype
    )
    kernel = _rms_norm_gather_kernel(cfg['num_worker_warps'])
    ct.launch(
        stream, (x.shape[0],),
        kernel,
        (x, weight, y, rstd, N, eps, cfg["tile_size"]),
    )
    return y


def _rms_norm_standard_tiled_base(stream, x, weight, y, rstd, N, eps):
    cfg = benchmark_tuning.get_tuned_config(
        tune_rms_norm,
        algo='regular',
        shape=x.shape,
        dtype=x.dtype
    )
    kernel = _rms_norm_regular_kernel(cfg['num_worker_warps'])
    ct.launch(
        stream, (x.shape[0],), kernel,
        (x, weight, y, rstd, N, eps, cfg["tile_size"]),
    )
    return y


def cutile_rms_norm(x, weight, eps, static_persistent, gather):
    x = x.contiguous()
    weight = weight.contiguous()

    # Allocate output tensor
    y = torch.empty_like(x)
    M, N = x.shape

    if static_persistent:
        _rms_norm_static_persistent_base(torch.cuda.current_stream(), x, y, weight, eps)
    else:
        rstd = torch.empty((M,), dtype=torch.float32, device='cuda')
        if gather:
            _rms_norm_standard_gather_base(torch.cuda.current_stream(),
                                           x, weight, y, rstd, N, eps)
        else:
            _rms_norm_standard_tiled_base(
                torch.cuda.current_stream(), x, weight, y, rstd, N, eps
            )
    return y.view(*x.shape)


def torch_rms_norm(input, weight, eps, static_persistent=False, gather=False):
    # layer norm should always be calculated in float32
    normalized_shape = weight.shape
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


def ref_rms_norm(input, weight, eps):
    return torch_rms_norm(input, weight, eps)
