# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import cache
from math import ceil, sqrt
from itertools import product
import benchmark_tuning
from conftest import dtype_id, shape_id
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import sdpa_kernel, SDPBackend
import cuda.tile as ct
from cuda.tile.tune import exhaustive_search

import pytest
import torch

from util import estimate_bench_iter
from kernels.attention import fmha_kernel


def qkv_id(qkv_shape: tuple[tuple[int, ...], tuple[int, ...]]) -> str:
    q_shape, kv_shape = qkv_shape
    if q_shape[2] == 1:
        prefix = "decode-"
    else:
        prefix = "prefill-"
    b, q_head, q_len, d = q_shape
    _, k_head, k_len, _ = kv_shape
    return prefix + shape_id((b, q_head, k_head, q_len, k_len, d))


@pytest.fixture(
    params=[
        # B, H, L, D
        ((1, 32, 1024, 128), (1, 32, 1024, 128)),  # prefill
        ((1, 32, 1024, 128), (1, 8, 1024, 128)),  # prefill + gqa
        ((1, 32, 8192, 128), (1, 32, 8192, 128)),  # prefill
        ((1, 32, 8192, 128), (1, 8, 8192, 128)),  # prefill + gqa
        ((1, 32, 1, 128), (1, 32, 1024, 128)),  # decode
        ((8, 32, 1, 128), (8, 32, 1024, 128)),  # decode
        ((1, 32, 1, 128), (1, 8, 1024, 128)),  # decode + gqa
        ((8, 32, 1, 128), (8, 8, 1024, 128)),  # decode + gqa
    ],
    ids=qkv_id)
def qkv_shape(request):
    return request.param


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=dtype_id)
def dtype(request):
    return request.param


@pytest.mark.benchmark(group='attention')
def bench_fmha(qkv_shape, dtype, backend, benchmark):
    q_shape, kv_shape = qkv_shape

    q = torch.randn(q_shape, dtype=dtype, device='cuda')
    k = torch.randn(kv_shape, dtype=dtype, device='cuda')
    v = torch.randn(kv_shape, dtype=dtype, device='cuda')
    o = torch.empty_like(q)
    ref = torch.empty_like(q)
    is_causal = q_shape[2] == kv_shape[2]
    enable_gqa = q_shape[1] != kv_shape[1]

    backend(q, k, v, o, is_causal, enable_gqa)
    ref_fmha(q, k, v, ref, is_causal, enable_gqa)
    torch.testing.assert_close(o, ref, atol=1e-2, rtol=5e-2)
    torch.cuda.synchronize()

    warmup_rounds, iterations, rounds = estimate_bench_iter(
        backend, (q, k, v, o, is_causal, enable_gqa),
        cudagraph=True
    )

    benchmark.pedantic(
        backend, (q, k, v, o, is_causal, enable_gqa),
        rounds=rounds, warmup_rounds=warmup_rounds, iterations=iterations,
        cudagraph=True
    )

    B, H, L, D = q.shape
    # first gemm mma(q, k): 2 * B * H * L * L * D
    # second gemm mma(p, v): 2 * B * H * L * L * D
    flop_count = 4 * B * H * L * L * D

    if is_causal:
        flop_count /= 2

    bytes_rw = sum([t.numel() * t.dtype.itemsize for t in (q, k, v, o)])
    benchmark.extra_info['flop_count'] = flop_count
    benchmark.extra_info['bytes_rw'] = bytes_rw


def cutile_fmha(q, k, v, o, is_causal, enable_gqa):
    b, qh, q_len, d = q.shape
    _, kh, k_len, _ = k.shape
    qk_scale = 1 / sqrt(d)
    cfg = benchmark_tuning.get_tuned_config(tune_fmha, is_causal=is_causal)
    TILE_M, TILE_N = cfg["tile_m"], cfg["tile_n"]
    query_group_size = qh // kh
    grid = (ceil(q_len / TILE_M), b * qh, 1)
    input_pos = 0 if q_len == k_len else (k_len - 1)
    EVEN_K = (k_len % TILE_N) == 0
    kernel = _fmha_kernel(cfg["occupancy"])
    ct.launch(torch.cuda.current_stream(), grid, kernel,
              (q, k, v, o,
               qk_scale,
               input_pos,
               d, qh,
               TILE_M, TILE_N,
               query_group_size, is_causal, EVEN_K))


@cache
def _fmha_kernel(occupancy):
    return fmha_kernel.replace_hints(occupancy=occupancy)


@pytest.mark.parametrize("is_causal", [False, True])
def tune_fmha(is_causal):
    if is_causal:
        q_shape, kv_shape = (1, 32, 8192, 128), (1, 32, 8192, 128)
    else:
        q_shape, kv_shape = (1, 32, 1, 128), (1, 32, 1024, 128)
    dtype = torch.float16
    q = torch.randn(q_shape, dtype=dtype, device='cuda')
    k = torch.randn(kv_shape, dtype=dtype, device='cuda')
    v = torch.randn(kv_shape, dtype=dtype, device='cuda')
    o = torch.empty_like(q)

    b, qh, q_len, d = q.shape
    _, kh, k_len, _ = k.shape
    qk_scale = 1 / sqrt(d)
    query_group_size = qh // kh
    input_pos = 0 if q_len == k_len else (k_len - 1)
    search_space = [
        {"tile_m": tile_m, "tile_n": tile_n, "occupancy": occupancy}
        for tile_m, tile_n, occupancy in product(
            (64, 128, 256),
            (64, 128, 256),
            (1, 2, 4),
        )
    ]
    return exhaustive_search(
        search_space,
        torch.cuda.current_stream(),
        grid_fn=lambda cfg: (ceil(q_len / cfg["tile_m"]), b * qh, 1),
        kernel=fmha_kernel,
        args_fn=lambda cfg: (
            q, k, v, o,
            qk_scale,
            input_pos,
            d, qh,
            cfg["tile_m"], cfg["tile_n"],
            query_group_size, is_causal, (k_len % cfg["tile_n"]) == 0,
        ),
        hints_fn=lambda cfg: {"occupancy": cfg["occupancy"]},
    )


def torch_fmha(q, k, v, o, is_causal, enable_gqa):
    backend = SDPBackend.CUDNN_ATTENTION \
            if (q.shape[2] == k.shape[2]) \
            else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(backend):
        ret = scaled_dot_product_attention(q, k, v,
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
        o.copy_(ret)


def ref_fmha(q, k, v, o, is_causal, enable_gqa):
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        ret = scaled_dot_product_attention(q, k, v,
                                           is_causal=is_causal,
                                           enable_gqa=enable_gqa)
        o.copy_(ret)
