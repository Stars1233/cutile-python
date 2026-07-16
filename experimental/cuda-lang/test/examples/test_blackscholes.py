# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math

import cuda.lang as cl
import torch


__doc__ = '''
cuda.lang port of the `Black-Scholes <BS>`__ example program from the CUDA samples repository.

.. _BS: <https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh>
'''  # noqa: E501


THREAD_N = 128
OPT_N = 4096
RISKFREE = 0.02
VOLATILITY = 0.30


@cl.function
def cnd_gpu(d):
    a1 = 0.31938153
    a2 = -0.356563782
    a3 = 1.781477937
    a4 = -1.821255978
    a5 = 1.330274429
    rsqrt2pi = 0.39894228040143267793994605993438

    k = 1.0 / (1.0 + 0.2316419 * cl.abs(d))
    cnd = rsqrt2pi * cl.exp((-0.5) * d * d) * (
        k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5))))
    )
    if d > 0.0:
        cnd = 1.0 - cnd
    return cnd


@cl.function
def black_scholes_body_gpu(s, x, t, r, v):
    sqrt_t = 1.0 / cl.rsqrt(t)
    d1 = (cl.log(s / x) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    cnd_d1 = cnd_gpu(d1)
    cnd_d2 = cnd_gpu(d2)
    exp_rt = cl.exp((-1.0) * r * t)

    call_result = s * cnd_d1 - x * exp_rt * cnd_d2
    put_result = x * exp_rt * (1.0 - cnd_d2) - s * (1.0 - cnd_d1)
    return call_result, put_result


def black_scholes_body_cpu(s, x, t, r, v):
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / x) + (r + 0.5 * v * v) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    def cnd_cpu(d):
        a1 = 0.31938153
        a2 = -0.356563782
        a3 = 1.781477937
        a4 = -1.821255978
        a5 = 1.330274429
        rsqrt2pi = 0.39894228040143267793994605993438
        k = 1.0 / (1.0 + 0.2316419 * abs(d))
        cnd = rsqrt2pi * math.exp(-0.5 * d * d) * \
            (k * (a1 + k * (a2 + k * (a3 + k * (a4 + k * a5)))))
        return 1.0 - cnd if d > 0.0 else cnd

    cnd_d1 = cnd_cpu(d1)
    cnd_d2 = cnd_cpu(d2)
    exp_rt = math.exp(-r * t)
    call_result = s * cnd_d1 - x * exp_rt * cnd_d2
    put_result = x * exp_rt * (1.0 - cnd_d2) - s * (1.0 - cnd_d1)
    return call_result, put_result


def test_blackscholes():
    """
    https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/BlackScholes/BlackScholes_kernel.cuh
    """

    @cl.kernel
    def black_scholes_gpu(
        call_result,
        put_result,
        stock_price,
        option_strike,
        option_years,
        riskfree,
        volatility,
        opt_n,
    ):
        tx = cl.thread_index(0)
        bx = cl.block_index(0)
        bdx = cl.thread_count(0)

        opt = bdx * bx + tx

        # The CUDA sample uses float2 buffers and does two options per thread.
        if opt < (opt_n // 2):
            i = 2 * opt

            call_result1, put_result1 = black_scholes_body_gpu(
                stock_price[i],
                option_strike[i],
                option_years[i],
                riskfree,
                volatility,
            )
            call_result2, put_result2 = black_scholes_body_gpu(
                stock_price[i + 1],
                option_strike[i + 1],
                option_years[i + 1],
                riskfree,
                volatility,
            )

            call_result[i] = call_result1
            put_result[i] = put_result1
            call_result[i + 1] = call_result2
            put_result[i + 1] = put_result2

    generator = torch.Generator(device="cuda").manual_seed(5347)
    stock_price = 5.0 + 25.0 * torch.rand(
        OPT_N, generator=generator, dtype=torch.float32, device="cuda"
    )
    option_strike = 1.0 + 99.0 * torch.rand(
        OPT_N, generator=generator, dtype=torch.float32, device="cuda"
    )
    option_years = 0.25 + 9.75 * torch.rand(
        OPT_N, generator=generator, dtype=torch.float32, device="cuda"
    )
    call_result = torch.zeros(OPT_N, dtype=torch.float32, device="cuda")
    put_result = torch.zeros(OPT_N, dtype=torch.float32, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        ((OPT_N // 2 + THREAD_N - 1) // THREAD_N,),
        (THREAD_N,),
        black_scholes_gpu,
        (
            call_result,
            put_result,
            stock_price,
            option_strike,
            option_years,
            RISKFREE,
            VOLATILITY,
            OPT_N,
        ),
    )
    torch.cuda.synchronize()

    ref_call = torch.empty(OPT_N, dtype=torch.float32)
    ref_put = torch.empty(OPT_N, dtype=torch.float32)
    stock_cpu = stock_price.cpu()
    strike_cpu = option_strike.cpu()
    years_cpu = option_years.cpu()

    for i in range(OPT_N):
        ref_call[i], ref_put[i] = black_scholes_body_cpu(
            float(stock_cpu[i]),
            float(strike_cpu[i]),
            float(years_cpu[i]),
            RISKFREE,
            VOLATILITY,
        )

    assert torch.allclose(call_result.cpu(), ref_call, atol=1e-5, rtol=1e-5)
    assert torch.allclose(put_result.cpu(), ref_put, atol=1e-5, rtol=1e-5)
