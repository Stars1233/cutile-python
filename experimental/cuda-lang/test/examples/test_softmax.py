# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch


__doc__ = """
Port of softmax kernels from Karpathy's llm.c.

https://github.com/karpathy/llm.c/blob/master/dev/cuda/softmax_forward.cu
"""  # noqa: E501


N = 32
C = 256
BLOCK_SIZE = 128


@cl.kernel
def softmax_forward_kernel1(out, inp, n: cl.Constant[int], c: cl.Constant[int]):
    row = cl.block_idx()[0] * cl.block_dim()[0] + cl.thread_idx()[0]

    if row < n:
        base = row * c

        maxval = cl.float32(-float("inf"))
        for j in range(c):
            maxval = cl.libdevice.fmaxf(maxval, inp[base + j])

        sumval = cl.float64(0.0)
        for j in range(c):
            expval = cl.libdevice.expf(inp[base + j] - maxval)
            out[base + j] = expval
            sumval += cl.float64(expval)

        for j in range(c):
            out[base + j] /= cl.float32(sumval)


def test_softmax_forward_kernel1():
    generator = torch.Generator(device="cpu").manual_seed(42)
    inp_cpu = torch.randn((N, C), generator=generator, dtype=torch.float32)

    inp = inp_cpu.reshape(N * C).contiguous().cuda()
    out = torch.empty_like(inp)

    cl.launch(
        torch.cuda.current_stream(),
        ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,),
        (BLOCK_SIZE,),
        softmax_forward_kernel1,
        (out, inp, N, C),
    )
    torch.cuda.synchronize()

    actual = out.reshape(N, C).cpu()
    expected = torch.nn.functional.softmax(inp_cpu, dim=-1)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
