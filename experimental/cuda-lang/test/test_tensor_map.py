# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.lang as cl
from .util import require_hopper_or_newer


@require_hopper_or_newer()
def test_mbar_manager():
    @cl.kernel()
    def kernel(x, y, i, j, H: cl.Constant[int], W: cl.Constant[int]):
        x_tm = cl.tensor_map_tiled(x, (W, H), order="F")
        mbar = cl.shared_array(
            shape=(), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        smem = cl.shared_array(shape=(W * H,), dtype=cl.int32, alignment=512)

        if cl.thread_index(0) == 0:
            cl.mbarrier_init(mbar, cl.thread_count(0))

        cl.barrier_sync_block()
        if cl.elect_sync():
            cl.cp_async_bulk_tensor_global_to_shared(
                x_tm, (j, i), smem.get_base_pointer(), mbar
            )
            tok = cl.mbarrier_arrive_expect_tx(mbar, W * H * 4)
        else:
            tok = cl.mbarrier_arrive(mbar)

        while not cl.mbarrier_try_wait(mbar, tok, time_hint=10_000):
            pass

        y[cl.thread_index(0)] = smem[cl.thread_index(0)]

    x = (
        torch.arange(37 * 48, dtype=torch.int32, device="cuda")
        .reshape((37, 48))
        .contiguous()
    )
    H, W = 32, 8

    for i, j in [(0, 0), (1, 12)]:
        y = torch.zeros(256, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (256,), kernel, (x, y, i, j, H, W))
        y = y.reshape((H, W))
        ref = x[i:i + H, j:j + W]
        torch.testing.assert_close(y, ref)


@require_hopper_or_newer()
def test_tensor_map_tiled():
    @cl.kernel
    def kern(x, y, i, j, H: cl.Constant[int], W: cl.Constant[int]):
        barrier = cl.shared_array(shape=(), dtype=cl.mbarrier, alignment=8)
        smem = cl.shared_array(shape=(W * H,), dtype=cl.int32, alignment=512)
        x_tm = cl.tensor_map_tiled(x, (W, H), order="F")

        if cl.thread_index(0) == 0:
            cl.mbarrier_init(barrier.get_base_pointer(), cl.thread_count(0))

        cl.barrier_sync_block()
        if cl.elect_sync():
            cl.cp_async_bulk_tensor_global_to_shared(
                x_tm,
                (j, i),
                smem.get_base_pointer(),
                barrier.get_base_pointer(),
            )
            tok = cl.mbarrier_arrive_expect_tx(barrier.get_base_pointer(), W * H * 4)
        else:
            tok = cl.mbarrier_arrive(barrier.get_base_pointer())

        while not cl.mbarrier_try_wait(barrier.get_base_pointer(), tok):
            # TODO: back off (see __cccl_thread_poll_with_backoff in CUDA C++ stdlib)
            cl._nvvm.nanosleep(10000)

        y[cl.thread_index(0)] = smem[cl.thread_index(0)]

    x = (
        torch.arange(37 * 48, dtype=torch.int32, device="cuda")
        .reshape((37, 48))
        .contiguous()
    )
    H, W = 32, 8

    for i, j in [(0, 0), (1, 12)]:
        y = torch.zeros(256, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (256,), kern, (x, y, i, j, H, W))
        y = y.reshape((H, W))
        ref = x[i:i + H, j:j + W]
        torch.testing.assert_close(y, ref)
