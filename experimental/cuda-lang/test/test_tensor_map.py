# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

import cuda.lang as cl
from .util import require_hopper_or_newer


@require_hopper_or_newer()
def test_tensor_map_tiled():
    @cl.kernel
    def kern(x, y, i, j, W: cl.Constant, H: cl.Constant):
        # TODO: barrier API
        barrier = cl.shared_array(shape=(), dtype=cl.uint64)
        smem = cl.shared_array(shape=(W * H,), dtype=cl.int32, alignment=512)
        x_tm = cl.tensor_map_tiled(x, (W, H))

        if cl.thread_idx(0) == 0:
            cl.nvvm.mbarrier_init_shared(barrier.get_base_pointer(), cl.block_dim()[0])

        cl.syncthreads()
        if cl.elect_sync():
            # TODO: proper cp.async API
            cl.nvvm.cp_async_bulk_tensor_g2s_cta_tile_2d(
                smem.get_base_pointer(),
                barrier.get_base_pointer(),
                x_tm.as_opaque_ptr(),
                j, i,
                0, False
            )
            tok = cl.nvvm.mbarrier_arrive_expect_tx_scope_cta_space_cta(
                barrier.get_base_pointer(),
                W * H * 4
            )
        else:
            tok = cl.nvvm.mbarrier_arrive_scope_cta_space_cta(barrier.get_base_pointer(), 1)

        while not cl.nvvm.mbarrier_try_wait_scope_cta_space_cta(barrier.get_base_pointer(), tok):
            # TODO: back off (see __cccl_thread_poll_with_backoff in CUDA C++ stdlib)
            cl.nvvm.nanosleep(10000)

        y[cl.thread_idx(0)] = smem[cl.thread_idx(0)]

    x = torch.arange(37 * 48, dtype=torch.int32, device="cuda").reshape((37, 48)).contiguous()
    H, W = 32, 8

    for i, j in [(0, 0), (1, 12)]:
        y = torch.zeros(256, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (256,), kern, (x, y, i, j, H, W))
        y = y.reshape((H, W))
        ref = x[i:i+H, j:j+W]
        torch.testing.assert_close(y, ref)
