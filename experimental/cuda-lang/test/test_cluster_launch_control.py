# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._compile import get_compute_capability
import torch
import pytest

from cuda.lang._exception import TileTypeError

cc = get_compute_capability()
if cc < (10, 0):
    pytest.skip(allow_module_level=True)

# Use non-portable target
arch = cc.arch + "a"
gpu_name = cc.gpu_name + "a"
target = dict(arch=arch, gpu_name=gpu_name)
clc_bytes = cl.clusterlaunchcontrol_token.bitwidth // 8


@cl.kernel(**target)
def bad_clc_memspace_1():
    smem = cl.shared_array(1, cl.mbarrier).get_base_pointer()
    with cl.local_array(1, cl.clusterlaunchcontrol_token) as larr:
        cl.clusterlaunchcontrol_try_cancel(larr.get_base_pointer(), smem)


@cl.kernel(**target)
def bad_clc_memspace_2():
    smem = cl.shared_array(1, cl.clusterlaunchcontrol_token).get_base_pointer()
    with cl.local_array(1, cl.mbarrier) as larr:
        cl.clusterlaunchcontrol_try_cancel(smem, larr.get_base_pointer())


@cl.kernel(**target)
def bad_clc_type():
    smem = cl.shared_array(2, cl.int64).get_base_pointer()
    cl.clusterlaunchcontrol_is_canceled(smem)


@pytest.mark.parametrize(
    "kernel,match",
    (
        pytest.param(
            bad_clc_memspace_1,
            "Expected a pointer to a cluster launch control token in shared memory",
            id="bad clc token memory space",
        ),
        pytest.param(
            bad_clc_memspace_2,
            "Expected a pointer to an mbarrier in shared memory",
            id="bad mbarrier memory space",
        ),
        pytest.param(
            bad_clc_type,
            "Expected type to be one of.+clusterlaunchcontrol_token.+but got.+int",
            id="bad clc token type",
        ),
    ),
)
def test_invalid_usage(kernel, match):
    with pytest.raises(TileTypeError, match=match):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def compute():
    return 5.0


@cl.kernel(**target)
def worksteal(data, n: cl.Constant[int], stolen):
    clc_resp = cl.shared_array(1, cl.clusterlaunchcontrol_token, alignment=16).get_base_pointer()
    mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

    tx = cl.thread_idx(0)
    bdx = cl.block_dim(0)
    bx = cl.block_idx(0)
    phase = 0

    if tx == 0:
        cl.mbarrier_init(mbar, 1)

    alpha = compute()
    while True:
        cl.syncthreads()

        if tx == 0:
            cl.nvvm.fence_proxy_async_generic_acquire_sync_restrict_space_cluster_scope_cluster()
            cl.clusterlaunchcontrol_try_cancel(clc_resp, mbar)
            cl.mbarrier_arrive_expect_tx(
                mbar,
                clc_bytes,
                ordering=cl.MemoryOrder.RELAXED,
            )

        i = bx * bdx + tx
        if i < n:
            data[i] *= alpha

        while not cl.mbarrier_try_wait_parity(mbar, phase, time_hint=10000):
            pass
        phase ^= 1

        tok = clc_resp.load()
        if not cl.clusterlaunchcontrol_is_canceled(tok):
            break

        if tx == 0:
            cl.atomic_add(stolen, 0, 1)
        bx = cl.clusterlaunchcontrol_get_first_block_idx(tok, axis=0)
        cl.nvvm.fence_proxy_async_generic_release_sync_restrict_space_cta_scope_cluster()


@cl.kernel(**target)
def worksteal_cluster(data, n: cl.Constant[int], stolen):
    clc_resp = cl.shared_array(1, cl.clusterlaunchcontrol_token, alignment=16).get_base_pointer()
    mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

    tx = cl.thread_idx(0)
    bdx = cl.block_dim(0)
    bx = cl.block_idx(0)
    local_block = cl.block_in_cluster_idx(0)
    phase = 0

    if tx == 0:
        cl.mbarrier_init(mbar, 1)
        cl.nvvm.fence_mbarrier_init_release_cluster()

    alpha = compute()

    while True:
        cl.nvvm.barrier_cluster_arrive_aligned()
        cl.nvvm.barrier_cluster_wait_aligned()

        if local_block == 0 and tx == 0:
            cl.nvvm.fence_proxy_async_generic_acquire_sync_restrict_space_cluster_scope_cluster()
            cl.clusterlaunchcontrol_try_cancel(clc_resp, mbar, multicast=True)

        if tx == 0:
            cl.mbarrier_arrive_expect_tx(
                mbar,
                clc_bytes,
                scope=cl.MbarrierScope.CLUSTER,
                ordering=cl.MemoryOrder.RELAXED,
            )

        i = bx * bdx + tx
        if i < n:
            data[i] *= alpha

        while not cl.mbarrier_try_wait_parity(
            mbar, phase, scope=cl.MbarrierScope.CLUSTER
        ):
            pass
        phase ^= 1

        token = clc_resp.load()
        if not cl.clusterlaunchcontrol_is_canceled(token):
            break  # no more work to steal

        if local_block == 0 and tx == 0:
            cl.atomic_add(stolen, 0, 1)

        bx = cl.clusterlaunchcontrol_get_first_block_idx(token, axis=0) + local_block
        cl.nvvm.fence_proxy_async_generic_release_sync_restrict_space_cta_scope_cluster()


def launch_configs():
    for gscale, bscale in (
        (1, 1),
        (2, 1),
        (4, 2),
    ):
        grid = gscale * 8 * 1024
        block = bscale * 128
        yield pytest.param((grid,), (block,), id=f"{grid=},{block=}")


@pytest.mark.parametrize(
    "kernel,cluster_dim",
    (
        pytest.param(worksteal, None, id="worksteal"),
        pytest.param(worksteal_cluster, (2, 1, 1), id="worksteal-cluster"),
    ),
)
@pytest.mark.parametrize("grid,block", launch_configs())
def test_worksteal(kernel, cluster_dim, grid, block):
    # Adapted from programming guide examples with added output tensor
    # to track number of stolen jobs.
    # https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html#use-case-thread-blocks
    n = grid[0] * block[0]
    data = torch.arange(n, dtype=torch.float32).cuda()
    stolen = torch.zeros(1, dtype=torch.int64).cuda()
    expect = (data * compute()).cpu()
    cl.launch(
        torch.cuda.current_stream(),
        grid,
        block,
        kernel,
        (data, n, stolen),
        cluster_dim=cluster_dim,
    )
    torch.testing.assert_close(data.cpu(), expect)
    stolen = stolen.cpu().item()
    assert stolen > 0, (
        f"With {grid=} larger than possible active CTAs, expected at least one stolen job"
    )
