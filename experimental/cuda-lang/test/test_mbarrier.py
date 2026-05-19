# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

import cuda.lang as cl
from cuda.lang._ir.ops import RawNVVMIntrinsic

from .util import compile_for_arguments, require_hopper_or_newer


@require_hopper_or_newer()
def test_cluster_barriers():
    '''
    Allocate an mbarrier and get a pointer to the rank-0 CTA's barrier after
    a cluster-wide sync.
    Arrive at rank 0's mbarrier, and then rank 0 observes it's completion.
    '''

    @cl.kernel()
    def kernel(out):
        rank = cl.block_in_cluster_idx(0)
        cdx = cl.block_in_cluster_dim(0)
        tx = cl.thread_idx(0)
        bdx = cl.block_dim(0)
        mbar = cl.shared_array(shape=(), dtype=cl.mbarrier, alignment=8)
        mbar = mbar.get_base_pointer()

        if tx == 0:
            cl.mbarrier_init(mbar, cdx * bdx)

        cl.nvvm.fence_mbarrier_init_release_cluster()
        cl.syncthreads()
        cl.nvvm.barrier_cluster_arrive_aligned()
        cl.nvvm.barrier_cluster_wait_aligned()

        mbar0 = cl.map_shared_to_cluster(mbar, 0)
        cl.mbarrier_arrive(mbar0, scope=cl.MemorySpace.SHARED_CLUSTER)

        if rank == 0 and tx == 0:
            while not cl.mbarrier_test_wait_parity(mbar, 0):
                pass
            out[0] = 1

    out = torch.zeros(1, dtype=torch.int32).cuda()
    # Grid == cluster so there's exactly one cluster of 2 CTAs. 32 threads/CTA
    # gives 64 total arrives at rank 0's mbarrier.
    # Initialize cdx * bdx barrier participants.
    cl.launch(
        torch.cuda.current_stream(),
        (2, 1, 1),
        (32, 1, 1),
        kernel,
        (out,),
        cluster_dim=(2, 1, 1),
    )
    assert out.cpu().tolist() == [1]


# compile-only tests that cover the full api


SCOPES = [cl.MemorySpace.SHARED, cl.MemorySpace.SHARED_CLUSTER]


def _scope_suffix(scope):
    return "cluster" if scope is cl.MemorySpace.SHARED_CLUSTER else "cta"


def _get_intrinsics(kernel):
    result = compile_for_arguments(kernel, ())
    return [
        op.intrinsic
        for block in result.final_ir.blocks
        for op in block.traverse()
        if isinstance(op, RawNVVMIntrinsic)
    ]


@require_hopper_or_newer()
def test_init_and_invalidate():
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        cl.mbarrier_init(mbar, 32)
        cl.mbarrier_invalidate(mbar)

    names = _get_intrinsics(kernel)
    assert "llvm.nvvm.mbarrier.init.shared" in names
    assert "llvm.nvvm.mbarrier.inval.shared" in names


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("expect_tx", [False, True])
def test_arrive_intrinsic_name(expect_tx, drop, relaxed, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if expect_tx:
            cl.mbarrier_arrive_expect_tx(
                mbar, 128, drop=drop, scope=scope, relaxed=relaxed
            )
        else:
            cl.mbarrier_arrive(mbar, 1, drop=drop, scope=scope, relaxed=relaxed)

    expected = "llvm.nvvm.mbarrier.arrive"
    if drop:
        expected += ".drop"
    if expect_tx:
        expected += ".expect.tx"
    if relaxed:
        expected += ".relaxed"
    expected += f".scope.{_scope_suffix(scope)}.space.cta"
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("op", ["expect_tx", "complete_tx"])
def test_expect_complete_tx_intrinsic_name(op, scope):
    fn = getattr(cl, f"mbarrier_{op}")

    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        fn(mbar, 64, scope=scope)

    expected = (
        f"llvm.nvvm.mbarrier.{op.replace('_', '.')}"
        + f".scope.{_scope_suffix(scope)}.space.cta"
    )
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("parity", [False, True])
def test_test_wait_intrinsic_name(parity, relaxed, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if parity:
            cl.mbarrier_test_wait_parity(mbar, 0, scope=scope, relaxed=relaxed)
        else:
            cl.mbarrier_test_wait(mbar, cl.uint64(0), scope=scope, relaxed=relaxed)

    expected = "llvm.nvvm.mbarrier.test.wait"
    if parity:
        expected += ".parity"
    if relaxed:
        expected += ".relaxed"
    expected += f".scope.{_scope_suffix(scope)}.space.cta"
    assert expected in _get_intrinsics(kernel)


@require_hopper_or_newer()
@pytest.mark.parametrize("scope", SCOPES)
@pytest.mark.parametrize("time_hint", [None, 1000])
@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("parity", [False, True])
def test_try_wait_intrinsic_name(parity, relaxed, time_hint, scope):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(
            shape=(1,), dtype=cl.mbarrier, alignment=8
        ).get_base_pointer()
        if parity:
            cl.mbarrier_try_wait_parity(
                mbar, 0, time_hint=time_hint, scope=scope, relaxed=relaxed
            )
        else:
            cl.mbarrier_try_wait(
                mbar, cl.uint64(0), time_hint=time_hint, scope=scope, relaxed=relaxed
            )

    expected = "llvm.nvvm.mbarrier.try.wait"
    if parity:
        expected += ".parity"
    if time_hint is not None:
        expected += ".tl"
    if relaxed:
        expected += ".relaxed"
    expected += f".scope.{_scope_suffix(scope)}.space.cta"
    assert expected in _get_intrinsics(kernel)
