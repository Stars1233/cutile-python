# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch


def test_tid():
    @cl.kernel
    def kernel(A):
        tidx, tidy, tidz = cl.thread_idx()
        A[tidx, tidy, tidz] = tidx + tidy + tidz

    A = torch.zeros(3, 3, 3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (3, 3, 3), kernel, (A,))
    expected = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
        ],
        device="cpu",
        dtype=torch.int32,
    )
    assert (expected == A.cpu()).all()


def test_bid():
    @cl.kernel
    def kernel(A):
        bidx, bidy, bidz = cl.block_idx()
        A[bidx, bidy, bidz] = bidx + bidy + bidz

    A = torch.zeros(3, 3, 3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (3, 3, 3), (1,), kernel, (A,))
    expected = torch.tensor(
        [
            [[0, 1, 2], [1, 2, 3], [2, 3, 4]],
            [[1, 2, 3], [2, 3, 4], [3, 4, 5]],
            [[2, 3, 4], [3, 4, 5], [4, 5, 6]],
        ],
        device="cpu",
        dtype=torch.int32,
    )
    assert (expected == A.cpu()).all()


def test_block_dim():
    @cl.kernel
    def kernel(out):
        tidx, _, _ = cl.thread_idx()
        if tidx == 0:
            out[0], out[1], out[2] = cl.block_dim()

    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (4, 3, 2), kernel, (out,))
    assert (out.cpu() == torch.tensor([4, 3, 2], dtype=torch.int32)).all()


def test_grid_dim():
    @cl.kernel
    def kernel(out):
        tidx, _, _ = cl.thread_idx()
        if tidx == 0:
            out[0], out[1], out[2] = cl.grid_dim()

    out = torch.zeros(3, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (5, 6, 7), (1,), kernel, (out,))
    assert (out.cpu() == torch.tensor([5, 6, 7], dtype=torch.int32)).all()


def test_saxpy():
    '''
    Taken from https://developer.nvidia.com/blog/six-ways-saxpy/

    __global__
    void saxpy(int n, float a, float * restrict x, float * restrict y)
    {
        int i = blockIdx.x*blockDim.x + threadIdx.x;
        if (i < n) y[i] = a*x[i] + y[i];
    }
    '''

    @cl.kernel
    def kernel(N: cl.Constant[int], a: cl.Constant[float], X, Y):
        tidx, _, _ = cl.thread_idx()
        bidx, _, _ = cl.block_idx()
        block_dim_x, _, _ = cl.block_dim()
        idx = tidx + bidx * block_dim_x
        if idx < N:
            Y[idx] = a * X[idx] + Y[idx]

    N = 256
    alpha = 2.0
    X = torch.ones(N, dtype=torch.float32, device="cuda")
    Y = torch.ones(N, dtype=torch.float32, device="cuda")
    expected = (alpha * X + Y).cpu()
    cl.launch(torch.cuda.current_stream(), (64,), (64,), kernel, (N, alpha, X, Y))
    assert torch.allclose(expected, Y.cpu())


class TestSyncwarp:
    '''
    _CUDA C++ Programming Guide 10.6 Synchronization Functions_ describes
    these functions and _20.6.2. Independent Thread Scheduling_ has example programs
    using them. These tests are based on those examples.
    '''

    def test_syncwarp(self):
        """
        __syncwarp(mask) waits until all named lanes execute the same warp barrier.
        """
        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            lane, _, _ = cl.thread_idx()

            shmem[lane] = lane
            cl.syncwarp()
            out[lane] = shmem[lane ^ 1]

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [lane ^ 1 for lane in range(32)],
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_syncwarp_reduction(self):
        """
        warp lanes can communicate via memory by storing, synchronizing with syncwarp
        and then reading peers' values.
        """

        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            lane, _, _ = cl.thread_idx()

            shmem[lane] = 1
            cl.syncwarp()

            if lane < 16:
                shmem[lane] = shmem[lane] + shmem[lane + 16]
            cl.syncwarp()
            if lane < 8:
                shmem[lane] = shmem[lane] + shmem[lane + 8]
            cl.syncwarp()
            if lane < 4:
                shmem[lane] = shmem[lane] + shmem[lane + 4]
            cl.syncwarp()
            if lane < 2:
                shmem[lane] = shmem[lane] + shmem[lane + 2]
            cl.syncwarp()
            if lane < 1:
                shmem[lane] = shmem[lane] + shmem[lane + 1]
            cl.syncwarp()

            if lane == 0:
                out[0] = shmem[0]

        out = torch.zeros(1, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        assert (out.cpu() == torch.tensor([32], dtype=torch.int32)).all()

    def test_syncwarp_half_mask(self):
        """
        each calling thread must have its own bit set in the mask, and only lanes
        named in the mask participate in the warp barrier.
        """

        @cl.kernel
        def kernel(out):
            shmem = cl.shared_array(shape=(16,), dtype=cl.int32)
            lane, _, _ = cl.thread_idx()
            mask = 0x0000FFFF

            if lane < 16:
                shmem[lane] = lane + 100
                cl.syncwarp(mask)
                out[lane] = shmem[lane ^ 1]
            else:
                out[lane] = -1

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [101, 100, 103, 102, 105, 104, 107, 106,
             109, 108, 111, 110, 113, 112, 115, 114] + [-1] * 16,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()


class TestShuffle:
    """
    Shuffle tests based on the CUDA sample
    Samples/2_Concepts_and_Techniques/shfl_scan/shfl_scan.cu and direct
    single-warp sanity checks.
    """

    def test_shfl_up_scan_width_8(self):
        @cl.kernel
        def kernel(inp, out):
            tid, _, _ = cl.thread_idx()
            lane = tid % 32
            sublane = lane % 8
            value = cl.int32(inp[tid])
            width = cl.int32(8)
            mask = cl.int32(0xFFFFFFFF)

            delta = cl.int32(1)
            for _ in range(4):
                other = cl.shfl_up_sync(mask, value, delta, width)
                if sublane >= delta:
                    value += other
                delta *= 2

            out[tid] = value

        inp = torch.ones(32, dtype=torch.int32, device="cuda")
        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (inp, out))
        expected = torch.tensor(
            [1, 2, 3, 4, 5, 6, 7, 8] * 4,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_shfl_sync_idx(self):
        @cl.kernel
        def kernel(out):
            lane, _, _ = cl.thread_idx()
            out[lane] = cl.shfl_sync(cl.int32(0xFFFFFFFF), lane, 4)

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.full((32,), 4, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_shfl_down_sync(self):
        @cl.kernel
        def kernel(out):
            lane, _, _ = cl.thread_idx()
            out[lane] = cl.shfl_down_sync(cl.int32(0xFFFFFFFF), lane, 4)

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.arange(32, dtype=torch.int32)
        expected[:-4] += 4
        assert (out.cpu() == expected).all()

    def test_shfl_xor_sync(self):
        @cl.kernel
        def kernel(out):
            lane, _, _ = cl.thread_idx()
            out[lane] = cl.shfl_xor_sync(cl.int32(0xFFFFFFFF), lane, 16)

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor([lane ^ 16 for lane in range(32)], dtype=torch.int32)
        assert (out.cpu() == expected).all()


class TestBarrierSync:
    """
    Barrier tests based on the CUDA guide's block synchronization semantics and
    PTX named barrier examples.
    """

    def test_barrier_sync_shared_exchange(self):
        @cl.kernel
        def kernel(out):
            lane, _, _ = cl.thread_idx()
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)

            shmem[lane] = lane * 2
            cl.nvvm.barrier_cta_sync_all(cl.int32(0))

            if lane < 16:
                out[lane] = shmem[lane + 16]
            else:
                out[lane] = shmem[lane - 16]

        out = torch.zeros(32, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (out,))
        expected = torch.tensor(
            [2 * (lane + 16) for lane in range(16)] +
            [2 * (lane - 16) for lane in range(16, 32)],
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()

    def test_barrier_sync_count_warp_subset(self):
        @cl.kernel
        def kernel(out):
            lane, _, _ = cl.thread_idx()
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)

            if lane < 32:
                shmem[lane] = lane + 100
                cl.nvvm.barrier_cta_sync_count(1, 32)
                out[lane] = shmem[lane ^ 1]
            else:
                out[lane] = -1

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor(
            [101, 100, 103, 102, 105, 104, 107, 106,
             109, 108, 111, 110, 113, 112, 115, 114,
             117, 116, 119, 118, 121, 120, 123, 122,
             125, 124, 127, 126, 129, 128, 131, 130] + [-1] * 32,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()


class TestVoteSync:
    """
    Vote tests based on simpleVoteIntrinsics and the CUDA guide's warp vote
    semantics.
    """

    def test_all_sync(self):
        @cl.kernel
        def kernel(out):
            tid, _, _ = cl.thread_idx()
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = True
            else:
                pred = lane < 16

            if cl.nvvm.vote_all_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor([1] * 32 + [0] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_any_sync(self):
        @cl.kernel
        def kernel(out):
            tid, _, _ = cl.thread_idx()
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = False
            else:
                pred = lane == 0

            if cl.nvvm.vote_any_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor([0] * 32 + [1] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_uni_sync(self):
        @cl.kernel
        def kernel(out):
            tid, _, _ = cl.thread_idx()
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = True
            elif tid < 64:
                pred = False
            else:
                pred = lane < 16

            if cl.nvvm.vote_uni_sync(mask, pred):
                out[tid] = 1
            else:
                out[tid] = 0

        out = torch.zeros(96, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (96,), kernel, (out,))
        expected = torch.tensor([1] * 64 + [0] * 32, dtype=torch.int32)
        assert (out.cpu() == expected).all()

    def test_ballot_sync(self):
        @cl.kernel
        def kernel(out):
            tid, _, _ = cl.thread_idx()
            lane = tid % 32
            mask = cl.int32(0xFFFFFFFF)

            if tid < 32:
                pred = lane < 8
            else:
                pred = (lane % 2) == 0

            out[tid] = cl.nvvm.vote_ballot_sync(mask, pred)

        out = torch.zeros(64, dtype=torch.int32, device="cuda")
        cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (out,))
        expected = torch.tensor(
            [0x000000FF] * 32 + [0x55555555] * 32,
            dtype=torch.int32,
        )
        assert (out.cpu() == expected).all()
