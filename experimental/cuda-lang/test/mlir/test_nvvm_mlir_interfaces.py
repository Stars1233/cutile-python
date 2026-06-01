# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.lang._exception import TileCompilerExecutionError, TileTypeError
from cuda.lang._compile import compile_simt, KernelSignature
import cuda.lang._stub.nvvm_mlir_interfaces as nvvm
import torch

from cuda.tile._exception import TileValueError
from ..util import require_hopper_or_newer, filecheck


@require_hopper_or_newer()
def test_mlir_interface_enums():
    @cl.kernel
    def kernel(tensor):
        nvvm.fence_proxy_acquire(
            scope=cl.MemoryScope.BLOCK,
            addr=tensor.get_base_pointer(),
            size=128,
            from_proxy=nvvm.ProxyKind.GENERIC,
            to_proxy=nvvm.ProxyKind.TENSORMAP,
        )

    z = torch.zeros(1, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (z,))


def test_mlir_interface_results():
    @cl.kernel
    def kernel():
        permuted = nvvm.prmt(lo=12, hi=16, selector=7, mode=nvvm.PermuteMode.F4E)
        print(permuted)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


@require_hopper_or_newer()
def test_cp_async_bulk_tensor_mlir_interface():
    @cl.kernel
    def kernel(src, dst):
        tma_descriptor = cl.tensor_map_tiled(src, (8, 8)).as_opaque_ptr()
        mbar = cl.shared_array(1, dtype=cl.mbarrier).get_base_pointer()
        smem = cl.shared_array(64, dtype=cl.int32, alignment=512)

        if cl.thread_idx(0) == 0:
            cl.mbarrier_init(mbar, cl.block_dim(0))

        cl.syncthreads()
        if cl.elect_sync():
            nvvm.cp_async_bulk_tensor_shared_cluster_global(
                dst_mem=smem.get_base_pointer(),
                tma_descriptor=tma_descriptor,
                coordinates=(0, 0),
                mbar=mbar,
                im2col_offsets=(),
                mode=nvvm.TMALoadMode.TILE,
                is_cta_only=True,
            )
            token = cl.mbarrier_arrive_expect_tx(mbar, 64 * 4)
        else:
            token = cl.mbarrier_arrive(mbar)

        while not cl.mbarrier_try_wait(mbar, token, time_hint=10_000):
            pass

        dst[cl.thread_idx(0)] = smem[cl.thread_idx(0)]

    src = torch.arange(64, dtype=torch.int32).cuda().reshape((8, 8)).contiguous()
    dst = torch.zeros(64, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (64,), kernel, (src, dst))
    torch.testing.assert_close(dst.reshape((8, 8)), src)


def test_mlir_interface_error_on_non_constant_enum():
    @cl.kernel
    def kernel(cond):
        if cond:
            dyn_kind = nvvm.ProxyKind.GENERIC
        else:
            dyn_kind = nvvm.ProxyKind.TENSORMAP
        nvvm.fence_proxy(kind=dyn_kind)

    with pytest.raises(
        TileTypeError,
        match="Expected ProxyKind constant, but given value is not constant",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (False,))


def test_mlir_interface_error_on_non_constant_attr():
    # cluster arrive takes a bool attr, so the value must be known at compile-time
    @cl.kernel
    def kernel(cond):
        nvvm.cluster_arrive(aligned=cond)

    with pytest.raises(
        TileTypeError,
        match="Expected a boolean constant, but given value is not constant",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (False,))


def test_wrong_enum_class():
    # the nvvm operation takes a MemScopeKind, but we map enums already exposed
    # in cuda.lang/cutile to nvvm enums when they exist, so this intrinsic
    # takes a cuda.tile._memory_model.MemoryScope instead. Ensure we give an
    # error on the NVVM enum and we accept the cuda.tile one.
    @cl.kernel
    def kernel():
        cl.memory_barrier(scope=nvvm.MemScopeKind.CLUSTER)

    with pytest.raises(
        TileTypeError,
        match=r"Expected MemoryScope, but given value has type Enum\[MemScopeKind\]",
    ):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


def test_correct_enum_class():
    # the nvvm operation takes a MemScopeKind, but we map enums already exposed
    # in cuda.lang/cutile to nvvm enums when they exist, so this intrinsic
    # takes a cuda.tile._memory_model.MemoryScope instead. Ensure we give an
    # error on the NVVM enum and we accept the cuda.tile one.
    @cl.kernel
    def kernel():
        cl.memory_barrier(scope=cl.MemoryScope.BLOCK)

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, ())


@pytest.mark.parametrize(
    "enum,expect",
    (
        (cl.MemoryScope.BLOCK, "cta"),
        (cl.MemoryScope.DEVICE, "gpu"),
        (cl.MemoryScope.SYS, "sys"),
        (cl.MemoryScope.NONE, None),
        # TODO: add cluster scope
        # (cl.MemoryScope.CLUSTER, ""),
    ),
)
def test_memory_scope_enum_mappings(enum, expect):
    def kernel():
        cl.memory_barrier(scope=enum)

    if expect is None:
        with pytest.raises(
            TileValueError,
            match="Expected one of MemoryScope.BLOCK, MemoryScope.DEVICE, "
            "MemoryScope.SYS, got MemoryScope.NONE",
        ):
            result = compile_simt(kernel, [KernelSignature(())])
    else:
        result = compile_simt(kernel, [KernelSignature(())])
        filecheck(
            result.mlir,
            f"""
            CHECK: nvvm.memory.barrier
            CHECK-SAME: scope = #nvvm<mem_scope <{expect}>>
            """,
        )


@require_hopper_or_newer()
@pytest.mark.parametrize(
    "enum,expect",
    (
        (cl.MemoryOrder.ACQUIRE, "acquire"),
        (cl.MemoryOrder.RELEASE, "release"),
        (cl.MemoryOrder.ACQ_REL, None),
        (cl.MemoryOrder.RELAXED, None),
        (cl.MemoryOrder.WEAK, None),
    ),
)
def test_memory_space_enum_mappings(enum, expect):
    def kernel():
        nvvm.fence_sync_restrict(order=enum)

    if expect is None:
        with pytest.raises(
            TileCompilerExecutionError,
            match="op only acquire and release semantics are supported",
        ):
            compile_simt(kernel, [KernelSignature(())])
    else:
        result = compile_simt(kernel, [KernelSignature(())])
        filecheck(
            result.mlir,
            f"""
            CHECK: nvvm.fence.sync_restrict
            CHECK-SAME: order = #nvvm<mem_order <{expect}>>
            """,
        )
