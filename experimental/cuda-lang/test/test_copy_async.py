# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._exception import TileTypeError
from cuda.lang.compilation import KernelSignature

from .util import (
    compile_kernel,
    make_symbolic_scalar,
    make_symbolic_tensor,
    require_blackwell_cc100,
    require_hopper_or_newer,
)


class CopyAsyncPtxTestBase:
    @staticmethod
    def signature():
        return KernelSignature(
            [
                make_symbolic_tensor((1, 1), cl.int32),
                make_symbolic_scalar(cl.bool_),
                make_symbolic_scalar(cl.int32),
                make_symbolic_scalar(cl.int32),
                32,
                8,
            ]
        )

    def check_ptx_source(self, kernel, *expect: str):
        compiled = cl.compile_simt(kernel, [self.signature()], log_ptx=True)
        ptx = compiled.ptx
        assert ptx is not None
        for expected in expect:
            assert expected in ptx, ptx


@require_hopper_or_newer()
class TestG2S(CopyAsyncPtxTestBase):
    @pytest.mark.parametrize("cluster", (True, False))
    def test_minimal(self, cluster):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = smem.get_base_pointer()
            if cluster:
                smem = cl.map_shared_to_cluster(smem, 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(tensor_map, (i, j), smem, mbar)

        shared_mode = "cluster" if cluster else "cta"
        expect = (
            f"cp.async.bulk.tensor.2d.shared::{shared_mode}"
            ".global.tile.mbarrier::complete_tx::bytes"
        )
        self.check_ptx_source(kernel, expect)

    @require_blackwell_cc100()
    @pytest.mark.parametrize(
        "cta_group,expect_group",
        (
            (cl.CTAGroup.CTA_1, "cta_group::1"),
            (cl.CTAGroup.CTA_2, "cta_group::2"),
        ),
    )
    def test_shared_cluster_group(self, cta_group, expect_group):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                cta_group=cta_group,
            )

        self.check_ptx_source(
            kernel,
            "cp.async.bulk.tensor.2d.shared::cluster.global",
            expect_group,
        )

    @require_blackwell_cc100()
    def test_shared_cluster_group_with_predicate_and_multicast(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                multicast_mask=0x3,
                cta_group=cl.CTAGroup.CTA_2,
                predicate=pred,
            )

        self.check_ptx_source(
            kernel,
            "cp.async.bulk.tensor.2d.shared::cluster.global",
            "multicast::cluster",
        )

    @require_blackwell_cc100()
    def test_shared_cluster_mbarrier_address_space(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()
            mbar = cl.map_shared_to_cluster(mbar, 0)

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                multicast_mask=0x3,
                cta_group=cl.CTAGroup.CTA_2,
            )

        match = (
            "Expected pointer memory space to be MemorySpace.SHARED "
            "but got MemorySpace.SHARED_CLUSTER"
        )
        with pytest.raises(TileTypeError, match=match):
            self.check_ptx_source(kernel)

    def test_unsupported_kwargs_for_cta_mode(self, subtests):
        @cl.kernel
        def k1(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                predicate=pred,
            )

        @cl.kernel
        def k2(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                multicast_mask=0xFF,
            )

        @cl.kernel
        def k3(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                cta_group=cl.CTAGroup.CTA_1,
            )

        def compile(kernel):
            match = (
                "When the destination memory is in shared memory, the "
                "predicate, multicast mask, and cta_group arguments are invalid."
            )
            with pytest.raises(
                TileTypeError,
                match=match,
            ):
                self.check_ptx_source(kernel)

        with subtests.test("predicate"):
            compile(k1)

        with subtests.test("multicast mask"):
            compile(k2)

        with subtests.test("cta_group"):
            compile(k3)

    @pytest.mark.parametrize("cluster", (True, False))
    def test_im2col_offsets_without_required_load_mode(self, cluster):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = smem.get_base_pointer()
            if cluster:
                smem = cl.map_shared_to_cluster(smem, 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map, (i, j), smem, mbar, im2col_offsets=(0, 1)
            )

        with pytest.raises(
            TileTypeError, match="TILE mode does not accept im2col_offsets"
        ):
            self.check_ptx_source(kernel)

    @require_blackwell_cc100()
    @pytest.mark.parametrize("cluster", (True, False))
    def test_tile_gather4_load_mode(self, cluster):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = smem.get_base_pointer()
            if cluster:
                smem = cl.map_shared_to_cluster(smem, 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j, 0, 0, 0),
                smem,
                mbar,
                mode=cl.TMALoadMode.TILE_GATHER4,
            )

        shared_mode = "cluster" if cluster else "cta"
        expect = (
            f"cp.async.bulk.tensor.2d.shared::{shared_mode}"
            ".global.tile::gather4.mbarrier::complete_tx::bytes"
        )
        self.check_ptx_source(kernel, expect)

    @pytest.mark.parametrize(
        "mode",
        (cl.TMALoadMode.IM2COL, cl.TMALoadMode.IM2COL_W, cl.TMALoadMode.IM2COL_W_128),
    )
    def test_im2col_load_modes_require_offsets(self, mode):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                mode=mode,
            )

        with pytest.raises(
            TileTypeError, match=f"{mode.name} mode requires im2col_offsets"
        ):
            self.check_ptx_source(kernel)

    def test_im2col_load_mode_rank3(self):
        @cl.kernel
        def kernel(
            x,
            pred,
            i,
            j,
            k,
            D: cl.Constant[int],
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            tensor_map = cl.tensor_map_tiled(x, (D, H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j, k),
                smem.get_base_pointer(),
                mbar,
                im2col_offsets=(0,),
                mode=cl.TMALoadMode.IM2COL,
            )

        compiled = cl.compile_simt(
            kernel,
            [
                KernelSignature(
                    [
                        make_symbolic_tensor((1, 1, 1), cl.int32),
                        make_symbolic_scalar(cl.bool_),
                        make_symbolic_scalar(cl.int32),
                        make_symbolic_scalar(cl.int32),
                        make_symbolic_scalar(cl.int32),
                        4,
                        32,
                        8,
                    ]
                )
            ],
            log_ptx=True,
        )
        assert compiled.ptx is not None
        assert "cp.async.bulk.tensor.3d.shared::cta.global.im2col" in compiled.ptx

    def test_tile_gather4_rejects_im2col_offsets(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                im2col_offsets=(0, 1),
                mode=cl.TMALoadMode.TILE_GATHER4,
            )

        with pytest.raises(
            TileTypeError, match="TILE_GATHER4 mode does not accept im2col_offsets"
        ):
            self.check_ptx_source(kernel)

    def test_invalid_tensor_map_pointer(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.copy_async_bulk_tensor_global_to_shared(
                smem.get_base_pointer(),
                (i, j),
                smem.get_base_pointer(),
                mbar,
            )

        with pytest.raises(
            TileTypeError,
            match="Expected tensor map or opaque tensor map pointer",
        ):
            self.check_ptx_source(kernel)


@require_hopper_or_newer()
class TestS2G(CopyAsyncPtxTestBase):
    def test_minimal(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j),
            )

        self.check_ptx_source(kernel, "cp.async.bulk.tensor.2d.global.shared::cta")

    def test_predicate(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j),
                predicate=pred,
            )

        self.check_ptx_source(kernel, "cp.async.bulk.tensor.2d.global.shared::cta")

    @require_blackwell_cc100()
    def test_tile_scatter4_store_mode(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j, 0, 0, 0),
                mode=cl.TMAStoreMode.TILE_SCATTER4,
            )

        self.check_ptx_source(
            kernel, "cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4"
        )

    def test_im2col_store_mode_rank2_is_rejected(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j),
                mode=cl.TMAStoreMode.IM2COL,
            )

        with pytest.raises(Exception, match="im2col|IM2COL|expected"):
            self.check_ptx_source(kernel)

    def test_im2col_store_mode_rank3(self):
        @cl.kernel
        def kernel(
            x,
            pred,
            i,
            j,
            k,
            D: cl.Constant[int],
            H: cl.Constant[int],
            W: cl.Constant[int],
        ):
            tensor_map = cl.tensor_map_tiled(x, (D, H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.copy_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j, k),
                mode=cl.TMAStoreMode.IM2COL,
            )

        compiled = cl.compile_simt(
            kernel,
            [
                KernelSignature(
                    [
                        make_symbolic_tensor((1, 1, 1), cl.int32),
                        make_symbolic_scalar(cl.bool_),
                        make_symbolic_scalar(cl.int32),
                        make_symbolic_scalar(cl.int32),
                        make_symbolic_scalar(cl.int32),
                        4,
                        32,
                        8,
                    ]
                )
            ],
            log_ptx=True,
        )
        assert compiled.ptx is not None
        assert "cp.async.bulk.tensor.3d.global.shared::cta.im2col" in compiled.ptx


@require_hopper_or_newer()
def test_copy_async_bulk_wait_group_read():
    def k():
        cl.copy_async_bulk_wait_group(0, read=False)
        cl.copy_async_bulk_wait_group(1, read=False)

    compile_kernel(
        k,
        filecheck_ptx="""
        CHECK-NOT: cp.async.bulk.wait_group.read
        CHECK: cp.async.bulk.wait_group 0
        CHECK-NEXT: cp.async.bulk.wait_group 1
        """,
    )


@require_hopper_or_newer()
def test_copy_async_bulk_wait_group():
    def k():
        cl.copy_async_bulk_wait_group(0, read=True)
        cl.copy_async_bulk_wait_group(1, read=True)

    compile_kernel(
        k,
        filecheck_ptx="""
        CHECK: cp.async.bulk.wait_group.read 0
        CHECK-NEXT: cp.async.bulk.wait_group.read 1
        """,
    )


@require_hopper_or_newer()
def test_copy_async_bulk_commit_group():
    def k():
        cl.copy_async_bulk_commit_group()

    compile_kernel(
        k,
        assert_in_ptx="cp.async.bulk.commit_group",
    )


@require_hopper_or_newer()
def test_copy_async_bulk_wait_group_non_immediate_group():
    def k(input):
        cl.copy_async_bulk_wait_group(input[0])

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_tensor(1, dtype=cl.int32)]),
        raises=pytest.raises(Exception, match="Expected constant of type int"),
    )


@require_hopper_or_newer()
def test_copy_async_bulk_wait_group_non_immediate_read():
    def k(input):
        cl.copy_async_bulk_wait_group(0, read=input[0] > 0)

    compile_kernel(
        k,
        signature=KernelSignature([make_symbolic_tensor(1, dtype=cl.int32)]),
        raises=pytest.raises(Exception, match="Expected constant of type bool"),
    )
