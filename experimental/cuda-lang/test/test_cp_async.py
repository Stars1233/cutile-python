# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._exception import TileTypeError
from cuda.lang.compilation import KernelSignature

from .util import make_symbolic_scalar, make_symbolic_tensor, require_hopper_or_newer


class CpAsyncPtxTestBase:
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
class TestG2S(CpAsyncPtxTestBase):
    def test_minimal(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(
                shape=(), dtype=cl.mbarrier, alignment=8
            ).get_base_pointer()

            cl.cp_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
            )

        self.check_ptx_source(kernel, "cp.async.bulk.tensor.2d.shared::cta.global")

    def test_shared_clsuter_nyi(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            smem = cl.map_shared_to_cluster(smem.get_base_pointer(), 0)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.cp_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem,
                mbar,
                predicate=pred,
            )

        with pytest.raises(
            TileTypeError,
            match="Copying from global to shared-cluster memory is not yet supported",
        ):
            self.check_ptx_source(kernel)

    def test_unsupported_kwargs_for_cta_mode(self, subtests):
        @cl.kernel
        def k1(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.cp_async_bulk_tensor_global_to_shared(
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

            cl.cp_async_bulk_tensor_global_to_shared(
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

            cl.cp_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                group=cl.CTAGroup.CTA_1,
            )

        def compile(kernel):
            match = (
                "When the destination memory is in shared memory, the "
                "predicate, multicast mask, and group arguments are invalid."
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

        with subtests.test("group"):
            compile(k3)

    def test_im2col_offsets_without_required_load_mode(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.cp_async_bulk_tensor_global_to_shared(
                tensor_map,
                (i, j),
                smem.get_base_pointer(),
                mbar,
                im2col_offsets=(0, 1),
            )

        with pytest.raises(
            TileTypeError, match="TILE mode does not accept im2col_offsets"
        ):
            self.check_ptx_source(kernel)

    def test_invalid_tensor_map_pointer(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)
            mbar = cl.shared_array(1, cl.mbarrier, alignment=8).get_base_pointer()

            cl.cp_async_bulk_tensor_global_to_shared(
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
class TestS2G(CpAsyncPtxTestBase):
    def test_minimal(self):
        @cl.kernel
        def kernel(x, pred, i, j, H: cl.Constant[int], W: cl.Constant[int]):
            tensor_map = cl.tensor_map_tiled(x, (H, W)).as_opaque_ptr()
            smem = cl.shared_array(shape=(H * W,), dtype=cl.int32, alignment=512)

            cl.cp_async_bulk_tensor_shared_to_global(
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

            cl.cp_async_bulk_tensor_shared_to_global(
                smem.get_base_pointer(),
                tensor_map,
                (i, j),
                predicate=pred,
            )

        self.check_ptx_source(kernel, "cp.async.bulk.tensor.2d.global.shared::cta")
