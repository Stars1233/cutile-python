# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

import cuda.lang as cl
from cuda.lang._compile import KernelSignature, get_compute_capability
from cuda.lang._exception import TileTypeError, TileValueError
from cuda.lang._logging import get_log_flags


cc = get_compute_capability()

if cc.major != 10:
    pytest.skip(reason="Blackwell only", allow_module_level=True)


@pytest.fixture
def log_ptx():
    get_log_flags().log_ptx = True


@pytest.mark.parametrize(
    "mc_mask,cta_group,expect",
    [
        [
            0xAB,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster"
            ".multicast::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_1,
            "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64",
        ],
        [
            None,
            cl.CTAGroup.CTA_2,
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64",
        ],
    ],
)
def test_commit(log_ptx, mc_mask, cta_group, expect):
    @cl.kernel
    def kernel():
        mbar = cl.shared_array(1, cl.mbarrier).get_base_pointer()
        cl.tcgen05_commit(mbar, multicast_mask=mc_mask, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.compiler_stderr.decode()
    assert expect in ptx


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
        ],
    ],
)
def test_alloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_alloc(p3, 5, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.compiler_stderr.decode()
    assert expect in ptx, ptx


def test_dealloc_requires_tensor_pointer():
    @cl.kernel
    def kernel():
        p3 = cl.shared_array(1, cl.uint32).get_base_pointer()
        cl.tcgen05_dealloc(p3, 5)

    with pytest.raises(TileTypeError, match="Expected pointer to be in tensor address space"):
        cl.compile_simt(kernel, [KernelSignature([])])


@pytest.mark.parametrize(
    "cta_group,expect",
    [
        [
            cl.CTAGroup.CTA_1,
            "tcgen05.dealloc.cta_group::1.sync.aligned.b32",
        ],
        [
            cl.CTAGroup.CTA_2,
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32",
        ],
    ],
)
def test_dealloc(log_ptx, cta_group, expect):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128, cta_group=cta_group)
        tmem_ptr = smem[0]
        cl.tcgen05_dealloc(tmem_ptr, 128, cta_group=cta_group)

    compiled = cl.compile_simt(kernel, [KernelSignature([])])
    ptx = compiled.compiler_stderr.decode()
    assert expect in ptx, ptx


@pytest.mark.parametrize("shape", cl.Tcgen05LdStShape._member_map_.values())
@pytest.mark.parametrize("count", (1, 2, 4, 8, 16, 32, 64, 128))
@pytest.mark.parametrize("pack", (True, False, None))
@pytest.mark.parametrize("offset", (True, False, None))
def test_ld(log_ptx, shape, count, pack, offset):
    @cl.kernel
    def kernel():
        tmem_dtype = cl.pointer_dtype(cl.int8, cl.MemorySpace.TENSOR)
        smem = cl.shared_array(1, tmem_dtype, alignment=4)
        cl.tcgen05_alloc(smem.get_base_pointer(), 128)
        tmem_ptr = smem[0]
        cl.tcgen05_ld(shape, tmem_ptr, count=count, pack=pack, offset=offset)
        cl.tcgen05_dealloc(tmem_ptr, 128)

    def do_compile():
        compiled = cl.compile_simt(kernel, [KernelSignature([])])
        ptx = compiled.compiler_stderr.decode()
        shape_str = str(shape).split("_")[1].lower()
        assert "tcgen05.ld.sync.aligned" in ptx and shape_str in ptx, ptx

    bad_args = offset is not None and shape is not cl.Tcgen05LdStShape.SHAPE_16X32BX2
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X256B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
    )
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X32BX2 and offset is None
    bad_args |= shape is cl.Tcgen05LdStShape.SHAPE_16X128B and count not in (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    if bad_args:
        with pytest.raises((TileTypeError, TileValueError)):
            do_compile()
    else:
        do_compile()
