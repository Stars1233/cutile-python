# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import pytest
import torch
from torch.testing import make_tensor

import cuda.tile as ct
from cuda.tile._exception import TileTypeError
from conftest import arithmetic_dtypes, dtype_id
from util import assert_equal

ConstInt = ct.Constant[int]


def check_tiled_view_properties(tiled_view, dtype, tile_shape):
    tv_dtype, tv_tile_shape = tiled_view.dtype, tiled_view.tile_shape
    ct.static_assert(tv_dtype == dtype)
    ct.static_assert(tv_tile_shape == tile_shape)


@pytest.mark.parametrize("shape", [64, (128,), (225,)])
@pytest.mark.parametrize("tile_size", [64, 128])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("allow_tma", [False, True])
def test_tiled_view_copy_1d(shape, tile_size, dtype, allow_tma):
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        bid = ct.bid(0)
        tv_x = x.tiled_view(TILE)
        check_tiled_view_properties(tv_x, x.dtype, (TILE,))
        tv_y = y.tiled_view(TILE)
        tv_y.store(bid, tv_x.load(bid, allow_tma=allow_tma), allow_tma=allow_tma)

    x = make_tensor(shape, dtype=dtype, device='cuda')
    y = torch.zeros_like(x)
    shape = shape[0] if isinstance(shape, tuple) else shape
    grid = (ct.cdiv(shape, tile_size),)
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, tile_size))
    assert_equal(y, x)


@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("shape", [(128, 256), (192, 134)])
@pytest.mark.parametrize("tile_size", [(64, 64), (128, 128)])
@pytest.mark.parametrize("dtype", arithmetic_dtypes, ids=dtype_id)
def test_tiled_view_copy_2d(shape, tile_size, dtype, noncontiguous):

    @ct.kernel
    def kernel(x, y, n, TILE_M: ConstInt, TILE_N: ConstInt):
        bidm = ct.bid(0)
        bidn = ct.bid(1)
        tv_x = x.tiled_view((TILE_M, TILE_N))
        check_tiled_view_properties(tv_x, x.dtype, (TILE_M, TILE_N))
        tv_y = y.tiled_view((TILE_M, TILE_N))
        tv_y.store((bidm, bidn), tv_x.load((bidm, bidn)))
        tv_n = n.tiled_view(())
        check_tiled_view_properties(tv_n, n.dtype, ())
        if bidm == 0 and bidn == 0:
            nt1, nt2 = tv_x.num_tiles(0), tv_x.num_tiles(1)
            tv_n.store(0, nt1)
            tv_n.store(1, nt2)

    x = make_tensor(shape, dtype=dtype, device='cuda', noncontiguous=noncontiguous)
    y = torch.zeros_like(x)
    n = torch.zeros(len(shape), dtype=torch.int32, device='cuda')
    ref_n = torch.tensor([ct.cdiv(shape[0], tile_size[0]), ct.cdiv(shape[1], tile_size[1])],
                         dtype=torch.int32,
                         device='cuda')

    grid = (ct.cdiv(shape[0], tile_size[0]), ct.cdiv(shape[1], tile_size[1]))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, n, tile_size[0], tile_size[1]))
    assert_equal(y, x)
    assert_equal(n, ref_n)


_padding_mode_to_val = {
    ct.PaddingMode.ZERO: 0.0,
    ct.PaddingMode.NEG_ZERO: -0.0,
    ct.PaddingMode.NAN: math.nan,
    ct.PaddingMode.POS_INF: math.inf,
    ct.PaddingMode.NEG_INF: -math.inf,
}


@pytest.mark.parametrize("padding_mode", [
    ct.PaddingMode.ZERO,
    ct.PaddingMode.NEG_ZERO,
    ct.PaddingMode.NAN,
    ct.PaddingMode.POS_INF,
    ct.PaddingMode.NEG_INF
], ids=str)
def test_tiled_view_padding_mode(padding_mode):
    @ct.kernel
    def kernel(x, z, TILE: ConstInt):
        tv = x.tiled_view(TILE, padding_mode=padding_mode)
        tile = tv.load(1)
        ct.store(z, 0, tile=tile)

    x = make_tensor((100,), dtype=torch.float32, device='cuda')
    z = torch.zeros(1, dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, z, 128))

    if padding_mode == ct.PaddingMode.NAN:
        assert math.isnan(z.item())
    else:
        assert z.item() == _padding_mode_to_val[padding_mode]


@pytest.mark.parametrize("tile_size", [(1, 2), (1, 2, 3), (1, 2, 3, 4)])
def test_tiled_view_rank_mismatch(tile_size):
    @ct.kernel
    def kernel(x):
        x.tiled_view(tile_size)

    x = torch.zeros(16, dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError, match=f"Expected shape length to be 1, got {len(tile_size)}"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_store_tile_shape_mismatch():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        wrong_tile = ct.load(x, 0, (TILE * 2,))
        y.tiled_view(TILE).store(0, wrong_tile)

    x = torch.zeros(16, dtype=torch.float32, device='cuda')
    y = torch.zeros(16, dtype=torch.float32, device='cuda')
    match = r"Tile shape \(8,\) is not broadcastable to the tiled view's tile shape \(4,\)"
    with pytest.raises(TileTypeError, match=match):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 4))


@pytest.mark.parametrize("src_shape,dst_shape", [
    ((),       (16, 16)),
    ((1, 16),  (128, 16)),
    ((16, 1),  (16, 64)),
    ((1, 1),   (32, 16)),
])
def test_tiled_view_store_broadcast(src_shape, dst_shape):
    @ct.kernel
    def kernel(x, y):
        tile = x.tiled_view(src_shape).load((0, 0))
        y.tiled_view(dst_shape).store((0, 0), tile)

    x_shape = src_shape if len(src_shape) > 0 else (1, 1)
    x = make_tensor(x_shape, dtype=torch.float32, device='cuda')
    y = torch.zeros(dst_shape, dtype=torch.float32, device='cuda')
    ref = torch.broadcast_to(x, dst_shape)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


@pytest.mark.parametrize("use_x", [True, False])
def test_tiled_view_ifelse_result(use_x):
    @ct.kernel
    def kernel(x, y, z, TILE: ConstInt, USE_X: ct.Constant[bool]):
        tv = x.tiled_view(TILE) if USE_X else y.tiled_view(TILE)
        for i in range(tv.num_tiles(0)):
            z.tiled_view(TILE).store(i, tv.load(i))

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = make_tensor((128,), dtype=torch.float32, device='cuda')
    z = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z, 64, use_x))
    assert_equal(z, x if use_x else y)


def test_tiled_view_loop_carried():
    @ct.kernel
    def kernel(x, y, z, TILE: ConstInt):
        tv = x.tiled_view(TILE)
        tv_z = z.tiled_view(TILE)
        for i in range(tv_z.num_tiles(0)):
            tv_z.store(i, tv.load(0))
            tv = y.tiled_view(TILE)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = make_tensor((128,), dtype=torch.float32, device='cuda')
    z = torch.zeros((256,), dtype=torch.float32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, z, 128))
    ref_z = torch.cat((x, y))
    assert_equal(z, ref_z)


def test_tiled_view_ifelse_type_mismatch():
    @ct.kernel
    def kernel(x, cond: bool, TILE_A: ConstInt, TILE_B: ConstInt):
        if cond:
            tv = x.tiled_view(TILE_A)
        else:
            tv = x.tiled_view(TILE_B)
        tv.store(0, ct.full(TILE_A, 1.0, ct.float32))

    x = torch.zeros(128, dtype=torch.float32, device='cuda')
    with pytest.raises(TileTypeError, match="depends on path taken"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, True, 64, 32))


def test_tiled_view_helper_func():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        def get_view(arr, tile_size):
            return arr.tiled_view(tile_size)

        def copy_tile(tv_src, tv_dst, i):
            tv_dst.store(i, tv_src.load(i))

        tv_x = get_view(x, TILE)
        tv_y = get_view(y, TILE)
        for i in range(tv_x.num_tiles(0)):
            copy_tile(tv_x, tv_y, i)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 64))
    assert_equal(y, x)


def test_tiled_view_closure():
    @ct.kernel
    def kernel(x, y, TILE: ConstInt):
        tv_x = x.tiled_view(TILE)

        def make_closure():
            tv_y = y.tiled_view(TILE)

            def copy(i):
                tv_y.store(i, tv_x.load(i))

            return copy

        func = make_closure()
        for i in range(tv_x.num_tiles(0)):
            func(i)

    x = make_tensor((128,), dtype=torch.float32, device='cuda')
    y = torch.zeros_like(x)
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, 64))
    assert_equal(y, x)
