# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest
from typing import Annotated
from unittest.mock import patch

import cuda.tile
import cuda.tile as ct
import torch
import math
from cuda.tile import TileTypeError
from cuda.tile._bytecode import BytecodeVersion
from util import assert_equal
from conftest import requires_tileiras


@ct.kernel
def array_attr_kernel(X, out):
    ndim = X.ndim
    shape = X.shape
    strides = X.strides
    ct.static_assert(ndim == 3)
    ct.static_assert(len(shape) == ndim)
    ct.static_assert(len(strides) == ndim)

    ct.store(out, (0,), shape[0])
    ct.store(out, (1,), shape[1])
    ct.store(out, (2,), shape[2])
    ct.store(out, (3,), strides[0])
    ct.store(out, (4,), strides[1])
    ct.store(out, (5,), strides[2])


def test_array_attr():
    x = torch.zeros((2, 3, 4), device='cuda')
    out = torch.zeros(6, device='cuda', dtype=torch.int64)
    ct.launch(torch.cuda.current_stream(),
              (1,),
              array_attr_kernel, (x, out))
    assert list(out[0:3]) == list(x.shape)
    assert list(out[3:6]) == list(x.stride())


def test_array_getitem():
    @ct.kernel
    def kernel(x):
        x[0]

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays are not directly subscriptable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_array_setitem():
    @ct.kernel
    def kernel(x):
        x[0] = 3.0

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays do not support item assignment. Use store()"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


def test_array_aug_setitem():
    @ct.kernel
    def kernel(x):
        x[0] += 3

    x = torch.zeros((10,), device='cuda')
    with pytest.raises(TileTypeError, match="Arrays are not directly subscriptable"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))


@ct.kernel
def int64_index_inc1(x: ct.IndexedWithInt64, y: ct.IndexedWithInt64, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid, 0), shape=(TILE, 1))
    ct.store(y, index=(bid, 0), tile=tx + 1)


@requires_tileiras(BytecodeVersion.V_13_3)
def test_int64_index_inc1():
    """
    This test may be excluded from selected CI jobs with
    ``-k "not int64_index"`` because it requires a very large allocation.
    Keep ``int64_index`` in the test name unless those CI filters are updated.
    """
    n = (1 << 32) + 5

    x = torch.randint(-128, 127, (n, 1), device='cuda', dtype=torch.int8)
    y = torch.zeros(n, 1, device='cuda', dtype=torch.int8)

    TILE = 2048
    grid = (math.ceil(n / TILE), 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, int64_index_inc1, (x, y, TILE))
    assert_equal(y, x + 1)


def test_int64_index_overflow_without_annotation():
    # Stride > INT32_MAX triggers OverflowError without allocating 6 GiB.
    # dim-0 stride 2**32 exceeds INT32_MAX; dim-1 stride 0 keeps storage at 128 elements.
    base = torch.zeros(128, device='cuda', dtype=torch.bfloat16)
    x = torch.as_strided(base, (1, 25165824, 1, 128), (2**32, 0, 0, 1))
    out = torch.as_strided(base, (1, 25165824, 1, 128), (2**32, 0, 0, 1))

    @ct.kernel
    def kernel(value, out_):
        pass

    with pytest.raises(OverflowError):
        ct.launch(torch.cuda.current_stream(),
                  (1,),
                  kernel, (x, out))


@ct.kernel
def load_static_shaped(
        x: Annotated[ct.Array, ct.ArrayAnnotation(static_shape_dims=(0, 1))], out):
    t = ct.load(x, (0, 0), (16, 16))
    ct.store(out, (0, 0), t)


def test_static_shape_standalone_recompile():
    k = cuda.tile.kernel(load_static_shaped._pyfunc)
    shapes = [(16, 16), (32, 32), (16, 16)]
    with patch('cuda.tile._compile.compile_tile',
               side_effect=cuda.tile._compile.compile_tile) as mock_compile:
        for shape in shapes:
            x = torch.randint(0, 100, shape, dtype=torch.int32, device='cuda')
            out = torch.zeros((16, 16), dtype=torch.int32, device='cuda')
            ct.launch(torch.cuda.current_stream(), (1,), k, (x, out))
            assert_equal(out, x[:16, :16])
    assert mock_compile.call_count == 2


def test_static_shape_annotation_validation():
    with pytest.raises(TypeError, match="must be int"):
        ct.ArrayAnnotation(static_shape_dims=(False,))
