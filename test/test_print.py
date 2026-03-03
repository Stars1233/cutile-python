# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import math
import sys
import subprocess
import ast
import torch
import numpy as np
import pytest

from math import ceil
import cuda.tile as ct

# FIXME: Default opt_level causes print to be out of order.
# Remove when it is fixed in tile compiler.


@ct.kernel(opt_level=0)
def kernel_printf_float(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.printf("tile[%d]:%.5f\n", bid, tx)


@ct.kernel(opt_level=0)
def kernel_printf_int(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.printf("tile[%d]:%d\n", bid, tx)


@ct.kernel(opt_level=0)
def kernel_print_int(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"tile[{bid}]:{tx}")


@ct.kernel(opt_level=0)
def kernel_print_float(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"tile[{bid}]:{tx:.5f}")


@ct.kernel(opt_level=0)
def kernel_print_sep(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print("tile:", tx, sep='')


@ct.kernel(opt_level=0)
def kernel_print_two_vars_with_expr(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(f"tile[{bid}]: a={tx:.6f} b={tx + tx:.6f}")


@ct.kernel(opt_level=0)
def kernel_print_no_end(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ct.print(tx, end='')


@ct.kernel(opt_level=0)
def kernel_builtin_print_int(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    print(f"tile[{bid}]:{tx}")


@ct.kernel(opt_level=0)
def kernel_builtin_print_float(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    print(f"tile[{bid}]:{tx:.5f}")


_KERNELS_MAP_ = {
    "kernel_printf_float": kernel_printf_float,
    "kernel_printf_int": kernel_printf_int,
    "kernel_print_int": kernel_print_int,
    "kernel_print_float": kernel_print_float,
    "kernel_print_sep": kernel_print_sep,
    "kernel_print_two_vars_with_expr": kernel_print_two_vars_with_expr,
    "kernel_print_no_end": kernel_print_no_end,
    "kernel_builtin_print_int": kernel_builtin_print_int,
    "kernel_builtin_print_float": kernel_builtin_print_float,
}


def _run_kernel_subprocess(kernel_name: str, shape: str, dtype_str: str, tile: str):
    shape = ast.literal_eval(shape)
    dtype = getattr(torch, dtype_str)
    tile = int(tile)
    x = torch.arange(torch.prod(torch.tensor(shape)), device='cuda').reshape(shape).to(dtype)
    grid = (ceil(shape[0] / tile), 1, 1)
    kernel = _KERNELS_MAP_[kernel_name]
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, tile))
    torch.cuda.synchronize()


def _run_kernel_proc(kernel_name, shape, dtype_str, tile):
    return subprocess.run(
        [sys.executable, __file__, "run_kernel",
         kernel_name, str(shape), dtype_str, str(tile)],
        capture_output=True,
    )


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype_str", ["float32", "float16", "int32"])
@pytest.mark.parametrize("float_kernel,int_kernel", [
    ("kernel_printf_float", "kernel_printf_int"),
    ("kernel_print_float", "kernel_print_int"),
], ids=["ct_printf", "ct_print"])
def test_print_1d(shape, tile, dtype_str, float_kernel, int_kernel):
    kernel_name = float_kernel if "float" in dtype_str else int_kernel
    proc = _run_kernel_proc(kernel_name, shape, dtype_str, tile)
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0

    actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines()
                   if line]
    dtype = getattr(np, dtype_str)
    x = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile*i, tile*(i+1)
        if "float" in dtype_str:
            formatted_x = ', '.join([f"{elem:.5f}" for elem in x[start_idx:end_idx]])
        elif "int" in dtype_str:
            formatted_x = ', '.join([f"{elem}" for elem in x[start_idx:end_idx]])
        else:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        expected_out = f"tile[{i}]:[{formatted_x}]"
        assert expected_out in actual_outs


@pytest.mark.parametrize("shape", [(8,),])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_sep(shape, tile):
    proc = _run_kernel_proc("kernel_print_sep", shape, "int32", tile)
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0

    actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines() if line]
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    expected = f"tile:[{formatted_x}]"
    assert expected in actual_outs


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_two_vars(shape, tile):
    proc = _run_kernel_proc("kernel_print_two_vars_with_expr", shape, "float32", tile)
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0

    actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines() if line]
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.float32)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        formatted_a = ', '.join([f"{elem:.6f}" for elem in x[start_idx:end_idx]])
        formatted_b = ', '.join([f"{elem * 2:.6f}" for elem in x[start_idx:end_idx]])
        expected = f"tile[{i}]: a=[{formatted_a}] b=[{formatted_b}]"
        assert expected in actual_outs


@pytest.mark.parametrize("shape", [(8,),])
@pytest.mark.parametrize("tile", [8])
def test_ct_print_no_end(shape, tile):
    proc = _run_kernel_proc("kernel_print_no_end", shape, "int32", tile)
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0
    stdout = proc.stdout.decode("UTF-8")
    x = np.arange(np.prod(shape)).reshape(shape).astype(np.int32)
    formatted_x = ', '.join([f"{elem}" for elem in x[:tile]])
    assert f"[{formatted_x}]" in stdout


@pytest.mark.parametrize("shape", [(8,), (16,)])
@pytest.mark.parametrize("tile", [8])
@pytest.mark.parametrize("dtype_str", ["float32", "int32"])
def test_builtin_print(shape, tile, dtype_str):
    kernel_name = ("kernel_builtin_print_float" if "float" in dtype_str
                   else "kernel_builtin_print_int")
    proc = _run_kernel_proc(kernel_name, shape, dtype_str, tile)
    print(proc.stderr.decode(), file=sys.stderr)
    assert proc.returncode == 0

    actual_outs = [line for line in proc.stdout.decode("UTF-8").splitlines() if line]
    x = np.arange(np.prod(shape)).reshape(shape)
    num_tiles = math.ceil(shape[0] / tile)
    for i in range(num_tiles):
        start_idx, end_idx = tile * i, tile * (i + 1)
        if "float" in dtype_str:
            formatted = ', '.join([f"{elem:.5f}"
                                   for elem in x[start_idx:end_idx].astype(np.float32)])
        else:
            formatted = ', '.join([f"{elem}" for elem in x[start_idx:end_idx].astype(np.int32)])
        expected = f"tile[{i}]:[{formatted}]"
        assert expected in actual_outs


def test_ct_print_error_conversion():
    from cuda.tile._exception import TileSyntaxError

    @ct.kernel(opt_level=0)
    def bad_kernel(x, TILE: ct.Constant[int]):
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ct.print(f"{tx!r}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileSyntaxError, match="!r, !s, !a"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))


def test_ct_print_error_dynamic_format_spec():
    from cuda.tile._exception import TileSyntaxError

    @ct.kernel(opt_level=0)
    def bad_kernel(x, TILE: ct.Constant[int]):
        width = 5
        tx = ct.load(x, index=(0,), shape=(TILE,))
        ct.print(f"{tx:{width}}")

    x = torch.zeros(8, device='cuda', dtype=torch.int32)
    with pytest.raises(TileSyntaxError, match="dynamic format specs"):
        ct.launch(torch.cuda.current_stream(), (1, 1, 1), bad_kernel, (x, 8))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "run_kernel":
        _run_kernel_subprocess(*sys.argv[2:])
