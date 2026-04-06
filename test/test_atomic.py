# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
import pytest
from math import ceil

import torch
from torch.testing import make_tensor

import cuda.tile as ct
from cuda.tile._bytecode.version import BytecodeVersion
from cuda.tile._exception import TileTypeError
from cuda.tile._ir.ops_utils import _is_implicit_cast_ok
from cuda.tile._ir.typing_support import to_dtype
from util import (
    assert_equal, filecheck, get_int_dtype_of_same_size, jit_kernel,
    get_bytecode, raises_if
)
from conftest import arithmetic_dtypes, dtype_id, get_tileiras_version


# === Helpers ===
kernel_cache = {}


def _jit_helper(name: str, source: str, tmp_path):
    if source not in kernel_cache:
        kernel_cache[source] = jit_kernel(name, source, tmp_path)
    return kernel_cache[source]


array_kernel_template = """
def {name}(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    val = ct.gather(y, offset)
    old_val = {body}(x, offset, val,
                     memory_order=ct.MemoryOrder.ACQ_REL,
                     memory_scope=ct.MemoryScope.DEVICE)
    ct.scatter(z, offset, old_val)"""


array_order_scope_template = """
def {name}(x, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    val = ct.full((TILE,), 1, dtype=ct.int32)
    old_val = ct.atomic_add(x, offset, val{args})"""


def array_kernel(name: str, body: str, tmp_path):
    name = 'array_' + name
    source = array_kernel_template.format(name=name, body=body)
    return _jit_helper(name, source, tmp_path)


def launch_array_kernel(kernel, x, y, z, tile: int):
    assert z.ndim >= 1 and z.ndim <= 3
    grid = tuple(map(lambda d: ceil(d / tile), z.shape))
    ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z, tile))


scalar_kernel_template = """
def {name}(x, y, z):
    val = ct.gather(y, 0)
    old_val = {body}(x, 0, val)
    ct.scatter(z, 0, old_val)"""


def scalar_kernel(name: str, body: str, tmp_path):
    name = 'scalar_' + name
    source = scalar_kernel_template.format(name=name, body=body)
    return _jit_helper(name, source, tmp_path)


# === End of Helpers ===


def ref_atomic_arith(x, y, operation):
    if x.dtype in [torch.uint32, torch.uint64]:
        # Cast to float64 because torch cuda maximum, minimum do not support uint32/64
        ref_x = operation(x.to(torch.float64), y.to(torch.float64))
        ref_x = ref_x.to(x.dtype)
    else:
        ref_x = operation(x, y.to(x.dtype))
    ref_z = x.clone()
    return ref_x, ref_z


def create_atomic_test_params(ops_config):
    params = []
    for op_name, torch_op, supported_dtypes in ops_config:
        for x_dtype in supported_dtypes:
            param_id = f"{op_name}-{dtype_id(x_dtype)}"
            params.append(pytest.param(op_name, torch_op, x_dtype, id=param_id))
    return params


int_32_64_dtypes = [torch.uint32, torch.uint64, torch.int32, torch.int64]
float_32_64_dtypes = [torch.float32, torch.float64]
int_float_32_64_dtypes = int_32_64_dtypes + float_32_64_dtypes

atomic_arith_config = [
    ("xchg", lambda _, y: y, int_float_32_64_dtypes),
    ("add", torch.add, int_float_32_64_dtypes + [torch.float16]),
    ("max", torch.maximum, int_32_64_dtypes),
    ("min", torch.minimum, int_32_64_dtypes),
]


@pytest.mark.parametrize("op_name,torch_op,x_dtype",
                         create_atomic_test_params(atomic_arith_config))
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
def test_atomic_arith(op_name, torch_op, x_dtype, y_dtype, tmp_path, mode):
    if op_name == "xchg" and get_tileiras_version() == BytecodeVersion.V_13_3:
        pytest.xfail(reason="unblock development only. TODO: remove before release")

    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        kernel = array_kernel(f"atomic_{op_name}", f"ct.atomic_{op_name}", tmp_path)

        def launch():
            launch_array_kernel(kernel, x, y, z, 128)
    else:  # scalar
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        kernel = scalar_kernel(f"atomic_{op_name}", f"ct.atomic_{op_name}", tmp_path)
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z))

    invalid_cast = not _is_implicit_cast_ok(to_dtype(y_dtype), to_dtype(x_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ref_x, ref_z = ref_atomic_arith(x, y, torch_op)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


def ref_atomic_bitwise(x, y, operation):
    int_dtype = get_int_dtype_of_same_size(x.dtype)
    ref_x = operation(x.view(int_dtype), y.view(int_dtype)).view(x.dtype)
    ref_z = x.clone()
    return ref_x, ref_z


atomic_bitwise_config = [
    ("and", lambda x, y: x & y, int_float_32_64_dtypes),
    ("or", lambda x, y: x | y, int_float_32_64_dtypes),
    ("xor", lambda x, y: x ^ y, int_float_32_64_dtypes),
]


@pytest.mark.parametrize("op_name,torch_op,x_dtype",
                         create_atomic_test_params(atomic_bitwise_config))
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
def test_atomic_bitwise(op_name, torch_op, x_dtype, y_dtype, tmp_path, mode):
    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        kernel = array_kernel(f"atomic_{op_name}", f"ct.atomic_{op_name}", tmp_path)

        def launch():
            launch_array_kernel(kernel, x, y, z, 128)
    else:  # scalar
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        kernel = scalar_kernel(f"atomic_{op_name}", f"ct.atomic_{op_name}", tmp_path)
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, kernel, (x, y, z))

    x_dtype = to_dtype(x_dtype)
    y_dtype = to_dtype(y_dtype)
    if x_dtype in (ct.float32, ct.float64):
        with pytest.raises(TileTypeError, match="Unsupported array dtype"):
            launch()
    elif x_dtype != y_dtype:
        msg = re.escape(f"Bitwise atomic read-modify-write operations require that the "
                        f"update dtype ({y_dtype}) exactly matches the array dtype ({x_dtype})")
        with pytest.raises(TileTypeError, match=msg):
            launch()
    else:
        ref_x, ref_z = ref_atomic_bitwise(x, y, torch_op)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


@ct.kernel
def atomic_cas(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    offset = ct.arange(TILE, dtype=ct.int64)
    offset += bid*TILE
    cmp = ct.gather(x, offset)
    val = ct.gather(y, offset)
    old_val = ct.atomic_cas(x, offset, cmp, val,
                            memory_order=ct.MemoryOrder.ACQ_REL,
                            memory_scope=ct.MemoryScope.DEVICE)
    ct.scatter(z, offset, old_val)


@ct.kernel
def scalar_atomic_cas(x, y, z):
    cmp = ct.gather(x, 0)
    val = ct.gather(y, 0)
    old_val = ct.atomic_cas(x, 0, cmp, val)
    ct.scatter(z, 0, old_val)


def ref_atomic_cas(x, y):
    ref_x = y.to(x.dtype)
    ref_z = x.clone()
    return ref_x, ref_z


atomic_cas_dtypes = [torch.uint32, torch.uint64, torch.int32, torch.int64,
                     torch.float32, torch.float64]


@pytest.mark.parametrize("x_dtype", atomic_cas_dtypes, ids=dtype_id)
@pytest.mark.parametrize("y_dtype", arithmetic_dtypes, ids=dtype_id)
@pytest.mark.parametrize("mode", ["array", "scalar"])
def test_atomic_cas(x_dtype, y_dtype, mode):
    if get_tileiras_version() == BytecodeVersion.V_13_3:
        pytest.xfail(reason="unblock development only. TODO: remove before release")

    if mode == "array":
        x = make_tensor((512,), dtype=x_dtype, device='cuda')
        y = make_tensor((512,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = tuple(map(lambda d: ceil(d / 128), z.shape))

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, atomic_cas, (x, y, z, 128))
    else:  # scalar
        x = make_tensor((1,), dtype=x_dtype, device='cuda')
        y = make_tensor((1,), dtype=y_dtype, device='cuda')
        z = torch.zeros_like(x, device="cuda")
        grid = (1,)

        def launch():
            ct.launch(torch.cuda.current_stream(), grid, scalar_atomic_cas, (x, y, z))

    invalid_cast = not _is_implicit_cast_ok(to_dtype(y_dtype), to_dtype(x_dtype))
    msg = "cannot implicitly cast"
    with raises_if(invalid_cast, TileTypeError, match=re.escape(msg)):
        ref_x, ref_z = ref_atomic_cas(x, y)
        launch()
        assert_equal(x, ref_x)
        assert_equal(z, ref_z)


ct_scope_to_tileir_scope = {
    ct.MemoryScope.BLOCK: "tl_blk",
    ct.MemoryScope.DEVICE: "device",
    ct.MemoryScope.SYS: "sys"
}


@pytest.mark.use_mlir
@pytest.mark.parametrize("order", [None, "RELAXED", "ACQUIRE", "RELEASE", "ACQ_REL"])
@pytest.mark.parametrize("scope", [None, "BLOCK", "DEVICE", "SYS"])
def test_atomic_order_scope(order, scope, tmp_path):
    name = f"atomic_order_scope_{order}_{scope}"
    args = ""
    check_directive = "// CHECK: atomic_rmw_tko"

    # set up expected order
    if order:
        order_enum = f"ct.MemoryOrder.{order}"
        args += f", memory_order={order_enum}"
    else:
        order_enum = "ct.MemoryOrder.ACQ_REL"
    check_directive += f" {eval(order_enum).value}"

    # set up expected scope
    if scope:
        scope_enum = f"ct.MemoryScope.{scope}"
        args += f", memory_scope={scope_enum}"
    else:
        scope_enum = "ct.MemoryScope.DEVICE"
    check_directive += f" {ct_scope_to_tileir_scope[eval(scope_enum)]}"

    source = array_order_scope_template.format(name=name, args=args)
    kernel = jit_kernel(name, source, tmp_path)
    x = make_tensor((512,), dtype=torch.int32, device='cuda')
    bytecode = get_bytecode(kernel, (x, 128))
    filecheck(bytecode, check_directive)


@ct.kernel
def mixed_scalar_tile_atomic(x, y):
    cmp = ct.gather(x, 0)
    val = ct.gather(y, 0)
    ct.atomic_cas(x, 0, cmp, val)
    ct.atomic_xchg(x, 1, val)
    ct.atomic_add(x, 2, val)
    ct.atomic_xor(x, 3, val)
    ct.atomic_max(x, 4, val)


def test_mixed_scalar_tile_atomic():
    x = make_tensor((1,), dtype=torch.int32, device="cuda")
    y = make_tensor((1,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), mixed_scalar_tile_atomic, (x, y))


class TestInvalidAtomicMemoryOrderAndScope:
    def test_atomic_cas_weak_ordering(self):
        @ct.kernel
        def kernel(x):
            ct.atomic_cas(x, 0, 0, 0, memory_order=ct.MemoryOrder.WEAK,
                          memory_scope=ct.MemoryScope.DEVICE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(TileTypeError, match="Invalid memory order for tile_atomic_cas"):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    def test_atomic_rmw_weak_ordering(self):
        @ct.kernel
        def kernel(x):
            ct.atomic_add(x, 0, 0, memory_order=ct.MemoryOrder.WEAK,
                          memory_scope=ct.MemoryScope.DEVICE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(TileTypeError, match="Invalid memory order for tile_atomic_rmw"):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    @pytest.mark.parametrize(
        "memory_order",
        [
            ct.MemoryOrder.ACQ_REL,
            ct.MemoryOrder.ACQUIRE,
            ct.MemoryOrder.RELEASE,
            ct.MemoryOrder.RELAXED,
        ],
    )
    def test_atomic_rmw_none_scope(self, memory_order):
        @ct.kernel
        def kernel(x):
            ct.atomic_add(x, 0, 0, memory_order=memory_order, memory_scope=ct.MemoryScope.NONE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(
            TileTypeError,
            match="tile_atomic_rmw with (.+) memory ordering requires a memory scope",
        ):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))

    @pytest.mark.parametrize(
        "memory_order",
        [
            ct.MemoryOrder.ACQ_REL,
            ct.MemoryOrder.ACQUIRE,
            ct.MemoryOrder.RELEASE,
            ct.MemoryOrder.RELAXED,
        ],
    )
    def test_atomic_cas_none_scope(self, memory_order):
        @ct.kernel
        def kernel(x):
            ct.atomic_cas(x, 0, 0, 0, memory_order=memory_order, memory_scope=ct.MemoryScope.NONE)
        x = make_tensor((1,), dtype=torch.int32, device="cuda")
        with pytest.raises(
            TileTypeError,
            match="tile_atomic_cas with (.+) memory ordering requires a memory scope",
        ):
            ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
