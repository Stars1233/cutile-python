# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import pytest

import cuda.tile as ct
import torch

from cuda.tile import TileTypeError, TileSyntaxError
from util import assert_equal


def test_tuple_comprehension_basic():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        rotated = tuple(tiles[(i + 1) % 3] for i in ct.static_iter(range(3)))
        for i, t in ct.static_iter(enumerate(rotated)):
            ct.store(y, (i,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    a, b, c = x[:16], x[16:32], x[32:]
    ref = torch.cat([b, c, a])

    y = torch.zeros((3 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_unpack_target():
    @ct.kernel
    def kernel(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        pairs = ((a, b), (b, a))
        diffs = tuple(u - v for u, v in ct.static_iter(pairs))
        for i, t in ct.static_iter(enumerate(diffs)):
            ct.store(y, (i,), t)

    x = torch.arange(2 * 16, dtype=torch.int32, device="cuda")
    a, b = x[:16], x[16:]
    ref = torch.cat([a - b, b - a])

    y = torch.zeros((2 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_nested_unpack_target():
    @ct.kernel
    def kernel(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        c = ct.load(x, (2,), (16,))
        d = ct.load(x, (3,), (16,))
        triples = ((a, (b, c)), (b, (c, d)))
        result = tuple(u + v + w for u, (v, w) in ct.static_iter(triples))
        for i, t in ct.static_iter(enumerate(result)):
            ct.store(y, (i,), t)

    x = torch.arange(4 * 16, dtype=torch.int32, device="cuda")
    a, b, c, d = x[:16], x[16:32], x[32:48], x[48:]
    ref = torch.cat([a + b + c, b + c + d])

    y = torch.zeros((2 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_multiple_generators():
    @ct.kernel
    def kernel(x, y):
        a = ct.load(x, (0,), (16,))
        b = ct.load(x, (1,), (16,))
        tiles = (a, b)
        tiles_square = (a * 2, b * 2)
        products = tuple(u + v for u in ct.static_iter(tiles)
                         for v in ct.static_iter(tiles_square))
        for i, t in ct.static_iter(enumerate(products)):
            ct.store(y, (i,), t)

    x = torch.arange(2 * 16, dtype=torch.int32, device="cuda")
    a, b = x[:16], x[16:]
    ref = torch.cat([a + a * 2, a + b * 2, b + a * 2, b + b * 2])

    y = torch.zeros((4 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_if_on_outer_generator():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        scales = (1, 2)
        result = tuple(t * s
                       for i, t in ct.static_iter(enumerate(tiles)) if i != 1
                       for s in ct.static_iter(scales))
        for k, t in ct.static_iter(enumerate(result)):
            ct.store(y, (k,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    a, c = x[:16], x[32:]
    ref = torch.cat([a * 1, a * 2, c * 1, c * 2])

    y = torch.zeros((4 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_nested():
    @ct.kernel
    def kernel(x):
        # inner range depends on outer induction var; inner tuple length varies per row
        result = tuple(tuple(i * 2 for i in ct.static_iter(range(j)))
                       for j in ct.static_iter(range(5)))
        # result = ((), (0,), (0, 2), (0, 2, 4), (0, 2, 4, 6)) — 0+1+2+3+4 = 10 elements
        idx = 0
        for row in ct.static_iter(result):
            for v in ct.static_iter(row):
                ct.scatter(x, idx, v)
                idx += 1

    x = torch.zeros(10, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x,))
    assert x.tolist() == [0, 0, 2, 0, 2, 4, 0, 2, 4, 6]


def test_tuple_comprehension_nested_in_for_loop():
    @ct.kernel
    def kernel(y):
        for offset in range(2):
            matrix = tuple(
                tuple(offset + i + j for i in ct.static_iter(range(2)))
                for j in ct.static_iter(range(2))
            )
            for i, row in ct.static_iter(enumerate(matrix)):
                for j, v in ct.static_iter(enumerate(row)):
                    ct.scatter(y, offset * 4 + i * 2 + j, v)

    y = torch.zeros(8, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [offset + i + j
                          for offset in range(2) for i in range(2) for j in range(2)]


def test_tuple_comprehension_iter_var_not_leaked():
    @ct.kernel
    def kernel(y):
        tiles = (1, 2)
        i = 99
        # comprehension uses i as its loop variable; it must not overwrite the outer i
        _ = tuple(t for i, t in ct.static_iter(enumerate(tiles)))
        ct.scatter(y, (), i)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y, ))
    assert y.item() == 99


def test_tuple_comprehension_closure_var_not_shadowed():
    n = 99

    @ct.kernel
    def kernel(y):
        # comprehension uses n as its induction variable; the captured outer n must remain 99
        _ = tuple(n for n in ct.static_iter(range(3)))
        ct.scatter(y, (), n)

    y = torch.zeros((), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.item() == 99


def test_tuple_comprehension_closure_var_in_for_loop_not_shadowed():
    j = 99

    @ct.kernel
    def kernel(y):
        tiles = (1, 2)
        # comprehension uses j as its loop variable; it must not overwrite the outer j
        for i in range(10):
            _ = tuple(t for j, t in ct.static_iter(enumerate(tiles)))
            ct.scatter(y, (i, ), j)
        ct.scatter(y, (10, ), j)

    y = torch.zeros((11,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y, ))
    assert (y == 99).all()


def test_tuple_comprehension_lambda():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        results = tuple((lambda t: t * scale)(tile)
                        for scale, tile in ct.static_iter(zip(range(1, 4), tiles)))
        for k, t in ct.static_iter(enumerate(results)):
            ct.store(y, (k,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    y = torch.zeros(3 * 16, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))

    ref = torch.cat([x[:16] * 1, x[16:32] * 2, x[32:] * 3])
    assert_equal(y, ref)


def test_tuple_comprehension_lambda_nested_capture():
    @ct.kernel
    def kernel(y):
        outer = tuple(
            tuple(lambda: j for j in ct.static_iter(range(1, 4)))
            for _ in ct.static_iter(range(1))
        )
        fns = outer[0]
        for k, f in ct.static_iter(enumerate(fns)):
            ct.scatter(y, k, f())

    y = torch.zeros(3, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [3, 3, 3]


def test_tuple_comprehension_lambda_stored():
    # Matches Python: all lambdas share one `scale` binding that holds the last value.
    @ct.kernel
    def kernel(y):
        fns = tuple(lambda: scale for scale in ct.static_iter(range(4)))
        for k, f in ct.static_iter(enumerate(fns)):
            ct.scatter(y, k, f())

    y = torch.zeros(4, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [3, 3, 3, 3]


def test_tuple_comprehension_lambda_immediate_call():
    @ct.kernel
    def kernel(y):
        x = 123  # noqa: F841
        results = tuple((lambda: x)() for x in ct.static_iter(range(3)))
        for i, v in ct.static_iter(enumerate(results)):
            ct.scatter(y, i, v)

    y = torch.zeros(3, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [0, 1, 2]


def test_tuple_comprehension_lambda_stored_outer_reassign():
    @ct.kernel
    def kernel(y):
        fns = tuple(lambda: x for x in ct.static_iter(range(3)))
        x = 123  # noqa: F841
        for k, f in ct.static_iter(enumerate(fns)):
            ct.scatter(y, k, f())

    y = torch.zeros(3, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [2, 2, 2]


def test_tuple_comprehension_outer_iter_from_scope():
    @ct.kernel
    def kernel(x, y):
        n = 2
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        result = tuple(tiles[i] * n
                       for i in ct.static_iter(range(n))
                       for n in ct.static_iter(range(3)))
        for k, t in ct.static_iter(enumerate(result)):
            ct.store(y, (k,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    y = torch.zeros((6 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))

    n = 2
    ref = torch.cat([x[i * 16:(i + 1) * 16] * n for i in range(n) for n in range(3)])
    assert_equal(y, ref)


def test_tuple_comprehension_inner_iter_from_scope():
    @ct.kernel
    def kernel(x, y):
        n = 3
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(n)))
        result = tuple(tiles[i] * j
                       for i in ct.static_iter(range(2))
                       for j in ct.static_iter(range(n)))
        for k, t in ct.static_iter(enumerate(result)):
            ct.store(y, (k,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    a, b = x[:16], x[16:32]
    ref = torch.cat([a * 0, a * 1, a * 2, b * 0, b * 1, b * 2])

    y = torch.zeros(6 * 16, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_inner_iter_uses_outer_induction_var():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        result = tuple(tiles[n] * (j + 1)
                       for n in ct.static_iter(range(3))
                       for j in ct.static_iter(range(n)))
        for k, t in ct.static_iter(enumerate(result)):
            ct.store(y, (k,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    b, c = x[16:32], x[32:]
    # n=0: range(0) → nothing
    # n=1: j=0 → tiles[1]*(0+1) = b
    # n=2: j=0 → tiles[2]*(0+1) = c; j=1 → tiles[2]*(1+1) = c*2
    ref = torch.cat([b, c, c * 2])

    y = torch.zeros(3 * 16, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_inner_iter_uses_outside_variable():
    # n=10 is a kernel-local variable, but `for n in ...` makes n local to the
    # entire genexp scope.  range(n) in the second generator therefore cannot see
    # the kernel's n — same as Python's UnboundLocalError in this situation.
    @ct.kernel
    def kernel():
        n = 10
        _ = tuple(i + j
                  for i in ct.static_iter(range(3))
                  for j in ct.static_iter(range(n))
                  for n in ct.static_iter(range(4)))

    with pytest.raises(TileSyntaxError, match="Undefined variable n"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_comprehension_first_iter_uses_own_induction_var():
    # Python: `n = 4; tuple(n for n in range(n))` = (0, 1, 2, 3).
    # The first iterable range(n) uses the outer n=4; the element n is the induction var.
    n = 4

    @ct.kernel
    def kernel(y):
        ns = tuple(n for n in ct.static_iter(range(n)))
        for i, v in ct.static_iter(enumerate(ns)):
            ct.scatter(y, i, v)

    y = torch.zeros(n, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == list(range(n))


def test_tuple_comprehension_second_iter_uses_own_induction_var():
    # Python: `n = 4; tuple(1 for i in range(3) for n in range(n))`
    @ct.kernel
    def kernel():
        n = 4
        _ = tuple(i for i in ct.static_iter(range(3))
                  for n in ct.static_iter(range(n)))

    with pytest.raises(TileSyntaxError, match="Undefined variable n"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_comprehension_inner_iter_uses_later_induction_var():
    @ct.kernel
    def kernel():
        _ = tuple(i + j
                  for i in ct.static_iter(range(3))
                  for j in ct.static_iter(range(n))  # noqa: F821
                  for n in ct.static_iter(range(4)))

    with pytest.raises(TileSyntaxError, match="Undefined variable n"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_comprehension_if_uses_later_induction_var():
    # The `if` condition references n which is only introduced by a later generator.
    @ct.kernel
    def kernel():
        _ = tuple(i
                  for i in ct.static_iter(range(3))
                  if n > 0  # noqa: F821
                  for n in ct.static_iter(range(4)))

    with pytest.raises(TileSyntaxError, match="Undefined variable n"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())


def test_tuple_comprehension_duplicate_induction_var():
    @ct.kernel
    def kernel(y):
        result = tuple(n for n in ct.static_iter(range(3))
                       for n in ct.static_iter(range(4)))
        for i, v in ct.static_iter(enumerate(result)):
            ct.scatter(y, i, v)

    y = torch.zeros(12, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [0, 1, 2, 3] * 3


def test_tuple_comprehension_with_if():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        evens = tuple(t for i, t in ct.static_iter(enumerate(tiles)) if i % 2 == 0)
        for i, t in ct.static_iter(enumerate(evens)):
            ct.store(y, (i,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    ref = torch.cat([x[:16], x[32:]])

    y = torch.zeros((2 * 16,), dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_with_multiple_ifs():
    @ct.kernel
    def kernel(x, y):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(4)))
        middle = tuple(t for i, t in ct.static_iter(enumerate(tiles)) if i > 0 if i < 3)
        for i, t in ct.static_iter(enumerate(middle)):
            ct.store(y, (i,), t)

    x = torch.arange(4 * 16, dtype=torch.int32, device="cuda")
    ref = torch.cat([x[16:32], x[32:48]])

    y = torch.zeros(2 * 16, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y))
    assert_equal(y, ref)


def test_tuple_comprehension_ifelse_in_element():
    @ct.kernel
    def kernel(y):
        result = tuple(i if i % 2 == 0 else -i for i in ct.static_iter(range(6)))
        for k, v in ct.static_iter(enumerate(result)):
            ct.scatter(y, k, v)

    y = torch.zeros(6, dtype=torch.int32, device="cuda")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (y,))
    assert y.tolist() == [0, -1, 2, -3, 4, -5]


def test_tuple_comprehension_dynamic_if():
    @ct.kernel
    def kernel(x, y, cond: bool):
        tiles = tuple(ct.load(x, (i,), (16,)) for i in ct.static_iter(range(3)))
        result = tuple(t for i, t in ct.static_iter(enumerate(tiles)) if cond)
        for i, t in ct.static_iter(enumerate(result)):
            ct.store(y, (i,), t)

    x = torch.arange(3 * 16, dtype=torch.int32, device="cuda")
    y = torch.zeros((3 * 16,), dtype=torch.int32, device="cuda")
    with pytest.raises(TileTypeError,
                       match="Tuple comprehension if-conditions must be statically known"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, y, True))


def test_tuple_comprehension_walrus_operator_in_if():
    # Walrus operator `:=` requires support ast.NamedExpr (`:=`).
    # TODO: support ast.NamedExpr.
    @ct.kernel
    def kernel():
        _ = tuple(y
                  for x in ct.static_iter(range(5))
                  if (y := x * 2) > 3)  # noqa: F841

    with pytest.raises(TileSyntaxError, match="Unsupported syntax"):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, ())
