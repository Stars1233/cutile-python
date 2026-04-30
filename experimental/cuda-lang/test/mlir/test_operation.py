# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir


def test_addf_printing():
    f32t = mlir.Float32Type()
    lhs, rhs = mlir.Value(f32t), mlir.Value(f32t)
    lhs.value_id = "x"
    rhs.value_id = "y"
    with mlir.Block().append_here() as block:
        mlir.arith.add_AddFOp(lhs=lhs, rhs=rhs)
    op = block[-1]
    expected = (
        '%0 = "arith.addf"(%x, %y)'
        " <{fastmath = #arith<fastmath <none>>}> : (f32, f32) -> f32"
    )
    assert str(op) == expected


def test_cond_br_printing():
    i1t = mlir.IntegerType.signless(1)
    f32t = mlir.Float32Type()
    cond = mlir.Value(i1t)
    cond.value_id = "c"
    x = mlir.Value(f32t)
    x.value_id = "x"
    y = mlir.Value(f32t)
    y.value_id = "y"
    true_label = mlir.BlockLabel("foo")
    false_label = mlir.BlockLabel("bar")
    with mlir.Block().append_here() as block:
        mlir.cf.add_CondBranchOp(
            condition=cond,
            trueDestOperands=[x, y],
            falseDestOperands=[x],
            trueDest=true_label,
            falseDest=false_label,
        )
    op = block[-1]
    expected = (
        '"cf.cond_br"(%c, %x, %y, %x) [^foo, ^bar]'
        " <{operandSegmentSizes = array<i32: 1, 2, 1>}> : (i1, f32, f32, f32) -> ()"
    )
    assert str(op) == expected
