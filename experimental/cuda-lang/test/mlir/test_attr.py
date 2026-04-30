# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir


def test_integer_attr_printing():
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 123)) == "123 : i32"
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), -5)) == "-5 : i32"


def test_string_attr_printing():
    assert str(mlir.StringAttr(value="")) == '""'
    assert str(mlir.StringAttr(value="Hello, world")) == '"Hello, world"'
    assert str(mlir.StringAttr(value='x"y\nz\\w')) == '"x\\22y\\0Az\\\\w"'


def test_dense_array_printing():
    assert str(mlir.DenseI32ArrayAttr([10, -20, 0, 30])) == "array<i32: 10, -20, 0, 30>"
    assert str(mlir.DenseI32ArrayAttr([])) == "array<i32>"


def test_type_attr_printing():
    assert str(mlir.TypeAttr(value=mlir.IntegerType.signed(16))) == "si16"
