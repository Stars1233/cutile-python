# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir
import pytest


def test_integer_attr_printing():
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), 123)) == "123 : i32"
    assert str(mlir.IntegerAttr.make(mlir.IntegerType.signless(32), -5)) == "-5 : i32"


def test_float_attr_printing():
    assert str(mlir.FloatAttr(type=mlir.Float32Type(), value=mlir.APFloat(1.25))) == "1.25 : f32"


@pytest.mark.parametrize(
    "float_type,value,expected",
    [
        (mlir.Float16Type(), float("inf"), "0x7C00 : f16"),
        (mlir.Float16Type(), -float("inf"), "0xFC00 : f16"),
        (mlir.Float16Type(), float("nan"), "0x7E00 : f16"),
        (mlir.BFloat16Type(), float("inf"), "0x7F80 : bf16"),
        (mlir.BFloat16Type(), -float("inf"), "0xFF80 : bf16"),
        (mlir.BFloat16Type(), float("nan"), "0x7FC0 : bf16"),
        (mlir.Float32Type(), float("inf"), "0x7F800000 : f32"),
        (mlir.Float32Type(), -float("inf"), "0xFF800000 : f32"),
        (mlir.Float32Type(), float("nan"), "0x7FC00000 : f32"),
        (mlir.Float64Type(), float("inf"), "0x7FF0000000000000 : f64"),
        (mlir.Float64Type(), -float("inf"), "0xFFF0000000000000 : f64"),
        (mlir.Float64Type(), float("nan"), "0x7FF8000000000000 : f64"),
    ],
)
def test_special_float_attr_printing(float_type, value, expected):
    assert str(mlir.FloatAttr(type=float_type, value=mlir.APFloat(value))) == expected


def test_string_attr_printing():
    assert str(mlir.StringAttr(value="")) == '""'
    assert str(mlir.StringAttr(value="Hello, world")) == '"Hello, world"'
    assert str(mlir.StringAttr(value='x"y\nz\\w')) == '"x\\22y\\0Az\\\\w"'


def test_dense_array_printing():
    assert str(mlir.DenseI32ArrayAttr([10, -20, 0, 30])) == "array<i32: 10, -20, 0, 30>"
    assert str(mlir.DenseI32ArrayAttr([])) == "array<i32>"


def test_type_attr_printing():
    assert str(mlir.TypeAttr(value=mlir.IntegerType.signed(16))) == "si16"
