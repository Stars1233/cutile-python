# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._mlir as mlir


def test_int_float_printing():
    assert str(mlir.Float32Type()) == "f32"
    assert str(mlir.IntegerType.signless(32)) == "i32"
    assert str(mlir.IntegerType.signed(32)) == "si32"
    assert str(mlir.IntegerType.unsigned(32)) == "ui32"
