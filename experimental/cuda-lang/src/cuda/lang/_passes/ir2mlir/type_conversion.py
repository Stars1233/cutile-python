# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch, partial
from typing import Any

import cuda.lang._ir.type as ir_type
import cuda.lang._mlir as mlir
import cuda.lang._datatype as dtype
from cuda.lang._exception import TileInternalError


@singledispatch
def ir_type_to_mlir_type(ir_type: Any) -> mlir.Type:
    raise NotImplementedError(f"Unable to convert {ir_type=} to MLIR type")


@ir_type_to_mlir_type.register
def pointer_type_to_mlir_type(src_type: ir_type.PointerTy) -> mlir.Type:
    return mlir.llvm.LLVMPointerType(addressSpace=int(src_type.memory_space.value))


@ir_type_to_mlir_type.register
def opaque_pointer_type_to_mlir_type(src_type: ir_type.OpaquePointerTy) -> mlir.Type:
    return mlir.llvm.LLVMPointerType(addressSpace=int(src_type.memory_space.value))


@ir_type_to_mlir_type.register
def tile_type_to_mlir_type(ir_type: ir_type.TileTy) -> mlir.Type:
    if ir_type.shape != ():
        raise NotImplementedError(f"Unable to convert {ir_type=} to MLIR type")
    return ir_type_to_mlir_type(ir_type.dtype)


@ir_type_to_mlir_type.register
def basic_scalar_type_to_mlir_type(src_type: dtype.DType) -> mlir.Type:
    match src_type:
        case (
            dtype.int8
            | dtype.int16
            | dtype.int32
            | dtype.int64
            | dtype.bool_
            | dtype.uint8
            | dtype.uint16
            | dtype.uint32
            | dtype.uint64
        ):
            return mlir.IntegerType(
                width=src_type.bitwidth, signedness=mlir.SignednessSemantics.SIGNLESS
            )
        case dtype.float16:
            return mlir.Float16Type()
        case dtype.bfloat16:
            return mlir.BFloat16Type()
        case dtype.float32:
            return mlir.Float32Type()
        case dtype.float64:
            return mlir.Float64Type()
        case _:
            raise NotImplementedError(f"Unable to convert {src_type=} to MLIR type")


@ir_type_to_mlir_type.register
def vector_type_to_mlir_type(src_type: ir_type.VectorTy) -> mlir.Type:
    element_type = ir_type_to_mlir_type(src_type.dtype)
    return mlir.VectorType(
        shape=src_type.shape,
        elementType=element_type,
        scalableDims=(False,) * len(src_type.shape),
    )


@singledispatch
def mlir_constant_of_type(mlir_type: mlir.Type, value) -> mlir.Value:
    raise NotImplementedError(f"Unable to convert {value=} to MLIR type {mlir_type=}")


@mlir_constant_of_type.register
def scalar_to_vector_constant(mlir_type: mlir.VectorType, value) -> mlir.Value:
    if any(mlir_type.scalableDims):
        raise NotImplementedError('Scalable vectors')
    if isinstance(mlir_type.elementType, mlir.FloatType):
        value = float(value)
    elif isinstance(mlir_type.elementType, mlir.IntegerType | mlir.IndexType):
        value = int(value)
    else:
        raise NotImplementedError(
            f"MLIR vector constant of element type {mlir_type.elementType}"
        )
    value_attr = mlir.DenseTypedElementsAttr(type=mlir_type, rawData=value)
    res = mlir.llvm.add_ConstantOp(
        res_type=mlir_type,
        value=value_attr
    )
    return res


@mlir_constant_of_type.register
def float_to_mlir_constant(mlir_type: mlir.FloatType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.FloatAttr(type=mlir_type, value=mlir.APFloat(float(value))),
    )


@mlir_constant_of_type.register
def int_to_mlir_constant(mlir_type: mlir.IntegerType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.IntegerAttr(
            type=mlir_type, value=mlir.APInt(int(value), mlir_type.width)
        ),
    )


@mlir_constant_of_type.register
def int_to_mlir_index(mlir_type: mlir.IndexType, value) -> mlir.Value:
    return mlir.arith.add_ConstantOp(
        value=mlir.IntegerAttr(
            type=mlir_type, value=mlir.APInt(int(value), mlir_type.width)
        ),
    )


def _get_type_conversion_encoder(
    from_dtype: dtype.ArithmeticDType, to_dtype: dtype.ArithmeticDType
):

    if from_dtype == to_dtype:
        return lambda x: x

    to_mlir_type = ir_type_to_mlir_type(to_dtype)

    def kind(t):
        if dtype.is_float(t):
            return "f"
        if dtype.is_integral(t) or dtype.is_boolean(t):
            return "si" if dtype.is_signed(t) else "ui"
        raise TileInternalError(f"Unsupported dtype: {t}")

    from_kind, to_kind = kind(from_dtype), kind(to_dtype)
    lhs_width = from_dtype.bitwidth
    rhs_width = to_dtype.bitwidth

    # TODO: rounding modes
    match from_kind, to_kind:
        case "f", "f":
            if lhs_width < rhs_width:
                return partial(mlir.arith.add_ExtFOp, out_type=to_mlir_type)
            else:
                return partial(mlir.arith.add_TruncFOp, out_type=to_mlir_type)
        case "f", "si":
            return partial(mlir.arith.add_FPToSIOp, out_type=to_mlir_type)
        case "f", "ui":
            return partial(mlir.arith.add_FPToUIOp, out_type=to_mlir_type)
        case "si", "f":
            return partial(mlir.arith.add_SIToFPOp, out_type=to_mlir_type)
        case "ui", "f":
            return partial(mlir.arith.add_UIToFPOp, out_type=to_mlir_type)

    if from_dtype.bitwidth < to_dtype.bitwidth:
        assert from_kind in ("si", "ui")
        if dtype.is_signed(from_dtype):
            return partial(mlir.arith.add_ExtSIOp, out_type=to_mlir_type)
        else:
            return partial(mlir.arith.add_ExtUIOp, out_type=to_mlir_type)
    elif from_dtype.bitwidth > to_dtype.bitwidth:
        return partial(mlir.arith.add_TruncIOp, out_type=to_mlir_type)
    elif from_kind in ("si", "ui") and to_kind in ("si", "ui"):
        return lambda in_: in_
    raise NotImplementedError(
        f"Conversion from {from_dtype} to {to_dtype} not implemented"
    )


def convert_dtype(
    src_type: dtype.DType, dst_type: dtype.DType, value: mlir.Value
) -> mlir.Value:
    encoder = _get_type_conversion_encoder(src_type, dst_type)
    return encoder(in_=value)


def mlir_integer_cast(
    int_value: mlir.Value, to_type: mlir.IntegerType, signed: bool
) -> mlir.Value:
    if int_value.type == to_type:
        return int_value

    if isinstance(int_value.type, mlir.IndexType) or isinstance(
        to_type, mlir.IndexType
    ):
        return mlir.arith.add_IndexCastOp(out_type=to_type, in_=int_value)

    src_bw = int_value.type.width
    dst_bw = to_type.width

    if src_bw < dst_bw:
        if signed:
            return mlir.arith.add_ExtSIOp(out_type=to_type, in_=int_value)
        else:
            return mlir.arith.add_ExtUIOp(out_type=to_type, in_=int_value)
    else:
        return mlir.arith.add_TruncIOp(out_type=to_type, in_=int_value)


__all__ = (
    "ir_type_to_mlir_type",
    "mlir_constant_of_type",
    "mlir_integer_cast",
    "convert_dtype",
)
