# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def get_i1_same_shape(ty):
    from . import ShapedType, UnrankedTensorType, IntegerType
    i1_type = IntegerType.signless(1)
    if isinstance(ty, ShapedType):
        return ty.clone_with(None, i1_type)
    elif isinstance(ty, UnrankedTensorType):
        return UnrankedTensorType(elementType=i1_type)
    else:
        return i1_type


def get_val_and_bool_struct_type(ty):
    from . import IntegerType
    from .llvm import LLVMStructType
    i1_type = IntegerType.signless(1)
    return LLVMStructType.make_literal((ty, i1_type))
