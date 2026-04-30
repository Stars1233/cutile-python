# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence, Any

from cuda.lang._execution import kernel
from cuda.tile._cext import CallingConvention
from cuda.tile.compilation._name_mangling import mangle_kernel_name
from cuda.tile.compilation import (
    KernelSignature as TileKernelSignature,
    ParameterConstraint,
    ScalarConstraint,
    ArrayConstraint,
    ListConstraint,
    ConstantConstraint,
)


class KernelSignature(TileKernelSignature):
    def __init__(
        self,
        parameters: Sequence[ParameterConstraint | bool | int | float],
        symbol: str | None = None,
    ):

        # TODO: Is this meaningful for SIMT programs?
        calling_convention = CallingConvention.cutile_python_v1()
        super().__init__(parameters, calling_convention, symbol)

    def with_mangled_symbol(self, function_name: str) -> "KernelSignature":
        symbol = mangle_kernel_name(function_name, self)
        return self.with_symbol(symbol)

    def with_symbol(self, symbol: str) -> "KernelSignature":
        return self.__class__(self.parameters, symbol)

    @staticmethod
    def from_kernel_args(
        kernel: kernel, kernel_args: Sequence[Any], *, symbol: str | None = None
    ) -> "KernelSignature":
        calling_convention = CallingConvention.cutile_python_v1()
        return TileKernelSignature.from_kernel_args(
            kernel,
            kernel_args,
            calling_convention,
            symbol=symbol,
        )


__all__ = (
    "KernelSignature",
    "ArrayConstraint",
    "ScalarConstraint",
    "ListConstraint",
    "ConstantConstraint",
)
