# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TypeAlias, Any
from types import FunctionType
from typing import TYPE_CHECKING

from cuda.lang._ir import ir
from cuda.tile import _cext
from cuda.tile._cext import launch_extended
from cuda.tile._execution import function, stub

if TYPE_CHECKING:
    from cuda.lang.compilation import KernelSignature


__all__ = [
    "function",
    "kernel",
    "launch",
    "stub",
]


Dim3: TypeAlias = tuple[int] | tuple[int, int] | tuple[int, int, int]


def launch(
    stream,
    block_count: Dim3,
    thread_count: Dim3,
    kernel,
    kernel_args: tuple[Any, ...],
    /,
    *,
    cooperative: bool = False,
    block_in_cluster_count: Dim3 | None = None,
    preferred_block_in_cluster_count: Dim3 | None = None,
    programmatic_dependent_launch: bool = False,
):
    """Launch a cuda.lang kernel.

    Args:
        stream: Stream-like object, such as ``torch.cuda.current_stream()``.
            Streams from cuda.bindings, numba, and raw pointers are also
            supported.
        block_count (Dim3):
        thread_count (Dim3):
        kernel: Kernel to be launched, decorated with ``cl.kernel``
        kernel_args (tuple[Any, ...]):
        cooperative (bool):
        block_in_cluster_count (Dim3 | None):
        preferred_block_in_cluster_count (Dim3 | None):
        programmatic_dependent_launch (bool):
    """
    launch_extended(
        stream,
        block_count,
        thread_count,
        kernel,
        kernel_args,
        cooperative=cooperative,
        block_in_cluster_count=block_in_cluster_count,
        preferred_block_in_cluster_count=preferred_block_in_cluster_count,
        programmatic_dependent_launch=programmatic_dependent_launch,
    )


class kernel(_cext.TileDispatcher):
    """A |kernel| is a function executed by each |thread| in each |block| in a |grid|.

    Examples:

        .. testcode::
            :template: setup_only.py

            @cl.kernel
            def kernel():
                print("Hello!")

            cl.launch(stream, (1,), (3,), kernel, ())

        .. testoutput::

            Hello!
            Hello!
            Hello!

    """

    def __new__(cls, function=None, /, **kwargs):
        if function is None:

            def decorate(func):
                return kernel(func, **kwargs)

            return decorate

        return super().__new__(cls, function, **kwargs)

    def __init__(
        self,
        function=None,
        /,
        *,
        opt_level: None | int = 3,
        arch: str | None = None,
        gpu_name: str | None = None,
    ):
        if not isinstance(function, FunctionType):
            raise TypeError("`kernel` decorator must be applied to a Python function")

        from cuda.tile._compiler_options import CompilerOptions
        from cuda.tile._annotated_function import get_annotated_function

        ann_func = get_annotated_function(function)
        compiler_options = CompilerOptions(opt_level=opt_level)
        super().__init__(ann_func.parameter_annotations)
        self._annotated_function = ann_func
        self._compiler_options = compiler_options
        self._arch = arch
        self._gpu_name = gpu_name

    def _compile(self, signature: KernelSignature, ctx: ir.IRContext):
        from cuda.lang._compile import compile_simt

        result = compile_simt(
            self._annotated_function,
            (signature,),
            arch=self._arch,
            gpu_name=self._gpu_name,
            compiler_options=self._compiler_options,
            ctx=None,  # the launcher currently provides a cutile context
        )
        [kernel_sig] = result.kernel_signatures
        return (
            result.cubin,
            kernel_sig.symbol,
            result.dyn_smem_size_program,
            result.hoisted_tensor_maps,
        )

    @property
    def _pyfunc(self):
        return self._annotated_function.pyfunc

    def __call__(self, *args, **kwargs):
        raise TypeError(
            "kernels cannot be called directly. Use cuda.lang.launch() instead."
        )
