# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import Sequence
from dataclasses import dataclass
import os.path
from types import FunctionType
import subprocess

from cuda.tile._passes.hir2ir import hir2ir
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
from cuda.tile._compile import _create_kernel_parameters, get_sm_arch
from cuda.tile._annotated_function import AnnotatedFunction, get_annotated_function
from cuda.tile._cext import get_compute_capability
from cuda.tile._compiler_options import CompilerOptions
from cuda.tile._exception import TileCompilerExecutionError
from cuda.lang._logging import get_log_flags
from cuda.lang._ir import ir, hir
from cuda.lang._ir.type import MemorySpace
from cuda.lang._passes.ast2hir import get_function_hir
from cuda.lang._passes.ir2mlir import ir2mlir
from cuda.lang._passes.flatten_cfg import flatten_cfg
from cuda.lang._passes.simt_semantics import simt_semantic_analysis
from cuda.lang._passes.canonicalize_parameters import canonicalize_parameters
from cuda.lang._passes.handle_dyn_shared_mem import handle_dynamic_shared_memory
from cuda.lang._passes.hoist_tensor_map import hoist_tensor_maps, HoistedTensorMap
from cuda.lang.compilation import (
    KernelSignature,
    ParameterConstraint,
    ScalarConstraint,
    ArrayConstraint,
    ListConstraint,
    ConstantConstraint,
)
from ._execution import kernel
from cuda.lang._ir.ops import cuda_lang_impl_registry
from ._ir._host_program import HostProgram, get_host_programs_by_var


def mlir2cubin(mlir_text: str, gpu_name: str, arch: str) -> bytes:
    executable = get_compiler_binary_path()
    argv = [executable, "-", "-o", "-", f"--gpu-name={gpu_name}", f"--arch={arch}"]
    custom_flags = os.environ.get("CUDA_LANG_MLIR2CUBIN_FLAGS", None)
    if custom_flags is not None:
        argv.extend(custom_flags.split())

    log_flags = get_log_flags()
    if log_flags.log_ptx:
        argv.extend(['--dump-ptx'])

    try:
        completed = subprocess.run(
            argv, input=mlir_text.encode(), capture_output=True, check=True
        )
    except subprocess.CalledProcessError as e:
        raise TileCompilerExecutionError(
            return_code=e.returncode,
            stderr=e.stderr.decode(),
            compiler_flags=argv,
            compiler_version=None,
        )
    if custom_flags is not None or log_flags.log_ptx:
        compiler_stderr = completed.stderr.decode()
        if len(compiler_stderr) > 0:
            print("==== mlir2cubin stderr: ====", file=sys.stderr)
            print(compiler_stderr, file=sys.stderr)
            print("^^^^ End of mlir2cubin stderr ^^^^", file=sys.stderr)
    return completed.stdout


def get_compiler_binary_path() -> str:
    binary_name = "mlir2cubin"
    if os.name == "nt":
        binary_name += ".exe"
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "bin", binary_name)


@dataclass
class CompilationResult:
    kernel_signatures: Sequence[KernelSignature]
    dyn_smem_size_program: HostProgram | None
    hoisted_tensor_maps: list[HoistedTensorMap]
    final_ir: ir.Region | None = None
    mlir: str | None = None
    cubin: bytes | None = None


def get_function_ir(
    function: hir.Function,
    signature: KernelSignature,
    ctx: ir.IRContext,
    constant_mask: Sequence[bool] | None = None,
) -> ir.Block:
    if constant_mask is None:
        constant_mask = [False] * len(signature.parameters)
    parameter_names = function.signature.parameters.keys()
    with ir.TileBuilder(ctx, function.body.loc) as builder:
        params = _create_kernel_parameters(
            signature.parameters,
            constant_mask,
            parameter_names,
            function.param_locs,
            ctx,
            array_memory_space=MemorySpace.GENERIC
        )
        canonicalize_parameters(params, builder)
        with cuda_lang_impl_registry.as_current():
            hir2ir(function, params.aggregate_vars, ctx)
    func_body = ctx.make_block("entry", function.body.loc)
    func_body.params = sum((vars for vars, _ in params.nonconstant_flat_vars), ())
    func_body.extend(builder.ops)
    return func_body


def _transform_ir(func_ir: ir.Block, ctx: ir.IRContext) \
        -> tuple[HostProgram | None, list[HoistedTensorMap]]:
    simt_semantic_analysis(func_ir, ctx)

    host_program_by_var = get_host_programs_by_var(func_ir)
    dyn_smem_size_program = handle_dynamic_shared_memory(func_ir, host_program_by_var)
    hoisted_tensor_maps = hoist_tensor_maps(func_ir, host_program_by_var)

    eliminate_assign_ops(func_ir)
    dead_code_elimination_pass(func_ir)

    return dyn_smem_size_program, hoisted_tensor_maps


def compile_simt(
    function: AnnotatedFunction | FunctionType,
    signatures: Sequence[KernelSignature],
    gpu_name: str | None = None,
    arch: str | None = None,
    compiler_options: CompilerOptions = CompilerOptions(),
    ctx: ir.IRContext | None = None,
) -> CompilationResult:
    match function:
        case FunctionType():
            function = get_annotated_function(function)
        case kernel():
            function = get_annotated_function(function._pyfunc)

    log_flags = get_log_flags()
    logging_template = (
        '=' * 20 + ' cuda.lang {header} dump: ' + '=' * 20 + '\n' + '{body}' + '\n'
    )

    func_hir = get_function_hir(function.pyfunc, entry_point=True)
    if log_flags.log_hir:
        print(logging_template.format(header='HIR', body=func_hir.body), file=sys.stderr)

    [signature] = signatures
    if signature.symbol is None:
        signature = signature.with_mangled_symbol(function.pyfunc.__name__)

    ctx = ctx or ir.IRContext(log_ir_on_error=log_flags.log_hir or log_flags.log_ir)

    func_ir = get_function_ir(
        func_hir, signature, ctx, function.constant_parameter_mask
    )

    if log_flags.log_ir:
        print(logging_template.format(header='IR (pre-transforms)', body=func_ir), file=sys.stderr)

    dyn_smem_size_program, hoisted_tensor_maps = _transform_ir(func_ir, ctx)

    if log_flags.log_ir:
        print(logging_template.format(header='IR (post-transforms)', body=func_ir), file=sys.stderr)

    flattened_ir = flatten_cfg(func_ir, ctx)

    if log_flags.log_flattened_ir:
        print(logging_template.format(header='FLATIR', body=flattened_ir), file=sys.stderr)

    mlir_module = ir2mlir(signature, flattened_ir, ctx)

    if log_flags.log_mlir:
        print(logging_template.format(header='MLIR', body=mlir_module), file=sys.stderr)

    if gpu_name is None or arch is None:
        major, minor = get_compute_capability()
        gpu_name = gpu_name or f"sm_{major}{minor}"
        arch = arch or f"compute_{major}{minor}"

    cubin = mlir2cubin(str(mlir_module), gpu_name=gpu_name, arch=arch)

    return CompilationResult(
        kernel_signatures=[signature],
        dyn_smem_size_program=dyn_smem_size_program,
        hoisted_tensor_maps=hoisted_tensor_maps,
        final_ir=flattened_ir,
        mlir=str(mlir_module),
        cubin=cubin,
    )


__all__ = (
    "mlir2cubin",
    "get_compiler_binary_path",
    "compile_simt",
    "get_sm_arch",
    "get_function_hir",
    "KernelSignature",
    "ParameterConstraint",
    "ScalarConstraint",
    "ArrayConstraint",
    "ListConstraint",
    "ConstantConstraint",
    "CompilationResult",
)
