# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Literal

from cuda.lang._ir import ir
from cuda.lang._ir.ops import AllocDynSharedMemory, get_dyn_shared_memory_base_ptr, \
    _pointer_with_offset, _reinterpret_pointer
from cuda.lang._datatype import int32
from cuda.lang._exception import TileTypeError
from cuda.tile._ir.ops import raw_binary_arithmetic, assign, \
    loosely_typed_const
from cuda.tile._ir.type import TileTy


SizeOpcode = Literal["Const", "KernelArgI32", "Mul", "Add"]


@dataclass
class SizeProgram:
    """
    Bytecode for a simple stack machine that computes the required dynamic shared memory size.

    This bytecode has two targets:
        1) The host-side launch code that needs to compute the "sharedMemBytes" argument
           of cuLaunchKernel() from the kernel arguments. (See size_program_eval in tile_kernel.cpp)
        2) The device-size bump allocator code that needs to partition the dynamic shared memory
           region into multiple arrays. (See _size_program_to_ir below)
    """
    opcodes: list[SizeOpcode]
    op_attrs: list[int]  # attributes for "Const" and "KernelArg" opcodes

    def extend(self, other: "SizeProgram"):
        self.opcodes.extend(other.opcodes)
        self.op_attrs.extend(other.op_attrs)


def handle_dynamic_shared_memory(kernel_body: ir.Block) -> SizeProgram | None:
    alloc_ops = [op for op in kernel_body.traverse() if isinstance(op, AllocDynSharedMemory)]
    if len(alloc_ops) == 0:
        return None

    # Sort allocations by decreasing alignment so that we avoid padding
    alloc_ops.sort(key=_get_alignment, reverse=True)

    # Per each array, build a SizeProgram that computes its total size in bytes
    kernel_param_names = tuple(v.name for v in kernel_body.params)
    array_programs = tuple(_build_size_program(op, kernel_param_names) for op in alloc_ops)

    # Bump-allocate array pointers
    array_pointers = []
    with ir.TileBuilder(kernel_body.ctx, kernel_body.loc) as builder:
        ptr = get_dyn_shared_memory_base_ptr()
        array_pointers.append(ptr)
        for prev_op, prev_prog in zip(alloc_ops[:-1], array_programs[:-1], strict=True):
            prev_arr_size = _size_program_to_ir(prev_prog, kernel_body.params)
            ptr = _pointer_with_offset(ptr, prev_arr_size)
            array_pointers.append(ptr)

        for op, ptr in zip(alloc_ops, array_pointers, strict=True):
            ptr = _reinterpret_pointer(ptr, op.result_var.get_type())
            assign(ptr, op.result_var)

    # Remove AllocDynSharedMemory operations
    removed_count = _remove_alloc_ops(kernel_body)
    assert removed_count == len(alloc_ops)

    # Prepend the newly generated code
    kernel_body[:0] = builder.ops

    # Make a final program that computes the total shared memory size
    total_program = SizeProgram([], [])
    for i, array_prog in enumerate(array_programs):
        total_program.extend(array_prog)
        if i > 0:
            total_program.opcodes.append("Add")

    return total_program


def _size_program_to_ir(program: SizeProgram, kernel_params: tuple[ir.Var, ...]) -> ir.Var:
    attrs = iter(program.op_attrs)
    stack: list[ir.Var] = []
    for opcode in program.opcodes:
        match opcode:
            case "Const": stack.append(loosely_typed_const(next(attrs)))
            case "KernelArgI32": stack.append(kernel_params[next(attrs)])
            case "Mul":
                b = stack.pop()
                stack[-1] = raw_binary_arithmetic("mul", stack[-1], b)
            case "Add":
                b = stack.pop()
                stack[-1] = raw_binary_arithmetic("add", stack[-1], b)
            case _:
                assert False
    assert next(attrs, None) is None
    assert len(stack) == 1
    return stack[0]


def _remove_alloc_ops(block: ir.Block):
    to_remove = {i for i, op in enumerate(block)
                 if isinstance(op, AllocDynSharedMemory)}
    remove_count = len(to_remove)
    if remove_count > 0:
        new_ops = [block[i] for i in range(len(block)) if i not in to_remove]
        block[:] = new_ops

    for op in block:
        for nb in op.nested_blocks:
            remove_count += _remove_alloc_ops(nb)

    return remove_count


def _get_alignment(alloc_op: AllocDynSharedMemory) -> int:
    return _get_item_size(alloc_op)


def _get_item_size(alloc_op: AllocDynSharedMemory) -> int:
    pointer_tile_ty = alloc_op.result_var.get_type()
    assert isinstance(pointer_tile_ty, TileTy)
    poinee_ty = pointer_tile_ty.dtype.pointee_type
    assert isinstance(poinee_ty, TileTy)
    assert poinee_ty.shape == ()
    assert poinee_ty.dtype.bitwidth % 8 == 0
    return poinee_ty.dtype.bitwidth // 8


def _build_size_program(alloc_op: AllocDynSharedMemory,
                        kernel_param_names: tuple[str, ...]) -> SizeProgram:
    program = SizeProgram([], [])
    constant_factor = _get_item_size(alloc_op)
    kernel_params = []
    for size_var in alloc_op.shape:
        if size_var.is_constant():
            constant_factor *= size_var.get_constant()
        elif size_var.name in kernel_param_names:
            if size_var.get_type() != TileTy(int32, ()):
                raise TileTypeError(f"Kernel parameter used as shared array size must be int32,"
                                    f" got {size_var.get_type()}",
                                    loc=size_var.loc)
            kernel_params.append(kernel_param_names.index(size_var.name))
        else:
            raise TileTypeError("Size of shared array must be either a constant"
                                " or a kernel parameter", loc=size_var.loc)

    first_factor = True
    if constant_factor != 1:
        program.opcodes.append("Const")
        program.op_attrs.append(constant_factor)
        first_factor = False

    for param_idx in kernel_params:
        program.opcodes.append("KernelArgI32")
        program.op_attrs.append(param_idx)
        if first_factor:
            first_factor = False
        else:
            program.opcodes.append("Mul")

    return program
