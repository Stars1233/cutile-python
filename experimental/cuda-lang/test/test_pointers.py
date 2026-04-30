# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch
from cuda.lang._exception import TileError

from .util import make_symbolic_tensor, make_symbolic_scalar, compile_for_arguments


def test_pointer_ldst_offset():
    @cl.kernel
    def kernel(A):
        a_ptr = A.get_base_pointer()
        a_element = a_ptr.load_offset(0)
        a_ptr.store_offset(0, a_element + 1)

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A[0, 0] == 1


def test_pointer_gep():
    @cl.kernel
    def kernel(A):
        A.get_element_pointer((0, 0)).store(1)
        A.get_element_pointer((1, 1)).store(2)
        A.get_element_pointer((2, 2)).store(3)

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [[1, 0, 0], [0, 2, 0], [0, 0, 3]]


def test_ptr_roundtrip():
    @cl.kernel
    def kernel(A):
        tx, ty, tz = cl.thread_idx()
        B = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        smem = B.get_base_pointer()
        B2 = cl.reinterpret_pointer_as_array(smem, cl.int32, 1)
        B2[0] = 1
        A[0] = B[0, 0]
        B2[0] = 2
        A[1] = B[0, 0]

    A = torch.zeros(2, dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    assert A.cpu().tolist() == [1, 2]


def test_pointer_smem():
    @cl.kernel
    def kernel(A):
        B = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        B.get_element_pointer((0, 0)).store(1)
        A[0, 0] = B[0, 0]

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_pointer_ldst():
    @cl.kernel
    def kernel(A):
        a_ptr = A.get_base_pointer()
        a_element = a_ptr.load()
        a_ptr.store(a_element + 1)

    A = torch.zeros(3, 3, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A[0, 0] == 1


@pytest.mark.parametrize(
    "allocator,expected_memspace",
    [
        (cl.local_array, [1, 0, 0]),
        (cl.shared_array, [0, 0, 1]),
    ],
)
def test_device_alloc_memspace(allocator, expected_memspace):
    @cl.kernel
    def kernel(memspace):
        A = allocator(shape=(3, 3), dtype=cl.int32)
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        if cl.thread_idx()[0] == 0:
            memspace[0] = cl.int32(cl.nvvm.isspacep_local(p))
            memspace[1] = cl.int32(cl.nvvm.isspacep_global(p))
            memspace[2] = cl.int32(cl.nvvm.isspacep_shared(p))
        cl.syncthreads()

    memspace = torch.zeros(3, dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (memspace,),
    )
    assert memspace.cpu().tolist() == expected_memspace


@pytest.mark.parametrize("allocator", [cl.local_array, cl.shared_array])
@pytest.mark.parametrize(
    "torch_dtype,cl_dtype",
    [
        (torch.int32, cl.int32),
        (torch.float32, cl.float32),
        (torch.int64, cl.int64),
        (torch.float64, cl.float64),
    ],
)
def test_device_allocations(allocator, torch_dtype, cl_dtype):

    @cl.kernel
    def kernel(out, memspace):
        A = allocator(shape=(3, 3), dtype=cl_dtype)
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        A[0, 0] = cl_dtype(1)
        A[1, 1] = cl_dtype(2)
        A[2, 2] = cl_dtype(3)
        out[0, 0] = A[0, 0]
        out[1, 1] = A[1, 1]
        out[2, 2] = A[2, 2]
        cl.syncthreads()

        memspace[0] = cl.int32(cl.nvvm.isspacep_local(p))
        memspace[1] = cl.int32(cl.nvvm.isspacep_global(p))
        memspace[2] = cl.int32(cl.nvvm.isspacep_shared(p))

    A = torch.zeros(3, 3, dtype=torch_dtype).cuda()
    memspace = torch.zeros(1, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A, memspace),
    )
    A = A.cpu()
    assert A[0, 0] == 1
    assert A[1, 1] == 2
    assert A[2, 2] == 3


@pytest.mark.parametrize("allocator", [cl.local_array, cl.shared_array])
def test_allocate_in_runtime_conditional(allocator):
    def kernel(tensor, cond):
        if tensor[0]:
            allocator(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    bool_constraint = make_symbolic_scalar(dtype=cl.bool_)
    with pytest.raises(TileError, match="Memory allocated in dynamic control flow"):
        compile_for_arguments(kernel, (tensor_constraint, bool_constraint))


@pytest.mark.parametrize("allocator", [cl.local_array, cl.shared_array])
def test_allocate_in_runtime_loop(allocator):
    def kernel(tensor, cond):
        for i in range(cl.int32(tensor[0])):
            allocator(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    bool_constraint = make_symbolic_scalar(dtype=cl.bool_)
    with pytest.raises(TileError, match="Memory allocated in dynamic control flow"):
        compile_for_arguments(kernel, (tensor_constraint, bool_constraint))
