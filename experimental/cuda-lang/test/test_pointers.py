# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
from cuda.tile import static_iter
import torch
from cuda.lang._compile import compile_simt
from cuda.lang._exception import TileError
from cuda.lang.compilation import KernelSignature

from .util import make_symbolic_tensor, make_symbolic_scalar, compile_for_arguments


@pytest.mark.parametrize("volatile", [True, False])
@pytest.mark.parametrize("element_count", [2, 4, 8])
@pytest.mark.parametrize(
    "torch_dtype,dtype",
    [
        (
            torch.float16,
            cl.float16,
        ),
        (
            torch.float32,
            cl.float32,
        ),
        (
            torch.float64,
            cl.float64,
        ),
        (
            torch.int8,
            cl.int8,
        ),
        (
            torch.int16,
            cl.int16,
        ),
        (
            torch.int32,
            cl.int32,
        ),
        (
            torch.int64,
            cl.int64,
        ),
    ],
)
def test_pointer_vector_ldst(volatile, element_count, torch_dtype, dtype):
    assert (element_count & (element_count - 1)) == 0
    num_bytes = dtype.bitwidth // 8
    alignment = num_bytes * element_count

    @cl.kernel
    def kernel(A):
        sarr = cl.shared_array(element_count, dtype, alignment=alignment)
        with cl.local_array(element_count, dtype, alignment=alignment) as larr:
            for i in static_iter(range(element_count)):
                larr[i] = dtype(i)
            v = larr.get_base_pointer().load(
                count=element_count,
                alignment=alignment,
                volatile=volatile,
            )
        sarr.get_base_pointer().store(
            v,
            alignment=alignment,
            volatile=volatile,
        )
        for i in static_iter(range(element_count)):
            A[i] = sarr[i]

    A = torch.zeros(element_count, dtype=torch_dtype).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    got = A.cpu().tolist()
    expect = list(range(element_count))
    assert got == expect, f"{expect=} {got=}"


def test_vector_apis():
    @cl.kernel
    def kernel(out):
        with cl.local_array(4, cl.int32, alignment=16) as larr:
            p = larr.get_base_pointer()
            vec = p.load(count=4, alignment=16)
            out[0] = cl.int32(vec.dtype == p.dtype)
            out[1] = cl.int32(p.dtype == cl.int32)
            out[2] = cl.int32(p.dtype == larr.dtype)
            out[3] = vec.element_count

    out = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (out,))
    assert out.cpu().tolist() == [1, 1, 1, 4]


def test_pointer_vector_ldst_bool():
    alignment = (cl.bool_.bitwidth // 8) * 4

    @cl.kernel
    def kernel(A):
        with cl.local_array(4, cl.bool_, alignment=alignment) as larr:
            sarr = cl.shared_array(4, cl.bool_, alignment=alignment)
            value = True
            for i in static_iter(range(4)):
                larr[i] = value
                value = not value
            v = larr.get_base_pointer().load(
                count=4,
                alignment=alignment,
                volatile=True,
            )
            sarr.get_base_pointer().store(
                v,
                alignment=alignment,
                volatile=True,
            )
            for i in static_iter(range(4)):
                A[i] = sarr[i]

    A = torch.zeros(4, dtype=torch.bool).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (A,))
    got = A.cpu().tolist()
    expect = [True, False, True, False]
    assert got == expect, f"{expect=} {got=}"


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


def test_pointer_sub_ldst():
    @cl.kernel
    def kernel(A):
        p = A.get_element_pointer(3)
        for i in range(A.shape[0]):
            (p - i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [9, 4, 1, 0]


def test_pointer_add_ldst():
    @cl.kernel
    def kernel(A):
        p = A.get_base_pointer()
        for i in range(A.shape[0]):
            (p + i).store(i * i)

    A = torch.zeros(4, dtype=torch.int32).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    assert A.cpu().tolist() == [0, 1, 4, 9]


def test_device_alloc_memspace():
    @cl.kernel
    def kernel(memspace):
        A = cl.shared_array(shape=(3, 3), dtype=cl.int32)
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        if cl.thread_idx()[0] == 0:
            memspace[0] = cl.int32(cl.nvvm.isspacep_local(p))
            memspace[1] = cl.int32(cl.nvvm.isspacep_global(p))
            memspace[2] = cl.int32(cl.nvvm.isspacep_shared(p))

        with cl.local_array(shape=(3, 3), dtype=cl.int32) as B:
            p = B.get_base_pointer()
            p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
            if cl.thread_idx()[0] == 0:
                memspace[3] = cl.int32(cl.nvvm.isspacep_local(p))
                memspace[4] = cl.int32(cl.nvvm.isspacep_global(p))
                memspace[5] = cl.int32(cl.nvvm.isspacep_shared(p))

    memspace = torch.zeros(6, dtype=torch.int32, device="cuda")
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (memspace,),
    )
    assert memspace.cpu().tolist() == [0, 0, 1, 1, 0, 0]


@pytest.mark.parametrize(
    "torch_dtype,cl_dtype",
    [
        (torch.int32, cl.int32),
        (torch.float32, cl.float32),
        (torch.int64, cl.int64),
        (torch.float64, cl.float64),
    ],
)
def test_static_shared_array(torch_dtype, cl_dtype):

    @cl.kernel
    def kernel(out):
        A = cl.shared_array(shape=(3, 3), dtype=cl_dtype)
        p = A.get_base_pointer()
        p = cl.address_space_cast(p, cl.MemorySpace.GENERIC)
        A[0, 0] = cl_dtype(1)
        A[1, 1] = cl_dtype(2)
        A[2, 2] = cl_dtype(3)
        out[0, 0] = A[0, 0]
        out[1, 1] = A[1, 1]
        out[2, 2] = A[2, 2]
        cl.syncthreads()

    A = torch.zeros(3, 3, dtype=torch_dtype).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
    A = A.cpu()
    assert A[0, 0] == 1
    assert A[1, 1] == 2
    assert A[2, 2] == 3


def test_device_allocation_alignment_lowering():
    @cl.kernel
    def kernel():
        shared = cl.shared_array(shape=(4,), dtype=cl.int32, alignment=128)
        with cl.local_array(shape=(4,), dtype=cl.int32, alignment=16) as local:
            local[0] = cl.int32(1)
            shared[0] = local[0]

    result = compile_simt(
        kernel,
        [KernelSignature(())],
        gpu_name="sm_80",
        arch="compute_80",
    )

    assert "alignment = 16 : i64" in result.mlir
    assert "alignment = 128 : i64" in result.mlir


def make_local_array(shape, dtype, alignment):
    with cl.local_array(shape, dtype, alignment):
        pass


@pytest.mark.parametrize("allocator", [make_local_array, cl.shared_array])
@pytest.mark.parametrize("alignment", [0, -1, 3, True])
def test_device_allocation_invalid_alignment(allocator, alignment):
    def kernel():
        allocator(shape=(1,), dtype=cl.int32, alignment=alignment)

    match = (
        "Expected an integer constant"
        if isinstance(alignment, bool)
        else "alignment must be a positive power of two"
    )
    with pytest.raises(TileError, match=match):
        compile_for_arguments(kernel, ())


@pytest.mark.parametrize("allocator", [make_local_array, cl.shared_array])
def test_device_allocation_alignment_must_be_constant(allocator):
    def kernel(alignment):
        allocator(shape=(1,), dtype=cl.int32, alignment=alignment)

    with pytest.raises(TileError, match="Expected an integer constant"):
        compile_for_arguments(kernel, (make_symbolic_scalar(cl.int32),))


def test_allocate_shmem_in_runtime_conditional():
    def kernel(tensor):
        if tensor[0]:
            cl.shared_array(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    with pytest.raises(TileError, match="Memory allocated in dynamic control flow"):
        compile_for_arguments(kernel, (tensor_constraint,))


def test_allocate_shmem_in_runtime_loop():
    def kernel(tensor, cond):
        for i in range(cl.int32(tensor[0])):
            cl.shared_array(shape=(1,), dtype=cl.int32)

    tensor_constraint = make_symbolic_tensor(shape=(2,), dtype=cl.float32)
    bool_constraint = make_symbolic_scalar(dtype=cl.bool_)
    with pytest.raises(TileError, match="Memory allocated in dynamic control flow"):
        compile_for_arguments(kernel, (tensor_constraint, bool_constraint))
