# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, TypeVar, Generic

from cuda.lang._execution import stub, function
from cuda.tile._stub import (
    Constant,
    Array as TileArray,
    static_eval,
    static_assert,
    static_iter,
)
from cuda.tile._memory_model import MemoryOrder
from cuda.lang._datatype import DType, MemorySpace
from . import nvvm

T = TypeVar("T")


class LocalArrayContextManager(Generic[T]):
    @stub
    def __enter__(self) -> "Array[T]": ...

    @stub
    def __exit__(self, exc_type, exc_val, exc_tb): ...


class Array(TileArray, Generic[T]):
    """
    N-dimensional array type.
    """

    @stub
    def get_base_pointer(self) -> "Pointer[T]": ...

    @stub
    def get_element_pointer(self, indices: int | tuple[int, ...]) -> "Pointer[T]": ...

    @stub
    def __setitem__(self, indices: int | tuple[int, ...], value: T): ...

    @stub
    def __getitem__(self, indices: int | tuple[int, ...]) -> T: ...


class Vector(Generic[T]):

    @property
    @stub
    def dtype(self) -> "DType": ...

    @property
    @stub
    def element_count(self) -> int: ...

    @stub
    def __getitem__(self, item): ...

    @stub
    def __setitem__(self, key, value): ...


class Pointer(Generic[T]):

    @property
    @stub
    def dtype(self) -> "DType": ...

    @stub
    def load(
        self,
        *,
        count: int | None = None,
        alignment: int | None = None,
        volatile: bool = False,
        ordering: MemoryOrder | None = None,
    ) -> T | Vector[T]:
        '''
        Low-level API to read from memory.

        Args:
            count: If count is provided, a vector will be returned.
                For best performance, vector loads should be aligned to the
                number of bytes in the vector.
            alignment: Inform the compiler that the address being loaded from
                is aligned to at least this many bytes.
                The user is responsible for ensuring aligned loads occur only
                on appropriately aligned pointers.
                If alignment is None, do not give the compiler any alignment
                hints.
            volatile: If True, the compiler will not modify the number of times
                this load is performed nor the order of execution with respect
                to other volatile operations.
            ordering: When ordering is specified, the load is atomic.
                Alignment must be explicitly specified on atomic loads.
                Atomic loads require a pointee type with a bit width that
                is a power of two greater than or equal to one byte.
        '''

    @stub
    def store(
        self,
        value: T | Vector[T],
        *,
        alignment: int | None = None,
        volatile: bool = False,
        ordering: Literal[MemoryOrder.RELAXED, MemoryOrder.RELEASE, MemoryOrder.WEAK]
        | None = None,
    ) -> None:
        '''
        Low-level API to store to memory.

        Args:
            value: Scalar or vector to be stored to the given address.
            alignment: Inform the compiler that the address being stored to
                is aligned to at least this many bytes.
                The user is responsible for ensuring aligned loads occur only
                on appropriately aligned pointers.
                If alignment is None, do not give the compiler any alignment
                hints.
            volatile: If True, the compiler will not modify the number of times
                this store is performed nor the order of execution with respect
                to other volatile operations.
            ordering: When ordering is specified, the store is atomic.
                Alignment must be explicitly specified on atomic stores.
                Atomic loads require a pointee type with a bit width that
                is a power of two greater than or equal to one byte.
                Only relaxed, release, and weak are valid memory orders on
                stores.
        '''


@function
def thread_idx() -> tuple[int, int, int]:
    """Gets the current thread indices as ``(x, y, z)``."""
    return (
        nvvm.read_ptx_sreg_tid_x(),
        nvvm.read_ptx_sreg_tid_y(),
        nvvm.read_ptx_sreg_tid_z(),
    )


@function
def block_idx() -> tuple[int, int, int]:
    """Gets the current block indices as ``(x, y, z)``."""
    return (
        nvvm.read_ptx_sreg_ctaid_x(),
        nvvm.read_ptx_sreg_ctaid_y(),
        nvvm.read_ptx_sreg_ctaid_z(),
    )


@function
def block_dim() -> tuple[int, int, int]:
    """Gets the current block dimensions as ``(x, y, z)``."""
    return (
        nvvm.read_ptx_sreg_ntid_x(),
        nvvm.read_ptx_sreg_ntid_y(),
        nvvm.read_ptx_sreg_ntid_z(),
    )


@function
def grid_dim() -> tuple[int, int, int]:
    """Gets the current grid dimensions as ``(x, y, z)``."""
    return (
        nvvm.read_ptx_sreg_nctaid_x(),
        nvvm.read_ptx_sreg_nctaid_y(),
        nvvm.read_ptx_sreg_nctaid_z(),
    )


@stub
def shared_array(
    shape: int | tuple[int, ...],
    dtype: DType,
    dynamic: bool = False,
    alignment: int | None = None,
) -> Array[T]:
    """Create an on-device array in shared memory.

    Shared arrays must be declared at the beginning of the kernel.
    The optional alignment is specified in bytes and must be a positive power of
    two.

    If `dynamic` is `False` (default), the array will be placed in the statically allocated
    shared memory. In this case, `shape` must be a compile-time constant.

    If `dynamic` is `True`, the array will be placed in the dynamically allocated shared memory
    (regardless of whether the provided shape is actually constant).
    In this case, `shape` is allowed to be dynamic. However, only a restricted set of expressions
    is allowed to be used for the dynamic shape: currently, only referencing an integer
    kernel parameter directly is permitted.

    Static shared memory example:

    .. testcode::
        :template: setup_only.py

        @cl.kernel
        def kernel():
            shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
            tx, _, _ = cl.thread_idx()
            if tx == 0:
                shmem[0] = 42

            cl.syncthreads()

            if tx == 1:
                cl.printf("thread id %d sees shmem[0] = %d\\n", tx, shmem[0])

        cl.launch(stream, (1,), (2,), kernel, ())

    .. testoutput::

        thread id 1 sees shmem[0] = 42

    Dynamic shared memory example:

    .. testcode::
        :template: setup_only.py

        @cl.kernel
        def kernel(n):
            shmem = cl.shared_array(shape=(n,), dtype=cl.int32, dynamic=True)
            tx, _, _ = cl.thread_idx()
            if tx == 0:
                shmem[0] = 42

            cl.syncthreads()

            if tx == 1:
                cl.printf("thread id %d sees shmem[0] = %d\\n", tx, shmem[0])

        cl.launch(stream, (1,), (2,), kernel, (32,))

    .. testoutput::

        thread id 1 sees shmem[0] = 42
    """


@stub
def local_array(
    shape: int | tuple[int, ...],
    dtype: DType,
    alignment: int | None = None,
) -> LocalArrayContextManager:
    """Create an on-device array in local memory.

    Local arrays must be declared at the beginning of the kernel
    and must have a dynamic shape. The optional alignment is specified in bytes
    and must be a positive power of two.

    Examples:

        .. testcode::
            :template: setup_only.py

            @cl.kernel
            def kernel():
                local_array = cl.local_array(shape=(32,), dtype=cl.int32)
                tx, _, _ = cl.thread_idx()
                local_array[0] = tx
                if tx == 0:
                    cl.printf("thread id %d sees local_array[0] = %d\\n", tx, local_array[0])
                else:
                    cl.printf("thread id %d sees local_array[0] = %d\\n", tx, local_array[0])

            cl.launch(stream, (1,), (2,), kernel, ())

        .. testoutput::

            thread id 0 sees local_array[0] = 0
            thread id 1 sees local_array[0] = 1
    """


@function
def syncthreads() -> None:
    """
    Synchronizes all threads in the current thread block.

    It is equivalent to ``__syncthreads`` in CUDA C++.

    Examples:

        .. testcode::
            :template: setup_only.py

            @cl.kernel
            def kernel():
                shmem = cl.shared_array(shape=(32,), dtype=cl.int32)
                tx, _, _ = cl.thread_idx()
                if tx == 0:
                    shmem[0] = 42

                cl.syncthreads()

                # Write to shared memory now reflected in all threads
                if tx != 0:
                    cl.printf("shmem[0] = %d\\n", shmem[0])

            cl.launch(stream, (1,), (2,), kernel, ())

        .. testoutput::

            shmem[0] = 42

    """
    nvvm.barrier_cta_sync_all(0)


@stub
def printf(format, *args) -> None:
    """Print the values at runtime from the device

    Args:
        format (str): a c-printf style format string
            in the form of ``%[flags][width][.precision][length]specifier``,
            where specifier is limited to integer and float for now, i.e.
            ``[diuoxXeEfFgGaA]``

        *args (tuple[int | float, ...]):
            Only arithmetic types are supported.

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            cl.printf("value: %d\\n", 42)

        .. testoutput::

            value: 42

    Notes:
        This operation has significant overhead, and should only be used
        for debugging purpose.
    """


@stub
def inline_ptx(ptx_code: str, *constraint_pairs: tuple) -> tuple:
    """Execute inline PTX.

    The API mirrors CUDA C++'s device-side `asm` statement:
    `cl.inline_ptx(ptx_code, (constraint1, value1), (constraint2, value2), ...)`.

    Args:

        ptx_code (str):
            The PTX source string.

        *constraint_pairs:
            Constraint/value pairs.
            Constraints must be compile-time constant strings.

            Read-only operands use constraints ``"h"``, ``"r"``, ``"l"``,
            ``"f"``, ``"d"``, or ``"C"`` and are paired with runtime values.

            Write-only operands use constraints ``"=h"``, ``"=r"``, ``"=l"``,
            ``"=f"``, or ``"=d"`` and are paired with dtype specs.
            This determines the type of the output.

            Read-write operands use constraints ``"+h"``, ``"+r"``, ``"+l"``,
            ``"+f"``, ``"+d"``, or ``"+C"`` and are paired with runtime values.

    Returns:

        The returned tuple depends on the number of write-only input arguments:
        - no write-only outputs: `()`
        - one write-only output: `(value,)`
        - multiple write-only outputs: `(value0, value1, ...)`

    Examples:

        .. testcode::
            :template: kernel_wrapper.py

            i = 12
            j = 30

            # CUDA C++ would use:
            # asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(i), "r"(j));

            (result,) = cl.inline_ptx(
                "add.s32 %0, %1, %2;",
                ("=r", cl.int32),
                ("r", i),
                ("r", j),
            )
            cl.printf("result: %d\\n", result)

        .. testoutput::

            result: 42

    Notes:
        - See CUDA C++ documentation for more details on the `asm` statement.
            https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
        - Constraint type strings map to data types as follows:
            - ``h``: ``cl.int16``
            - ``r``: ``cl.int32``
            - ``l``: ``cl.int64``
            - ``f``: ``cl.float32``
            - ``d``: ``cl.float64``
            - ``C``: pointer value from ``array.get_base_pointer()``

    """


@stub
def atomic_add(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] += val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_add(array, (2, 3), 1)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 0
            after: 1, prev_val: 0

    """


@stub
def atomic_sub(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] -= val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_sub(array, (2, 3), 1)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 0
            after: -1, prev_val: 0

    """


@stub
def atomic_and(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] &= val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 14  # 0b1110
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_and(array, (2, 3), 11)  # 0b1011
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 14
            after: 10, prev_val: 14

    """


@stub
def atomic_or(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] |= val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 12  # 0b1100
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_or(array, (2, 3), 3)  # 0b0011
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 12
            after: 15, prev_val: 12

    """


@stub
def atomic_xor(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] ^= val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 12  # 0b1100
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_xor(array, (2, 3), 10)  # 0b1010
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 12
            after: 6, prev_val: 12

    """


@stub
def atomic_min(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] = min(A[idx], val)`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_min(array, (2, 3), 3)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 7
            after: 3, prev_val: 7

    """


@stub
def atomic_max(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic `A[idx] = max(A[idx], val)`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_max(array, (2, 3), 11)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 7
            after: 11, prev_val: 7

    """


@stub
def atomic_inc(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic increment of `A[idx]` with wrap at `val`.

    This behaves as `A[idx] = 0 if A[idx] >= val else A[idx] + 1`.
    Supports uint32, and uint64 only.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            cl.printf("before: %u\\n", array[2, 3])
            prev_val = cl.atomic_inc(array, (2, 3), 7)
            cl.printf("after: %u, prev_val: %u\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 7
            after: 0, prev_val: 7

    """


@stub
def atomic_dec(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic decrement of `A[idx]` with wrap at `val`.

    This behaves as `A[idx] = val if (A[idx] == 0) or (A[idx] > val) else A[idx] - 1`.
    Supports uint32, and uint64 only.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 0
            cl.printf("before: %u\\n", array[2, 3])
            prev_val = cl.atomic_dec(array, (2, 3), 7)
            cl.printf("after: %u, prev_val: %u\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 0
            after: 7, prev_val: 0

    """


@stub
def atomic_exch(A: Array[T], idx: int | tuple[int, ...], val: T) -> T:
    """
    Perform atomic exchange `A[idx] = val`.

    Returns the old value at the index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_exch(array, (2, 3), 4)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 7
            after: 4, prev_val: 7

    """


@stub
def atomic_cas(A: Array[T], idx: int | tuple[int, ...], old: T, val: T) -> T:
    """
    Perform atomic compare-and-swap on `A[idx]`.

    If the current value equals `old`, store `val`. Returns the old value at the
    index location as if it is loaded atomically.

    Examples:

        .. testcode::
            :template: kernel_2d_array_wrapper.py

            array[2, 3] = 7
            cl.printf("before: %d\\n", array[2, 3])
            prev_val = cl.atomic_cas(array, (2, 3), 7, 4)
            cl.printf("after: %d, prev_val: %d\\n", array[2, 3], prev_val)

        .. testoutput::

            before: 7
            after: 4, prev_val: 7

    """


@stub
def shfl_sync(mask: int, value: int, src_lane: int, width: int = 32) -> int:
    """
    Return ``value`` from lane ``src_lane`` within the logical warp subdivision.
    """


@stub
def shfl_up_sync(mask: int, value: int, delta: int, width: int = 32) -> int:
    """
    Return ``value`` from the lane ``delta`` positions lower in the logical warp
    subdivision.
    """


@stub
def shfl_down_sync(mask: int, value: int, delta: int, width: int = 32) -> int:
    """
    Return ``value`` from the lane ``delta`` positions higher in the logical
    warp subdivision.
    """


@stub
def shfl_xor_sync(mask: int, value: int, lane_mask: int, width: int = 32) -> int:
    """
    Return ``value`` from the lane addressed by XORing the caller lane with
    ``lane_mask`` within the logical warp subdivision.
    """


@function
def syncwarp(mask: int = 0xFFFFFFFF) -> None:
    """
    Performs barrier synchronization for threads within a warp.

    This operation causes the executing thread to wait until all threads
    corresponding to the mask operand have executed a bar.warp.sync with
    the same mask value before resuming execution.

    The mask operand specifies the threads participating in the barrier,
    where each bit position corresponds to the thread's lane ID within the
    warp. Only threads with their corresponding bit set in the mask
    participate in the barrier synchronization.
    """
    nvvm.bar_warp_sync(mask)


@stub
def address_space_cast(value: Pointer[T], memory_space: MemorySpace) -> Pointer[T]:
    """
    Cast a pointer to the given memory space.

    .. testcode::
        :template: kernel_wrapper.py

        smem = cl.shared_array(1, cl.int32)
        smem_ptr = smem.get_base_pointer()
        generic_ptr = cl.address_space_cast(smem_ptr, cl.MemorySpace.GENERIC)

    """


@stub
def reinterpret_pointer_as_array(
    pointer: Pointer[T],
    dtype: DType,
    shape: tuple[int, ...],
    strides: tuple[int, ...] | None = None,
) -> Array[T]:
    """
    Args:
        pointer: Pointer[T]
        dtype: DType
        shape: tuple[int, ...]
        strides: tuple[int, ...] | None = None

    Returns:
        Array with the specified base pointer, dtype, shape, and strides.

    Examples:

    .. testcode::
        :template: kernel_wrapper.py

        smem_array = cl.shared_array(1, cl.int32)
        smem_array[0] = 5

        smem_ptr = smem_array.get_base_pointer()
        smem_array_2 = cl.reinterpret_pointer_as_array(smem_ptr, shape=1, dtype=cl.int32)

        # Assignment through the reconstructed array is equivalent
        # to assignment through the original.
        smem_array_2[0] = 7
        cl.printf("%d\\n", smem_array[0])

    .. testoutput::

        7

    """


__all__ = (
    "block_idx",
    "block_dim",
    "grid_dim",
    "thread_idx",
    "atomic_add",
    "atomic_sub",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_min",
    "atomic_max",
    "atomic_inc",
    "atomic_dec",
    "atomic_exch",
    "atomic_cas",
    "shfl_sync",
    "shfl_up_sync",
    "shfl_down_sync",
    "shfl_xor_sync",
    "Constant",
    "printf",
    "Array",
    "Pointer",
    "Vector",
    "shared_array",
    "local_array",
    "syncwarp",
    "static_eval",
    "static_assert",
    "static_iter",
    "address_space_cast",
    "reinterpret_pointer_as_array",
)
