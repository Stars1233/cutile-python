# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Generic, TypeVar, Literal

from cuda.tile import MemoryOrder, DType
from cuda.tile._execution import stub
from cuda.tile._memory_model import MemorySpace

T = TypeVar("T")


class Scalar(Generic[T]):
    pass


class Vector(Generic[T]):
    """Fixed-size collection returned by vectorized pointer operations."""

    _cutile_is_builtin = True

    @stub
    def __init__(self, *elements: T, dtype: DType | None = None) -> None:
        """Constructs a vector from scalar elements, optionally with an explicit dtype."""

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
    """Typed address into a CUDA memory space with low-level load and store operations."""

    @stub
    def load(
            self,
            *,
            count: int | None = None,
            alignment: int | None = None,
            volatile: bool = False,
            ordering: MemoryOrder | None = None,
    ) -> T | Vector[T]:
        """
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
        """

    @stub
    def store(
            self,
            value: T | Vector[T],
            *,
            alignment: int | None = None,
            volatile: bool = False,
            ordering: Literal[MemoryOrder.RELAXED,
                              MemoryOrder.RELEASE, MemoryOrder.WEAK] | None = None,
    ) -> None:
        """
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
        """

    @property
    @stub
    def opaque(self) -> bool:
        """
        Whether the pointer is opaque, i.e. doesn't point to a value of a specific data type.
        This is a compile-time constant boolean.
        """

    @property
    @stub
    def pointee_dtype(self) -> DType:
        """
        Data type of the value that this pointer points to.
        Raises a compilation error if the pointer is opaque.
        """

    @property
    @stub
    def memory_space(self) -> MemorySpace:
        """
        Memory space of this pointer.
        """
