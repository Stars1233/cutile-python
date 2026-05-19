# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._datatype import MemorySpace
from cuda.lang._execution import stub
from cuda.lang._datatype import uint64, bool_


@stub
def mbarrier_init(mbar, participants: int) -> None:
    ...


@stub
def mbarrier_invalidate(mbar) -> None:
    ...


@stub
def mbarrier_arrive(
    mbar,
    count: int = 1,
    *,
    drop: bool = False,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "uint64 | None":
    """Arrive at ``mbar``. When the mbarrier resides in ``MemorySpace.SHARED``,
    an opaque 64-bit value capturing the phase of the mbarrier object _prior_
    to this arrive operation is returned. ``drop=True`` drops a participant
    from the barrier.
    """


@stub
def mbarrier_arrive_expect_tx(
    mbar,
    bytes: int,
    *,
    drop: bool = False,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "uint64 | None":
    """Arrive at ``mbar`` and set the expected transaction count to ``bytes``.
    """


@stub
def mbarrier_expect_tx(
    mbar,
    bytes: int,
    *,
    scope: MemorySpace = MemorySpace.SHARED,
) -> None:
    ...


@stub
def mbarrier_complete_tx(
    mbar,
    bytes: int,
    *,
    scope: MemorySpace = MemorySpace.SHARED,
) -> None:
    ...


@stub
def mbarrier_test_wait(
    mbar,
    state,
    *,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "bool_":
    """Non-blocking test whether ``mbar`` has completed.
    """


@stub
def mbarrier_test_wait_parity(
    mbar,
    parity: int,
    *,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_test_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.
    """


@stub
def mbarrier_try_wait(
    mbar,
    state,
    *,
    time_hint: int | None = None,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "bool_":
    """Bounded-wait test whether ``mbar`` has completed.
    """


@stub
def mbarrier_try_wait_parity(
    mbar,
    parity: int,
    *,
    time_hint: int | None = None,
    scope: MemorySpace = MemorySpace.SHARED,
    relaxed: bool = False,
) -> "bool_":
    """Phase-parity variant of ``mbarrier_try_wait``.
    ``parity`` is the 0/1 integer parity of the phase to test for.
    """


__all__ = (
    "mbarrier_init",
    "mbarrier_invalidate",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_expect_tx",
    "mbarrier_complete_tx",
    "mbarrier_test_wait",
    "mbarrier_test_wait_parity",
    "mbarrier_try_wait",
    "mbarrier_try_wait_parity",
)
