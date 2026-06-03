# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import enum
from typing import Any

from cuda.lang._execution import stub
from .nvvm import P3, P6


class CTAGroup(enum.Enum):
    CTA_1 = "cg1"
    CTA_2 = "cg2"


class Tcgen05LdStShape(enum.Enum):
    SHAPE_16X64B = "16x64b"
    SHAPE_16X128B = "16x128b"
    SHAPE_16X256B = "16x256b"
    SHAPE_32X32B = "32x32b"
    SHAPE_16X32BX2 = "16x32bx2"


@stub
def tcgen05_alloc(
    addr: P3,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None: ...


@stub
def tcgen05_dealloc(
    addr: P6,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None: ...


@stub
def tcgen05_commit(
    mbar: P3,
    *,
    multicast_mask: int | None = None,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None: ...


@stub
def tcgen05_ld(
    shape: Tcgen05LdStShape,
    tmem_addr: P6,
    *,
    count: int = 1,
    pack: bool | None = None,
    offset: int | None = None,
) -> Any: ...


__all__ = (
    "CTAGroup",
    "Tcgen05LdStShape",
    "tcgen05_alloc",
    "tcgen05_dealloc",
    "tcgen05_commit",
    "tcgen05_ld",
)
