# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from .core_api import Pointer
from cuda.lang._datatype import (
    bool_,
    clusterlaunchcontrol_token,
    mbarrier,
    int32,
    MemorySpace,
)
from cuda.lang._execution import stub


@stub
def clusterlaunchcontrol_try_cancel(
    addr: "Pointer[clusterlaunchcontrol_token, MemorySpace.SHARED]",
    mbar: "Pointer[mbarrier, MemorySpace.SHARED]",
    multicast: bool = False,
) -> None: ...


@stub
def clusterlaunchcontrol_is_canceled(token: clusterlaunchcontrol_token) -> "bool_": ...


@stub
def clusterlaunchcontrol_get_first_block_idx(
    token: clusterlaunchcontrol_token, axis: int | None = None
) -> "int32 | tuple[int32, int32, int32]": ...
