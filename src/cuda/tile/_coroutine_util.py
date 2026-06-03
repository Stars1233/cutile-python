# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Awaitable

from cuda.tile._cext import run_coroutine


# Replace `await foo()` with `await resume_after(foo())` to bypass the recursion limit.
@dataclass
class resume_after:
    awaitable: Awaitable

    def __await__(self):
        return (yield self.awaitable)


__all__ = ["run_coroutine", "resume_after"]
