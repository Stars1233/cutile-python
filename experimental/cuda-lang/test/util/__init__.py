# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.lang._compile import get_compute_capability

from .filecheck_utils import filecheck, get_source
from .ir_utils import (
    get_ir,
    make_symbolic_scalar,
    make_symbolic_tensor,
    compile_for_arguments,
)


def require_blackwell_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (10, 0),
        reason="feature requires Blackwell or newer",
    )


def require_hopper_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (9, 0),
        reason="feature requires Hopper or newer",
    )


__all__ = (
    "filecheck",
    "get_source",
    "get_ir",
    "make_symbolic_scalar",
    "make_symbolic_tensor",
    "compile_for_arguments",
    "require_hopper_or_newer",
    "require_blackwell_or_newer",
)
