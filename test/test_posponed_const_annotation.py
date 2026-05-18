# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Postpone evaluation of type annotations.
from __future__ import annotations

import torch
import cuda.tile as ct

ConstInt = ct.Constant[int]


def needs_constant(x: ct.Constant):
    pass


def needs_constant_int(x: ConstInt):
    pass


def needs_constant_bool(x: ct.Constant[bool]):
    pass


# TODO: Run with `mypy --check-untyped-defs` or another static type checker.
def test_constant_type_hints() -> None:
    int_constant: ct.Constant[int] = 42
    float_constant: ct.Constant[float] = 3.14
    bool_constant: ct.Constant[bool] = True

    needs_constant(int_constant)
    needs_constant(float_constant)
    needs_constant(bool_constant)
    needs_constant_int(int_constant)
    needs_constant_bool(bool_constant)


@ct.kernel
def _arange_kernel(out, N: ConstInt):
    x = ct.arange(N, dtype=ct.int32)
    ct.store(out, 0, tile=x)


def test_kernel_with_postponed_annotations() -> None:
    N = 8
    out = torch.zeros(N, dtype=torch.int32, device='cuda')
    ct.launch(torch.cuda.current_stream(), (1,), _arange_kernel, (out, N))
    torch.testing.assert_close(out.cpu(), torch.arange(N, dtype=torch.int32))
