# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import cuda.lang as cl
import torch


@pytest.mark.parametrize(
    "dtype,format_string",
    [
        (torch.int32, "%d"),
        (torch.int64, "%ld"),
        (torch.float32, "%f"),
        (torch.float64, "%lf"),
    ],
)
def test_print(dtype, format_string):

    @cl.kernel
    def kernel(A):
        cl.printf(format_string, A[0])

    A = torch.tensor([5], dtype=dtype).cuda()
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (1,),
        kernel,
        (A,),
    )
