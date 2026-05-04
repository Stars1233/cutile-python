# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch


def test_local_array_in_if_else():
    @cl.kernel
    def kern(x):
        res = 0
        if cl.thread_idx()[0] == 0:
            res = 3
            with cl.local_array(shape=4, dtype=cl.int32) as a:
                a[0] = 3
                res = a[0]
        elif cl.thread_idx()[0] == 1:
            res = 5
            with cl.local_array(shape=4, dtype=cl.int32) as b:
                b[0] = 5
                res = b[0]

        x[cl.thread_idx()[0]] = res

    x = torch.zeros((2,), dtype=torch.int32, device="cuda")
    cl.launch(torch.cuda.current_stream(), (1,), (2,), kern, (x, ))
    assert x.tolist() == [3, 5]
