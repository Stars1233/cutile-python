# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch
from .util import require_hopper_or_newer


@require_hopper_or_newer()
def test_pdl():
    @cl.kernel
    def dependee(a):
        tx = cl.thread_index(0)
        a[tx] = tx * 2.0

        cl.memory_barrier(scope=cl.MemoryScope.DEVICE)
        cl.grid_dependency_control_launch_dependents()

    @cl.kernel
    def dependent(a, b):

        # --- overlap some work with parent
        tx = cl.thread_index(0)
        val = tx * 4.0 + b[tx]
        # ---

        cl.grid_dependency_control_wait()
        a[tx] = val + a[tx]

    a = torch.zeros(32, dtype=torch.float32).cuda()
    b = torch.ones(32, dtype=torch.float32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (32,), dependee, (a,))
    cl.launch(
        torch.cuda.current_stream(),
        (1,),
        (32,),
        dependent,
        (a, b),
        programmatic_dependent_launch=True,
    )
    torch.cuda.synchronize()
    assert a.cpu().tolist() == list(i * 2 + i * 4 + 1 for i in range(32))
