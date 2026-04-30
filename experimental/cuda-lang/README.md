<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# cuda.lang

`cuda.lang` compiles Python to cubin, providing a SIMT programming model.

```python
import cuda.lang as cl
import torch

@cl.kernel
def saxpy(
    N: cl.Constant[int],
    a: cl.Constant[float],
    X, Y
):
    tidx, _, _ = cl.thread_idx()
    bidx, _, _ = cl.block_idx()
    block_dim_x, _, _ = cl.block_dim()
    idx = tidx + bidx * block_dim_x
    if idx < N:
        Y[idx] = a * X[idx] + Y[idx]

N = 256
alpha = 2.0
X = torch.ones(N, dtype=torch.float32, device="cuda")
Y = torch.ones(N, dtype=torch.float32, device="cuda")
expected = (alpha * X + Y).cpu()
cl.launch(
  stream=torch.cuda.current_stream(),
  grid=(64,),
  block=(64,),
  kernel=saxpy,
  kernel_args=(N, alpha, X, Y),
)
assert torch.allclose(expected, Y.cpu())
```
