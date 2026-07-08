# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch
import pytest


def test_too_many_kwargs():
    @cl.kernel()
    def kernel():
        pass

    bad_kwargs = {f"kw{i}": i for i in range(20)}

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (), **bad_kwargs)
