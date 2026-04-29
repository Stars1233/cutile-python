# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from io import BytesIO

import pytest
import torch

import cuda.tile as ct
from cuda.tile._compile import get_sm_arch
from cuda.tile._exception import TileUnsupportedFeatureError
from cuda.tile._numeric_semantics import RoundingMode
from cuda.tile.compilation import CallingConvention, KernelSignature


def compile_with_version(pyfunc, args, version: str):
    kernel = ct.kernel(pyfunc)
    cconv = CallingConvention.cutile_python_v1()
    sig = KernelSignature.from_kernel_args(kernel, args, cconv)
    ct.compilation.export_kernel(kernel, [sig], output_file=BytesIO(),
                                 gpu_code=get_sm_arch(), output_format="cubin",
                                 bytecode_version=version)


def tensor(dtype=torch.float32):
    return torch.zeros(64, dtype=dtype, device='cuda')


def test_atan2_requires_13_2():
    def kernel(x, y, z):
        tx = ct.load(x, 0, shape=64)
        ty = ct.load(y, 0, shape=64)
        ct.store(z, 0, tile=ct.atan2(tx, ty))

    with pytest.raises(TileUnsupportedFeatureError, match=r"atan2 requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor(), tensor()), "13.1")


def test_tanh_rounding_mode_requires_13_2():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx, rounding_mode=RoundingMode.APPROX))

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"tanh rounding_mode=approx requires tileiras 13\.2"):
        compile_with_version(kernel, (tensor(), tensor()), "13.1")


def test_tanh_without_rounding_mode_works_on_13_1():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.tanh(tx))

    # Should not raise version error
    compile_with_version(kernel, (tensor(), tensor()), "13.1")


def test_exp_rounding_mode_requires_13_3():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.exp(tx, rounding_mode=RoundingMode.APPROX))

    with pytest.raises(TileUnsupportedFeatureError,
                       match=r"exp rounding_mode=approx requires tileiras 13\.3"):
        compile_with_version(kernel, (tensor(), tensor()), "13.2")


def test_exp_without_rounding_mode_works_on_13_1():
    def kernel(x, y):
        tx = ct.load(x, 0, shape=64)
        ct.store(y, 0, tile=ct.exp(tx))

    # Should not raise version error
    compile_with_version(kernel, (tensor(), tensor()), "13.1")
