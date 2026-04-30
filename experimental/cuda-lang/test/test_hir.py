# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.tile._passes.ast2hir import get_function_hir

from .util import filecheck


def filecheck_hir(func_hir: cl.kernel, check_directives: str) -> None:
    func_hir = get_function_hir(func_hir._pyfunc, entry_point=True)
    hir_string = str(func_hir.body)
    filecheck(hir_string, check_directives)


def test_load_store_in_hir():
    @cl.kernel
    def my_kernel(A):
        val = cl.load(A, 0, (1,))
        cl.store(A, 0, val + 1)

    filecheck_hir(
        my_kernel,
        """
        CHECK-LABEL: ^{{[0-9]+}}():
        CHECK: [[LOAD:%[0-9]+]] = <fn:getattr>{{.+}}'load'
        CHECK: %{{[0-9]+}} = [[LOAD]]({{.+}})
        CHECK: [[STORE:%[0-9]+]] = <fn:getattr>{{.+}}'store'
        CHECK: [[STORE]]({{.+}}, %{{[0-9]+}})
        CHECK: return
        """,
    )
