// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "py.h"

#include "tile_kernel.h"
#include "cuda_helper.h"
#include "coroutine_util.h"
#include "xla_ffi_py.h"

#ifdef _WIN32
extern "C" int _fltused = 0;
#endif


static PyObject* dev_features_enabled(PyObject*, PyObject*) {
#ifdef CUDA_TILE_ENABLE_DEV_FEATURES
    Py_RETURN_TRUE;
#else
    Py_RETURN_FALSE;
#endif
}

static PyMethodDef module_methods[] = {
    {"dev_features_enabled", dev_features_enabled, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

static PyModuleDef module_def = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "cuda.tile._cext",
    .m_size = 0,
    .m_methods = module_methods,
};

PyMODINIT_FUNC PyInit__cext() {
    PyPtr m = steal(PyModule_Create(&module_def));
    if (!m) return nullptr;

#ifdef Py_GIL_DISABLED
    if (PyUnstable_Module_SetGIL(m.get(), Py_MOD_GIL_NOT_USED) != 0 )
        return nullptr;
#endif

    if (!tile_kernel_init(m.get()))
        return nullptr;

    if (!cuda_helper_init(m.get()))
        return nullptr;

    if (!coroutine_util_init(m.get()))
        return nullptr;

    if (!xla_ffi_init(m.get()))
        return nullptr;


    return m.release();
}

