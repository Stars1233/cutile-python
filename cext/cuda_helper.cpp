// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "cuda_helper.h"
#include "cuda_loader.h"


const char* get_cuda_error(const DriverApi* driver, CUresult res) {
    const char* str = nullptr;
    driver->cuGetErrorString(res, &str);
    return str ? str : "Unknown error";
}

Status check_driver_version(const DriverApi* driver, int minimum_version) {
    int version;
    CUresult res = driver->cuDriverGetVersion(&version);
    if (res != CUDA_SUCCESS) {
        PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(driver, res));
        return ErrorRaised;
    }
    if (version < minimum_version) {
        int major = version / 1000;
        int minor = (version % 1000) / 10;
        int required_major = minimum_version / 1000;
        PyErr_Format(PyExc_RuntimeError,
                     "Minimum driver version required is %d.0, got %d.%d",
                     required_major, major, minor);
        return ErrorRaised;
    }
    return OK;
}

PyObject* get_max_grid_size(PyObject *self, PyObject *args) {
    int device_id;
    if (!PyArg_ParseTuple(args, "i", &device_id))
        return NULL;

    Result<const DriverApi*> driver = get_driver_api();
    if (!driver.is_ok()) return NULL;

    CUdevice dev;
    CUresult res = (*driver)->cuDeviceGet(&dev, device_id);
    if (res != CUDA_SUCCESS)
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(*driver, res));

    int max_grid_size[3];
    for (int i = 0; i < 3; ++i) {
        res = (*driver)->cuDeviceGetAttribute(&max_grid_size[i],
            static_cast<CUdevice_attribute>(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + i),
            dev);
        if (res != CUDA_SUCCESS) {
            return PyErr_Format(PyExc_RuntimeError,
                                "cuDeviceGetAttribute: %s", get_cuda_error(*driver, res));
        }
    }
    return Py_BuildValue("(iii)", max_grid_size[0], max_grid_size[1], max_grid_size[2]);
}

PyObject* get_compute_capability(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int major, minor;
    CUdevice dev;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuDeviceGet(&dev, 0);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGet: %s", get_cuda_error(d, res));
    }
    res = d->cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(d, res));
    }
    res = d->cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDeviceGetAttribute: %s", get_cuda_error(d, res));
    }
    return Py_BuildValue("(ii)", major, minor);
}

PyObject* get_driver_version(PyObject *self, PyObject *Py_UNUSED(ignored)) {
    int major, minor;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuDriverGetVersion(&major);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError, "cuDriverGetVersion: %s", get_cuda_error(d, res));
    }
    minor = (major % 1000) / 10;
    major = major / 1000;
    return Py_BuildValue("(ii)", major, minor);
}

// ========== Context helpers ==========

PyObject* synchronize_context(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuCtxSynchronize();
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuCtxSynchronize: %s", get_cuda_error(d, res));
    }
    Py_RETURN_NONE;
}

// ========== Stream helpers ==========

PyObject* create_stream(PyObject* self, PyObject* Py_UNUSED(ignored)) {
    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUstream stream;
    CUresult res = d->cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuStreamCreate: %s", get_cuda_error(d, res));
    }
    return PyLong_FromVoidPtr(stream);
}

PyObject* destroy_stream(PyObject* self, PyObject* arg) {
    CUstream stream = static_cast<CUstream>(PyLong_AsVoidPtr(arg));
    if (PyErr_Occurred()) return NULL;

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return NULL;
    const DriverApi* d = *driver_result;

    CUresult res = d->cuStreamDestroy(stream);
    if (res != CUDA_SUCCESS) {
        return PyErr_Format(PyExc_RuntimeError,
                            "cuStreamDestroy: %s", get_cuda_error(d, res));
    }
    Py_RETURN_NONE;
}

static decltype(cuLaunchKernel)* g_real_cuLaunchKernel;
static PyObject* g_cuLaunchKernel_spy_callback;

static CUresult shim_cuLaunchKernel(
        CUfunction f,
        unsigned int gridDimX,
        unsigned int gridDimY,
        unsigned int gridDimZ,
        unsigned int blockDimX,
        unsigned int blockDimY,
        unsigned int blockDimZ,
        unsigned int sharedMemBytes,
        CUstream hStream,
        void** kernelParams,
        void** extra) {

    PyPtr res = steal(PyObject_CallFunction(
            g_cuLaunchKernel_spy_callback,
            "(K III III I K)",
            reinterpret_cast<unsigned long long>(f),
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            reinterpret_cast<unsigned long long>(hStream)
    ));
    if (!res) return CUDA_ERROR_LAUNCH_FAILED;

    return g_real_cuLaunchKernel(f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
                                 sharedMemBytes, hStream, kernelParams, extra);
}

static PyObject* spy_on_cuLaunchKernel_begin(PyObject* self, PyObject* arg) {
    if (g_real_cuLaunchKernel)
        return PyErr_Format(PyExc_RuntimeError, "Already spying");

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;

    DriverApi* api = const_cast<DriverApi*>(*driver_result);
    g_real_cuLaunchKernel = api->cuLaunchKernel;
    g_cuLaunchKernel_spy_callback = Py_NewRef(arg);
    api->cuLaunchKernel = shim_cuLaunchKernel;
    return Py_NewRef(Py_None);
}

static PyObject* spy_on_cuLaunchKernel_end(PyObject* self, PyObject* arg) {
    if (!g_real_cuLaunchKernel)
        return PyErr_Format(PyExc_RuntimeError, "Not spying");

    Result<const DriverApi*> driver_result = get_driver_api();
    if (!driver_result.is_ok()) return nullptr;

    DriverApi* api = const_cast<DriverApi*>(*driver_result);
    api->cuLaunchKernel = g_real_cuLaunchKernel;
    g_real_cuLaunchKernel = nullptr;
    Py_CLEAR(g_cuLaunchKernel_spy_callback);
    return Py_NewRef(Py_None);
}

static PyMethodDef functions[] = {
    {"get_compute_capability", get_compute_capability, METH_NOARGS,
        "Get compute capability of the default CUDA device"},
    {"get_driver_version", get_driver_version, METH_NOARGS,
        "Get the cuda driver version"},
    {"_get_max_grid_size", get_max_grid_size, METH_VARARGS,
        "Get max grid size of a CUDA device, given device id"},
    {"_synchronize_context", synchronize_context, METH_NOARGS,
        "Synchronize the current CUDA context (drain all streams)."},
    {"_create_stream", create_stream, METH_NOARGS,
        "Create a non-blocking CUDA stream. Returns int handle."},
    {"_destroy_stream", destroy_stream, METH_O,
        "Destroy a CUDA stream given its int handle."},
    {"_spy_on_cuLaunchKernel_begin", spy_on_cuLaunchKernel_begin, METH_O, nullptr},
    {"_spy_on_cuLaunchKernel_end", spy_on_cuLaunchKernel_end, METH_NOARGS, nullptr},
    NULL
};

Status cuda_helper_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;

    return OK;
}
