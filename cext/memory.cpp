/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "memory.h"
#include <Python.h>


void* xcalloc(size_t nmemb, size_t size) {
    void* ret = PyMem_RawCalloc(nmemb, size);
    CHECK(ret);
    return ret;
}

void* operator new (size_t len) {
    void* ret = PyMem_RawMalloc(len);
    CHECK(ret);
    return ret;
}

void* operator new[] (size_t len) {
    void* ret = PyMem_RawMalloc(len);
    CHECK(ret);
    return ret;
}

void operator delete (void* ptr, size_t) {
    PyMem_RawFree(ptr);
}

void operator delete[] (void* ptr) {
    PyMem_RawFree(ptr);
}

void mem_free(void* p) {
    PyMem_RawFree(p);
}

#ifdef _WIN32
// Prevent msvc to optimize this into a memcpy call
// because there isn't equivalent of -fno-builtin
#pragma optimize("", off)
#endif
void mem_copy(void* RESTRICT dst, const void* RESTRICT src, size_t n) {
    char* dst_c = static_cast<char*>(dst);
    const char* src_c = static_cast<const char*>(src);
    for (size_t i = 0; i < n; ++i)
        dst_c[i] = src_c[i];
}
#ifdef _WIN32
#pragma optimize("", on)
#endif
