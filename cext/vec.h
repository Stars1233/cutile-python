/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "memory.h"
#include "hash.h"

#include <new>
#include <utility>


// Replacement for std::vector<T>


template <typename T>
struct DefaultAllocation {
    T* data;

    DefaultAllocation(const DefaultAllocation&) = delete;
    void operator=(const DefaultAllocation&) = delete;

    DefaultAllocation() : data(nullptr) {}

    explicit DefaultAllocation(size_t count) : data(xcalloc<T>(count)) {}

    DefaultAllocation(DefaultAllocation&& other) : data(other.data) {
        other.data = nullptr;
    }

    DefaultAllocation& operator=(DefaultAllocation&& other) {
        if (this != &other) {
            if (data) mem_free(data);
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    ~DefaultAllocation() { mem_free(data); }
};


template <typename T, size_t AlignmentBytes>
struct AlignedAllocation {
    static_assert((AlignmentBytes & (AlignmentBytes - 1)) == 0);
    static_assert(AlignmentBytes % sizeof(T) == 0);

    AlignedAllocation() = default;

    explicit AlignedAllocation(size_t count)
        : raw_allocation_(raw_allocation_size(count))
    {
        uintptr_t address = reinterpret_cast<uintptr_t>(raw_allocation_.data);
        uintptr_t correction = (-address) & (AlignmentBytes - 1);
        data = reinterpret_cast<T*>(raw_allocation_.data + correction);
    }

    AlignedAllocation(AlignedAllocation&& other)
        : raw_allocation_(std::move(other.raw_allocation_)), data(other.data)
    {
        other.data = nullptr;
    }

    AlignedAllocation& operator=(AlignedAllocation&& other) {
        if (this != &other) {
            raw_allocation_ = std::move(other.raw_allocation_);
            data = other.data;
            other.data = nullptr;
        }
        return *this;
    }

    T* data = nullptr;
private:
    DefaultAllocation<char> raw_allocation_;

    static size_t raw_allocation_size(size_t count) {
        size_t bytes = count * sizeof(T);
        CHECK(bytes / sizeof(T) == count); // check for overflow
        size_t ret = bytes + (AlignmentBytes - 1);
        CHECK(ret >= bytes); // check for overflow
        return ret;
    }
};



template <typename T, typename Allocation = DefaultAllocation<T>>
class Vec {
public:
    Vec() : size_(0), capacity_(0) {}

    Vec(size_t size)
      : allocation_(size), size_(size), capacity_(size)
    {
        T* ptr = allocation_.data;
        while (size--)
            new (ptr++) T();
    }

    Vec(const Vec& other)
      : allocation_(other.size_), size_(other.size_), capacity_(other.size_)
    {
        _copy_from(other);
    }

    Vec(Vec&& other)
      : allocation_(std::move(other.allocation_)),
        size_(other.size_),
        capacity_(other.capacity_)
    {
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Vec(std::initializer_list<T> list)
      : allocation_(list.size()), size_(list.size()), capacity_(list.size())
    {
        _copy_from(list);
    }

    ~Vec() {
        clear();
    }

    Vec& operator=(const Vec& other) {
        if (this != &other) {
            clear();
            _ensure_capacity(other.size_);
            size_ = other.size_;
            _copy_from(other);
        }
        return *this;
    }

    Vec& operator=(Vec&& other) {
        if (this != &other) {
            clear();
            allocation_ = std::move(other.allocation_);
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T* data() {
        return allocation_.data;
    }

    const T* data() const {
        return allocation_.data;
    }

    size_t size() const {
        return size_;
    }

    bool empty() const {
        return !size_;
    }

    void clear() {
        T* ptr = allocation_.data;
        for (size_t n = size_; n; --n)
            (ptr++)->~T();
        size_ = 0;
    }

    void resize(size_t new_size) {
        if (new_size > size_) {
            _ensure_capacity(new_size);
            do {
                new (&allocation_.data[size_++]) T();
            } while (new_size > size_);
        } else {
            while (new_size < size_)
                allocation_.data[--size_].~T();
        }
    }

    void reserve(size_t capacity) {
        _ensure_capacity(capacity);
    }

    void push_back(const T& value) {
        _ensure_capacity(size_ + 1);
        new (&allocation_.data[size_++]) T(value);
    }

    void push_back(T&& value) {
        _ensure_capacity(size_ + 1);
        new (&allocation_.data[size_++]) T(std::move(value));
    }

    T* begin() {
        return allocation_.data;
    }

    T* end() {
        return allocation_.data + size_;
    }

    const T* begin() const {
        return allocation_.data;
    }

    const T* end() const {
        return allocation_.data + size_;
    }

    T& operator[] (size_t i) {
        return allocation_.data[i];
    }

    const T& operator[] (size_t i) const {
        return allocation_.data[i];
    }

    T& back() {
        return allocation_.data[size_ - 1];
    }

    const T& back() const {
        return allocation_.data[size_ - 1];
    }

    void pop_back() {
        back().~T();
        --size_;
    }

    bool operator== (const Vec& other) const {
        size_t n = size_;
        if (n != other.size_) return false;
        const T *a = allocation_.data, *b = other.allocation_.data;
        while (n--) {
            if (*a++ != *b++)
                return false;
        }
        return true;
    }

    bool operator!= (const Vec& other) const {
        return !(*this == other);
    }

private:
    Allocation allocation_;
    size_t size_;
    size_t capacity_;

    void _ensure_capacity(size_t required_size) {
        if (capacity_ >= required_size) return;

        size_t min_capacity = capacity_ + capacity_ / 2 + 1;
        if (min_capacity < capacity_) min_capacity = required_size;

        size_t new_capacity = required_size < min_capacity ? min_capacity : required_size;
        Allocation new_allocation(new_capacity);

        T *dst = new_allocation.data, *src = allocation_.data;
        for (size_t n = size_; n; --n, ++src, ++dst) {
            new (dst) T(std::move(*src));
            src->~T();
        }

        allocation_ = std::move(new_allocation);
        capacity_ = new_capacity;
    }

    template <typename Seq>
    void _copy_from(Seq&& src) {
        T* dst = allocation_.data;
        for (const T& src_item : src)
            new (dst++) T(src_item);
    }
};


template <typename T>
struct Hash<Vec<T>> {
    static void hash(const Vec<T>& vec, Hasher& h) {
        h.hash(vec.size());
        for (const T& x : vec)
            Hash<T>::hash(x, h);
    }
};

