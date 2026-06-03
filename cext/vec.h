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
class Vec {
public:
    Vec() : data_(nullptr), size_(0), capacity_(0) {}

    Vec(size_t size)
      : data_(xcalloc<T>(size)),
        size_(size),
        capacity_(size)
    {
        T* ptr = data_;
        while (size--)
            new (ptr++) T();
    }

    Vec(const Vec& other)
      : data_(xcalloc<T>(other.size_)),
        size_(other.size_),
        capacity_(other.size_)
    {
        _copy_from(other);
    }

    Vec(Vec&& other)
      : data_(other.data_),
        size_(other.size_),
        capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    Vec(std::initializer_list<T> list)
      : data_(xcalloc<T>(list.size())),
        size_(list.size()),
        capacity_(list.size())
    {
        _copy_from(list);
    }

    ~Vec() {
        clear();
        mem_free(data_);
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
            mem_free(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    T* data() {
        return data_;
    }

    const T* data() const {
        return data_;
    }

    size_t size() const {
        return size_;
    }

    bool empty() const {
        return !size_;
    }

    void clear() {
        T* ptr = data_;
        for (size_t n = size_; n; --n)
            (ptr++)->~T();
        size_ = 0;
    }

    void resize(size_t new_size) {
        if (new_size > size_) {
            _ensure_capacity(new_size);
            do {
                new (&data_[size_++]) T();
            } while (new_size > size_);
        } else {
            while (new_size < size_)
                data_[--size_].~T();
        }
    }

    void reserve(size_t capacity) {
        _ensure_capacity(capacity);
    }

    void push_back(const T& value) {
        _ensure_capacity(size_ + 1);
        new (&data_[size_++]) T(value);
    }

    void push_back(T&& value) {
        _ensure_capacity(size_ + 1);
        new (&data_[size_++]) T(std::move(value));
    }

    T* begin() {
        return data_;
    }

    T* end() {
        return data_ + size_;
    }

    const T* begin() const {
        return data_;
    }

    const T* end() const {
        return data_ + size_;
    }

    T& operator[] (size_t i) {
        return data_[i];
    }

    const T& operator[] (size_t i) const {
        return data_[i];
    }

    T& back() {
        return data_[size_ - 1];
    }

    const T& back() const {
        return data_[size_ - 1];
    }

    void pop_back() {
        back().~T();
        --size_;
    }

    bool operator== (const Vec& other) const {
        size_t n = size_;
        if (n != other.size_) return false;
        const T *a = data_, *b = other.data_;
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
    T* data_;
    size_t size_;
    size_t capacity_;

    void _ensure_capacity(size_t required_size) {
        if (capacity_ >= required_size) return;

        size_t min_capacity = capacity_ + capacity_ / 2 + 1;
        if (min_capacity < capacity_) min_capacity = required_size;

        size_t new_capacity = required_size < min_capacity ? min_capacity : required_size;
        T* new_data = xcalloc<T>(new_capacity);

        T *dst = new_data, *src = data_;
        for (size_t n = size_; n; --n, ++src, ++dst) {
            new (dst) T(std::move(*src));
            src->~T();
        }

        mem_free(data_);
        data_ = new_data;
        capacity_ = new_capacity;
    }

    template <typename Seq>
    void _copy_from(Seq&& src) {
        T* dst = data_;
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

