// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "vec.h"
#include <memory>

namespace {
template <typename T, size_t InitialSize = 16>
class Arena {
public:
    Arena()
      : cur_chunk_(new T[InitialSize])
      , cur_chunk_avail_(InitialSize)
      , cur_chunk_size_(InitialSize)
    { }

    T* alloc(size_t count) {
        if (count > cur_chunk_avail_)
            allocate_chunk(count);
        cur_chunk_avail_ -= count;
        return &cur_chunk_[cur_chunk_avail_];
    }

    template <size_t AlignmentBytes>
    T* alloc_aligned(size_t count) {
        static_assert(AlignmentBytes % sizeof(T) == 0);
        static_assert((AlignmentBytes & (AlignmentBytes - 1)) == 0);
        T* ret = try_alloc_aligned<AlignmentBytes>(count);
        if (ret) return ret;

        // Request enough items so that we can always find an aligned segment
        allocate_chunk(count + AlignmentBytes / sizeof(T) - 1);
        ret = try_alloc_aligned<AlignmentBytes>(count);
        CHECK(ret);
        return ret;
    }

    void clear() {
        // Preserve the current (i.e. the biggest chunk) for future reuse
        old_chunks_.clear();
        cur_chunk_avail_ = cur_chunk_size_;
    }

private:
    template <size_t AlignmentBytes>
    T* try_alloc_aligned(size_t count) {
        if (count > cur_chunk_avail_) return nullptr;
        uintptr_t cur_chunk_start = reinterpret_cast<uintptr_t>(cur_chunk_.get());
        uintptr_t addr = (cur_chunk_start + (cur_chunk_avail_ - count) * sizeof(T))
                        & ~(AlignmentBytes - 1);
        if (addr < cur_chunk_start) return nullptr;
        cur_chunk_avail_ = (addr - cur_chunk_start) / sizeof(T);
        return &cur_chunk_[cur_chunk_avail_];
    }

    void allocate_chunk(size_t min_capacity) {
        // Always grow the chunk at least by a factor of two, so that eventually,
        // after clearing the arena, we have one big enough chunk to satisfy all allocations.
        cur_chunk_size_ *= 2;
        if (cur_chunk_size_ < min_capacity)
            cur_chunk_size_ = min_capacity;
        old_chunks_.push_back(std::move(cur_chunk_));
        cur_chunk_.reset(new T[cur_chunk_size_]);
        cur_chunk_avail_ = cur_chunk_size_;
    }

public:
    // Keep public for ease of testing
    std::unique_ptr<T[]> cur_chunk_;
    size_t cur_chunk_avail_;
    size_t cur_chunk_size_;
    Vec<std::unique_ptr<T[]>> old_chunks_;
};
}

