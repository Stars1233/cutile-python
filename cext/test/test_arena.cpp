// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "../arena.h"
#include "../check.h"


template <typename T, size_t InitialSize>
static inline bool in_cur_chunk(T* p, const Arena<T, InitialSize>& arena) {
    T* chunk = arena.cur_chunk_.get();
    return p >= chunk && p < chunk + arena.cur_chunk_size_;
}

int main() {
    int64_t* p[8];

    Arena<int64_t, 2> arena;
    CHECK(arena.cur_chunk_size_ == 2);

    p[0] = arena.alloc(1);
    *p[0] = 100;
    CHECK(arena.old_chunks_.empty());
    CHECK(in_cur_chunk(p[0], arena));

    p[1] = arena.alloc(1);
    *p[1] = 101;
    CHECK(arena.old_chunks_.empty());
    CHECK(in_cur_chunk(p[1], arena));

    for (int i = 0; i < 3; ++i) {
        p[2 + i] = arena.alloc(1);
        *p[2 + i] = 102 + i;
        CHECK(arena.old_chunks_.size() == 1);
        CHECK(in_cur_chunk(p[2 + i], arena));
        CHECK(arena.cur_chunk_size_ == 4);
    }

    p[5] = arena.alloc(2);
    *p[5] = 105;
    *(p[5] + 1) = 205;
    CHECK(arena.old_chunks_.size() == 2);
    CHECK(in_cur_chunk(p[5], arena));
    CHECK(arena.cur_chunk_size_ == 8);
    CHECK(arena.cur_chunk_avail_ == 6);

    p[6] = arena.alloc_aligned<sizeof(int64_t) * 2>(2);
    CHECK(reinterpret_cast<uintptr_t>(p[6]) % (sizeof(int64_t) * 2) == 0);
    *p[6] = 106;
    *(p[6] + 1) = 206;
    CHECK(arena.old_chunks_.size() == 2);
    CHECK(in_cur_chunk(p[6], arena));
    CHECK(arena.cur_chunk_avail_ == 3 || arena.cur_chunk_avail_ == 4);

    p[7] = arena.alloc_aligned<sizeof(int64_t) * 2>(2);
    CHECK(reinterpret_cast<uintptr_t>(p[6]) % (sizeof(int64_t) * 2) == 0);
    *p[7] = 107;
    *(p[7] + 1) = 207;
    CHECK(arena.old_chunks_.size() == 2);
    CHECK(in_cur_chunk(p[7], arena));
    CHECK(arena.cur_chunk_avail_ == 1 || arena.cur_chunk_avail_ == 2);

    for (int i = 0; i <= 7; ++i) {
        CHECK(*p[i] == 100 + i);
        if (i == 5 || i == 6 || i == 7)
            CHECK(*(p[i] + 1) == 200 +i);
    }

    arena.clear();
    CHECK(arena.old_chunks_.size() == 0);
    CHECK(arena.cur_chunk_size_ == 8);
    CHECK(arena.cur_chunk_avail_ == 8);

    for (int i = 0; i < 2; ++i) {
        int64_t* q = arena.alloc_aligned<sizeof(int64_t) * 2>(2);
        CHECK(reinterpret_cast<uintptr_t>(q) % (sizeof(int64_t) * 2) == 0);
        CHECK(arena.old_chunks_.size() == 0);
        CHECK(in_cur_chunk(q, arena));
        arena.alloc(1);
    }

    int64_t* q = arena.alloc_aligned<sizeof(int64_t) * 2>(4);
    CHECK(reinterpret_cast<uintptr_t>(q) % (sizeof(int64_t) * 2) == 0);
    CHECK(arena.old_chunks_.size() == 1);
    CHECK(in_cur_chunk(q, arena));

    arena.alloc(777);
    CHECK(arena.cur_chunk_size_ == 777);

    return 0;
}
