// SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "../vec.h"
#include "../check.h"

int main() {
    // Growing the vector
    Vec<int> v;
    CHECK(v.empty());
    for (int i = 0; i < 100; ++i) {
        CHECK(v.size() == static_cast<size_t>(i));
        for (int j = 0; j < i; ++j)
            CHECK(v[j] == j * 10);
        v.push_back(i * 10);
        CHECK(!v.empty());
    }

    // Copy constructor
    Vec<int> v2(v);
    CHECK(v2.size() == v.size());
    CHECK(v2.size() == 100);
    for (int i = 0; i < 100; ++i)
        CHECK(v2[i] == i * 10);

    // Move constructor
    Vec<int> v3(std::move(v2));
    CHECK(v2.size() == 0);
    CHECK(v3.size() == 100);
    for (int i = 0; i < 100; ++i)
        CHECK(v3[i] == i * 10);

    // Copy assignment
    v2 = v3;
    CHECK(v2.size() == 100);
    for (int i = 0; i < 100; ++i)
        CHECK(v2[i] == i * 10);

    // Clear
    v3.clear();
    CHECK(v3.size() == 0);

    // Move assignment
    v3 = std::move(v2);
    CHECK(v2.size() == 0);
    CHECK(v3.size() == 100);
    for (int i = 0; i < 100; ++i)
        CHECK(v3[i] == i * 10);

    // Resize to same size
    v3.resize(100);
    CHECK(v3.size() == 100);
    for (int i = 0; i < 100; ++i)
        CHECK(v3[i] == i * 10);

    // Resize to shrink
    v3.resize(50);
    CHECK(v3.size() == 50);
    for (int i = 0; i < 50; ++i)
        CHECK(v3[i] == i * 10);

    // Resize to grow
    v3.resize(100);
    CHECK(v3.size() == 100);
    for (int i = 0; i < 50; ++i)
        CHECK(v3[i] == i * 10);
    for (int i = 50; i < 100; ++i)
        CHECK(v3[i] == 0);

    // Resize to grow much bigger, requiring a realloc
    v3.resize(1000);
    CHECK(v3.size() == 1000);
    for (int i = 0; i < 50; ++i)
        CHECK(v3[i] == i * 10);
    for (int i = 50; i < 1000; ++i)
        CHECK(v3[i] == 0);

    // Sized constructor
    Vec<int> v4(10);
    CHECK(v4.size() == 10);
    for (int i = 0; i < 10; ++i)
        CHECK(v4[i] == 0);

    // Initializer list constructor
    Vec<int> v5 = {1, 2, 3};
    CHECK(v5.size() == 3);
    for (int i = 0; i < 3; ++i)
        CHECK(v5[i] == i + 1);

    // Comparison
    CHECK(Vec<int>{} == Vec<int>{});

    CHECK(!(Vec<int>{} == Vec<int>{0}));
    CHECK(Vec<int>{} != Vec<int>{0});

    CHECK(!(Vec<int>{1} == Vec<int>{0}));
    CHECK(Vec<int>{1} != Vec<int>{0});

    // Aligned vector
    Vec<int, AlignedAllocation<int, 4096>> v_aligned;
    v_aligned.push_back(3);
    CHECK(reinterpret_cast<uintptr_t>(v_aligned.data()) % 4096 == 0);

    return 0;
}
