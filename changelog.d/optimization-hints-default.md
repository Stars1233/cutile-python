<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

 - `ByTarget(default=...)` and scalar kernel hints (e.g.
    `@ct.kernel(num_ctas=8)`) now apply to every GPU architecture, with
    per-arch entries (`ByTarget(sm_100=8, default=2)`) acting as
    overrides on the matching arch.
  - `latency` and `allow_tma` on `ct.load`/`ct.store` likewise apply to
    every target.
  - Requires tileiras 13.3 or later; falls back to the legacy
    single-arch encoding for older versions.
