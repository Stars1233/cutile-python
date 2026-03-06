<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `ct.mma_scaled()` operation for block-scaled matrix multiply-accumulate.
  Supported input dtypes: `float8_e4m3fn`, `float8_e5m2`, `float4_e2m1fn`.
  Supported scale dtypes: `float8_e8m0fnu`, `float8_e4m3fn` (f4 inputs only).
