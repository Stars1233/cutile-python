<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Added comparison and constructor support for Python ``enum.Enum`` inside kernels. Enum members can now be compared with ``==`` / ``!=`` and constructed from a constant value (e.g. ``Color(0)``).
