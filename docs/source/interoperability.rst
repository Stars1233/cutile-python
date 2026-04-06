.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

Interoperability
================

.. _interoperability-machine-representation:

Machine Representation
----------------------

cuTile executes Python |tile code| on NVIDIA GPUs by translating the Python code into a *machine representation* that can be executed by CUDA devices.
Functions, types, and objects all have a machine representation.

Machine representations are defined in terms of corresponding CUDA C++ entities.
Example: ``cuda.tile.float16`` has the same machine representation as ``__half`` in CUDA C++.

Interoperability with SIMT
--------------------------

Inter-Kernel
~~~~~~~~~~~~

Inter-kernel interoperability refers to all interoperability concerns that do not cross the kernel boundary - everything except mixing tile and SIMT code in a kernel.
This includes:

- Writing tile and SIMT kernels in the same source file.
- Linking tile and SIMT kernels into the same binary.
- Passing the same kinds of arrays to both tile and SIMT kernels.

Intra-kernel interoperability will be supported in the future.
