.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

.. currentmodule:: cuda.lang


Operations
==========


.. _operations-array-creation:

Array Creation
--------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   local_array
   shared_array


.. _operations-pointer-utilities:

Pointer Utilities
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

   is_pointer_dtype
   pointer_dtype
   opaque_pointer_dtype
   address_space_cast
   map_shared_to_cluster

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   PointerInfo

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

   MemorySpace


SIMT Model
----------
.. autosummary::
   :toctree: generated
   :nosignatures:

    thread_index
    thread_count
    block_index
    block_count
    cluster_index
    cluster_count
    block_in_cluster_index
    block_in_cluster_count
    lane_index
    lane_count
    warp_index
    full_mask
    elect_sync


Atomics
-------
.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    MemoryOrder
    MemoryScope

.. autosummary::
   :toctree: generated
   :nosignatures:

    atomic_add
    atomic_sub
    atomic_and
    atomic_or
    atomic_xor
    atomic_min
    atomic_max
    atomic_inc
    atomic_dec
    atomic_xchg
    atomic_cas


.. _operations-math:

Math
----
.. currentmodule:: cuda.lang._stub.math
.. autosummary::
   :toctree: generated
   :nosignatures:

    abs
    ceil
    floor
    exp
    exp2
    log
    log2
    sqrt
    rsqrt
    sin
    cos
    tan
    sinh
    cosh
    tanh
    atan2
    isnan
    isinf
    isfinite
    isnormal
.. currentmodule:: cuda.lang


Warp shuffle
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    shfl_sync
    shfl_up_sync
    shfl_down_sync
    shfl_xor_sync


.. _operations-tensor-map:

TensorMap
---------
.. autosummary::
   :toctree: generated
   :nosignatures:

    tensor_map_tiled

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    SwizzleMode


Synchronization
---------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    barrier_sync_warp
    barrier_sync_block
    barrier_arrive_block
    barrier_reduce_block
    barrier_arrive_cluster
    barrier_wait_cluster
    barrier_sync_cluster

    mbarrier_init
    mbarrier_invalidate
    mbarrier_arrive
    mbarrier_arrive_expect_tx
    mbarrier_expect_tx
    mbarrier_complete_tx
    mbarrier_test_wait
    mbarrier_test_wait_parity
    mbarrier_try_wait
    mbarrier_try_wait_parity

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    MbarrierScope
    BarrierReductionKind


Memory Fence
------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    memory_barrier


TensorCore (Gen5)
-----------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    tcgen05_alloc
    tcgen05_dealloc
    tcgen05_commit
    tcgen05_load
    tcgen05_store

.. autosummary::
   :toctree: generated
   :nosignatures:
   :template: autosummary/class_no_init.rst

    Tcgen05LdStShape
    CTAGroup


Cluster Launch Control
----------------------
.. autosummary::
   :toctree: generated
   :nosignatures:

    clusterlaunchcontrol_try_cancel
    clusterlaunchcontrol_is_canceled
    clusterlaunchcontrol_get_first_block_index
    griddepcontrol_wait
    griddepcontrol_launch_dependents


Utility
-------
.. autosummary::
   :toctree: generated
   :nosignatures:

    nanosleep


.. _operations-classes:

Classes
-------
.. autosummary::
   :nosignatures:

   Array
   Pointer
   Vector
   TensorMap

.. toctree::
   :hidden:

   data/array
   data/pointer
   data/vector
   data/tensor_map
