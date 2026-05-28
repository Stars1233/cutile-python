# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.tile as ct
import numpy as np


PAD_ZERO = ct.PaddingMode.ZERO


@ct.kernel(occupancy=ct.ByTarget(sm_100=16))
def rms_norm_kernel(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Standard RMSNorm kernel for non-static persistent mode with tiled loads"""
    row = ct.bid(0)
    _rms = ct.full((1, TILE_SIZE), 0.0, dtype=np.float32)
    num_tiles = ct.cdiv(x.shape[1], TILE_SIZE)

    for j in range(0, num_tiles):
        xj = ct.load(
            x, index=(row, j), shape=(1, TILE_SIZE),
            allow_tma=False,
            latency=1,
            padding_mode=PAD_ZERO
        )
        xj = ct.astype(xj, np.float32)
        _rms += xj * xj

    # Calculate RMS Norm
    rms = ct.rsqrt(ct.sum(_rms, axis=1, keepdims=False) / N + eps)
    ct.store(Rstd, index=(row,), tile=rms)

    for j in range(0, num_tiles):
        wj = ct.load(
            w, index=(j,), shape=(TILE_SIZE,),
            allow_tma=False,
            latency=1,
        )
        wj = ct.astype(wj, np.float32)
        xj = ct.load(
            x, index=(row, j), shape=(1, TILE_SIZE),
            allow_tma=False,
            latency=1,
        )
        xj = ct.astype(xj, np.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.store(
            out, index=(row, j), tile=yj,
            allow_tma=False,
            latency=1,
        )


@ct.kernel
def rms_norm_kernel_gather(
    x,
    w,
    out,
    Rstd,
    N: ct.Constant[int],
    eps: ct.Constant[float],
    TILE_SIZE: ct.Constant[int],
):
    """Standard RMSNorm kernel for non-static persistent mode with ptr loads"""
    row = ct.bid(0)
    _rms = ct.full((TILE_SIZE,), 0.0, dtype=np.float32)
    num_tiles = ct.cdiv(N, TILE_SIZE)
    offsets = ct.arange(TILE_SIZE, dtype=np.int32)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, np.float32)
        _rms += xj * xj

    # Calculate RMS Norm
    rms = ct.rsqrt(ct.sum(_rms, axis=0, keepdims=False) / N + eps)
    ct.scatter(Rstd, row, rms)

    for j in range(0, num_tiles):
        offs = j * TILE_SIZE + offsets
        wj = ct.gather(w, offs, latency=1)
        wj = ct.astype(wj, np.float32)
        xj = ct.gather(x, (row, offs), latency=1)
        xj = ct.astype(xj, np.float32)
        yj = xj * rms * wj
        yj = ct.astype(yj, x.dtype)
        ct.scatter(out, (row, offs), yj, latency=1)


@ct.kernel
def rms_norm_kernel_static_persistent(
    X,  # Input tensor
    Y,  # Output tensor
    W,  # Weight tensor
    TILE_SIZE_M: ct.Constant[int],  # rows per tile
    TILE_SIZE_N: ct.Constant[int],  # columns per tile
    N: ct.Constant[int],
    eps: ct.Constant[float],  # Epsilon value
):
    """
    CuTile static persistent RMSNorm kernel that uses a persistent approach,
    where NUM_SMS tile blocks are launched and each tile block processes multiple output tiles
    for better efficiency.
    """
    # Get program ID
    bid = ct.bid(0)

    # Infer tensor dimensions from input shape
    M = X.shape[0]  # Number of rows

    # Calculate upper bound
    upper_bound = (M + TILE_SIZE_M - 1) // TILE_SIZE_M
    num_tiles_n = ct.cdiv(N, TILE_SIZE_N)

    # Load W once
    w = ct.load(W, index=(0,), shape=(N,), padding_mode=PAD_ZERO)

    # Static persistent loop: each  processes multiple tiles
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        x2_sum = ct.full((TILE_SIZE_M, 1), 0.0, dtype=np.float32)
        for j in range(0, num_tiles_n):
            # Load input tile
            x = ct.load(
                X, index=(current_bid, j), shape=(TILE_SIZE_M, TILE_SIZE_N),
                latency=10,
                padding_mode=PAD_ZERO,
            )
            x = ct.astype(x, np.float32)

            # Step 1: Compute x^2
            x_squared = ct.mul(x, x)

            # Step 2: Reduce sum along axis=1 (columns)
            x2_sum += ct.sum(
                x_squared, axis=1, keepdims=True
            )  # Shape: [TILE_SIZE_M, 1]

        # Step 3: Compute variance (divide by N)
        N_f32 = ct.full((TILE_SIZE_M, 1), N * 1.0, dtype=np.float32)
        variance = ct.truediv(x2_sum, N_f32)

        # Step 4: Add epsilon and compute rsqrt
        eps_tensor = ct.full((TILE_SIZE_M, 1), eps, dtype=np.float32)
        variance_eps = ct.add(variance, eps_tensor)
        rsqrt = ct.rsqrt(variance_eps)

        for j in range(0, num_tiles_n):
            # Load input and weight tiles
            x = ct.load(
                X, index=(current_bid, j), shape=(TILE_SIZE_M, TILE_SIZE_N),
                latency=10,
                padding_mode=PAD_ZERO,
            )
            x = ct.astype(x, np.float32)

            # Step 5: Apply normalization
            x_normalized = x * rsqrt

            # Step 6: Apply linear transformation
            w_sub = w.extract((j,), (TILE_SIZE_N,))
            y = x_normalized * w_sub

            # Convert back to original dtype
            y = ct.astype(y, X.dtype)

            # Store result
            ct.store(
                Y, index=(current_bid, j), tile=y,
                allow_tma=False,
                latency=3,
            )
