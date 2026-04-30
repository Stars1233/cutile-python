# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
import torch

"""
cuda.lang port of the CUDA sample:
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/2_Concepts_and_Techniques/convolutionSeparable/README.md

Both the CUDA version and the CPU version were replicated here.
Note that the CUDA version uses cooperative groups to more selectively
synchronize, but we just use syncthreads.

The original example also uses `pitch` for tile size, but then
only tests with a tile size of the image width.

Rather than incrementing the base of the shared memory pointer
like the CUDA version does, we just use the offset in the index
calculation. I found this easier to read.
"""

KERNEL_RADIUS = 8
KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

ROWS_BLOCKDIM_X = 16
ROWS_BLOCKDIM_Y = 4
ROWS_RESULT_STEPS = 8
ROWS_HALO_STEPS = 1

COLUMNS_BLOCKDIM_X = 16
COLUMNS_BLOCKDIM_Y = 8
COLUMNS_RESULT_STEPS = 8
COLUMNS_HALO_STEPS = 1

IMAGE_W = 128
IMAGE_H = 64


def cpu_separable_conv2d(src: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    tmp = torch.zeros_like(src)
    out = torch.zeros_like(src)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            acc = 0.0
            for k in range(KERNEL_LENGTH):
                src_x = x + k - KERNEL_RADIUS
                if 0 <= src_x < src.shape[1]:
                    acc += float(kernel[k]) * float(src[y, src_x])
            tmp[y, x] = acc

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            acc = 0.0
            for k in range(KERNEL_LENGTH):
                src_y = y + k - KERNEL_RADIUS
                if 0 <= src_y < src.shape[0]:
                    acc += float(kernel[k]) * float(tmp[src_y, x])
            out[y, x] = acc

    return out


def test_convolution_separable():

    @cl.kernel
    def convolution_rows(
        dst,
        src,
        kernel,
        image_w: cl.Constant[int],
        image_h: cl.Constant[int],
        pitch: cl.Constant[int],
    ):
        tile = cl.shared_array(
            shape=(
                ROWS_BLOCKDIM_Y,
                (ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X,
            ),
            dtype=cl.float32,
        )

        bx, by, _ = cl.block_idx()
        tx, ty, _ = cl.thread_idx()

        base_x = (bx * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + tx
        base_y = by * ROWS_BLOCKDIM_Y + ty
        row_offset = base_y * pitch
        src_offset = row_offset + base_x

        for step in range(ROWS_HALO_STEPS, ROWS_HALO_STEPS + ROWS_RESULT_STEPS):
            tile[ty, tx + step * ROWS_BLOCKDIM_X] = src[src_offset + step * ROWS_BLOCKDIM_X]

        for step in range(ROWS_HALO_STEPS):
            src_x = base_x + step * ROWS_BLOCKDIM_X
            tile[ty, tx + step * ROWS_BLOCKDIM_X] = (
                src[row_offset + src_x] if src_x >= 0 else cl.float32(0.0)
            )

        for step in range(
            ROWS_HALO_STEPS + ROWS_RESULT_STEPS,
            ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS,
        ):
            src_x = base_x + step * ROWS_BLOCKDIM_X
            tile[ty, tx + step * ROWS_BLOCKDIM_X] = (
                src[row_offset + src_x] if src_x < image_w else cl.float32(0.0)
            )

        cl.syncthreads()

        for step in range(ROWS_HALO_STEPS, ROWS_HALO_STEPS + ROWS_RESULT_STEPS):
            acc = cl.float32(0.0)
            tile_x = tx + step * ROWS_BLOCKDIM_X
            for k in range(KERNEL_LENGTH):
                acc = acc + kernel[k] * tile[ty, tile_x + k - KERNEL_RADIUS]
            dst[src_offset + step * ROWS_BLOCKDIM_X] = acc

    @cl.kernel
    def convolution_columns(
        dst,
        src,
        kernel,
        image_w: cl.Constant[int],
        image_h: cl.Constant[int],
        pitch: cl.Constant[int],
    ):
        tile = cl.shared_array(
            shape=(
                COLUMNS_BLOCKDIM_X,
                (COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1,
            ),
            dtype=cl.float32,
        )

        bx, by, _ = cl.block_idx()
        tx, ty, _ = cl.thread_idx()

        base_x = bx * COLUMNS_BLOCKDIM_X + tx
        base_y = (by * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + ty
        src_offset = base_y * pitch + base_x

        for step in range(COLUMNS_HALO_STEPS, COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS):
            tile[tx, ty + step * COLUMNS_BLOCKDIM_Y] = src[
                src_offset + step * COLUMNS_BLOCKDIM_Y * pitch
            ]

        for step in range(COLUMNS_HALO_STEPS):
            src_y = base_y + step * COLUMNS_BLOCKDIM_Y
            tile[tx, ty + step * COLUMNS_BLOCKDIM_Y] = (
                src[src_y * pitch + base_x] if src_y >= 0 else cl.float32(0.0)
            )

        for step in range(
            COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS,
            COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS,
        ):
            src_y = base_y + step * COLUMNS_BLOCKDIM_Y
            tile[tx, ty + step * COLUMNS_BLOCKDIM_Y] = (
                src[src_y * pitch + base_x] if src_y < image_h else cl.float32(0.0)
            )

        cl.syncthreads()

        for step in range(COLUMNS_HALO_STEPS, COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS):
            acc = cl.float32(0.0)
            tile_y = ty + step * COLUMNS_BLOCKDIM_Y
            for k in range(KERNEL_LENGTH):
                acc = acc + kernel[k] * tile[tx, tile_y + k - KERNEL_RADIUS]
            dst[src_offset + step * COLUMNS_BLOCKDIM_Y * pitch] = acc

    src = (
        torch.arange(IMAGE_W * IMAGE_H, dtype=torch.float32, device="cuda")
        .reshape(IMAGE_H, IMAGE_W)
        / 64.0
    )
    kernel = torch.linspace(-1.0, 1.0, steps=KERNEL_LENGTH, dtype=torch.float32, device="cuda")
    tmp = torch.zeros(IMAGE_W * IMAGE_H, dtype=torch.float32, device="cuda")
    dst = torch.zeros(IMAGE_W * IMAGE_H, dtype=torch.float32, device="cuda")

    cl.launch(
        torch.cuda.current_stream(),
        (IMAGE_W // (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), IMAGE_H // ROWS_BLOCKDIM_Y),
        (ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y),
        convolution_rows,
        (tmp, src.reshape(-1), kernel, IMAGE_W, IMAGE_H, IMAGE_W),
    )
    cl.launch(
        torch.cuda.current_stream(),
        (IMAGE_W // COLUMNS_BLOCKDIM_X, IMAGE_H // (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y)),
        (COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y),
        convolution_columns,
        (dst, tmp, kernel, IMAGE_W, IMAGE_H, IMAGE_W),
    )
    torch.cuda.synchronize()

    expected = cpu_separable_conv2d(src.cpu(), kernel.cpu())
    assert torch.allclose(dst.reshape(IMAGE_H, IMAGE_W).cpu(), expected, atol=1e-5, rtol=1e-5)
