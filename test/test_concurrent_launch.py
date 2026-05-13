# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
import subprocess
import threading
import torch
import cuda.tile as ct
import cuda.tile._cext as _cext
import traceback
import pytest


@ct.kernel
def vector_add(x, y, z, TILE: ct.Constant[int]):
    bid = ct.bid(0)
    tx = ct.load(x, index=(bid,), shape=(TILE,))
    ty = ct.load(y, index=(bid,), shape=(TILE,))
    ct.store(z, index=(bid,), tile=tx + ty)


_TILES = (32, 64, 128, 256)

_LAUNCH_FUNCS = {
    "ct.launch": ct.launch,
    "_cext.launch": _cext.launch,
    "_cext._benchmark": _cext._benchmark,
}


def _worker(thread_id, n_iters, x, y, expected, errors, launch_func):
    try:
        for i in range(n_iters):
            tile = _TILES[(thread_id + i) % len(_TILES)]
            grid = (x.shape[0] // tile, 1, 1)
            z = torch.zeros_like(x)
            launch_func(torch.cuda.current_stream(), grid, vector_add, (x, y, z, tile))
            torch.cuda.synchronize()
            if not torch.equal(z, expected):
                errors.append(f"thread={thread_id} iter={i} tile={tile}:"
                              f" result mismatch: {z} != {expected}")
                return
    except Exception:
        errors.append(f"thread={thread_id} exception: {traceback.format_exc()}")


def _run_concurrent_launch(launch_func_name):
    launch_func = _LAUNCH_FUNCS[launch_func_name]
    N_THREADS = 8
    N_ITERS = 800
    N_ELEMS = 8192

    x = torch.randint(0, 100, (N_ELEMS,), device='cuda', dtype=torch.int32)
    y = torch.randint(0, 100, (N_ELEMS,), device='cuda', dtype=torch.int32)
    expected = x + y

    errors = []
    threads = []
    for t in range(N_THREADS):
        t = threading.Thread(target=_worker, args=(t, N_ITERS, x, y, expected, errors, launch_func))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if errors:
        print("concurrent launch errors:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)


@pytest.mark.parametrize("launch_func_name", list(_LAUNCH_FUNCS))
def test_concurrent_launch(launch_func_name):
    # Free-threading mode may result in segfaults, run in a subprocess to isolate the test.
    result = subprocess.run(
        [sys.executable, __file__, "run_concurrent_launch", launch_func_name], capture_output=True,
    )
    if result.returncode != 0:
        print("--- Captured stdout ---")
        print(result.stdout.decode(errors='replace'))
        print("--- End Captured stdout ---\n--- Captured stderr ---")
        print(result.stderr.decode(errors='replace'))
        print("--- End Captured stderr ---")
        pytest.fail(f"concurrent launch subprocess failed with return code {result.returncode}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run_concurrent_launch":
        _run_concurrent_launch(sys.argv[2])
