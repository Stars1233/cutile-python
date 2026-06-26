# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from multiprocessing.connection import Client

from cuda.tile.tune._tune_utils import _benchmark_worker_main


def main() -> None:
    if len(sys.argv) != 3:
        raise RuntimeError("Expected the benchmark worker connection address and authkey")
    address, authkey_hex = sys.argv[1], sys.argv[2]
    conn = Client(address, authkey=bytes.fromhex(authkey_hex))
    _benchmark_worker_main(conn)


if __name__ == "__main__":
    main()
