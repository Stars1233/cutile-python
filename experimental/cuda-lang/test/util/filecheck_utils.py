# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
import tempfile
import inspect
from types import FunctionType


def _find_filecheck_bin() -> str:
    filecheck_path = shutil.which("FileCheck")
    if filecheck_path:
        return filecheck_path
    raise FileNotFoundError("'FileCheck' not found")


def get_source(scrutinee: FunctionType | None = None) -> str:
    match scrutinee:
        case FunctionType() as func:
            # Get source of function passed in directly
            return inspect.getsource(func)
        case None:
            # Get the caller's source
            return inspect.getsource(inspect.currentframe().f_back.f_code)
        case _:
            raise ValueError(f"Could not get source from {scrutinee=}")


def filecheck(
    text: str, check_directives: str, check_prefixes: tuple[str, ...] = ("CHECK",)
) -> None:
    filecheck_bin = _find_filecheck_bin()
    with (
        tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as check_file,
        tempfile.NamedTemporaryFile(suffix=".mlir", mode="w") as input_file,
    ):
        check_file.write(check_directives)
        check_file.flush()
        input_file.write(text)
        input_file.flush()
        result = subprocess.run(
            [
                filecheck_bin,
                "-input-file",
                input_file.name,
                check_file.name,
                "--check-prefixes",
                ",".join(check_prefixes),
            ],
            check=False,
            capture_output=True,
        )
        assert result.returncode == 0, f"FileCheck failed:\n{result.stderr.decode()}"
