# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import warnings
from functools import cache


def _warn(message):
    warnings.warn(message, stacklevel=3)


def lock(config, target):
    try:
        gpu_indices = _resolve_gpu_indices(target)
    except RuntimeError as exc:
        _warn(f"failed to resolve GPU clock locking target {target!r}: {exc}")
        return

    for gpu_index in gpu_indices:
        _lock_gpu(config, gpu_index)


def reset(config):
    for gpu_index in _locked_gpu_indices(config, "_gpu_locked"):
        try:
            _run_nvidia_smi(
                ["--reset-gpu-clocks"],
                gpu_index=gpu_index,
                use_sudo=_should_use_sudo(),
            )
        except RuntimeError as exc:
            _warn(f"failed to reset GPU {gpu_index} clocks: {exc}")

    for gpu_index in _locked_gpu_indices(config, "_gpu_memory_locked"):
        try:
            _run_nvidia_smi(
                ["--reset-memory-clocks"],
                gpu_index=gpu_index,
                use_sudo=_should_use_sudo(),
            )
        except RuntimeError as exc:
            _warn(f"failed to reset GPU {gpu_index} memory clocks: {exc}")


def _lock_gpu(config, gpu_index):
    try:
        gpu_name = _query_gpu_name(gpu_index)
    except RuntimeError as exc:
        _warn(f"failed to query GPU {gpu_index} name for clock locking: {exc}")
        return

    _enable_persistence_mode(gpu_index)
    _lock_graphics_clock(config, gpu_index, gpu_name)
    _lock_memory_clock(config, gpu_index, gpu_name)


def _enable_persistence_mode(gpu_index):
    try:
        _run_nvidia_smi(
            ["--persistence-mode=1"],
            gpu_index=gpu_index,
            use_sudo=_should_use_sudo(),
        )
    except RuntimeError as exc:
        _warn(f"failed to enable GPU {gpu_index} persistence mode: {exc}")
        return

    print(f"Enabled GPU {gpu_index} persistence mode")


def _lock_graphics_clock(config, gpu_index, gpu_name):
    clock_mhz = _resolve_clock_mhz(gpu_index, gpu_name, "graphics")
    if clock_mhz is None:
        return

    try:
        _run_nvidia_smi(
            ["-lgc", f"{clock_mhz}"],
            gpu_index=gpu_index,
            use_sudo=_should_use_sudo(),
        )
    except RuntimeError as exc:
        _warn(f"failed to lock GPU {gpu_index} clocks: {exc}")
        return

    _add_locked_gpu(config, "_gpu_locked", gpu_index)
    print(f"Locked GPU {gpu_index} graphics clock to {clock_mhz} MHz")


def _lock_memory_clock(config, gpu_index, gpu_name):
    clock_mhz = _resolve_clock_mhz(gpu_index, gpu_name, "memory")
    if clock_mhz is None:
        return

    try:
        _run_nvidia_smi(
            ["-lmc", f"{clock_mhz}"],
            gpu_index=gpu_index,
            use_sudo=_should_use_sudo(),
        )
    except RuntimeError as exc:
        _warn(f"failed to lock GPU {gpu_index} memory clocks: {exc}")
        return

    _add_locked_gpu(config, "_gpu_memory_locked", gpu_index)
    print(f"Locked GPU {gpu_index} memory clock to {clock_mhz} MHz")


def _resolve_clock_mhz(gpu_index, gpu_name, clock_name):
    try:
        return _query_max_supported_clocks(gpu_index)[clock_name]
    except RuntimeError as exc:
        _warn(
            f"failed to query supported clocks for GPU {gpu_index} "
            f"({gpu_name!r}); leaving {clock_name} clocks unchanged: {exc}"
        )
        return None


@cache
def _query_max_supported_clocks(gpu_index):
    output = _run_nvidia_smi([
        "--query-supported-clocks=mem,gr",
        "--format=csv,noheader,nounits",
    ], gpu_index=gpu_index)

    memory_clocks = []
    graphics_clocks = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(
                f"Unexpected nvidia-smi supported clocks output line: {line!r}"
            )
        try:
            memory_mhz, graphics_mhz = (int(field) for field in fields)
        except ValueError as exc:
            raise RuntimeError(
                f"Unexpected nvidia-smi supported clocks output line: {line!r}"
            ) from exc
        memory_clocks.append(memory_mhz)
        graphics_clocks.append(graphics_mhz)

    if not memory_clocks or not graphics_clocks:
        raise RuntimeError(
            f"Unexpected nvidia-smi supported clocks output: {output!r}"
        )

    return {
        "memory": max(memory_clocks),
        "graphics": max(graphics_clocks),
    }


def _resolve_gpu_indices(target):
    if target != "all":
        return [target]

    output = _run_nvidia_smi([
        "--query-gpu=index",
        "--format=csv,noheader,nounits",
    ])
    gpu_indices = [line.strip() for line in output.splitlines() if line.strip()]
    if not gpu_indices:
        raise RuntimeError("nvidia-smi did not report any GPUs")
    return gpu_indices


def _query_gpu_name(gpu_index):
    return _run_nvidia_smi([
        "--query-gpu=name",
        "--format=csv,noheader",
    ], gpu_index=gpu_index)


def query_gpu_graphics_clock(gpu_index="0"):
    output = _run_nvidia_smi([
        "--query-gpu=clocks.current.sm",
        "--format=csv,noheader,nounits",
    ], gpu_index=gpu_index)
    try:
        return int(output.splitlines()[0].strip())
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Unexpected nvidia-smi graphics clock output: {output!r}"
        ) from exc


def query_gpu_memory_clock(gpu_index="0"):
    output = _run_nvidia_smi([
        "--query-gpu=clocks.current.memory",
        "--format=csv,noheader,nounits",
    ], gpu_index=gpu_index)
    try:
        return int(output.splitlines()[0].strip())
    except (IndexError, ValueError) as exc:
        raise RuntimeError(
            f"Unexpected nvidia-smi memory clock output: {output!r}"
        ) from exc


def _should_use_sudo():
    geteuid = getattr(os, "geteuid", None)
    return os.name != "nt" and geteuid is not None and geteuid() != 0


def _add_locked_gpu(config, attr, gpu_index):
    locked = getattr(config, attr, None)
    if not isinstance(locked, set):
        locked = set()
        setattr(config, attr, locked)
    locked.add(gpu_index)


def _locked_gpu_indices(config, attr):
    locked = getattr(config, attr, None)
    if locked is True:
        return ("0",)
    if not locked:
        return ()
    return tuple(sorted(locked, key=int))


def _run_nvidia_smi(args, gpu_index=None, use_sudo=False):
    command = ["nvidia-smi"]
    if gpu_index is not None:
        command.extend(["-i", gpu_index])
    command.extend(args)
    if use_sudo:
        command.insert(0, "sudo")
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        rendered = " ".join(command)
        raise RuntimeError(f"failed to start `{rendered}`: {exc}") from exc

    if result.returncode != 0:
        rendered = " ".join(command)
        details = _format_command_failure(result)
        raise RuntimeError(f"`{rendered}` failed: {details}")
    return result.stdout.strip()


def _format_command_failure(result):
    details = [f"exit code {result.returncode}"]
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if stdout:
        details.append(f"stdout: {stdout}")
    if stderr:
        details.append(f"stderr: {stderr}")
    if not stdout and not stderr:
        details.append("no stdout/stderr captured")
    return "; ".join(details)
