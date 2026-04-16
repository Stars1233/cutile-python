# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, TypeVar

from cuda.tile._cext import (
    _benchmark,
    _synchronize_context,
)
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class TuningResult(Generic[T]):
    """Holds configurations and their timing result."""

    best_config: T
    """Config with the smallest timing"""

    best_time_us: float
    """Time (microseconds) for the best config"""

    timings: Sequence[tuple[T, float]]
    """`(config, time_us)` for each successful config"""

    errors: Sequence[tuple[T, str, str]]
    """`(config, exc_type, message)` for each failed config"""

    def summary(self, *, top_k=10) -> str:
        """Return a summary of the result.

        Args:
            top_k (int): Max number of configs to be included, sorted by timing.
        """

        n_ok = len(self.timings)
        n_fail = len(self.errors)
        header = f"{n_ok} succeeded, {n_fail} failed"
        if self.best_config is not None:
            header += f", best: {self.best_config} ({self.best_time_us:.1f} us)"
        lines = [header]
        ranked = sorted(self.timings, key=lambda t: t[1])
        for cfg, time_us in ranked[:top_k]:
            marker = "*" if cfg == self.best_config else " "
            lines.append(f"  {marker} {cfg}: {time_us:.1f} us")
        if n_ok > top_k:
            lines.append(f"    ... {n_ok - top_k} more not shown")
        if self.errors:
            lines.append(f"  {n_fail} failed:")
            for cfg, err_type, msg in self.errors[:top_k]:
                first_line = msg.split("\n", 1)[0]
                if len(first_line) > 60:
                    first_line = first_line[:57] + "..."
                lines.append(f"    {cfg}: {err_type}: {first_line}")
            if n_fail > top_k:
                lines.append(f"    ... {n_fail - top_k} more not shown")
        if n_ok > top_k or n_fail > top_k:
            lines.append("Use .timings and .errors for full results.")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()


def exhaustive_search(
    search_space: Sequence[T],
    stream,
    grid_fn: Callable[[T], tuple[int, ...]],
    kernel,
    args_fn: Callable[[T], tuple[Any, ...]],
    hints_fn: Callable[[T], dict[str, Any]] | None = None,
    *,
    callback: Callable[..., None] | None = None,
) -> TuningResult[T]:
    """Searches the entire search space and return the best configuration.

    Args:
        search_space: Sequence of configs to evaluate.
        stream: The CUDA stream to execute kernel on.
        grid_fn: Maps a config to grid dimensions.
        kernel: The kernel to tune.
        args_fn: Maps a config to kernel arguments for timing.
        hints_fn: Maps a config to compiler hints. Default: no hints.
        callback: Called after each config evaluation.

            - On success: ``callback(config, <time_us>, None)``
            - On error:   ``callback(config, None, (<err_type>, <err_msg>))``

    Returns:
        TuningResult with the best config and its time in microseconds.

    Examples:

    .. testcode::
        :template: setup_only.py

        # Define the kernel

        @ct.kernel
        def matmul(X, Y, Out,
                   tm: ct.Constant[int],
                   tn: ct.Constant[int],
                   tk: ct.Constant[int]):

            i, j =  ct.bid(0), ct.bid(1)

            x_view = X.tiled_view((tm, tk), padding_mode=ct.PaddingMode.ZERO)
            y_view = Y.tiled_view((tk, tn), padding_mode=ct.PaddingMode.ZERO)
            acc = ct.zeros((tm, tn), ct.float32)
            for k in range(x_view.num_tiles(1)):
                tx = x_view.load((i, k))
                ty = y_view.load((k, j))
                acc = ct.mma(tx, ty, acc)
            ct.store(Out, (i, j), acc.astype(Out.dtype))

        # Tune the kernel

        from itertools import product

        def tune(x, y, out) -> ct.tune.TuningResult:
            keys = ("tm", "tn", "tk", "num_ctas")
            search_space = [dict(zip(keys, vals))
                            for vals in product(
                            (64, 128),
                            (64, 128),
                            (32, 64),
                            (1, 2))]
            grid = lambda cfg: (ct.cdiv(M, cfg['tm']), ct.cdiv(N, cfg['tn']))
            args = lambda cfg: (x, y, out.clone(), cfg['tm'], cfg['tn'], cfg['tk'])
            hints = lambda cfg: {'num_ctas': cfg['num_ctas']}
            stream = torch.cuda.current_stream()
            tuning_result = ct.tune.exhaustive_search(search_space,
                                                      stream,
                                                      grid,
                                                      matmul,
                                                      args,
                                                      hints,
                                                      callback=lambda cfg, timing, error:
                                                        print('x' if error else '.', end='')
                                                     )
            print()
            return tuning_result

        M, N, K = 1024, 256, 512
        x = torch.rand((M, K), dtype=torch.float16, device='cuda')
        y = torch.rand((K, N), dtype=torch.float16, device='cuda')
        out = torch.zeros((M, N), dtype=torch.float16, device='cuda')

        result = tune(x, y, out)
        print(f"Best config: {result.best_config} ({result.best_time_us:.1f}us)")

        # Launch the kernel with tuned result

        tm, tn, tk, num_ctas = result.best_config.values()
        kernel = matmul.replace_hints(num_ctas=num_ctas)
        ct.launch(torch.cuda.current_stream(),
                  (ct.cdiv(M, tm), ct.cdiv(N, tn)),
                  kernel,
                  (x, y, out, tm, tn, tk))

        torch.testing.assert_close(out, x @ y)

    .. testoutput::

       ................
       Best config: {'tm': ..., 'tn': ..., 'tk': ..., 'num_ctas': ...} (...us)
    """

    timings = []
    errors = []

    best_time_us = float("inf")
    best_cfg = None

    for cfg in search_space:
        grid = grid_fn(cfg)
        hints = hints_fn(cfg) if hints_fn is not None else {}
        updated_kernel = kernel.replace_hints(**hints)

        try:
            avg_us, _ = _time_us(
                stream, grid, updated_kernel,
                lambda _cfg=cfg: args_fn(_cfg),
            )
        except Exception as e:
            err_type = type(e).__name__
            msg = str(e)
            errors.append((cfg, err_type, msg))
            if callback is not None:
                callback(cfg, None, (err_type, msg))
            continue

        timings.append((cfg, avg_us))
        if callback is not None:
            callback(cfg, avg_us, None)

        if avg_us < best_time_us:
            best_time_us = avg_us
            best_cfg = cfg

    if len(search_space) == 0:
        raise ValueError("Search space is empty.")
    elif best_cfg is None:
        cfg, exc_type, msg = errors[0]
        raise ValueError(f"No valid config found in search space."
                         f"\nConfig: {cfg}\n{exc_type}: {msg}")

    result = TuningResult(
        best_config=best_cfg,
        best_time_us=best_time_us,
        timings=tuple(timings),
        errors=tuple(errors),
    )

    return result


_MAX_MEASURE_TIME_US = 5_000_000  # 5s
_MAX_REPEATS = 500
_WARM_UP_STEPS = 10
_PILOT_STEPS = 10


def _time_us(stream, grid, kernel, get_args) -> tuple[float, int]:
    _synchronize_context()

    # Warmup
    pilot_time = 0.0
    for i in range(_WARM_UP_STEPS + _PILOT_STEPS):
        t = _benchmark(stream, grid, kernel, get_args())
        if i >= _WARM_UP_STEPS:
            pilot_time += t

    _synchronize_context()

    # Estimate number of repeats
    avg_time = pilot_time / _PILOT_STEPS
    repeats = min(pilot_time, _MAX_MEASURE_TIME_US) // (avg_time + 1e-5)
    repeats = min(max(1, int(repeats)), _MAX_REPEATS)

    # Run benchmark `repeats` number of times
    final_time = sum(_benchmark(stream, grid, kernel, get_args()) for _ in range(repeats))
    return final_time / repeats, repeats
