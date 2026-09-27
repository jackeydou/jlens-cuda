"""Flag-gated per-stage timers for the decode hot path.

Off by default; enable with JLENS_PERF=1. When enabled, `mark` synchronizes
the GPU so asynchronously launched work is attributed to the stage that
launched it — this adds sync points, so per-stage numbers slightly inflate
the total. Measure end-to-end baselines with the flag OFF.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict

import torch

ENABLED = os.environ.get("JLENS_PERF") == "1"
_acc: dict[str, list[float]] = defaultdict(list)


def begin() -> float | None:
    """Start a stage clock. Returns None (no-op) when disabled."""
    if not ENABLED:
        return None
    return time.perf_counter()


def mark(t0: float | None, name: str, *arrays) -> float | None:
    """End the current stage: synchronize, record elapsed time under `name`,
    and start the next stage clock."""
    if t0 is None:
        return None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    now = time.perf_counter()
    _acc[name].append(now - t0)
    return now


def report() -> dict[str, dict[str, float]]:
    """Per-stage stats in ms: median, mean, count. Resets the accumulator."""
    out = {}
    for name, xs in sorted(_acc.items()):
        s = sorted(xs)
        out[name] = {
            "median_ms": 1e3 * s[len(s) // 2],
            "mean_ms": 1e3 * sum(s) / len(s),
            "n": len(s),
        }
    _acc.clear()
    return out
