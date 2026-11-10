"""Pure time-series statistics for GK run diagnostics (shared by TACC and GCP adapters).

Implements the domain-check verdicts in docs/review/DOMAIN.md:

- PHYS-01 (`dt_collapse_stats`): a robust timestep-collapse ratio. The baseline is the median
  of the early post-initialization steps `dts[1:6]` and the recent value is the median of the
  last 50 steps, so a single transient eddy spike cannot trigger TIMESTEP_COLLAPSE. Logs with
  fewer than 6 steps keep the legacy `min(dts) / dts[0]` ratio.
- PHYS-02 (`flux_window_stats`): the heat flux label is the time average over the
  post-transient window `t >= transient_discard_fraction * t_reached` (default: second half of
  the run), with its sample standard deviation, the last sample, and the relative drift
  between the first and second half of the window (a secularly growing, un-saturated
  trajectory has a large drift and must not be called converged).

Stdlib only (TACC login-node allowlist). Callers decide thresholds; these functions only
compute statistics. Non-finite samples are ignored here; callers flag them as DIVERGENCE.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

MIN_WINDOW_SAMPLES = 4
DT_BASELINE_SLICE = slice(1, 6)
DT_RECENT_SAMPLES = 50


@dataclass(frozen=True)
class DtCollapseStats:
    """Baseline vs. recent timestep and their ratio (`recent_dt / initial_dt`)."""

    initial_dt: float
    recent_dt: float
    ratio: float
    method: str  # "median_window" (>= 6 steps) or "min_over_first" (< 6 steps, legacy)


@dataclass(frozen=True)
class FluxWindowStats:
    """Post-transient window statistics of a flux time series."""

    mean: float
    std: float  # sample standard deviation (ddof=1) over the window
    last: float
    drift: float  # |mean(second half) - mean(first half)| / max(|mean|, 1e-12)
    n_samples: int
    t_start: float
    t_end: float


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def dt_collapse_stats(dts: Sequence[float]) -> DtCollapseStats | None:
    """Return the robust PHYS-01 timestep ratio, or None if there is no usable dt sample.

    Only finite, strictly positive timesteps are used.
    """
    clean = [float(d) for d in dts if _finite(d) and float(d) > 0.0]
    if not clean:
        return None
    if len(clean) >= 6:
        initial = statistics.median(clean[DT_BASELINE_SLICE])
        recent = statistics.median(clean[-DT_RECENT_SAMPLES:])
        method = "median_window"
    else:
        initial = clean[0]
        recent = min(clean)
        method = "min_over_first"
    if initial <= 0.0:
        return None
    return DtCollapseStats(
        initial_dt=initial, recent_dt=recent, ratio=recent / initial, method=method
    )


def flux_window_stats(
    times: Sequence[float],
    values: Sequence[float],
    *,
    transient_discard_fraction: float = 0.5,
    min_samples: int = MIN_WINDOW_SAMPLES,
) -> FluxWindowStats | None:
    """Return PHYS-02 post-transient statistics, or None if fewer than `min_samples` exist.

    `times` and `values` are paired element-wise (extra elements of the longer one are
    ignored); pairs with a non-finite time or value are dropped. Samples are used in the
    given (chronological) order. The window is `t >= transient_discard_fraction * t_reached`
    where `t_reached` is the largest time; if it holds fewer than `min_samples` samples, the
    last `max(min_samples, n // 2)` samples are used instead.
    """
    if not 0.0 <= transient_discard_fraction < 1.0:
        raise ValueError("transient_discard_fraction must be in [0, 1)")
    min_samples = max(2, int(min_samples))
    pairs = [
        (float(t), float(v))
        for t, v in zip(times, values, strict=False)
        if _finite(t) and _finite(v)
    ]
    if len(pairs) < min_samples:
        return None
    t_reached = max(t for t, _ in pairs)
    t_cut = transient_discard_fraction * t_reached
    window = [(t, v) for t, v in pairs if t >= t_cut]
    if len(window) < min_samples:
        window = pairs[-max(min_samples, len(pairs) // 2) :]
    vals = [v for _, v in window]
    # Samples near the float limit (a run blowing up just before NaN) make these raise
    # OverflowError; callers must treat that as a divergence, never as a label.
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    half = len(vals) // 2
    mean_first = statistics.fmean(vals[:half])
    mean_second = statistics.fmean(vals[half:])
    drift = abs(mean_second - mean_first) / max(abs(mean), 1e-12)
    if not all(math.isfinite(x) for x in (mean, std, drift)):
        raise OverflowError("flux window statistics overflow the float range")
    return FluxWindowStats(
        mean=mean,
        std=std,
        last=vals[-1],
        drift=drift,
        n_samples=len(vals),
        t_start=window[0][0],
        t_end=window[-1][0],
    )
