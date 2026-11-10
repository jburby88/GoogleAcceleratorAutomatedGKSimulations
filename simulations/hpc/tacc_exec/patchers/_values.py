"""Strict value handling shared by the GX (`tomlkit`) and GENE (`f90nml`) patchers.

Fail closed: a value that is not exactly representable as the parameter's type is rejected,
never truncated or coerced (`int(64.7) == 64` and `float(True) == 1.0` are silent corruptions).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

# Restart-file names written into the run directory must be plain names (no paths). Same
# grammar as the GCP task runner (`gcp_exec/task_runner.py::_SAFE_NAME_RE`).
SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,200}$")

# Parameters that must be strictly positive (grid sizes, timestep, Courant factor) or
# non-negative (dissipation coefficient). Validity constraints only; the allowlist bounds in
# config `allowed_parameters` are enforced by the watchdog before any patch.
STRICTLY_POSITIVE_PARAMS = frozenset({"nx", "ny", "nz", "dt_max", "cfl_fraction"})
NON_NEGATIVE_PARAMS = frozenset({"hyperdiffusion_coeff"})


def coerce_number(kind: type, value: Any, label: str) -> int | float:
    """Return `value` as `kind` (int or float) or raise ValueError; never truncates."""
    if hasattr(value, "unwrap"):
        value = value.unwrap()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number, got {type(value).__name__} {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{label} must be finite, got {value!r}")
    if kind is int:
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"{label} must be an integer, got {value!r}")
        return int(value)
    return as_float


def validate_modifications(
    modifications: Mapping[str, Any],
    param_map: Mapping[str, tuple[str, str, type]],
    solver: str,
) -> dict[str, int | float]:
    """Validate a patch request against a solver parameter map; return typed values.

    Raises ValueError for non-allowlisted names, booleans, non-numeric, non-finite,
    non-integral (for int parameters) or non-positive (where required) values.
    """
    if not isinstance(modifications, Mapping):
        raise ValueError(f"{solver} modifications must be a mapping, got {type(modifications)}")
    typed: dict[str, int | float] = {}
    for name, raw in modifications.items():
        if name not in param_map:
            raise ValueError(f"Forbidden {solver} parameter modification: {name!r}")
        kind = param_map[name][2]
        value = coerce_number(kind, raw, f"{solver} parameter {name!r}")
        if name in STRICTLY_POSITIVE_PARAMS and value <= 0:
            raise ValueError(f"{solver} parameter {name!r} must be > 0, got {raw!r}")
        if name in NON_NEGATIVE_PARAMS and value < 0:
            raise ValueError(f"{solver} parameter {name!r} must be >= 0, got {raw!r}")
        typed[name] = value
    return typed


def changed_leaf_paths(
    before: Any, after: Any, prefix: tuple[str, ...] = ()
) -> set[tuple[str, ...]]:
    """Return the leaf paths whose value differs between two nested dicts (added/removed too)."""
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        paths: set[tuple[str, ...]] = set()
        for key in set(before) | set(after):
            path = (*prefix, str(key))
            if key not in before or key not in after:
                paths |= _all_leaf_paths(after.get(key) if key in after else before[key], path)
            else:
                paths |= changed_leaf_paths(before[key], after[key], path)
        return paths
    return set() if before == after else {prefix}


def _all_leaf_paths(value: Any, prefix: tuple[str, ...]) -> set[tuple[str, ...]]:
    if isinstance(value, Mapping) and value:
        paths: set[tuple[str, ...]] = set()
        for key, sub in value.items():
            paths |= _all_leaf_paths(sub, (*prefix, str(key)))
        return paths
    return {prefix}


def values_match(expected: int | float | bool, actual: Any) -> bool:
    """Compare a requested value with what the re-parsed file contains."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected is actual
    if not isinstance(actual, (int, float)):
        return False
    if isinstance(expected, int):
        return isinstance(actual, int) and actual == expected
    return math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=0.0)


def ensure_only_allowed_changes(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    allowed: Iterable[tuple[str, ...]],
    solver: str,
) -> None:
    """Raise ValueError if `after` differs from `before` outside the `allowed` leaf paths."""
    unexpected = changed_leaf_paths(before, after) - set(allowed)
    if unexpected:
        rendered = sorted(".".join(p) for p in unexpected)
        raise ValueError(f"{solver} patch would change non-allowlisted keys: {rendered}")
