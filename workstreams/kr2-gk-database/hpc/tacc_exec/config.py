"""Executor config loading and validation shared by the tacc-* CLIs (stdlib only).

Relative paths in the config (`queue_dir`, `state_dir`, `dataset_output_dir`) are resolved
against the directory that contains the config file, never against the process CWD: cron
starts jobs in `$HOME`, so CWD-relative paths would silently point the driver at a different
queue than the one operators fill.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from tacc_exec.patchers.gene_namelist import GENE_PARAM_MAP
from tacc_exec.patchers.gx_toml import GX_PARAM_MAP

PATH_KEYS = ("queue_dir", "state_dir", "dataset_output_dir")
_VALID_TYPES = {"int", "float"}
# PHYS-03 (docs/review/DOMAIN.md) + aurora-steer allowed-parameters.md `warm_restart_allowed`:
# a checkpoint stores the phase-space array, so only pure timestep controls may change when a
# run resumes from it.
WARM_RESTART_ALLOWED_PARAMS: frozenset[str] = frozenset({"dt_max", "cfl_fraction"})


class ConfigError(ValueError):
    """Raised when the executor config is unusable. The driver must not start."""


def _resolve_path(key: str, raw: Any, base_dir: Path) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError(f"Config {key!r} must be a non-empty string path")
    expanded = os.path.expandvars(os.path.expanduser(raw.strip()))
    if "$" in expanded:
        raise ConfigError(
            f"Config {key!r}={raw!r} references an unset environment variable (cron does not "
            "load your login environment); use an absolute path"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def validate_allowed_parameters(allowed: Any) -> dict[str, dict[str, Any]]:
    """Validate the client-side allowlist mirror; raise ConfigError on any inconsistency.

    Every entry needs type int|float, finite min <= max, and gene_path/gx_path that match the
    patcher maps exactly, so a directive that passes the client guardrail can always be
    applied to both solvers, and config drift (e.g. a stale GX key) stops the driver early.
    """
    if not isinstance(allowed, dict):
        raise ConfigError("allowed_parameters must be an object")
    problems: list[str] = []
    for name, spec in allowed.items():
        if not isinstance(spec, dict):
            problems.append(f"{name}: spec must be an object")
            continue
        if spec.get("type") not in _VALID_TYPES:
            problems.append(f"{name}: type must be one of {sorted(_VALID_TYPES)}")
        bounds = []
        for bound in ("min", "max"):
            value = spec.get(bound)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                problems.append(f"{name}: {bound} must be a number")
            elif not math.isfinite(float(value)):
                problems.append(f"{name}: {bound} must be finite")
            else:
                bounds.append(float(value))
        if len(bounds) == 2 and bounds[0] > bounds[1]:
            problems.append(f"{name}: min > max")
        warm = spec.get("warm_restart_allowed", False)
        if not isinstance(warm, bool):
            problems.append(f"{name}: warm_restart_allowed must be a boolean")
        elif warm and name not in WARM_RESTART_ALLOWED_PARAMS:
            problems.append(
                f"{name}: warm_restart_allowed=true is only valid for "
                f"{sorted(WARM_RESTART_ALLOWED_PARAMS)} (checkpoints fix grids and dissipation)"
            )
        if name not in GX_PARAM_MAP or name not in GENE_PARAM_MAP:
            problems.append(f"{name}: no patcher mapping for both GX and GENE")
            continue
        gx_section, gx_key, _ = GX_PARAM_MAP[name]
        gene_group, gene_key, _ = GENE_PARAM_MAP[name]
        if spec.get("gx_path") is not None and spec.get("gx_path") != f"{gx_section}.{gx_key}":
            problems.append(
                f"{name}: gx_path {spec.get('gx_path')!r} != patcher {gx_section}.{gx_key}"
            )
        if (
            spec.get("gene_path") is not None
            and spec.get("gene_path") != f"{gene_group}.{gene_key}"
        ):
            problems.append(
                f"{name}: gene_path {spec.get('gene_path')!r} != patcher {gene_group}.{gene_key}"
            )
    if problems:
        raise ConfigError("Invalid allowed_parameters: " + "; ".join(problems))
    return allowed


def normalize_config(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Return a copy of `config` with absolute paths and a validated allowlist."""
    if not isinstance(config, dict):
        raise ConfigError("Config root must be a JSON object")
    cfg = dict(config)
    for key in PATH_KEYS:
        if cfg.get(key) is not None:
            cfg[key] = _resolve_path(key, cfg[key], base_dir)
    validate_allowed_parameters(cfg.get("allowed_parameters", {}))
    environment = cfg.get("environment", "prod")
    if environment not in ("prod", "sim", "synthetic", "preflight"):
        raise ConfigError(f"environment must be prod|sim|synthetic|preflight, got {environment!r}")
    return cfg


def load_config(path: Path | str) -> dict[str, Any]:
    """Read, path-resolve and validate an executor config file."""
    config_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot read config {config_path}: {exc}") from exc
    return normalize_config(raw, config_path.parent)
