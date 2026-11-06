"""Fortran namelist (`parameters`) reader & atomic patcher for GENE via `f90nml`.

Only the mapped keys below and `&in_out read_checkpoint` are ever written. Every patch is
rendered to a temporary file, re-parsed and compared with the original namelist; any change
outside those keys aborts the patch before the input file is replaced.
GENE parameter reference: https://genecode.org/ (`&box nx0/nky0/nz0`, `&general dt_max/courant/
hyp_z/simtimelim`, `&in_out read_checkpoint`).
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import f90nml
from hpc_core.fsutil import atomic_write_bytes, atomic_write_text

from tacc_exec.patchers._values import (
    coerce_number,
    ensure_only_allowed_changes,
    validate_modifications,
    values_match,
)

# Canonical mapping from OKF parameter name -> (namelist_group, namelist_key, converter)
GENE_PARAM_MAP: dict[str, tuple[str, str, type]] = {
    "nx": ("box", "nx0", int),
    "ny": ("box", "nky0", int),
    "nz": ("box", "nz0", int),
    "dt_max": ("general", "dt_max", float),
    "cfl_fraction": ("general", "courant", float),
    "hyperdiffusion_coeff": ("general", "hyp_z", float),
}

_RESTART_GROUP, _RESTART_KEY = "in_out", "read_checkpoint"


def _single_group(nml: Any, group_name: str) -> Any | None:
    """Return a namelist group, refusing duplicated groups (f90nml `Cogroup`, a list)."""
    group = nml.get(group_name)
    if group is None:
        return None
    if isinstance(group, list):
        raise ValueError(f"GENE parameters contain duplicate &{group_name} groups; ambiguous")
    return group


def read_gene_parameters(parameters_path: Path) -> dict[str, int | float]:
    """Extract allowlisted numerical parameters from a GENE `parameters` namelist file.

    Raises ValueError on duplicated groups or non-numeric / boolean / non-finite values.
    """
    nml = f90nml.read(str(parameters_path))
    extracted: dict[str, int | float] = {}
    for canonical_name, (group_name, key_name, caster) in GENE_PARAM_MAP.items():
        group = _single_group(nml, group_name)
        if group is not None and key_name in group:
            extracted[canonical_name] = coerce_number(
                caster, group[key_name], f"GENE &{group_name} {key_name}"
            )
    return extracted


def read_gene_run_controls(parameters_path: Path) -> dict[str, Any]:
    """Return run controls the babysitter reads but never treats as allowlisted parameters.

    Keys (raw parsed values, None when absent; callers validate):
    - `simtimelim`: `&general simtimelim`, the simulation time limit (DOM-02 fallback).
      (`&general timelim` is a wall-clock limit in seconds and is deliberately not used.)
    - `read_checkpoint`: `&in_out read_checkpoint` (True = continue from a checkpoint).
    """
    nml = f90nml.read(str(parameters_path))
    general = _single_group(nml, "general") or {}
    in_out = _single_group(nml, _RESTART_GROUP) or {}
    return {
        "simtimelim": general.get("simtimelim"),
        "read_checkpoint": in_out.get(_RESTART_KEY),
    }


def patch_gene_parameters(
    parameters_path: Path,
    modifications: dict[str, int | float],
    restart_from_checkpoint: bool,
) -> dict[str, int | float]:
    """Atomically patch a GENE Fortran namelist while preserving physics namelist blocks.

    - Writes only `GENE_PARAM_MAP` keys of the requested parameters and
      `&in_out read_checkpoint`; verified by re-parsing before the input is replaced.
    - `<input>.orig` keeps the input as it was before the first automated patch;
      `<input>.bak` keeps the input as it was before this patch.
    - Writes are durable and atomic (temp file + fsync + rename).
    """
    parameters_path = Path(parameters_path)
    typed = validate_modifications(modifications, GENE_PARAM_MAP, "GENE")
    if not isinstance(restart_from_checkpoint, bool):
        raise ValueError("restart_from_checkpoint must be a bool")

    original_bytes = parameters_path.read_bytes()
    existing_nml = f90nml.read(str(parameters_path))
    patch_dict: dict[str, dict[str, int | float | bool]] = {}
    allowed_changes: set[tuple[str, ...]] = {(_RESTART_GROUP, _RESTART_KEY)}
    expected: dict[tuple[str, str], int | float | bool] = {}
    for canonical_name, value in typed.items():
        group_name, key_name, _caster = GENE_PARAM_MAP[canonical_name]
        _single_group(existing_nml, group_name)
        patch_dict.setdefault(group_name, {})[key_name] = value
        allowed_changes.add((group_name, key_name))
        expected[(group_name, key_name)] = value
    expected[(_RESTART_GROUP, _RESTART_KEY)] = restart_from_checkpoint
    has_in_out = _single_group(existing_nml, _RESTART_GROUP) is not None
    if has_in_out:
        patch_dict[_RESTART_GROUP] = {_RESTART_KEY: restart_from_checkpoint}

    fd, tmp_name = tempfile.mkstemp(prefix=".parameters.tmp.", dir=str(parameters_path.parent))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        if patch_dict:
            f90nml.patch(str(parameters_path), patch_dict, str(tmp_path))
        else:
            tmp_path.write_bytes(original_bytes)
        rendered = tmp_path.read_text(encoding="utf-8")
        if not has_in_out:
            flag_str = ".true." if restart_from_checkpoint else ".false."
            if not rendered.endswith("\n"):
                rendered += "\n"
            rendered += f"\n&{_RESTART_GROUP}\n  {_RESTART_KEY} = {flag_str}\n/\n"
            tmp_path.write_text(rendered, encoding="utf-8")
        patched_nml = f90nml.read(str(tmp_path))
    finally:
        with contextlib.suppress(OSError):
            tmp_path.unlink()

    ensure_only_allowed_changes(
        existing_nml.todict(), patched_nml.todict(), allowed_changes, "GENE"
    )
    for (group_name, key_name), value in expected.items():
        group = _single_group(patched_nml, group_name) or {}
        if not values_match(value, group.get(key_name)):
            raise ValueError(
                f"GENE patch verification failed for &{group_name} {key_name}: "
                f"expected {value!r}, got {group.get(key_name)!r}"
            )

    orig_path = parameters_path.with_name(parameters_path.name + ".orig")
    if not orig_path.exists():
        atomic_write_bytes(orig_path, original_bytes)
    atomic_write_bytes(parameters_path.with_name(parameters_path.name + ".bak"), original_bytes)
    atomic_write_text(parameters_path, rendered)

    return read_gene_parameters(parameters_path)
