"""Comment-preserving TOML (`gx.in`) reader & atomic patcher for GX via `tomlkit`.

Parameter keys follow docs/review/DOMAIN.md (PHYS-04): GX hyperdiffusion is
`[Dissipation] D_hyper` (enabled by `hyper = true`); `hyperviscosity` is not a GX key and is
only read as a legacy fallback (and kept in sync when present) for older input decks.
GX reference: https://gx.readthedocs.io/en/latest/ and the repo's Cyclone decks under
`workstreams/kr2-gk-database/gcp/benchmarks/*/gx.in`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit
from hpc_core.fsutil import atomic_write_bytes, atomic_write_text

from tacc_exec.patchers._values import (
    SAFE_FILENAME_RE,
    coerce_number,
    ensure_only_allowed_changes,
    validate_modifications,
    values_match,
)

# Canonical mapping from OKF parameter name -> (section, key, converter)
GX_PARAM_MAP: dict[str, tuple[str, str, type]] = {
    "nx": ("Dimensions", "nx", int),
    "ny": ("Dimensions", "ny", int),
    "nz": ("Dimensions", "ntheta", int),
    "dt_max": ("Time", "dt", float),
    "cfl_fraction": ("Time", "cfl", float),
    "hyperdiffusion_coeff": ("Dissipation", "D_hyper", float),
}

# Legacy keys: read when the canonical key is absent, and updated alongside it when present.
GX_LEGACY_KEYS: dict[str, tuple[str, str]] = {
    "nz": ("Dimensions", "nz"),
    "hyperdiffusion_coeff": ("Dissipation", "hyperviscosity"),
}

_PATCHED_SECTIONS = ("Dimensions", "Time", "Dissipation", "Restart")


def _load_doc(gx_in_path: Path) -> tomlkit.TOMLDocument:
    return tomlkit.parse(Path(gx_in_path).read_text(encoding="utf-8"))


def _table(doc: Any, name: str) -> Any | None:
    section = doc.get(name)
    return section if isinstance(section, dict) else None


def _species_table(doc: Any) -> Any | None:
    # Official GX decks use a lowercase `[species]` table (DOM-04); `[Species]` is legacy.
    for name in ("species", "Species"):
        if name in doc:
            return doc[name]
    return None


def read_gx_parameters(gx_in_path: Path) -> dict[str, int | float]:
    """Extract allowlisted numerical parameters and velocity/species grid dimensions from `gx.in`.

    Raises ValueError if an allowlisted key holds a non-numeric, boolean, non-finite or
    (for integer grids) non-integral value.
    """
    doc = _load_doc(gx_in_path)
    extracted: dict[str, int | float] = {}
    for canonical_name, (section_name, key_name, caster) in GX_PARAM_MAP.items():
        section = _table(doc, section_name)
        if section is None:
            continue
        if key_name in section:
            extracted[canonical_name] = coerce_number(
                caster, section[key_name], f"gx.in [{section_name}] {key_name}"
            )
        elif canonical_name in GX_LEGACY_KEYS:
            legacy_section, legacy_key = GX_LEGACY_KEYS[canonical_name]
            legacy = _table(doc, legacy_section)
            if legacy is not None and legacy_key in legacy:
                extracted[canonical_name] = coerce_number(
                    caster, legacy[legacy_key], f"gx.in [{legacy_section}] {legacy_key}"
                )

    dims = _table(doc, "Dimensions")
    if dims is not None:
        for dim_key in ("nhermite", "nlaguerre", "nspecies"):
            if dim_key in dims:
                extracted[dim_key] = coerce_number(
                    int, dims[dim_key], f"gx.in [Dimensions] {dim_key}"
                )

    species_sec = _species_table(doc)
    if "nspecies" not in extracted and species_sec is not None:
        if isinstance(species_sec, (list, tuple)):
            extracted["nspecies"] = len(species_sec)
        elif isinstance(species_sec, dict):
            if "nspecies" in species_sec:
                extracted["nspecies"] = coerce_number(
                    int, species_sec["nspecies"], "gx.in [species] nspecies"
                )
            elif "z" in species_sec and isinstance(species_sec["z"], (list, tuple)):
                extracted["nspecies"] = len(species_sec["z"])
            else:
                extracted["nspecies"] = len(species_sec)

    return extracted


def read_gx_run_controls(gx_in_path: Path) -> dict[str, Any]:
    """Return run controls the babysitter reads but never treats as allowlisted parameters.

    Keys (raw parsed values, None when absent; callers validate):
    - `t_max`: `[Time] t_max`, the simulation end time (DOM-02 sim_time_target fallback).
    - `fluxes`: `[Diagnostics] fluxes` (GX defaults to False when absent; CLAUDE-P0-2).
    - `restart`: `[Restart] restart` (True = continue from a checkpoint).
    - `restart_to_file` / `restart_from_file`: restart file names if they are plain file names
      inside the run directory, else None (also reported in `unsafe_restart_names`).
    """
    doc = _load_doc(gx_in_path)
    time_sec = _table(doc, "Time") or {}
    diag_sec = _table(doc, "Diagnostics") or {}
    restart_sec = _table(doc, "Restart")
    if restart_sec is None:
        restart_sec = _table(doc, "restart") or {}
    controls: dict[str, Any] = {
        "t_max": _plain(time_sec.get("t_max")),
        "fluxes": _plain(diag_sec.get("fluxes")),
        "restart": _plain(restart_sec.get("restart")),
        "restart_to_file": None,
        "restart_from_file": None,
        "unsafe_restart_names": [],
    }
    for key in ("restart_to_file", "restart_from_file"):
        raw = _plain(restart_sec.get(key))
        if raw in (None, ""):
            continue
        if isinstance(raw, str) and SAFE_FILENAME_RE.match(raw) and ".." not in raw:
            controls[key] = raw
        else:
            controls["unsafe_restart_names"].append(str(raw))
    return controls


def _plain(value: Any) -> Any:
    return value.unwrap() if hasattr(value, "unwrap") else value


def _ensure_unambiguous_sections(doc: Any) -> None:
    """Refuse to patch when a target table also exists under another letter case."""
    top_level = [str(k) for k in doc]
    for canonical in _PATCHED_SECTIONS:
        variants = [k for k in top_level if k.lower() == canonical.lower() and k != canonical]
        if variants:
            raise ValueError(
                f"Ambiguous gx.in: table(s) {variants} duplicate [{canonical}] under another "
                "case; refusing to patch (GX would read only one of them)"
            )


def patch_gx_toml(
    gx_in_path: Path,
    modifications: dict[str, int | float],
    restart_from_checkpoint: bool,
) -> dict[str, int | float]:
    """Atomically patch a GX `gx.in` TOML file while preserving comments and physics tables.

    - Only the mapped keys of the requested parameters (plus legacy aliases already present,
      an explicit `[Time] dt_max` when patching `dt_max`, `[Dissipation] hyper` when enabling
      hyperdiffusion, and `[Restart] restart`) may change; the rendered result is re-parsed and
      verified before it replaces the input.
    - `<input>.orig` keeps the input as it was before the first automated patch;
      `<input>.bak` keeps the input as it was before this patch.
    - Writes are durable and atomic (temp file + fsync + rename).
    """
    gx_in_path = Path(gx_in_path)
    typed = validate_modifications(modifications, GX_PARAM_MAP, "GX")
    if not isinstance(restart_from_checkpoint, bool):
        raise ValueError("restart_from_checkpoint must be a bool")

    original_text = gx_in_path.read_text(encoding="utf-8")
    original_doc = tomlkit.parse(original_text)
    _ensure_unambiguous_sections(original_doc)
    before_plain = original_doc.unwrap()

    doc = tomlkit.parse(original_text)
    allowed_changes: set[tuple[str, ...]] = {("Restart", "restart")}
    expected: dict[tuple[str, str], int | float | bool] = {}

    for canonical_name, value in typed.items():
        section_name, key_name, _caster = GX_PARAM_MAP[canonical_name]
        if section_name not in doc:
            doc[section_name] = tomlkit.table()
        section = doc[section_name]
        if not isinstance(section, dict):
            raise ValueError(f"gx.in [{section_name}] is not a table; refusing to patch")
        section[key_name] = value
        allowed_changes.add((section_name, key_name))
        expected[(section_name, key_name)] = value
        if canonical_name == "dt_max" and "dt_max" in section:
            # GXSRC-02: GX defaults `dt_max = dt` and clamps adaptive steps to `dt <= dt_max`;
            # sync an explicit `[Time] dt_max` when already present in the deck.
            section["dt_max"] = value
            allowed_changes.add((section_name, "dt_max"))
            expected[(section_name, "dt_max")] = value
        if canonical_name in GX_LEGACY_KEYS:
            legacy_section, legacy_key = GX_LEGACY_KEYS[canonical_name]
            if legacy_section == section_name and legacy_key in section:
                section[legacy_key] = value
                allowed_changes.add((legacy_section, legacy_key))
                expected[(legacy_section, legacy_key)] = value
        if canonical_name == "hyperdiffusion_coeff" and value > 0.0:
            # PHYS-04: D_hyper only takes effect with `hyper = true`.
            if section.get("hyper") is not True:
                section["hyper"] = True
            allowed_changes.add((section_name, "hyper"))
            expected[(section_name, "hyper")] = True

    if "Restart" not in doc:
        doc["Restart"] = tomlkit.table()
    restart_sec = doc["Restart"]
    if not isinstance(restart_sec, dict):
        raise ValueError("gx.in [Restart] is not a table; refusing to patch")
    if restart_sec.get("restart") is not restart_from_checkpoint:
        restart_sec["restart"] = restart_from_checkpoint
    expected[("Restart", "restart")] = restart_from_checkpoint

    rendered = tomlkit.dumps(doc)
    after_doc = tomlkit.parse(rendered)
    ensure_only_allowed_changes(before_plain, after_doc.unwrap(), allowed_changes, "GX")
    for (section_name, key_name), value in expected.items():
        actual = _plain(after_doc.get(section_name, {}).get(key_name))
        if not values_match(value, actual):
            raise ValueError(
                f"GX patch verification failed for [{section_name}] {key_name}: "
                f"expected {value!r}, got {actual!r}"
            )

    original_bytes = original_text.encode("utf-8")
    orig_path = gx_in_path.with_name(gx_in_path.name + ".orig")
    if not orig_path.exists():
        atomic_write_bytes(orig_path, original_bytes)
    atomic_write_bytes(gx_in_path.with_name(gx_in_path.name + ".bak"), original_bytes)
    atomic_write_text(gx_in_path, rendered)

    return read_gx_parameters(gx_in_path)
