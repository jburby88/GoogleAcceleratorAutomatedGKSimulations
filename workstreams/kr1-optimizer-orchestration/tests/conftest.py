"""Shared fixtures: write KR2 dataset runs that follow the loader's manifest/fluxes contract."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

RunWriter = Callable[..., Path]


def _write_run(
    dataset_root: Path,
    campaign_id: str,
    run_id: str,
    equilibrium_id: str,
    *,
    solver: str = "GX",
    fidelity: str = "low",
    backend: str = "gcp_batch",
    heat_flux_es: Any = 2.5,
    manifest_overrides: dict[str, Any] | None = None,
    fluxes_overrides: dict[str, Any] | None = None,
    drop_manifest: tuple[str, ...] = (),
    drop_fluxes: tuple[str, ...] = (),
    write_equilibrium_meta: bool = True,
) -> Path:
    """Write one eligible schema-2.0 run (then apply overrides/drops); return its folder."""
    campaign_dir = dataset_root / "campaigns" / campaign_id
    run_dir = campaign_dir / "runs" / run_id
    run_dir.mkdir(parents=True)

    fluxes: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": run_id,
        "equilibrium_id": equilibrium_id,
        "solver": solver,
        "fidelity": fidelity,
        "flux_units": "gyroBohm",
        "heat_flux_es": heat_flux_es,
        "heat_flux_em": None,
        "particle_flux": 0.11,
        "maxwell_reynolds_stress": None,
        "momentum_flux": None,
        "growth_rate_gamma": 0.24,
        "omega_freq": None,
        "metrics": {"heat_flux_es_std": 0.3},
    }
    fluxes.update(fluxes_overrides or {})
    for key in drop_fluxes:
        fluxes.pop(key)
    fluxes_bytes = json.dumps(fluxes, sort_keys=True).encode()  # NaN overrides -> NaN token
    (run_dir / "fluxes.json").write_bytes(fluxes_bytes)

    manifest: dict[str, Any] = {
        "schema_version": "2.0",
        "run_id": run_id,
        "campaign_id": campaign_id,
        "equilibrium_id": equilibrium_id,
        "solver": solver,
        "fidelity": fidelity,
        "backend": backend,
        "converged": True,
        "convergence_method": "time_average_stationarity",
        "escalated": False,
        "simulated": False,
        "environment": "prod",
        "dataset_eligible": True,
        "eligibility_reasons": [],
        "hands_free": True,
        "retry_count": 0,
        "parameter_patches": [],
        "patched": False,
        "final_numerical_parameters": {"nx": 64, "dt_max": 0.02},
        "input_file": None,
        "files": {
            "fluxes.json": {
                "sha256": hashlib.sha256(fluxes_bytes).hexdigest(),
                "bytes": len(fluxes_bytes),
            }
        },
    }
    manifest.update(manifest_overrides or {})
    for key in drop_manifest:
        manifest.pop(key)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    eq_meta = campaign_dir / "equilibria" / equilibrium_id / "equilibrium_meta.json"
    if write_equilibrium_meta and not eq_meta.exists():
        eq_meta.parent.mkdir(parents=True)
        eq_meta.write_text(json.dumps({"equilibrium_id": equilibrium_id}), encoding="utf-8")
    return run_dir


@pytest.fixture
def write_run() -> RunWriter:
    return _write_run
