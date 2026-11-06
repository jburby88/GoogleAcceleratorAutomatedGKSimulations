"""Unit tests for KR3 active-learning validation queue hook (`kr3_design/active_learning_queue.py`)."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest
from hpc_core.slurm import SlurmBackend, find_batch_script

from kr3_design import RunAlreadyQueuedError, enqueue_validation_run
from kr3_design import active_learning_queue as alq


def test_enqueue_validation_run_gx_geometry_and_slurm_script(tmp_path: Path) -> None:
    pending_dir = tmp_path / "queue" / "pending"
    run_folder = enqueue_validation_run(
        kr2_pending_dir=pending_dir,
        run_id="gx_kr3_val_001",
        equilibrium_id="mrx_eq_0042",
        solver="GX",
        fidelity="high",
        surrogate_epistemic_std=0.18,
        beta_target=0.042,
    )

    gx_in = (run_folder / "gx.in").read_text(encoding="utf-8")
    # GXSRC-04a: GX reads geo_file (not geofilename) and needs geo_option="nc" for .eik.nc.
    assert tomllib.loads(gx_in)["Geometry"] == {
        "geo_option": "nc",
        "geo_file": "mrx_eq_0042.eik.nc",
    }
    assert "geofilename" not in gx_in

    slurm_script = find_batch_script(run_folder)
    assert slurm_script.name == "submit.slurm"
    assert os.access(slurm_script, os.X_OK)
    slurm_text = slurm_script.read_text(encoding="utf-8")
    assert "#SBATCH -p gh" in slurm_text
    assert "ibrun gx gx.in" in slurm_text

    backend = SlurmBackend(dry_run=True)
    job_id = backend.submit(run_folder, solver="GX")
    assert job_id.startswith("dryrun-")


def test_enqueue_validation_run_gene_geometry_and_slurm_script(tmp_path: Path) -> None:
    pending_dir = tmp_path / "queue" / "pending"
    run_folder = enqueue_validation_run(
        kr2_pending_dir=pending_dir,
        run_id="gene_kr3_val_002",
        equilibrium_id="mrx_eq_0099",
        solver="GENE",
        fidelity="low",
        beta_target=0.038,
    )

    params_text = (run_folder / "parameters").read_text(encoding="utf-8")
    assert "&geometry" in params_text
    assert "magn_geometry = 'gist'" in params_text
    assert "geomfile = 'mrx_eq_0099'" in params_text

    slurm_script = find_batch_script(run_folder)
    assert os.access(slurm_script, os.X_OK)
    slurm_text = slurm_script.read_text(encoding="utf-8")
    assert "#SBATCH -A TG-PHY240001" in slurm_text
    assert 'ibrun "${GENE_BIN:-gene}"' in slurm_text

    backend = SlurmBackend(dry_run=True)
    job_id = backend.submit(run_folder, solver="GENE")
    assert job_id.startswith("dryrun-")


@pytest.mark.parametrize(
    ("bad_run_id", "bad_eq_id"),
    [
        ("../escape", "mrx_eq_0001"),
        ("gx_01/../../escape", "mrx_eq_0001"),
        (".hidden_run", "mrx_eq_0001"),
        ("gx run space", "mrx_eq_0001"),
        ("gx_valid_01", "../bad_eq"),
        ("gx_valid_01", "eq;rm -rf /"),
        ("", "mrx_eq_0001"),
    ],
)
def test_enqueue_validation_run_rejects_unsafe_ids_and_path_traversal(
    tmp_path: Path, bad_run_id: str, bad_eq_id: str
) -> None:
    pending_dir = tmp_path / "queue" / "pending"
    with pytest.raises(ValueError):
        enqueue_validation_run(
            kr2_pending_dir=pending_dir,
            run_id=bad_run_id,
            equilibrium_id=bad_eq_id,
        )


def _queue_entries(pending_dir: Path) -> list[str]:
    return sorted(p.name for p in pending_dir.iterdir()) if pending_dir.exists() else []


def test_gx_deck_is_valid_toml_with_gx_keys_and_harvestable_outputs(tmp_path: Path) -> None:
    pending_dir = tmp_path / "queue" / "pending"
    run_folder = enqueue_validation_run(
        kr2_pending_dir=pending_dir, run_id="gx_em", equilibrium_id="mrx_eq_1", beta_target=0.04
    )
    deck = tomllib.loads((run_folder / "gx.in").read_text(encoding="utf-8"))
    # PHYS-04: GX's hyperdiffusion key is D_hyper; `hyperviscosity` does not exist in GX.
    assert deck["Dissipation"] == {"hyper": True, "D_hyper": 0.05}
    # DOM-05: restart files and flux/free-energy diagnostics are written; finite t_max.
    assert deck["Restart"] == {"restart": False, "save_for_restart": True}
    assert deck["Diagnostics"] == {"fluxes": True, "free_energy": True}
    assert deck["Time"]["t_max"] == 200.0
    dims = deck["Dimensions"]
    assert (dims["nhermite"], dims["nlaguerre"], dims["nspecies"]) == (16, 8, 2)
    assert deck["Physics"] == {"beta": 0.04, "nonlinear_mode": True}
    assert deck["Domain"] == {"y0": 28.2, "boundary": "linked"}
    # GXSRC-04b/c: fully kinetic at finite beta (GX zeroes beta with Boltzmann electrons).
    assert deck["species"] == {
        "z": [1.0, -1.0],
        "mass": [1.0, 2.72e-4],
        "dens": [1.0, 1.0],
        "temp": [1.0, 1.0],
        "tprim": [3.0, 3.0],
        "fprim": [1.0, 1.0],
        "vnewk": [0.01, 0.01],
        "type": ["ion", "electron"],
    }
    assert deck["Boltzmann"] == {"add_Boltzmann_species": False}

    electrostatic = enqueue_validation_run(
        kr2_pending_dir=pending_dir, run_id="gx_es", equilibrium_id="mrx_eq_1", beta_target=0.0
    )
    es_deck = tomllib.loads((electrostatic / "gx.in").read_text(encoding="utf-8"))
    assert es_deck["Dimensions"]["nspecies"] == 1
    assert es_deck["Physics"] == {"beta": 0.0, "nonlinear_mode": True}
    assert es_deck["Domain"] == {"y0": 28.2, "boundary": "linked"}
    assert es_deck["Geometry"] == {"geo_option": "nc", "geo_file": "mrx_eq_1.eik.nc"}
    assert es_deck["species"] == {
        "z": [1.0],
        "mass": [1.0],
        "dens": [1.0],
        "temp": [1.0],
        "tprim": [3.0],
        "fprim": [1.0],
        "vnewk": [0.01],
        "type": ["ion"],
    }
    assert es_deck["Boltzmann"] == {
        "add_Boltzmann_species": True,
        "Boltzmann_type": "electrons",
        "tau_fac": 1.0,
    }
    # Every per-species array has exactly nspecies entries in both decks.
    for d in (deck, es_deck):
        n = d["Dimensions"]["nspecies"]
        assert all(len(values) == n for values in d["species"].values())


def test_gene_deck_parses_as_namelist_with_dom05_keys(tmp_path: Path) -> None:
    f90nml = pytest.importorskip("f90nml")
    run_folder = enqueue_validation_run(
        kr2_pending_dir=tmp_path / "queue" / "pending",
        run_id="gene_v",
        equilibrium_id="mrx_eq_2",
        solver="GENE",
    )
    nml = f90nml.read(str(run_folder / "parameters"))
    assert (nml["box"]["nv0"], nml["box"]["nw0"]) == (16, 8)
    assert "kymin" not in nml["box"]  # per-equilibrium physics choice, deliberately omitted
    assert nml["general"]["calc_dt"] is True
    assert nml["general"]["simtimelim"] == 200.0
    assert nml["in_out"]["read_checkpoint"] is False


@pytest.mark.parametrize("state", ["pending", "running", "completed", "needs_human"])
def test_run_id_already_in_any_queue_state_is_refused(tmp_path: Path, state: str) -> None:
    queue = tmp_path / "queue"
    live = queue / state / "gx_dup"
    live.mkdir(parents=True)
    (live / "gx.in").write_text("# live run, must not be touched\n", encoding="utf-8")

    with pytest.raises(RunAlreadyQueuedError, match=state):
        enqueue_validation_run(
            kr2_pending_dir=queue / "pending", run_id="gx_dup", equilibrium_id="mrx_eq_3"
        )
    assert (live / "gx.in").read_text(encoding="utf-8") == "# live run, must not be touched\n"
    assert _queue_entries(queue / "pending") in ([], ["gx_dup"])  # no staging leftovers


def test_failed_write_leaves_no_partial_folder_for_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = alq._write_durable

    def flaky_write(path: Path, text: str) -> None:
        if path.name == "gx.in":
            raise OSError("disk full")
        real_write(path, text)

    monkeypatch.setattr(alq, "_write_durable", flaky_write)
    pending_dir = tmp_path / "queue" / "pending"
    with pytest.raises(OSError, match="disk full"):
        enqueue_validation_run(kr2_pending_dir=pending_dir, run_id="gx_p", equilibrium_id="eq_4")
    assert _queue_entries(pending_dir) == []


def test_folder_is_published_complete_by_one_rename_from_a_hidden_staging_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, list[str]]] = []
    real_rename = os.rename

    def spy_rename(src: os.PathLike[str], dst: os.PathLike[str]) -> None:
        seen.append((Path(src).name, sorted(p.name for p in Path(src).iterdir())))
        real_rename(src, dst)

    monkeypatch.setattr(alq.os, "rename", spy_rename)
    run_folder = enqueue_validation_run(
        kr2_pending_dir=tmp_path / "queue" / "pending", run_id="gx_atomic", equilibrium_id="eq_5"
    )
    assert len(seen) == 1
    staging_name, staged_files = seen[0]
    assert staging_name.startswith(".")  # driver.py skips dot-prefixed pending entries
    assert staged_files == ["gx.in", "run_meta.json", "submit.slurm"]
    assert _queue_entries(run_folder.parent) == ["gx_atomic"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"kr2_pending_dir_name": "inbox"},
        {"beta_target": -0.01},
        {"beta_target": 1.0},
        {"beta_target": "0.03"},
        {"beta_target": float("nan")},
        {"surrogate_epistemic_std": -0.1},
        {"surrogate_epistemic_std": True},
        {"solver": None},
        {"fidelity": ["high"]},
        {"slurm_account": "TG;rm"},
    ],
)
def test_enqueue_rejects_invalid_numeric_and_queue_inputs(tmp_path: Path, overrides: dict) -> None:
    kwargs = {"run_id": "gx_bad", "equilibrium_id": "eq_6", **overrides}
    pending_dir = tmp_path / "queue" / kwargs.pop("kr2_pending_dir_name", "pending")
    with pytest.raises(ValueError):
        enqueue_validation_run(kr2_pending_dir=pending_dir, **kwargs)
    assert _queue_entries(pending_dir) == []
