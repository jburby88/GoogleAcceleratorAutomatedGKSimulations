"""Cloud Batch & GKE Spot TPU campaign orchestrator (`gx-jax-campaign`).

Provides deterministic multi-worker manifest slicing, structural `jax.vmap`
auto-grouping (`run_ensemble` for compatible batches of size `2..B`, falling
back to `run_cli_simulation` for singletons), local SSD/tmpfs scratch staging
for GCS FUSE compatibility, `--skip-completed` idempotency, and graceful Spot
TPU preemption (`SIGTERM` / `SIGINT` -> atomic checkpoint promotion + partial
worker ledger + exit code `143` -> automatic `t0 -> t_max` resume).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import shutil
import signal
import threading
import time
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import netCDF4
import numpy as np

from gx_jax.cli import parse_mesh_arg, run_cli_simulation
from gx_jax.config import GXConfig, load_config
from gx_jax.diagnostics_nc import (
    DiagnosticsState,
    DiagnosticsWriter,
    compute_diagnostics_step,
    extract_kr2_diagnostics,
    sanitize_json_payload,
)
from gx_jax.ensemble import (
    EnsembleMemberSpec,
    _apply_cfg_overrides,
    _deck_stem,
    _index_batch_host_tree,
    batched_compute_diagnostics_step,
    batched_step_integrator_state,
    init_batched_ensemble,
    init_batched_integrator_state,
    run_ensemble,
    validate_ensemble_compatibility,
)
from gx_jax.fields import Fields, load_restart_nc, save_restart_nc
from gx_jax.geometry_nc import init_geometry
from gx_jax.integrator import (
    _scheme_honors_fixed_dt,
    advance_integrator_chunk,
    init_gyrokinetic_operator,
    init_integrator_state,
)
from gx_jax.shard_tpu import configure_tpu_precision

SIGTERM_EXIT_CODE: int = 128 + int(signal.SIGTERM)  # 143
SIGINT_EXIT_CODE: int = 128 + int(signal.SIGINT)  # 130


class SpotPreemptionInterrupt(BaseException):
    """Raised when `SIGTERM` or `SIGINT` signals Spot TPU VM preemption."""

    def __init__(self, signum: int) -> None:
        sig_int = int(signum)
        try:
            sig_name = signal.Signals(sig_int).name
        except ValueError:
            sig_name = f"SIG{sig_int}"
        super().__init__(f"Spot preemption signal received: {sig_name} ({sig_int})")
        self.signum: int = sig_int
        self.signal_name: str = sig_name
        self.exit_code: int = 128 + sig_int


@dataclass(frozen=True)
class CampaignItem:
    """Single simulation deck entry resolved from a campaign manifest."""

    index: int
    input_path: Path
    stem: str
    raw_name: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CampaignWorkerSlice:
    """Resolved deterministic `(task_index, task_count)` worker identity."""

    task_index: int
    task_count: int
    source: str


@dataclass(frozen=True)
class CampaignBatch:
    """Grouped execution unit of structurally compatible campaign decks."""

    batch_index: int
    mode: str
    items: tuple[CampaignItem, ...]
    specs: tuple[EnsembleMemberSpec, ...]


def _is_primary_process() -> bool:
    """Return True on coordinator rank 0 (compatible with `gx_jax.distributed`)."""
    try:
        from gx_jax.distributed import is_primary_process  # type: ignore[import-not-found]

        return bool(is_primary_process())
    except ImportError:
        return int(jax.process_index()) == 0


def _maybe_init_distributed(
    *,
    distributed: bool,
    task_index: int | None,
    task_count: int | None,
    coordinator_address: str | None,
    num_processes: int | None,
    process_id: int | None,
) -> bool:
    """Initialize multi-host JAX runtime only when `--distributed` is explicitly set."""
    configure_tpu_precision()
    if not distributed:
        return False
    if task_index is None or task_count is None:
        raise ValueError(
            "--distributed requires explicit --task-index and --task-count so all SPMD "
            "processes execute the identical campaign slice."
        )
    try:
        from gx_jax.distributed import init_distributed_runtime  # type: ignore[import-not-found]
    except ImportError:
        raise ValueError(
            "gx_jax.distributed is not available in this build; cannot use --distributed."
        ) from None

    init_distributed_runtime(
        distributed=True,
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )
    return True


def _maybe_shutdown_distributed(did_init: bool) -> None:
    """Cleanly shut down `gx_jax.distributed` if initialized."""
    if not did_init:
        return
    try:
        from gx_jax.distributed import (  # type: ignore[import-not-found]
            shutdown_distributed_runtime,
        )

        shutdown_distributed_runtime()
    except ImportError:
        pass


class PreemptionController:
    """Signal trap for graceful Spot TPU `SIGTERM` / `SIGINT` checkpoint & ledger flush.

    Installs handlers on the main thread and always restores previous signal handlers
    in ``__exit__``. Critical atomic rename / JSON flush sections wrapped in
    :meth:`shield` defer raising :class:`SpotPreemptionInterrupt` until the critical
    section finishes so ``.tmp`` promotions are never left half-written.
    """

    def __init__(self, *, trap_signals: bool = True) -> None:
        self._trap_signals = bool(trap_signals) and (
            threading.current_thread() is threading.main_thread()
        )
        self._lock = threading.RLock()
        self._shield_depth: int = 0
        self._pending_signum: int | None = None
        self._cleaning_up: bool = False
        self._prev_sigterm: Any = None
        self._prev_sigint: Any = None
        self._installed: bool = False

    @property
    def pending_signum(self) -> int | None:
        """Signal number received (`15` for `SIGTERM`, `2` for `SIGINT`), or `None`."""
        with self._lock:
            return self._pending_signum

    def mark_cleaning_up(self) -> None:
        """Prevent duplicate signals from aborting final preemption ledger flush."""
        with self._lock:
            self._cleaning_up = True

    def _on_signal(self, signum: int, _frame: Any) -> None:
        should_raise = False
        chosen_sig = int(signum)
        with self._lock:
            if self._pending_signum is None or chosen_sig == int(signal.SIGTERM):
                self._pending_signum = chosen_sig
            else:
                chosen_sig = self._pending_signum
            if not self._cleaning_up and self._shield_depth == 0:
                should_raise = True
        if should_raise:
            raise SpotPreemptionInterrupt(chosen_sig)

    @contextlib.contextmanager
    def shield(self) -> Iterator[None]:
        """Defer raising `SpotPreemptionInterrupt` until atomic file promotion completes."""
        is_main = threading.current_thread() is threading.main_thread()
        if not is_main:
            yield
            return
        with self._lock:
            self._shield_depth += 1
        exc_in_block = False
        try:
            yield
        except BaseException:
            exc_in_block = True
            raise
        finally:
            to_raise: int | None = None
            with self._lock:
                self._shield_depth = max(0, self._shield_depth - 1)
                if (
                    self._shield_depth == 0
                    and not self._cleaning_up
                    and not exc_in_block
                    and self._pending_signum is not None
                ):
                    to_raise = self._pending_signum
            if to_raise is not None:
                raise SpotPreemptionInterrupt(to_raise)

    def check(self) -> None:
        """Raise `SpotPreemptionInterrupt` immediately if a preemption signal is pending."""
        if threading.current_thread() is not threading.main_thread():
            return
        with self._lock:
            if not self._cleaning_up and self._shield_depth == 0 and self._pending_signum:
                sig = self._pending_signum
            else:
                sig = None
        if sig is not None:
            raise SpotPreemptionInterrupt(sig)

    def __enter__(self) -> PreemptionController:
        if self._trap_signals:
            self._prev_sigterm = signal.getsignal(signal.SIGTERM)
            self._prev_sigint = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
            self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._installed:
            try:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            finally:
                signal.signal(signal.SIGINT, self._prev_sigint)
                self._installed = False


# ---------------------------------------------------------------------------
# 1. Manifest Parsing, Stem Disambiguation & Deterministic Worker Slicing
# ---------------------------------------------------------------------------

_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_stem(raw: str, fallback: str) -> str:
    cleaned = _SAFE_STEM_RE.sub("_", raw.strip()).strip("._")
    return cleaned or fallback


def _parse_manifest_entry(
    entry: Any,
    *,
    base_dir: Path,
    idx: int,
) -> tuple[Path, str, dict[str, Any]]:
    if isinstance(entry, str):
        rel_str = entry.strip()
        if not rel_str:
            raise ValueError(f"Empty deck path at manifest index {idx}.")
        p = Path(rel_str)
        resolved = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
        return resolved, _deck_stem(resolved), {}

    if isinstance(entry, dict):
        raw_path = entry.get("input") or entry.get("deck") or entry.get("path") or entry.get("file")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError(
                f"Manifest object at index {idx} must contain a non-empty string "
                "'input' (or 'deck'/'path'/'file')."
            )
        p = Path(raw_path.strip())
        resolved = p.resolve() if p.is_absolute() else (base_dir / p).resolve()
        raw_name_val = entry.get("name") or entry.get("stem") or entry.get("id")
        if raw_name_val is not None:
            raw_stem = _sanitize_stem(str(raw_name_val), _deck_stem(resolved))
        else:
            raw_stem = _deck_stem(resolved)
        meta = {
            str(k): v
            for k, v in entry.items()
            if k not in ("input", "deck", "path", "file", "name", "stem", "id")
        }
        return resolved, raw_stem, meta

    raise ValueError(
        f"Unsupported manifest entry at index {idx}: expected string path or JSON object, "
        f"got {type(entry).__name__}."
    )


def load_campaign_manifest(manifest_path: str | Path) -> list[CampaignItem]:
    """Load and validate a campaign manifest (`.json`, `.jsonl`, `.txt`, or directory).

    Relative deck paths are resolved against the manifest's parent directory (or the
    directory itself). Collision-free output stems are deterministically assigned across
    the full manifest prior to worker slicing so multi-worker shares never collide.
    """
    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Campaign manifest path does not exist: {m_path}")

    raw_entries: list[tuple[Path, str, dict[str, Any]]] = []

    if m_path.is_dir():
        base_dir = m_path.resolve()
        deck_files = sorted(
            (
                p
                for p in base_dir.rglob("*")
                if p.is_file()
                and p.suffix.lower() in (".in", ".toml")
                and not p.name.startswith(".")
            ),
            key=lambda p: p.relative_to(base_dir).as_posix(),
        )
        for p in deck_files:
            raw_entries.append((p.resolve(), _deck_stem(p), {}))
    else:
        base_dir = m_path.resolve().parent
        suffix = m_path.suffix.lower()
        text = m_path.read_text(encoding="utf-8")

        if suffix == ".json":
            parsed = json.loads(text)
            if isinstance(parsed, list):
                seq = parsed
            elif isinstance(parsed, dict):
                seq = None
                for key in ("decks", "items", "runs"):
                    if key in parsed:
                        if not isinstance(parsed[key], list):
                            raise ValueError(
                                f"Manifest key {key!r} in {m_path} must map to a JSON list."
                            )
                        seq = parsed[key]
                        break
                if seq is None:
                    raise ValueError(
                        f"JSON manifest object {m_path} must contain 'decks', 'items', or 'runs'."
                    )
            else:
                raise ValueError(
                    f"JSON manifest {m_path} must be a list or object, got {type(parsed).__name__}."
                )
            for idx, entry in enumerate(seq):
                raw_entries.append(_parse_manifest_entry(entry, base_dir=base_dir, idx=idx))

        elif suffix == ".jsonl":
            idx = 0
            for line_no, raw_line in enumerate(text.splitlines(), start=1):
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {m_path}:{line_no}: {exc}") from exc
                raw_entries.append(_parse_manifest_entry(entry, base_dir=base_dir, idx=idx))
                idx += 1

        elif suffix in (".in", ".toml"):
            # Convenience: allow passing a single .in/.toml deck directly
            resolved = m_path.resolve()
            raw_entries.append((resolved, _deck_stem(resolved), {}))

        else:
            # Plain-text manifest (.txt, .manifest, .list, etc.): one deck path per line
            idx = 0
            for raw_line in text.splitlines():
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                raw_entries.append(_parse_manifest_entry(stripped, base_dir=base_dir, idx=idx))
                idx += 1

    if not raw_entries:
        raise ValueError(f"Campaign manifest {m_path} contains zero simulation decks.")

    stem_counts: dict[str, int] = {}
    for deck_path, raw_stem, _meta in raw_entries:
        if not deck_path.is_file():
            raise FileNotFoundError(f"Campaign deck file does not exist: {deck_path}")
        stem_counts[raw_stem] = stem_counts.get(raw_stem, 0) + 1

    used_stems: set[str] = set()
    items: list[CampaignItem] = []
    for idx, (deck_path, raw_stem, meta) in enumerate(raw_entries):
        if stem_counts[raw_stem] == 1 and raw_stem not in used_stems:
            eff_stem = raw_stem
        else:
            cand = f"deck_{idx:04d}_{raw_stem}"
            disambig = 1
            while cand in used_stems:
                cand = f"deck_{idx:04d}_{disambig}_{raw_stem}"
                disambig += 1
            eff_stem = cand
        used_stems.add(eff_stem)
        items.append(
            CampaignItem(
                index=idx,
                input_path=deck_path,
                stem=eff_stem,
                raw_name=raw_stem,
                metadata=meta,
            )
        )
    return items


def _parse_env_int(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"Environment variable {key}={raw!r} is not a valid integer.") from exc


def detect_worker_slice(
    *,
    task_index: int | None = None,
    task_count: int | None = None,
    env: Mapping[str, str] | None = None,
) -> CampaignWorkerSlice:
    """Resolve `(task_index, task_count)` from CLI flags, Cloud Batch, or GKE Indexed Job."""
    env_map = os.environ if env is None else env

    idx_src = "default"
    if task_index is not None:
        resolved_index = int(task_index)
        idx_src = "cli"
    elif (batch_idx := _parse_env_int(env_map, "BATCH_TASK_INDEX")) is not None:
        resolved_index = batch_idx
        idx_src = "cloud_batch"
    elif (gke_idx := _parse_env_int(env_map, "JOB_COMPLETION_INDEX")) is not None:
        resolved_index = gke_idx
        idx_src = "gke_indexed_job"
    else:
        resolved_index = 0

    cnt_src = "default"
    if task_count is not None:
        resolved_count = int(task_count)
        cnt_src = "cli"
    elif (batch_cnt := _parse_env_int(env_map, "BATCH_TASK_COUNT")) is not None:
        resolved_count = batch_cnt
        cnt_src = "cloud_batch"
    elif (gke_cnt := _parse_env_int(env_map, "INDEX_COMPLETION_COUNT")) is not None:
        resolved_count = gke_cnt
        cnt_src = "gke_indexed_job"
    elif (gke_tot := _parse_env_int(env_map, "JOB_COMPLETION_TOTAL")) is not None:
        resolved_count = gke_tot
        cnt_src = "gke_indexed_job"
    else:
        resolved_count = 1

    if resolved_count < 1:
        raise ValueError(f"task_count must be >= 1, got {resolved_count}.")
    if not (0 <= resolved_index < resolved_count):
        raise ValueError(
            f"task_index must satisfy 0 <= task_index < task_count ({resolved_count}), "
            f"got task_index={resolved_index}."
        )

    if idx_src == cnt_src:
        source = idx_src
    elif "cli" in (idx_src, cnt_src) and "gke_indexed_job" in (idx_src, cnt_src):
        source = "gke_indexed_job+cli"
    elif "cli" in (idx_src, cnt_src):
        source = "cli"
    else:
        source = f"{idx_src}+{cnt_src}"

    return CampaignWorkerSlice(
        task_index=resolved_index,
        task_count=resolved_count,
        source=source,
    )


def slice_campaign_items(
    items: Sequence[CampaignItem],
    *,
    task_index: int,
    task_count: int,
) -> list[CampaignItem]:
    """Deterministically partition `items` across `0..task_count-1` with zero duplicates."""
    if task_count < 1:
        raise ValueError(f"task_count must be >= 1, got {task_count}.")
    if not (0 <= task_index < task_count):
        raise ValueError(
            f"task_index must satisfy 0 <= task_index < task_count ({task_count}), "
            f"got {task_index}."
        )
    return list(items[task_index::task_count])


# ---------------------------------------------------------------------------
# 2. Structural Compatibility Key & Batch Auto-Grouping
# ---------------------------------------------------------------------------


def _align_nsave_to_nwrite(nsave_val: int, nwrite_val: int) -> int:
    """Round `nsave_val` up to the smallest positive multiple of `nwrite_val`."""
    nw = max(1, int(nwrite_val))
    ns = max(1, int(nsave_val))
    return ((ns + nw - 1) // nw) * nw


def _load_item_member_spec(
    item: CampaignItem,
    *,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    nsave: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    fft_backend: str | None = None,
    force_save_restart: bool = False,
    align_nsave_for_resume: bool = False,
) -> EnsembleMemberSpec:
    cfg_loaded = load_config(item.input_path)
    cfg_over = _apply_cfg_overrides(
        cfg_loaded,
        nstep=nstep,
        dt=dt,
        t_max=t_max,
        nwrite=nwrite,
        scheme=scheme,
        force_nonlinear=force_nonlinear,
        force_linear=force_linear,
        fft_backend=fft_backend,
    )
    extra: dict[str, Any] = {}
    eff_nsave = int(cfg_over.nsave)
    if nsave is not None:
        if int(nsave) < 1:
            raise ValueError(f"nsave must be >= 1, got {nsave}.")
        eff_nsave = int(nsave)
    if align_nsave_for_resume:
        eff_nsave = _align_nsave_to_nwrite(eff_nsave, int(cfg_over.nwrite))
    if eff_nsave != int(cfg_over.nsave):
        extra["nsave"] = eff_nsave
    if force_save_restart and not cfg_over.save_for_restart:
        extra["save_for_restart"] = True
    if extra:
        cfg_over = replace(cfg_over, **extra)
    cfg_fin, grids_fin, geo_fin = init_geometry(cfg_over)
    return EnsembleMemberSpec(
        cfg=cfg_fin,
        grids=grids_fin,
        geo=geo_fin,
        input_path=item.input_path,
        name=item.stem,
    )


def compatibility_key(spec: EnsembleMemberSpec) -> tuple[Any, ...]:
    """Build the structural shape/compile-time key matching `validate_ensemble_compatibility`."""
    c, g = spec.cfg, spec.grids
    c_closure = c.closure_model.strip().lower()
    return (
        int(g.nspecies),
        int(g.nhermite),
        int(g.nlaguerre),
        int(g.nz),
        int(g.naky),
        int(g.nakx),
        int(g.nx),
        int(g.ny),
        int(g.nj),
        int(g.nyc),
        tuple(int(d) for d in g.z.shape),
        tuple(int(d) for d in spec.geo.bmag.shape),
        c.scheme.strip().lower(),
        bool(c.nonlinear_mode),
        c_closure,
        int(c.smith_par_q) if c_closure == "smith_par" else None,
        int(c.smith_perp_q) if c_closure == "smith_perp" else None,
        bool(c.all_kinetic),
        int(c.boltzmann_opt) if not c.all_kinetic else None,
        bool(c.local_limit),
        bool(c.non_twist),
        bool(c.boundary_option_periodic),
        bool(c.dealias_kz),
        bool(c.long_wavelength_gk),
        bool(c.no_fields),
        str(getattr(c, "fft_backend", "auto")).strip().lower(),
    )


def group_compatible_items(
    items: Sequence[CampaignItem],
    *,
    batch_size: int = 4,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    nsave: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    fft_backend: str | None = None,
    force_save_restart: bool = False,
    align_nsave_for_resume: bool = False,
) -> list[CampaignBatch]:
    """Group structurally compatible decks into batches of up to `batch_size` (`B >= 1`).

    Batches with `>= 2` members execute via `run_ensemble` (`mode="ensemble"`), while
    singleton shapes (`len == 1` or `batch_size == 1`) execute via `run_cli_simulation`
    (`mode="singleton"`). Every multi-member group is verified with
    :func:`gx_jax.ensemble.validate_ensemble_compatibility`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    loaded: list[tuple[CampaignItem, EnsembleMemberSpec]] = [
        (
            it,
            _load_item_member_spec(
                it,
                nstep=nstep,
                dt=dt,
                t_max=t_max,
                nwrite=nwrite,
                nsave=nsave,
                scheme=scheme,
                force_nonlinear=force_nonlinear,
                force_linear=force_linear,
                fft_backend=fft_backend,
                force_save_restart=force_save_restart,
                align_nsave_for_resume=align_nsave_for_resume,
            ),
        )
        for it in items
    ]

    if batch_size == 1:
        return [
            CampaignBatch(
                batch_index=idx,
                mode="singleton",
                items=(it,),
                specs=(sp,),
            )
            for idx, (it, sp) in enumerate(loaded)
        ]

    open_buckets: dict[
        tuple[Any, ...], list[tuple[list[CampaignItem], list[EnsembleMemberSpec]]]
    ] = {}
    ordered_groups: list[tuple[list[CampaignItem], list[EnsembleMemberSpec]]] = []

    for it, sp in loaded:
        validate_ensemble_compatibility([sp])
        key = compatibility_key(sp)
        candidates = open_buckets.setdefault(key, [])
        placed = False
        for grp_items, grp_specs in candidates:
            if len(grp_items) >= batch_size:
                continue
            try:
                validate_ensemble_compatibility([*grp_specs, sp])
            except ValueError:
                continue
            grp_items.append(it)
            grp_specs.append(sp)
            placed = True
            break
        if not placed:
            new_grp = ([it], [sp])
            candidates.append(new_grp)
            ordered_groups.append(new_grp)

    batches: list[CampaignBatch] = []
    for idx, (grp_items, grp_specs) in enumerate(ordered_groups):
        validate_ensemble_compatibility(grp_specs)
        mode = "ensemble" if len(grp_items) >= 2 else "singleton"
        batches.append(
            CampaignBatch(
                batch_index=idx,
                mode=mode,
                items=tuple(grp_items),
                specs=tuple(grp_specs),
            )
        )
    return batches


# ---------------------------------------------------------------------------
# 3. Local Scratch Staging, Atomic GCS-FUSE Promotion & Idempotency Helpers
# ---------------------------------------------------------------------------


def _sweep_stale_worker_tmp_files(output_dir: Path, *, task_index: int) -> int:
    """Remove stale `.*.t<task_index>.p*.tmp` files left by a prior SIGKILL of this worker."""
    removed = 0
    if not output_dir.is_dir():
        return removed
    pattern = f".*.t{int(task_index)}.p*.tmp"
    for stale_p in output_dir.glob(pattern):
        if stale_p.is_file():
            try:
                stale_p.unlink(missing_ok=True)
                removed += 1
            except OSError:
                pass
    return removed


def atomic_promote_file(
    src_path: Path,
    dst_path: Path,
    *,
    task_index: int = 0,
) -> None:
    """Atomically promote `src_path` to `dst_path` via hidden temp file + `fsync` + `os.replace`."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_path.with_name(f".{dst_path.name}.t{task_index}.p{os.getpid()}.tmp")
    try:
        with src_path.open("rb") as f_in, tmp_dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
            f_out.flush()
            os.fsync(f_out.fileno())
        os.replace(tmp_dst, dst_path)
    except BaseException:
        tmp_dst.unlink(missing_ok=True)
        raise


def atomic_write_json(
    dst_path: Path,
    payload: Any,
    *,
    task_index: int = 0,
) -> dict[str, Any]:
    """Atomically write strict RFC-8259 JSON (`allow_nan=False`) via temp + `fsync`."""
    clean = sanitize_json_payload(payload)
    serialized = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dst = dst_path.with_name(f".{dst_path.name}.t{task_index}.p{os.getpid()}.tmp")
    try:
        with tmp_dst.open("w", encoding="utf-8") as fh:
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_dst, dst_path)
    except BaseException:
        tmp_dst.unlink(missing_ok=True)
        raise
    assert isinstance(clean, dict)
    return clean


def _compute_sim_time_target(
    cfg: GXConfig,
    *,
    nstep_cli: int | None,
    dt_cli: float | None,
    t_max_cli: float | None,
    t0: float = 0.0,
) -> float:
    dt_nom = float(cfg.dt) if cfg.dt > 0.0 else 0.05
    step_dt = float(dt_cli) if dt_cli is not None else dt_nom
    t_max_val = float(cfg.t_max)
    nstep_max = max(0, int(cfg.nstep))
    has_explicit_t_max = (t_max_cli is not None) or (
        math.isfinite(t_max_val) and t_max_val < 1.0e19
    )
    nstep_end_target = float(t0) + float(nstep_max) * step_dt
    if t_max_cli is not None:
        return t_max_val
    if nstep_cli is not None and has_explicit_t_max:
        return min(t_max_val, nstep_end_target)
    if has_explicit_t_max:
        return t_max_val
    return nstep_end_target


def is_deck_completed(
    output_dir: Path,
    stem: str,
    *,
    expected_sim_time_target: float | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Check whether `<stem>` has valid completed `.out.nc` and `.diagnostics_metrics.json`."""
    out_nc = output_dir / f"{stem}.out.nc"
    metrics_json = output_dir / f"{stem}.diagnostics_metrics.json"
    if not out_nc.is_file() or not metrics_json.is_file():
        return False, None
    try:
        if out_nc.stat().st_size <= 0:
            return False, None
    except OSError:
        return False, None

    try:
        raw_text = metrics_json.read_text(encoding="utf-8")
        payload = json.loads(raw_text)
    except (OSError, ValueError, TypeError):
        return False, None

    if not isinstance(payload, dict):
        return False, None
    if bool(payload.get("has_nan_or_inf", False)):
        return False, None

    status = payload.get("status")
    if status is not None and status != "completed":
        return False, None

    converged = bool(payload.get("converged", False))
    if not converged and status != "completed":
        return False, None

    t_final_raw = payload.get("t_final", payload.get("sim_time_reached"))
    target_raw = (
        expected_sim_time_target
        if expected_sim_time_target is not None
        else payload.get("sim_time_target")
    )
    if isinstance(t_final_raw, (int, float)) and isinstance(target_raw, (int, float)):
        t_fin = float(t_final_raw)
        t_tgt = float(target_raw)
        if math.isfinite(t_tgt) and t_tgt > 0.0:
            eps = max(1.0e-9, 1.0e-6 * abs(t_tgt))
            if t_fin < (t_tgt - eps):
                return False, None

    try:
        with netCDF4.Dataset(str(out_nc), "r") as ds:
            if "time" in ds.dimensions and len(ds.dimensions["time"]) <= 0:
                return False, None
    except OSError:
        return False, None

    return True, payload


class CheckpointMirrorThread:
    """Background mirror promoting atomic `<stem>.restart.nc` files from scratch to `--output-dir`.

    Watches `stage_dir` for completed `<stem>.restart.nc` files (never `.tmp` files) and
    atomically promotes updates to `output_dir` so periodic `nsave` checkpoints are safely
    persisted mid-run without performing random-access HDF5 writes on GCS FUSE.
    """

    def __init__(
        self,
        *,
        stage_dir: Path,
        output_dir: Path,
        stems: Sequence[str],
        task_index: int,
        enabled: bool = True,
        poll_interval_sec: float = 0.02,
    ) -> None:
        self._stage_dir = stage_dir
        self._output_dir = output_dir
        self._stems = tuple(stems)
        self._task_index = int(task_index)
        self._enabled = bool(enabled)
        self._poll_interval_sec = max(0.005, float(poll_interval_sec))
        self._stop_event = threading.Event()
        self._seen_stat: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def record_initial_baseline(self) -> None:
        """Record pre-staged input checkpoints so only newly written checkpoints are mirrored."""
        with self._lock:
            for stem in self._stems:
                cand = self._stage_dir / f"{stem}.restart.nc"
                if cand.is_file():
                    try:
                        st = cand.stat()
                        self._seen_stat[stem] = (st.st_mtime_ns, st.st_size)
                    except OSError:
                        pass

    def sync_once(self) -> list[str]:
        """Mirror any updated non-`.tmp` `<stem>.restart.nc` checkpoints once."""
        if not self._enabled:
            return []
        promoted: list[str] = []
        with self._lock:
            for stem in self._stems:
                src = self._stage_dir / f"{stem}.restart.nc"
                if not src.is_file():
                    continue
                try:
                    st = src.stat()
                except OSError:
                    continue
                if st.st_size <= 0:
                    continue
                sig = (st.st_mtime_ns, st.st_size)
                if self._seen_stat.get(stem) == sig:
                    continue
                dst = self._output_dir / f"{stem}.restart.nc"
                try:
                    atomic_promote_file(src, dst, task_index=self._task_index)
                except OSError:
                    continue
                self._seen_stat[stem] = sig
                promoted.append(stem)
        return promoted

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_sec):
            self.sync_once()

    def start(self) -> None:
        if not self._enabled or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"gx-jax-ckpt-mirror-{self._task_index}",
            daemon=True,
        )
        self._thread.start()

    def stop_and_flush(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.sync_once()


# ---------------------------------------------------------------------------
# 4. Exact `t0 -> t_max` Checkpoint Inspection, `.out.nc` Trimming & Alignment
# ---------------------------------------------------------------------------


def _copy_netcdf_group_up_to_time(
    src_grp: netCDF4.Dataset | netCDF4.Group,
    dst_grp: netCDF4.Dataset | netCDF4.Group,
    n_keep: int,
) -> None:
    for attr_name in src_grp.ncattrs():
        dst_grp.setncattr(attr_name, src_grp.getncattr(attr_name))

    for dim_name, dim_obj in src_grp.dimensions.items():
        dst_grp.createDimension(dim_name, None if dim_obj.isunlimited() else len(dim_obj))

    for var_name, src_var in src_grp.variables.items():
        dst_var = dst_grp.createVariable(var_name, src_var.datatype, src_var.dimensions)
        for v_attr in src_var.ncattrs():
            dst_var.setncattr(v_attr, src_var.getncattr(v_attr))
        if src_var.dimensions and src_var.dimensions[0] == "time":
            if n_keep > 0:
                dst_var[:n_keep, ...] = src_var[:n_keep, ...]
        else:
            dst_var[...] = src_var[...]

    for sub_name, sub_src in src_grp.groups.items():
        sub_dst = dst_grp.createGroup(sub_name)
        _copy_netcdf_group_up_to_time(sub_src, sub_dst, n_keep)


def _stage_prior_out_nc_up_to_time(
    src_out_nc: Path,
    dst_out_nc: Path,
    max_time: float,
) -> np.ndarray:
    """Stage `src_out_nc` into `dst_out_nc` retaining only frames with `time <= max_time`."""
    if not src_out_nc.is_file():
        return np.zeros((0,), dtype=np.float64)

    try:
        with netCDF4.Dataset(str(src_out_nc), "r") as ds:
            if "Grids" not in ds.groups or "time" not in ds.groups["Grids"].variables:
                return np.zeros((0,), dtype=np.float64)
            t_arr = np.asarray(ds.groups["Grids"].variables["time"][:], dtype=np.float64)
            eps = 1.0e-9 * max(abs(float(max_time)), 1.0)
            n_keep = 0
            for tv in t_arr:
                if math.isfinite(float(tv)) and float(tv) <= float(max_time) + eps:
                    n_keep += 1
                else:
                    break
            kept_times = np.asarray(t_arr[:n_keep], dtype=np.float64)

            if n_keep <= 0:
                dst_out_nc.unlink(missing_ok=True)
                return kept_times

            if n_keep == len(t_arr) and np.all(np.isfinite(t_arr)):
                need_trim = False
            else:
                need_trim = True

            if need_trim:
                dst_out_nc.parent.mkdir(parents=True, exist_ok=True)
                tmp_nc = dst_out_nc.with_name(f".{dst_out_nc.name}.{os.getpid()}.trim.tmp")
                try:
                    with netCDF4.Dataset(str(tmp_nc), "w", format="NETCDF4") as dst_ds:
                        _copy_netcdf_group_up_to_time(ds, dst_ds, n_keep)
                    os.replace(tmp_nc, dst_out_nc)
                except BaseException:
                    tmp_nc.unlink(missing_ok=True)
                    raise
                return kept_times
    except OSError:
        dst_out_nc.unlink(missing_ok=True)
        return np.zeros((0,), dtype=np.float64)

    dst_out_nc.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_out_nc, dst_out_nc)
    return kept_times


def _infer_completed_steps_at_checkpoint(
    *,
    t0: float,
    cfg: GXConfig,
    dt_cli: float | None,
    prior_frame_times: np.ndarray,
) -> tuple[int, str]:
    """Resolve `(k0, source)` (`fixed_dt`, `frame_match`, or `estimated`)."""
    dt_nom = float(cfg.dt) if cfg.dt > 0.0 else 0.05
    dt_eff = float(dt_cli) if dt_cli is not None else dt_nom
    is_fixed = _scheme_honors_fixed_dt(cfg.scheme) and (dt_cli is not None or bool(cfg.fixed_dt))
    if is_fixed and dt_eff > 0.0:
        return max(0, int(round(float(t0) / dt_eff))), "fixed_dt"

    nwrite_eff = max(1, int(cfg.nwrite))
    nsave_eff = max(1, int(cfg.nsave))
    m_frames = int(prior_frame_times.size)
    if m_frames > 0:
        t_last = float(prior_frame_times[-1])
        if abs(t_last - float(t0)) <= 1.0e-11 * max(abs(float(t0)), 1.0):
            return 1 + (m_frames - 1) * nwrite_eff, "frame_match"

    # Fallback when mirrored .restart.nc is newer than promoted .out.nc (e.g. hard SIGKILL)
    k_low = 0 if m_frames == 0 else 1 + (m_frames - 1) * nwrite_eff
    k_high = 1 + m_frames * nwrite_eff
    t_base = float(prior_frame_times[-1]) if m_frames > 0 else 0.0
    dt_est = (t_base / float(k_low)) if (k_low > 0 and t_base > 0.0) else dt_eff
    k_est = float(k_low) + max(0.0, (float(t0) - t_base) / max(dt_est, 1.0e-30))
    cands = [k for k in range(k_low + 1, k_high) if (k - 1) % nsave_eff == 0]
    if cands and k_est < float(k_high):
        return min(cands, key=lambda k: abs(float(k) - k_est)), "estimated"
    return max(k_low + 1, int(round(k_est))), "estimated"


def _advance_prealignment_steps(
    spec: EnsembleMemberSpec,
    *,
    g_ckpt: jax.Array,
    t0: float,
    k0: int,
    rem_nstep: int,
    dt_cli: float | None,
    real_dtype: jnp.dtype,
    staged_restart_path: Path,
    staged_out_nc_path: Path,
    preemption: PreemptionController,
) -> tuple[float, int, int]:
    """Advance `(-k0) % nwrite` steps so resumed `counter=0` cadence matches uninterrupted run."""
    cfg = spec.cfg
    nwrite_eff = max(1, int(cfg.nwrite))
    delta_align = (-int(k0)) % nwrite_eff
    t_max_val = float(cfg.t_max)
    eps_t = 1.0e-11 * max(abs(t_max_val), 1.0e-30)
    has_finite_tmax = math.isfinite(t_max_val) and t_max_val < 1.0e19

    if delta_align <= 0 or rem_nstep <= 0 or (has_finite_tmax and float(t0) >= (t_max_val - eps_t)):
        return float(t0), int(k0), int(rem_nstep)

    preemption.check()
    pre_steps = min(int(rem_nstep), int(delta_align))
    gk_op = init_gyrokinetic_operator(cfg, spec.grids, spec.geo, dtype=real_dtype)
    dt_nom = float(cfg.dt) if cfg.dt > 0.0 else 0.05
    dt_init = float(dt_cli) if dt_cli is not None else (dt_nom if cfg.fixed_dt else None)
    st0 = init_integrator_state(g_ckpt, gk_op, time=float(t0), dt=dt_init)

    chunk = advance_integrator_chunk(
        st0,
        gk_op,
        pre_steps,
        dt_override=dt_cli,
        scheme=cfg.scheme,
        t_max=t_max_val,
    )
    steps_done = int(chunk.steps_done)
    if steps_done <= 0:
        return float(t0), int(k0), int(rem_nstep)

    st_end = chunk.state
    t_after = float(st_end.time)
    k_after = int(k0) + steps_done
    rem_after = max(0, int(rem_nstep) - steps_done)

    with preemption.shield():
        # During k0+1 .. k0+delta_align, cadence writes never fire; only t_max overshoot can.
        if has_finite_tmax and t_after > (t_max_val + eps_t):
            d_prev = DiagnosticsState(
                G_old=chunk.G_prev,
                fields_old=chunk.fields_prev,
                time_old=chunk.time_prev,
                has_old=jnp.asarray(True, dtype=jnp.bool_),
            )
            snap, _ = compute_diagnostics_step(
                st_end.G,
                st_end.fields,
                d_prev,
                spec.grids,
                spec.geo,
                gk_op.linear_op.solver,
                time=st_end.time,
                dt=st_end.dt,
                beta=float(cfg.beta),
            )
            with DiagnosticsWriter(
                staged_out_nc_path,
                cfg,
                spec.grids,
                spec.geo,
                append=staged_out_nc_path.is_file(),
            ) as writer:
                g_snap = st_end.G if cfg.write_moms else None
                f_snap = st_end.fields if (cfg.write_fields or cfg.write_moms) else None
                writer.write_snapshot(
                    snap,
                    G=g_snap,
                    fields=f_snap,
                    solver=gk_op.linear_op.solver,
                )

        tmp_rst = staged_restart_path.with_name(
            f".{staged_restart_path.name}.{os.getpid()}.align.tmp"
        )
        try:
            save_restart_nc(tmp_rst, st_end.G, t_after, spec.grids)
            os.replace(tmp_rst, staged_restart_path)
        except BaseException:
            tmp_rst.unlink(missing_ok=True)
            raise

    return t_after, k_after, rem_after


def _advance_prealignment_ensemble_batch(
    specs: Sequence[EnsembleMemberSpec],
    *,
    g_ckpts: Sequence[jax.Array | None],
    t0_list: Sequence[float],
    k0_list: Sequence[int],
    rem_nstep_list: Sequence[int],
    resumed_flags: Sequence[bool],
    dt_cli: float | None,
    fft_backend: str | None,
    real_dtype: jnp.dtype,
    complex_dtype: jnp.dtype,
    staged_restart_paths: Sequence[Path],
    staged_out_nc_paths: Sequence[Path],
    preemption: PreemptionController,
) -> list[tuple[float, int, int]]:
    """Advance `(-k0_b) % nwrite_b` across an ensemble batch via `BatchedEnsembleOperators`."""
    b_sz = len(specs)
    pre_steps_per_member: list[int] = []
    for b in range(b_sz):
        if not resumed_flags[b]:
            pre_steps_per_member.append(0)
            continue
        cfg_b = specs[b].cfg
        nw_b = max(1, int(cfg_b.nwrite))
        delta_b = (-int(k0_list[b])) % nw_b
        tm_b = float(cfg_b.t_max)
        eps_b = 1.0e-11 * max(abs(tm_b), 1.0e-30)
        has_fin_tm = math.isfinite(tm_b) and tm_b < 1.0e19
        if (
            delta_b <= 0
            or rem_nstep_list[b] <= 0
            or (has_fin_tm and float(t0_list[b]) >= (tm_b - eps_b))
        ):
            pre_steps_per_member.append(0)
        else:
            pre_steps_per_member.append(min(int(rem_nstep_list[b]), int(delta_b)))

    max_pre = max(pre_steps_per_member, default=0)
    if max_pre <= 0:
        return [(float(t0_list[b]), int(k0_list[b]), int(rem_nstep_list[b])) for b in range(b_sz)]

    preemption.check()
    align_specs = [
        EnsembleMemberSpec(
            cfg=replace(specs[b].cfg, nstep=pre_steps_per_member[b]),
            grids=specs[b].grids,
            geo=specs[b].geo,
            input_path=specs[b].input_path,
            name=specs[b].name,
        )
        for b in range(b_sz)
    ]
    _, batch_ops = init_batched_ensemble(
        align_specs,
        dtype=real_dtype,
        dt=dt_cli,
        fft_backend=fft_backend,
    )
    g_init_list = [
        g_ckpts[b].astype(complex_dtype)
        if (g_ckpts[b] is not None and pre_steps_per_member[b] > 0)
        else jnp.zeros(specs[b].grids.g_shape, dtype=complex_dtype)
        for b in range(b_sz)
    ]
    st = init_batched_integrator_state(
        jnp.stack(g_init_list, axis=0),
        batch_ops,
        time=[float(t) for t in t0_list],
        dt_override=dt_cli,
    )
    step_fn = jax.jit(
        lambda s_in: batched_step_integrator_state(s_in, batch_ops, dt_override=dt_cli)
    )

    g_prev = st.G
    f_prev = st.fields
    t_prev = st.time
    for _ in range(max_pre):
        preemption.check()
        st_next = step_fn(st)
        advanced = st_next.step > st.step
        g_prev = jnp.where(advanced[:, None, None, None, None, None, None], st.G, g_prev)
        f_prev = jax.tree_util.tree_map(
            lambda cur_v, prv_v, adv=advanced: jnp.where(
                adv.reshape((b_sz,) + (1,) * (cur_v.ndim - 1)), cur_v, prv_v
            ),
            st.fields,
            f_prev,
        )
        t_prev = jnp.where(advanced, st.time, t_prev)
        st = st_next

    steps_np = np.asarray(jax.device_get(st.step), dtype=np.int64)
    times_np = np.asarray(jax.device_get(st.time), dtype=np.float64)
    g_host = jax.device_get(st.G)
    f_host = jax.device_get(st.fields)

    # Check if any member hit t_max overshoot during pre-alignment
    overshoot_members = []
    for b in range(b_sz):
        if int(steps_np[b]) <= 0:
            continue
        tm_b = float(specs[b].cfg.t_max)
        eps_b = 1.0e-11 * max(abs(tm_b), 1.0e-30)
        if math.isfinite(tm_b) and tm_b < 1.0e19 and float(times_np[b]) > (tm_b + eps_b):
            overshoot_members.append(b)

    snap_host = None
    solver_host = None
    if overshoot_members:
        dg_prev = DiagnosticsState(
            G_old=g_prev,
            fields_old=f_prev,
            time_old=t_prev,
            has_old=jnp.ones((b_sz,), dtype=jnp.bool_),
        )
        snaps_b, _ = batched_compute_diagnostics_step(
            st.G, st.fields, dg_prev, batch_ops, time=st.time, dt=st.dt
        )
        snap_host = jax.device_get(snaps_b)
        solver_host = jax.device_get(batch_ops.solver)

    out_triples: list[tuple[float, int, int]] = []
    with preemption.shield():
        for b in range(b_sz):
            sd_b = int(steps_np[b])
            if sd_b <= 0:
                out_triples.append((float(t0_list[b]), int(k0_list[b]), int(rem_nstep_list[b])))
                continue
            t_after_b = float(times_np[b])
            k_after_b = int(k0_list[b]) + sd_b
            rem_after_b = max(0, int(rem_nstep_list[b]) - sd_b)

            if b in overshoot_members and snap_host is not None and solver_host is not None:
                snap_b = _index_batch_host_tree(snap_host, b)
                solv_b = _index_batch_host_tree(solver_host, b)
                cfg_b = specs[b].cfg
                with DiagnosticsWriter(
                    staged_out_nc_paths[b],
                    cfg_b,
                    specs[b].grids,
                    specs[b].geo,
                    append=staged_out_nc_paths[b].is_file(),
                ) as writer:
                    g_snap_b = jnp.asarray(g_host[b]) if cfg_b.write_moms else None
                    f_snap_b = (
                        Fields(
                            phi=jnp.asarray(f_host.phi[b]),
                            apar=jnp.asarray(f_host.apar[b]),
                            bpar=jnp.asarray(f_host.bpar[b]),
                            phi_avg=jnp.asarray(f_host.phi_avg[b]),
                        )
                        if (cfg_b.write_fields or cfg_b.write_moms)
                        else None
                    )
                    writer.write_snapshot(snap_b, G=g_snap_b, fields=f_snap_b, solver=solv_b)

            rst_p = staged_restart_paths[b]
            tmp_rst = rst_p.with_name(f".{rst_p.name}.{os.getpid()}.{b}.align.tmp")
            try:
                save_restart_nc(tmp_rst, jnp.asarray(g_host[b]), t_after_b, specs[b].grids)
                os.replace(tmp_rst, rst_p)
            except BaseException:
                tmp_rst.unlink(missing_ok=True)
                raise
            out_triples.append((t_after_b, k_after_b, rem_after_b))

    return out_triples


def _format_toml_scalar(val: Any) -> str:
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError(f"Cannot serialize non-finite float {val!r} to TOML.")
        return repr(val)
    if isinstance(val, str):
        return json.dumps(val)
    if isinstance(val, (list, tuple)):
        return "[ " + ", ".join(_format_toml_scalar(x) for x in val) + " ]"
    raise TypeError(f"Unsupported TOML value type: {type(val).__name__}")


def _write_staged_singleton_deck(
    orig_input_path: Path,
    staged_deck_path: Path,
    *,
    nsave_override: int | None,
    force_append_on_restart: bool,
    force_save_for_restart: bool,
) -> Path:
    """Write a staged singleton `.in` TOML deck resolving relative geometry file paths."""
    with orig_input_path.open("rb") as fh:
        nml = tomllib.load(fh)

    orig_dir = orig_input_path.resolve().parent
    geo_sec = nml.get("Geometry")
    if isinstance(geo_sec, dict):
        for gkey in ("geo_file", "geofilename", "geofile", "vmec_file"):
            gval = geo_sec.get(gkey)
            if isinstance(gval, str) and gval.strip():
                gp = Path(gval.strip())
                if not gp.is_absolute():
                    cand = (orig_dir / gp).resolve()
                    if cand.exists():
                        geo_sec[gkey] = str(cand)

    rest_sec = nml.get("Restart")
    if not isinstance(rest_sec, dict):
        rest_sec = {}
        nml["Restart"] = rest_sec
    if nsave_override is not None:
        rest_sec["nsave"] = int(nsave_override)
    if force_append_on_restart:
        rest_sec["append_on_restart"] = True
    if force_save_for_restart:
        rest_sec["save_for_restart"] = True

    lines: list[str] = []
    for k, v in nml.items():
        if not isinstance(v, dict):
            lines.append(f"{k} = {_format_toml_scalar(v)}")
    for sec_name, sec_val in nml.items():
        if not isinstance(sec_val, dict):
            continue
        if lines:
            lines.append("")
        lines.append(f"[{sec_name}]")
        for k, v in sec_val.items():
            if isinstance(v, dict):
                lines.append(f"[{sec_name}.{k}]")
                for sk, sv in v.items():
                    lines.append(f" {sk} = {_format_toml_scalar(sv)}")
            else:
                lines.append(f" {k} = {_format_toml_scalar(v)}")

    staged_deck_path.parent.mkdir(parents=True, exist_ok=True)
    staged_deck_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return staged_deck_path


# ---------------------------------------------------------------------------
# 5. End-to-End Campaign Execution & Preemption Recovery
# ---------------------------------------------------------------------------


@dataclass
class _PreparedMember:
    item: CampaignItem
    base_spec: EnsembleMemberSpec
    exec_spec: EnsembleMemberSpec
    staged_out_nc: Path
    staged_restart_nc: Path
    resume_source_rst_nc: Path | None
    staged_metrics_json: Path
    full_sim_time_target: float
    want_restart: bool
    resumed: bool
    resumed_from_time: float
    resume_step_source: str | None
    prior_steps: int


def _salvage_preempted_batch(
    prepared: Sequence[_PreparedMember],
    *,
    output_dir: Path,
    task_index: int,
    complex_dtype: jnp.dtype,
) -> list[dict[str, Any]]:
    """Promote valid non-`.tmp` checkpoints & aligned `.out.nc` prefixes on `SIGTERM`."""
    checkpointed_records: list[dict[str, Any]] = []
    if not _is_primary_process():
        return checkpointed_records

    for pm in prepared:
        # Never promote .tmp or pre-aligned input files; check ONLY canonical <stem>.restart.nc
        staged_rst = pm.staged_restart_nc
        out_rst = output_dir / f"{pm.item.stem}.restart.nc"

        cand_rst: Path | None = None
        t_ckpt: float | None = None
        if staged_rst.is_file():
            try:
                _, t_val = load_restart_nc(
                    staged_rst,
                    pm.base_spec.grids,
                    scale=float(pm.base_spec.cfg.restart_scale),
                    dtype=complex_dtype,
                )
                cand_rst = staged_rst
                t_ckpt = float(t_val)
            except (OSError, ValueError):
                cand_rst = None

        if cand_rst is not None and t_ckpt is not None:
            # Promote trimmed .out.nc BEFORE .restart.nc so frame history is always present first
            if pm.staged_out_nc.is_file():
                clean_out_stage = pm.staged_out_nc.with_name(f"{pm.item.stem}.preempt_clean.out.nc")
                kept_t = _stage_prior_out_nc_up_to_time(pm.staged_out_nc, clean_out_stage, t_ckpt)
                if kept_t.size > 0 and clean_out_stage.is_file():
                    atomic_promote_file(
                        clean_out_stage,
                        output_dir / f"{pm.item.stem}.out.nc",
                        task_index=task_index,
                    )
                    clean_out_stage.unlink(missing_ok=True)
            atomic_promote_file(cand_rst, out_rst, task_index=task_index)
            checkpointed_records.append(
                {
                    "stem": pm.item.stem,
                    "input_file": str(pm.item.input_path),
                    "restart_nc": str(out_rst),
                    "t_checkpoint": float(t_ckpt),
                }
            )
        elif out_rst.is_file():
            try:
                _, t_existing = load_restart_nc(
                    out_rst,
                    pm.base_spec.grids,
                    scale=float(pm.base_spec.cfg.restart_scale),
                    dtype=complex_dtype,
                )
                if pm.staged_out_nc.is_file():
                    clean_out_stage = pm.staged_out_nc.with_name(
                        f"{pm.item.stem}.preempt_clean.out.nc"
                    )
                    kept_t = _stage_prior_out_nc_up_to_time(
                        pm.staged_out_nc, clean_out_stage, float(t_existing)
                    )
                    if kept_t.size > 0 and clean_out_stage.is_file():
                        atomic_promote_file(
                            clean_out_stage,
                            output_dir / f"{pm.item.stem}.out.nc",
                            task_index=task_index,
                        )
                        clean_out_stage.unlink(missing_ok=True)
                checkpointed_records.append(
                    {
                        "stem": pm.item.stem,
                        "input_file": str(pm.item.input_path),
                        "restart_nc": str(out_rst),
                        "t_checkpoint": float(t_existing),
                    }
                )
            except (OSError, ValueError):
                pass

    return checkpointed_records


def run_campaign(
    manifest_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    scratch_dir: str | Path = "/tmp/gx_jax_stage",
    ledger_json_path: str | Path | None = None,
    task_index: int | None = None,
    task_count: int | None = None,
    batch_size: int = 4,
    batch_devices: int | None = None,
    mesh_shape: tuple[int, int] | None = None,
    skip_completed: bool | None = None,
    auto_resume: bool = True,
    nstep: int | None = None,
    dt: float | None = None,
    t_max: float | None = None,
    nwrite: int | None = None,
    nsave: int | None = None,
    scheme: str | None = None,
    force_nonlinear: bool = False,
    force_linear: bool = False,
    precision: str = "fp32",
    fft_backend: str | None = None,
    async_io: bool = False,
    save_restart: bool | None = None,
    trap_signals: bool = True,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute a sliced, auto-batched, preemption-resilient GX-JAX simulation campaign."""
    configure_tpu_precision()
    prec_clean = precision.strip().lower()
    if prec_clean not in ("fp32", "fp64"):
        raise ValueError(f"Invalid precision={precision!r}; expected 'fp32' or 'fp64'.")
    use_fp64 = prec_clean == "fp64"
    real_dtype = jnp.float64 if use_fp64 else jnp.float32
    complex_dtype = jnp.complex128 if use_fp64 else jnp.complex64

    if force_nonlinear and force_linear:
        raise ValueError("force_nonlinear and force_linear are mutually exclusive.")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

    m_path = Path(manifest_path)
    all_items = load_campaign_manifest(m_path)
    worker = detect_worker_slice(task_index=task_index, task_count=task_count, env=env)
    assigned_items = slice_campaign_items(
        all_items,
        task_index=worker.task_index,
        task_count=worker.task_count,
    )

    if output_dir is not None:
        out_dir = Path(output_dir).resolve()
    else:
        base_out = m_path.resolve() if m_path.is_dir() else m_path.resolve().parent
        out_dir = base_out / "campaign_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if _is_primary_process():
        _sweep_stale_worker_tmp_files(out_dir, task_index=worker.task_index)

    ledger_path = (
        Path(ledger_json_path).resolve()
        if ledger_json_path is not None
        else out_dir / f"campaign_task_{worker.task_index:04d}.json"
    )

    eff_skip_completed = bool(auto_resume) if skip_completed is None else bool(skip_completed)
    force_save_restart = bool(auto_resume) if save_restart is None else bool(save_restart)

    scratch_root = Path(scratch_dir).resolve()
    worker_stage_dir = (
        scratch_root / f"worker_t{worker.task_index:04d}_of_{worker.task_count:04d}_p{os.getpid()}"
    )

    t_campaign_start = time.perf_counter()
    completed_stems: list[str] = []
    skipped_stems: list[str] = []
    resumed_stems: list[str] = []
    estimated_resume_stems: list[str] = []
    deck_records: dict[str, dict[str, Any]] = {}
    batch_summaries: list[dict[str, Any]] = []

    def _build_ledger_payload(
        *,
        status: str,
        exit_code: int,
        signal_name: str | None = None,
        checkpointed: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        ckpt_list = checkpointed or []
        ckpt_by_stem = {c["stem"]: c for c in ckpt_list}
        done_set = set(completed_stems) | set(skipped_stems)
        pending_stems = [it.stem for it in assigned_items if it.stem not in done_set]
        ordered_records: list[dict[str, Any]] = []
        for it in assigned_items:
            if it.stem in deck_records:
                ordered_records.append(deck_records[it.stem])
            elif it.stem in ckpt_by_stem:
                c_info = ckpt_by_stem[it.stem]
                ordered_records.append(
                    {
                        "index": it.index,
                        "stem": it.stem,
                        "input_file": str(it.input_path),
                        "status": "checkpointed",
                        "restart_nc": c_info["restart_nc"],
                        "t_checkpoint": c_info["t_checkpoint"],
                    }
                )
            else:
                ordered_records.append(
                    {
                        "index": it.index,
                        "stem": it.stem,
                        "input_file": str(it.input_path),
                        "status": "pending",
                    }
                )
        return {
            "status": status,
            "exit_code": int(exit_code),
            "signal": signal_name,
            "manifest": str(m_path.resolve()),
            "output_dir": str(out_dir),
            "scratch_dir": str(scratch_root),
            "task_index": worker.task_index,
            "task_count": worker.task_count,
            "worker_source": worker.source,
            "total_manifest_decks": len(all_items),
            "assigned_deck_count": len(assigned_items),
            "completed_count": len(completed_stems),
            "skipped_count": len(skipped_stems),
            "resumed_count": len(resumed_stems),
            "estimated_resume_count": len(estimated_resume_stems),
            "checkpointed_count": len(ckpt_list),
            "pending_count": len(pending_stems),
            "completed_decks": list(completed_stems),
            "skipped_decks": list(skipped_stems),
            "resumed_decks": list(resumed_stems),
            "estimated_resume_decks": list(estimated_resume_stems),
            "checkpointed_decks": ckpt_list,
            "pending_decks": pending_stems,
            "batches": list(batch_summaries),
            "decks": ordered_records,
            "precision": prec_clean,
            "batch_size_limit": int(batch_size),
            "auto_resume": bool(auto_resume),
            "skip_completed": bool(eff_skip_completed),
            "wall_seconds": float(time.perf_counter() - t_campaign_start),
        }

    with PreemptionController(trap_signals=trap_signals) as preemption:
        active_prepared: list[_PreparedMember] = []
        active_mirror: CheckpointMirrorThread | None = None
        try:
            worker_stage_dir.mkdir(parents=True, exist_ok=True)

            # 1. Filter completed decks (--skip-completed) before grouping active work
            todo_items: list[CampaignItem] = []
            for it in assigned_items:
                preemption.check()
                probe_spec = _load_item_member_spec(
                    it,
                    nstep=nstep,
                    dt=dt,
                    t_max=t_max,
                    nwrite=nwrite,
                    nsave=nsave,
                    scheme=scheme,
                    force_nonlinear=force_nonlinear,
                    force_linear=force_linear,
                    fft_backend=fft_backend,
                    force_save_restart=force_save_restart,
                    align_nsave_for_resume=bool(auto_resume),
                )
                full_target = _compute_sim_time_target(
                    probe_spec.cfg,
                    nstep_cli=nstep,
                    dt_cli=dt,
                    t_max_cli=t_max,
                    t0=0.0,
                )
                if eff_skip_completed:
                    is_done, existing_metrics = is_deck_completed(
                        out_dir,
                        it.stem,
                        expected_sim_time_target=full_target,
                    )
                    if is_done and existing_metrics is not None:
                        skipped_stems.append(it.stem)
                        deck_records[it.stem] = {
                            "index": it.index,
                            "stem": it.stem,
                            "input_file": str(it.input_path),
                            "status": "skipped",
                            "output_nc": str(out_dir / f"{it.stem}.out.nc"),
                            "metrics_json": str(out_dir / f"{it.stem}.diagnostics_metrics.json"),
                            "t_final": existing_metrics.get("t_final"),
                            "nstep": existing_metrics.get("nstep"),
                        }
                        continue
                todo_items.append(it)

            # 2. Group remaining active items into compatible batches of size <= batch_size
            batches = group_compatible_items(
                todo_items,
                batch_size=batch_size,
                nstep=nstep,
                dt=dt,
                t_max=t_max,
                nwrite=nwrite,
                nsave=nsave,
                scheme=scheme,
                force_nonlinear=force_nonlinear,
                force_linear=force_linear,
                fft_backend=fft_backend,
                force_save_restart=force_save_restart,
                align_nsave_for_resume=bool(auto_resume),
            )

            # 3. Execute each batch inside isolated scratch staging directory
            for batch in batches:
                preemption.check()
                batch_stage_dir = worker_stage_dir / f"batch_{batch.batch_index:04d}"
                batch_stage_dir.mkdir(parents=True, exist_ok=True)
                t_batch_start = time.perf_counter()

                raw_members: list[dict[str, Any]] = []
                for it, sp in zip(batch.items, batch.specs, strict=True):
                    preemption.check()
                    orig_cfg = sp.cfg
                    want_rst = (
                        bool(orig_cfg.save_for_restart)
                        if (save_restart is None and not auto_resume)
                        else force_save_restart
                    )
                    full_target = _compute_sim_time_target(
                        orig_cfg,
                        nstep_cli=nstep,
                        dt_cli=dt,
                        t_max_cli=t_max,
                        t0=0.0,
                    )
                    staged_out_nc = batch_stage_dir / f"{it.stem}.out.nc"
                    # Canonical staged_rst_nc stays absent until the solver writes a new checkpoint;
                    # incoming/pre-aligned restart input uses non-canonical resume_source_rst_nc!
                    staged_rst_nc = batch_stage_dir / f"{it.stem}.restart.nc"
                    resume_src_nc = batch_stage_dir / f".{it.stem}.resume_source.nc"
                    staged_metrics = batch_stage_dir / f"{it.stem}.diagnostics_metrics.json"

                    resumed = False
                    resumed_from_t = 0.0
                    resume_source_kind: str | None = None
                    k0 = 0
                    rem_nstep0 = max(0, int(orig_cfg.nstep))
                    g_ckpt_opt: jax.Array | None = None

                    existing_rst = out_dir / f"{it.stem}.restart.nc"
                    if auto_resume and existing_rst.is_file():
                        try:
                            g_loaded, t0_loaded = load_restart_nc(
                                existing_rst,
                                sp.grids,
                                scale=float(orig_cfg.restart_scale),
                                dtype=complex_dtype,
                            )
                            valid_ckpt = True
                        except (OSError, ValueError):
                            g_loaded = jnp.zeros((1,), dtype=complex_dtype)
                            t0_loaded = 0.0
                            valid_ckpt = False

                        if valid_ckpt:
                            resumed = True
                            resumed_from_t = float(t0_loaded)
                            g_ckpt_opt = g_loaded
                            shutil.copy2(existing_rst, resume_src_nc)
                            kept_times = _stage_prior_out_nc_up_to_time(
                                out_dir / f"{it.stem}.out.nc",
                                staged_out_nc,
                                float(t0_loaded),
                            )
                            k0, resume_source_kind = _infer_completed_steps_at_checkpoint(
                                t0=float(t0_loaded),
                                cfg=orig_cfg,
                                dt_cli=dt,
                                prior_frame_times=kept_times,
                            )
                            rem_nstep0 = max(0, int(orig_cfg.nstep) - k0)

                    raw_members.append(
                        {
                            "item": it,
                            "sp": sp,
                            "orig_cfg": orig_cfg,
                            "want_rst": want_rst,
                            "full_target": full_target,
                            "staged_out_nc": staged_out_nc,
                            "staged_rst_nc": staged_rst_nc,
                            "resume_src_nc": resume_src_nc if resumed else None,
                            "staged_metrics": staged_metrics,
                            "resumed": resumed,
                            "resumed_from_t": resumed_from_t,
                            "resume_step_source": resume_source_kind,
                            "g_ckpt": g_ckpt_opt,
                            "k0": k0,
                            "rem_nstep0": rem_nstep0,
                        }
                    )

                if len(raw_members) >= 2:
                    aligned_triples = _advance_prealignment_ensemble_batch(
                        [m["sp"] for m in raw_members],
                        g_ckpts=[m["g_ckpt"] for m in raw_members],
                        t0_list=[m["resumed_from_t"] for m in raw_members],
                        k0_list=[m["k0"] for m in raw_members],
                        rem_nstep_list=[m["rem_nstep0"] for m in raw_members],
                        resumed_flags=[m["resumed"] for m in raw_members],
                        dt_cli=dt,
                        fft_backend=fft_backend,
                        real_dtype=real_dtype,
                        complex_dtype=complex_dtype,
                        staged_restart_paths=[
                            m["resume_src_nc"]
                            if m["resume_src_nc"] is not None
                            else (batch_stage_dir / f".{m['item'].stem}.unused_align.nc")
                            for m in raw_members
                        ],
                        staged_out_nc_paths=[m["staged_out_nc"] for m in raw_members],
                        preemption=preemption,
                    )
                else:
                    m0 = raw_members[0]
                    if (
                        m0["resumed"]
                        and m0["g_ckpt"] is not None
                        and m0["resume_src_nc"] is not None
                    ):
                        aligned_triples = [
                            _advance_prealignment_steps(
                                m0["sp"],
                                g_ckpt=m0["g_ckpt"],
                                t0=m0["resumed_from_t"],
                                k0=m0["k0"],
                                rem_nstep=m0["rem_nstep0"],
                                dt_cli=dt,
                                real_dtype=real_dtype,
                                staged_restart_path=m0["resume_src_nc"],
                                staged_out_nc_path=m0["staged_out_nc"],
                                preemption=preemption,
                            )
                        ]
                    else:
                        aligned_triples = [(m0["resumed_from_t"], m0["k0"], m0["rem_nstep0"])]

                prepared_batch: list[_PreparedMember] = []
                for m, (_t_al, k_aligned, rem_aligned) in zip(
                    raw_members, aligned_triples, strict=True
                ):
                    it = m["item"]
                    sp = m["sp"]
                    orig_cfg = m["orig_cfg"]
                    want_rst = m["want_rst"]
                    resumed = m["resumed"]
                    staged_rst_nc = m["staged_rst_nc"]
                    resume_src_opt = m["resume_src_nc"]
                    if resumed and resume_src_opt is not None:
                        exec_cfg = replace(
                            orig_cfg,
                            nstep=rem_aligned,
                            restart=True,
                            restart_if_exists=False,
                            restart_from_file=str(resume_src_opt),
                            restart_with_perturb=False,
                            append_on_restart=True,
                            save_for_restart=want_rst,
                        )
                        prior_steps = k_aligned
                    else:
                        exec_cfg = replace(orig_cfg, save_for_restart=want_rst)
                        prior_steps = 0

                    exec_spec = EnsembleMemberSpec(
                        cfg=exec_cfg,
                        grids=sp.grids,
                        geo=sp.geo,
                        input_path=it.input_path,
                        name=it.stem,
                    )
                    prepared_batch.append(
                        _PreparedMember(
                            item=it,
                            base_spec=sp,
                            exec_spec=exec_spec,
                            staged_out_nc=m["staged_out_nc"],
                            staged_restart_nc=staged_rst_nc,
                            resume_source_rst_nc=resume_src_opt,
                            staged_metrics_json=m["staged_metrics"],
                            full_sim_time_target=m["full_target"],
                            want_restart=want_rst,
                            resumed=resumed,
                            resumed_from_time=m["resumed_from_t"],
                            resume_step_source=m["resume_step_source"],
                            prior_steps=prior_steps,
                        )
                    )

                active_prepared = prepared_batch
                mirror = CheckpointMirrorThread(
                    stage_dir=batch_stage_dir,
                    output_dir=out_dir,
                    stems=[pm.item.stem for pm in prepared_batch],
                    task_index=worker.task_index,
                    enabled=bool(
                        _is_primary_process() and any(pm.want_restart for pm in prepared_batch)
                    ),
                )
                mirror.record_initial_baseline()
                active_mirror = mirror
                mirror.start()

                if len(prepared_batch) >= 2:
                    eff_b_devs = (
                        batch_devices
                        if (batch_devices is not None and len(prepared_batch) % batch_devices == 0)
                        else None
                    )
                    ens_out = run_ensemble(
                        [pm.exec_spec for pm in prepared_batch],
                        output_dir=batch_stage_dir,
                        summary_json_path=batch_stage_dir / "batch_ensemble_summary.json",
                        dt=dt,
                        precision=prec_clean,
                        fft_backend=fft_backend,
                        batch_devices=eff_b_devs,
                        save_restart=any(pm.want_restart for pm in prepared_batch),
                    )
                    steps_ran_list = [int(s) for s in np.asarray(ens_out["nstep"]).tolist()]
                    t_final_list = [float(t) for t in np.asarray(ens_out["t_final"]).tolist()]
                else:
                    pm0 = prepared_batch[0]
                    cfg0 = pm0.exec_spec.cfg
                    raw_deck_cfg = load_config(pm0.item.input_path)
                    nsave_changed = int(cfg0.nsave) != int(raw_deck_cfg.nsave)
                    need_staged_deck = (
                        nsave_changed
                        or (pm0.resumed and not raw_deck_cfg.append_on_restart)
                        or (pm0.want_restart and not raw_deck_cfg.save_for_restart)
                    )
                    cli_deck = (
                        _write_staged_singleton_deck(
                            pm0.item.input_path,
                            batch_stage_dir / f"{pm0.item.stem}.staged.in",
                            nsave_override=int(cfg0.nsave) if nsave_changed else None,
                            force_append_on_restart=pm0.resumed,
                            force_save_for_restart=pm0.want_restart,
                        )
                        if need_staged_deck
                        else pm0.item.input_path
                    )
                    eff_mesh_shape = (
                        mesh_shape
                        if (
                            mesh_shape is not None
                            and int(cfg0.nspecies) % int(mesh_shape[0]) == 0
                            and int(cfg0.nhermite) % int(mesh_shape[1]) == 0
                        )
                        else None
                    )
                    cli_res = run_cli_simulation(
                        cli_deck,
                        output_path=pm0.staged_out_nc,
                        nstep=int(cfg0.nstep),
                        dt=dt,
                        t_max=t_max,
                        nwrite=int(cfg0.nwrite),
                        scheme=cfg0.scheme,
                        force_nonlinear=bool(cfg0.nonlinear_mode),
                        force_linear=not bool(cfg0.nonlinear_mode),
                        mesh_shape=eff_mesh_shape,
                        precision=prec_clean,
                        restart_file=pm0.resume_source_rst_nc if pm0.resumed else None,
                        save_restart_path=pm0.staged_restart_nc if pm0.want_restart else None,
                        metrics_json_path=pm0.staged_metrics_json,
                        async_io=async_io,
                        fft_backend=cfg0.fft_backend,
                    )
                    steps_ran_list = [int(cli_res["nstep"])]
                    t_final_list = [float(cli_res["t_final"])]

                mirror.stop_and_flush()
                active_mirror = None
                batch_wall = time.perf_counter() - t_batch_start

                # Atomically promote completed outputs & strict-JSON metrics per member
                with preemption.shield():
                    for b_idx, (pm, steps_ran, t_fin) in enumerate(
                        zip(prepared_batch, steps_ran_list, t_final_list, strict=True)
                    ):
                        final_out_nc = out_dir / f"{pm.item.stem}.out.nc"
                        final_rst_nc = (
                            out_dir / f"{pm.item.stem}.restart.nc" if pm.want_restart else None
                        )
                        final_metrics_json = out_dir / f"{pm.item.stem}.diagnostics_metrics.json"
                        total_nstep = int(pm.prior_steps) + int(steps_ran)
                        dt_nom = float(pm.base_spec.cfg.dt) if pm.base_spec.cfg.dt > 0.0 else 0.05

                        if _is_primary_process():
                            extract_kr2_diagnostics(
                                pm.staged_out_nc,
                                metrics_out_path=pm.staged_metrics_json,
                                sim_time_target=pm.full_sim_time_target,
                            )
                            raw_metrics = json.loads(
                                pm.staged_metrics_json.read_text(encoding="utf-8")
                            )
                            raw_metrics["status"] = "completed"
                            raw_metrics["campaign_mode"] = batch.mode
                            raw_metrics["batch_index"] = batch.batch_index
                            raw_metrics["member_index"] = b_idx
                            raw_metrics["member_name"] = pm.item.stem
                            raw_metrics["stem"] = pm.item.stem
                            raw_metrics["task_index"] = worker.task_index
                            raw_metrics["task_count"] = worker.task_count
                            raw_metrics["input_file"] = str(pm.item.input_path)
                            raw_metrics["output_nc"] = str(final_out_nc)
                            raw_metrics["restart_nc"] = (
                                str(final_rst_nc) if final_rst_nc is not None else None
                            )
                            raw_metrics["nstep"] = total_nstep
                            raw_metrics["dt"] = float(dt) if dt is not None else dt_nom
                            raw_metrics["t_final"] = float(t_fin)
                            raw_metrics["sim_time_target"] = float(pm.full_sim_time_target)
                            raw_metrics["scheme"] = pm.base_spec.cfg.scheme
                            raw_metrics["nonlinear"] = bool(pm.base_spec.cfg.nonlinear_mode)
                            raw_metrics["precision"] = prec_clean
                            raw_metrics["beta"] = float(pm.base_spec.cfg.beta)
                            raw_metrics["resumed"] = bool(pm.resumed)
                            raw_metrics["resumed_from_time"] = (
                                float(pm.resumed_from_time) if pm.resumed else None
                            )
                            raw_metrics["resume_step_source"] = pm.resume_step_source
                            raw_metrics["wall_seconds"] = float(batch_wall)

                            # Promote .out.nc and .restart.nc BEFORE committing metrics JSON
                            atomic_promote_file(
                                pm.staged_out_nc,
                                final_out_nc,
                                task_index=worker.task_index,
                            )
                            if final_rst_nc is not None and pm.staged_restart_nc.is_file():
                                atomic_promote_file(
                                    pm.staged_restart_nc,
                                    final_rst_nc,
                                    task_index=worker.task_index,
                                )
                            clean_metrics = atomic_write_json(
                                final_metrics_json,
                                raw_metrics,
                                task_index=worker.task_index,
                            )
                        else:
                            clean_metrics = {"converged": True, "has_nan_or_inf": False}

                        completed_stems.append(pm.item.stem)
                        if pm.resumed:
                            resumed_stems.append(pm.item.stem)
                            if pm.resume_step_source == "estimated":
                                estimated_resume_stems.append(pm.item.stem)
                        deck_records[pm.item.stem] = {
                            "index": pm.item.index,
                            "stem": pm.item.stem,
                            "input_file": str(pm.item.input_path),
                            "status": "completed",
                            "mode": batch.mode,
                            "batch_index": batch.batch_index,
                            "resumed": bool(pm.resumed),
                            "resumed_from_time": (
                                float(pm.resumed_from_time) if pm.resumed else None
                            ),
                            "resume_step_source": pm.resume_step_source,
                            "output_nc": str(final_out_nc),
                            "restart_nc": (str(final_rst_nc) if final_rst_nc is not None else None),
                            "metrics_json": str(final_metrics_json),
                            "nstep": total_nstep,
                            "t_final": float(t_fin),
                            "sim_time_target": float(pm.full_sim_time_target),
                            "converged": bool(clean_metrics.get("converged", False)),
                        }

                    batch_summaries.append(
                        {
                            "batch_index": batch.batch_index,
                            "mode": batch.mode,
                            "batch_size": len(prepared_batch),
                            "stems": [pm.item.stem for pm in prepared_batch],
                            "wall_seconds": float(batch_wall),
                        }
                    )
                    active_prepared = []
                    shutil.rmtree(batch_stage_dir, ignore_errors=True)
                    if _is_primary_process():
                        atomic_write_json(
                            ledger_path,
                            _build_ledger_payload(status="running", exit_code=0),
                            task_index=worker.task_index,
                        )

            with preemption.shield():
                final_ledger = _build_ledger_payload(status="completed", exit_code=0)
                if _is_primary_process():
                    final_ledger = atomic_write_json(
                        ledger_path,
                        final_ledger,
                        task_index=worker.task_index,
                    )
            return {
                "status": "completed",
                "exit_code": 0,
                "ledger_json": ledger_path,
                "ledger": final_ledger,
                "output_dir": out_dir,
                "task_index": worker.task_index,
                "task_count": worker.task_count,
            }

        except SpotPreemptionInterrupt as preempt_exc:
            preemption.mark_cleaning_up()
            if active_mirror is not None:
                active_mirror.stop_and_flush()
                active_mirror = None
            ckpt_records = _salvage_preempted_batch(
                active_prepared,
                output_dir=out_dir,
                task_index=worker.task_index,
                complex_dtype=complex_dtype,
            )
            preempt_ledger = _build_ledger_payload(
                status="preempted",
                exit_code=preempt_exc.exit_code,
                signal_name=preempt_exc.signal_name,
                checkpointed=ckpt_records,
            )
            if _is_primary_process():
                preempt_ledger = atomic_write_json(
                    ledger_path,
                    preempt_ledger,
                    task_index=worker.task_index,
                )
            return {
                "status": "preempted",
                "exit_code": preempt_exc.exit_code,
                "signal": preempt_exc.signal_name,
                "ledger_json": ledger_path,
                "ledger": preempt_ledger,
                "output_dir": out_dir,
                "task_index": worker.task_index,
                "task_count": worker.task_count,
            }
        finally:
            if active_mirror is not None:
                active_mirror.stop_and_flush()
            shutil.rmtree(worker_stage_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 6. `gx-jax-campaign` CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------


def build_campaign_arg_parser() -> argparse.ArgumentParser:
    """Construct the `gx-jax-campaign` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="gx-jax-campaign",
        description=(
            "Stellarator GX-JAX Cloud Batch & GKE Spot TPU campaign orchestrator "
            "(manifest slicing, jax.vmap auto-batching, local scratch staging, and "
            "SIGTERM -> 143 -> --auto-resume preemption recovery)."
        ),
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Campaign manifest path (.json, .jsonl, .txt, or directory of .in/.toml decks).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Destination output directory (e.g. GCS FUSE mount /mnt/disks/gcs/outputs).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp/gx_jax_stage"),
        help="Fast local SSD / tmpfs staging directory for HDF5/NetCDF4 writes.",
    )
    parser.add_argument(
        "--ledger-json",
        type=Path,
        default=None,
        help="Optional explicit path for worker summary ledger JSON.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Worker index in [0, task_count) (overrides BATCH_TASK_INDEX / JOB_COMPLETION_INDEX).",
    )
    parser.add_argument(
        "--task-count",
        type=int,
        default=None,
        help="Total number of workers >= 1 (overrides BATCH_TASK_COUNT / INDEX_COMPLETION_COUNT).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Maximum structurally compatible decks per jax.vmap ensemble batch (default: 4).",
    )
    parser.add_argument(
        "--batch-devices",
        type=int,
        default=None,
        help="Explicit device count for 1D ('batch',) ensemble Mesh sharding.",
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="2D ('sp', 'm') device mesh shape for singleton decks, e.g. '2x4' or 'auto'.",
    )
    parser.add_argument(
        "--skip-completed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip decks that already have valid completed .out.nc & .diagnostics_metrics.json.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume unfinished decks from <stem>.restart.nc checkpoints.",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=None,
        help="Override maximum number of timesteps across campaign decks.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Override fixed timestep size dt across campaign decks.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=None,
        help="Override maximum physical simulation time t_max across campaign decks.",
    )
    parser.add_argument(
        "--nwrite",
        type=int,
        default=None,
        help="Override diagnostic snapshot cadence nwrite across campaign decks.",
    )
    parser.add_argument(
        "--nsave",
        type=int,
        default=None,
        help="Override periodic .restart.nc checkpoint cadence nsave across campaign decks.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default=None,
        choices=["sspx2", "rk3", "sspx3", "rk4", "k10"],
        help="Override time integration scheme across campaign decks.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--nonlinear",
        action="store_true",
        default=False,
        help="Force nonlinear ExB advection on across campaign decks.",
    )
    mode_group.add_argument(
        "--linear",
        action="store_true",
        default=False,
        help="Force linear mode across campaign decks.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp64"],
        help="Floating-point precision: 'fp32' (complex64, default) or 'fp64' (complex128).",
    )
    parser.add_argument(
        "--fft-backend",
        type=str,
        choices=("auto", "fft", "mxu"),
        default=None,
        help="Spectral transform backend override ('auto', 'fft', or 'mxu').",
    )
    parser.add_argument(
        "--async-io",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable background NetCDF4 I/O thread on singleton runs (default: False).",
    )
    parser.add_argument(
        "--save-restart",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable/disable saving .restart.nc checkpoints.",
    )
    parser.add_argument(
        "--distributed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Explicitly initialize multi-host jax.distributed SPMD (default: False; "
            "requires explicit --task-index and --task-count)."
        ),
    )
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default=None,
        help="Explicit host:port coordinator address for multi-host jax.distributed.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Total number of multi-host JAX processes.",
    )
    parser.add_argument(
        "--process-id",
        type=int,
        default=None,
        help="Current multi-host JAX process ID in [0, num_processes).",
    )
    return parser


def campaign_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `gx-jax-campaign`."""
    parser = build_campaign_arg_parser()
    args = parser.parse_args(argv)

    try:
        mesh_shape = parse_mesh_arg(args.mesh)
    except ValueError as exc:
        parser.error(str(exc))

    did_init_dist = False
    try:
        did_init_dist = _maybe_init_distributed(
            distributed=bool(args.distributed),
            task_index=args.task_index,
            task_count=args.task_count,
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )
        result = run_campaign(
            args.manifest,
            output_dir=args.output_dir,
            scratch_dir=args.scratch_dir,
            ledger_json_path=args.ledger_json,
            task_index=args.task_index,
            task_count=args.task_count,
            batch_size=args.batch_size,
            batch_devices=args.batch_devices,
            mesh_shape=mesh_shape,
            skip_completed=args.skip_completed,
            auto_resume=args.auto_resume,
            nstep=args.nstep,
            dt=args.dt,
            t_max=args.t_max,
            nwrite=args.nwrite,
            nsave=args.nsave,
            scheme=args.scheme,
            force_nonlinear=args.nonlinear,
            force_linear=args.linear,
            precision=args.precision,
            fft_backend=args.fft_backend,
            async_io=args.async_io,
            save_restart=args.save_restart,
            trap_signals=True,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        _maybe_shutdown_distributed(did_init_dist)

    return int(result["exit_code"])


main = campaign_main

__all__ = [
    "CampaignBatch",
    "CampaignItem",
    "CampaignWorkerSlice",
    "CheckpointMirrorThread",
    "PreemptionController",
    "SIGINT_EXIT_CODE",
    "SIGTERM_EXIT_CODE",
    "SpotPreemptionInterrupt",
    "atomic_promote_file",
    "atomic_write_json",
    "build_campaign_arg_parser",
    "campaign_main",
    "compatibility_key",
    "detect_worker_slice",
    "group_compatible_items",
    "is_deck_completed",
    "load_campaign_manifest",
    "main",
    "run_campaign",
    "slice_campaign_items",
]

if __name__ == "__main__":
    raise SystemExit(main())
