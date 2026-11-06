"""Durable lifecycle telemetry emitter (`hpc_core/telemetry.py`).

Design guarantees:
- Zero AI/LLM or cloud SDK dependencies (stdlib + `requests` client only).
- Emits `workstream` and `environment` on every event. An unrecognised environment value is
  recorded as "sim", never as "prod", so misconfigured runs cannot inflate production KPIs.
- Durable local outbox (`outbox.jsonl`): events are appended and fsync'd to disk first, then
  flushed in batches (<=500) via HTTPS POST to `/v1/events`. After a flush the outbox is
  rewritten atomically (temp + fsync + rename) and keeps every event that was appended by
  another process while the batch was in flight.
- Poison-event isolation: if a batch receives HTTP 400/413/422 (`EventsValidationError`),
  isolates and quarantines only the offending event(s) into `outbox.quarantine.jsonl` so
  valid events are never blocked. Unparseable outbox lines are quarantined, never dropped.
- Exponential backoff when the network or Cloud Run is unreachable.
- Operator handles are salted-SHA-256 hashed. The salt comes from the environment, else from a
  random per-installation salt file next to the outbox (mode 0600); never a public default.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hpc_core.client import EventsValidationError
from hpc_core.fsutil import (
    append_line_durable,
    atomic_write_bytes,
    atomic_write_text,
    fsync_dir,
    is_safe_id,
)

logger = logging.getLogger(__name__)

MAX_EVENT_BYTES = 1024
MAX_BATCH_SIZE = 500
DEFAULT_PROGRESS_INTERVAL_S = 600.0
MAX_LOG_TAIL_LINES = 30
MAX_CAMPAIGN_ID_CHARS = 64
OPERATOR_SALT_FILENAME = ".operator_hash_salt"

VALID_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "submitted",
        "running",
        "progress",
        "completed",
        "failed",
        "resubmit_walltime",
        "walltime_resubmitted",
        "triage",
        "triage_requested",
        "directive_received",
        "patched",
        "resubmitted",
        "escalated",
        "human_resolved",
    }
)

VALID_SOLVERS: frozenset[str] = frozenset({"GENE", "GX", "MRX", "SFINCS"})
VALID_FIDELITIES: frozenset[str] = frozenset({"low", "high"})
VALID_ENVIRONMENTS: frozenset[str] = frozenset({"prod", "sim", "synthetic", "preflight"})
# Environment recorded when the configured value is not recognised (never "prod").
FALLBACK_ENVIRONMENT = "sim"

_SALT_RE = re.compile(r"^[\x21-\x7e]{16,512}$")
_PROCESS_SALT: str | None = None
_WARNED_ENVIRONMENTS: set[str] = set()


def _require_safe_id(field_name: str, value: Any, *, max_len: int = 128) -> str:
    if not isinstance(value, str) or not is_safe_id(value) or len(value) > max_len:
        raise ValueError(
            f"{field_name} {value!r} is not a safe identifier "
            f"([A-Za-z0-9][A-Za-z0-9_.-]*, at most {max_len} characters)"
        )
    return value


def _require_fidelity(value: Any) -> str:
    if isinstance(value, str) and value.strip().lower() in VALID_FIDELITIES:
        return value.strip().lower()
    raise ValueError(f"fidelity {value!r} must be one of {sorted(VALID_FIDELITIES)}")


def infer_run_metadata(run_dir: Path, config: dict[str, Any]) -> tuple[str, str, str]:
    """Return (campaign_id, fidelity, equilibrium_id) declared for a queued run (fail closed).

    The source of truth is `<run_dir>/run_meta.json`. Undeclared campaign_id / fidelity fall
    back to the executor config (`campaign_id`, `default_fidelity`). equilibrium_id (or its
    legacy alias geometry_id) is "" when undeclared: it is never guessed from directory names
    and never defaulted to a real equilibrium, so an unlabeled run cannot enter the dataset or
    an ML split group under a wrong id.

    Raises ValueError if run_meta.json exists but is unreadable or not a JSON object, or if a
    declared or configured value is invalid (unsafe identifier, fidelity not low/high).

    TODO(david): confirm exact low vs high fidelity grid criteria and equilibrium/geometry naming.
    """
    campaign_id: Any = config.get("campaign_id", "pilot_50")
    fidelity: Any = config.get("default_fidelity", "low")
    equilibrium_id: Any = ""

    meta_file = run_dir / "run_meta.json"
    if meta_file.exists():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError(f"{meta_file.name} in {run_dir.name!r} is unreadable: {exc}") from exc
        if not isinstance(meta, dict):
            raise ValueError(f"{meta_file.name} in {run_dir.name!r} must contain a JSON object")
        if meta.get("campaign_id") is not None:
            campaign_id = meta["campaign_id"]
        if meta.get("fidelity") is not None:
            fidelity = meta["fidelity"]
        declared = meta.get("equilibrium_id")
        if declared is None or declared == "":
            declared = meta.get("geometry_id")
        if declared is not None and declared != "":
            equilibrium_id = declared

    campaign_id = _require_safe_id("campaign_id", campaign_id, max_len=MAX_CAMPAIGN_ID_CHARS)
    fidelity = _require_fidelity(fidelity)
    if equilibrium_id != "":
        equilibrium_id = _require_safe_id("equilibrium_id", equilibrium_id)
    return campaign_id, fidelity, equilibrium_id


def _process_salt() -> str:
    global _PROCESS_SALT
    if _PROCESS_SALT is None:
        _PROCESS_SALT = secrets.token_hex(32)
        logger.warning(
            "No operator hash salt configured or persistable; using a random per-process salt "
            "(operator hashes will not be comparable across runs). Set TELEMETRY_OPERATOR_SALT."
        )
    return _PROCESS_SALT


def _load_or_create_salt(salt_file: Path) -> str:
    """Read the persisted salt, creating it atomically (0600, O_EXCL) on first use."""
    for attempt in range(4):
        try:
            text = salt_file.read_text(encoding="ascii", errors="replace").strip()
        except FileNotFoundError:
            text = None
        if text is not None:
            if _SALT_RE.fullmatch(text):
                with contextlib.suppress(OSError):
                    if salt_file.stat().st_mode & 0o077:
                        os.chmod(salt_file, 0o600)
                return text
            time.sleep(0.05 * (attempt + 1))  # a concurrent creator may still be writing it
            continue
        salt = secrets.token_hex(32)
        salt_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(str(salt_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="ascii") as fh:
            fh.write(salt + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        fsync_dir(salt_file.parent)
        return salt
    # Present but empty/malformed (crash mid-create or manual edit): no stable hash was ever
    # produced from it, so replace it.
    salt = secrets.token_hex(32)
    atomic_write_bytes(salt_file, (salt + "\n").encode("ascii"), mode=0o600)
    logger.warning("Replaced malformed operator hash salt file %s", salt_file)
    return salt


def _resolve_operator_salt(config: dict[str, Any] | None, salt_file: Path | None) -> str:
    cfg = config or {}
    salt_env = str(cfg.get("operator_hash_salt_env", "TELEMETRY_OPERATOR_SALT"))
    for name in (salt_env, "TELEMETRY_OPERATOR_SALT", "TACC_OPERATOR_SALT"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    if salt_file is not None:
        try:
            return _load_or_create_salt(Path(salt_file))
        except OSError as exc:
            logger.warning("Cannot use operator salt file %s: %s", salt_file, exc)
    return _process_salt()


def hash_operator_handle(
    operator: str | None,
    config: dict[str, Any] | None = None,
    *,
    salt_file: Path | None = None,
) -> str | None:
    """Hash operator username with SHA-256 + salt by default; plaintext only if TELEMETRY_STORE_USERNAMES=true.

    Salt precedence: the env var named by config `operator_hash_salt_env`, then
    TELEMETRY_OPERATOR_SALT, then TACC_OPERATOR_SALT, then the persisted `salt_file`
    (created with a random value on first use), then a random per-process salt.

    TODO(tan): confirm UT Austin data policy on operator usernames in telemetry.
    """
    if not operator:
        return None
    cleaned = operator.strip()
    if not cleaned:
        return None
    if cleaned.startswith("op_") and len(cleaned) == 19:
        return cleaned

    store_plaintext_env = os.environ.get("TELEMETRY_STORE_USERNAMES", "").strip().lower()
    if store_plaintext_env == "true":
        return cleaned[:64]

    salt = _resolve_operator_salt(config, salt_file)
    digest = hashlib.sha256(f"{salt}:{cleaned}".encode()).hexdigest()[:16]
    return f"op_{digest}"


def _normalize_environment(raw: str | None) -> str:
    value = (raw or "").strip().lower()
    if value in VALID_ENVIRONMENTS:
        return value
    if value not in _WARNED_ENVIRONMENTS:
        _WARNED_ENVIRONMENTS.add(value)
        logger.warning(
            "Unrecognised telemetry environment %r; recording %r (never 'prod')",
            raw,
            FALLBACK_ENVIRONMENT,
        )
    return FALLBACK_ENVIRONMENT


def _sanitize_and_bound_event(event: dict[str, Any]) -> dict[str, Any]:
    """Ensure the serialized event is strictly <1 KB (1024 bytes) and log_tail <=30 lines."""
    payload = dict(event.get("payload") or {})

    if "log_tail" in payload and isinstance(payload["log_tail"], str):
        lines = payload["log_tail"].splitlines()[-MAX_LOG_TAIL_LINES:]
        payload["log_tail"] = "\n".join(lines)[:400]

    if "metrics" in payload and isinstance(payload["metrics"], dict):
        condensed: dict[str, Any] = {}
        for idx, (k, v) in enumerate(payload["metrics"].items()):
            if idx >= 16:
                break
            if isinstance(v, float):
                condensed[str(k)] = round(v, 6)
            elif isinstance(v, int | str | bool) or v is None:
                condensed[str(k)] = v
        payload["metrics"] = condensed

    event["payload"] = payload
    encoded = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(encoded) < MAX_EVENT_BYTES:
        return event

    for text_key in ("issue_md_text", "log_tail", "note", "reason"):
        if text_key in payload and isinstance(payload[text_key], str):
            excess = len(encoded) - (MAX_EVENT_BYTES - 96)
            current_str = payload[text_key]
            new_len = max(0, len(current_str) - max(excess, 64))
            payload[text_key] = (current_str[:new_len] + "...") if new_len > 0 else ""
            event["payload"] = payload
            encoded = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
            if len(encoded) < MAX_EVENT_BYTES:
                return event

    minimal_payload: dict[str, Any] = {}
    for k in (
        "slurm_job_id",
        "slurm_state",
        "failure_type",
        "action",
        "minutes",
        "walltime_resubmits",
        "tacc_folder_path",
        "geometry_id",
    ):
        if k in payload:
            minimal_payload[k] = payload[k]
    event["payload"] = minimal_payload
    return event


def _encode_event(event: dict[str, Any]) -> str:
    return json.dumps(event, separators=(",", ":"), sort_keys=True)


class TelemetryEmitter:
    """Durable, rate-limited, non-blocking lifecycle event emitter with poison-event quarantine."""

    def __init__(
        self,
        outbox_file: Path,
        client: Any | None = None,
        *,
        default_workstream: str = "kr2-gk-database",
        default_campaign_id: str = "pilot_50",
        default_fidelity: str = "low",
        default_environment: str | None = None,
        default_backend: str = "tacc_slurm",
        progress_interval_s: float = DEFAULT_PROGRESS_INTERVAL_S,
        batch_size: int = MAX_BATCH_SIZE,
        backoff_base_s: float = 1.0,
        max_backoff_s: float = 300.0,
        now_fn: Callable[[], float] = time.time,
        operator_salt_file: Path | None = None,
    ) -> None:
        self.outbox_file = Path(outbox_file)
        self.quarantine_file = self.outbox_file.with_suffix(".quarantine.jsonl")
        self.operator_salt_file = (
            Path(operator_salt_file)
            if operator_salt_file is not None
            else self.outbox_file.with_name(OPERATOR_SALT_FILENAME)
        )
        self.client = client
        self.default_workstream = default_workstream
        self.default_campaign_id = default_campaign_id
        self.default_fidelity = default_fidelity if default_fidelity in VALID_FIDELITIES else "low"
        self.default_environment = _normalize_environment(
            default_environment or os.environ.get("ENVIRONMENT", "prod")
        )
        self.default_backend = default_backend
        self.progress_interval_s = float(progress_interval_s)
        self.batch_size = max(1, min(int(batch_size), MAX_BATCH_SIZE))
        self.backoff_base_s = float(backoff_base_s)
        self.max_backoff_s = float(max_backoff_s)
        self.now_fn = now_fn

        self._last_progress_by_run: dict[str, float] = {}
        self._consecutive_failures: int = 0
        self._next_retry_epoch: float = 0.0

    def emit(
        self,
        *,
        event_type: str,
        run_id: str,
        solver: str,
        retry_count: int = 0,
        walltime_resubmits: int | None = None,
        workstream: str | None = None,
        backend: str | None = None,
        environment: str | None = None,
        operator: str | None = None,
        campaign_id: str | None = None,
        fidelity: str | None = None,
        payload: dict[str, Any] | None = None,
        event_id: str | None = None,
        ts_utc: str | None = None,
        ts: str | None = None,
        is_synthetic: bool = False,
    ) -> dict[str, Any] | None:
        """Append a lifecycle event (<1 KB) to the local outbox without ever raising."""
        try:
            if event_type not in VALID_EVENT_TYPES:
                logger.warning("Ignoring unknown telemetry event_type=%r", event_type)
                return None

            now_epoch = float(self.now_fn())
            if event_type == "progress":
                last_ts = self._last_progress_by_run.get(run_id)
                if last_ts is not None and (now_epoch - last_ts) < self.progress_interval_s:
                    return None
                self._last_progress_by_run[run_id] = now_epoch

            solver_norm = solver.upper()
            if solver_norm not in VALID_SOLVERS:
                solver_norm = "GENE"

            fid_norm = (fidelity or self.default_fidelity).lower()
            if fid_norm not in VALID_FIDELITIES:
                fid_norm = "low"

            if environment:
                env_norm = _normalize_environment(environment)
            elif is_synthetic:
                env_norm = "synthetic"
            else:
                env_norm = self.default_environment

            payload_dict = dict(payload or {})
            wt_resubmits = (
                int(walltime_resubmits)
                if walltime_resubmits is not None
                else int(payload_dict.get("walltime_resubmits", 0))
            )
            op_raw = operator or (
                str(payload_dict["operator"]) if payload_dict.get("operator") else None
            )
            op_hashed = hash_operator_handle(op_raw, salt_file=self.operator_salt_file)
            if op_hashed and "operator" in payload_dict:
                payload_dict["operator"] = op_hashed

            ts_iso = ts or ts_utc or datetime.fromtimestamp(now_epoch, tz=UTC).isoformat()
            raw_event: dict[str, Any] = {
                "event_id": event_id or str(uuid.uuid4()),
                "schema_version": "1.0",
                "workstream": str(workstream or self.default_workstream),
                "backend": str(backend or self.default_backend),
                "run_id": str(run_id),
                "campaign_id": str(campaign_id or self.default_campaign_id),
                "solver": solver_norm,
                "fidelity": fid_norm,
                "event_type": event_type,
                "ts_utc": ts_iso,
                "ts": ts_iso,
                "retry_count": max(0, int(retry_count)),
                "walltime_resubmits": max(0, wt_resubmits),
                "environment": env_norm,
                "is_synthetic": bool(is_synthetic or env_norm in ("synthetic", "preflight")),
                "payload": payload_dict,
            }
            if op_hashed is not None:
                raw_event["operator"] = op_hashed

            bounded_event = _sanitize_and_bound_event(raw_event)
            self._append_to_outbox(bounded_event)
            return bounded_event
        except Exception as exc:  # noqa: BLE001
            logger.warning("Telemetry emit failed (non-fatal) for run %s: %s", run_id, exc)
            return None

    def _append_to_outbox(self, event: dict[str, Any]) -> None:
        append_line_durable(self.outbox_file, _encode_event(event))

    def _quarantine_record(self, record: dict[str, Any]) -> None:
        record = {"quarantined_at": datetime.now(UTC).isoformat(), **record}
        append_line_durable(
            self.quarantine_file, json.dumps(record, separators=(",", ":"), sort_keys=True)
        )

    def _quarantine_event(self, event: dict[str, Any], error_msg: str) -> None:
        self._quarantine_record({"error": error_msg[:300], "event": event})

    def _read_outbox(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (pending events deduplicated by event_id, unparseable lines), in file order."""
        if not self.outbox_file.is_file():
            return [], []
        try:
            text = self.outbox_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Failed reading telemetry outbox %s: %s", self.outbox_file, exc)
            return [], []
        seen_ids: set[str] = set()
        events: list[dict[str, Any]] = []
        malformed: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except ValueError:
                malformed.append(stripped)
                continue
            if not isinstance(ev, dict) or not str(ev.get("event_id", "")):
                malformed.append(stripped)
                continue
            eid = str(ev["event_id"])
            if eid not in seen_ids:
                seen_ids.add(eid)
                events.append(ev)
        return events, malformed

    def read_pending_events(self) -> list[dict[str, Any]]:
        """Read and deduplicate pending events from outbox.jsonl preserving order."""
        return self._read_outbox()[0]

    def _rewrite_outbox(self, remaining: list[dict[str, Any]], seen_ids: set[str]) -> None:
        """Atomically replace the outbox with `remaining` plus events appended since the read.

        `seen_ids` are the event ids read at the start of the flush; any other event now in the
        file was appended concurrently (e.g. by `tacc-resolve`) and is kept.
        """
        current, _malformed = self._read_outbox()
        keep: list[dict[str, Any]] = []
        kept_ids: set[str] = set()
        for ev in [*remaining, *(e for e in current if str(e.get("event_id")) not in seen_ids)]:
            eid = str(ev.get("event_id"))
            if eid not in kept_ids:
                kept_ids.add(eid)
                keep.append(ev)
        atomic_write_text(self.outbox_file, "".join(_encode_event(ev) + "\n" for ev in keep))

    def _flush_batch_isolating_poison(
        self, batch: list[dict[str, Any]]
    ) -> tuple[int, list[dict[str, Any]]]:
        """Attempt batch POST; on HTTP 422, isolate individual poison events into quarantine."""
        assert self.client is not None
        try:
            self.client.post_events(batch)
            return len(batch), []
        except EventsValidationError as batch_exc:
            if len(batch) == 1:
                logger.error(
                    "Quarantining poison telemetry event %s: %s",
                    batch[0].get("event_id"),
                    batch_exc,
                )
                self._quarantine_event(batch[0], str(batch_exc))
                return 0, []
            # Isolate each event individually so valid events succeed and only bad ones quarantine
            sent = 0
            unflushed_transient: list[dict[str, Any]] = []
            for idx, ev in enumerate(batch):
                try:
                    self.client.post_events([ev])
                    sent += 1
                except EventsValidationError as single_exc:
                    logger.error(
                        "Quarantining poison telemetry event %s: %s",
                        ev.get("event_id"),
                        single_exc,
                    )
                    self._quarantine_event(ev, str(single_exc))
                except Exception:  # noqa: BLE001
                    unflushed_transient.extend(batch[idx:])
                    break
            return sent, unflushed_transient

    def _defer(self, now_epoch: float) -> float:
        self._consecutive_failures += 1
        delay = min(
            self.max_backoff_s,
            self.backoff_base_s * (2 ** (self._consecutive_failures - 1)),
        )
        self._next_retry_epoch = now_epoch + delay
        return delay

    def flush(self, *, force: bool = False) -> int:
        """Flush queued events in batches (<=500) to Cloud Run `/v1/events`."""
        try:
            if self.client is None or not hasattr(self.client, "post_events"):
                return 0

            now_epoch = float(self.now_fn())
            if not force and now_epoch < self._next_retry_epoch:
                return 0

            pending, malformed = self._read_outbox()
            for raw_line in malformed:
                self._quarantine_record(
                    {"error": "unparseable outbox line", "raw_line": raw_line[:2000]}
                )
            if not pending and not malformed:
                return 0
            seen_ids = {str(ev.get("event_id")) for ev in pending}

            flushed_count = 0
            idx = 0
            total = len(pending)
            while idx < total:
                batch = pending[idx : idx + self.batch_size]
                try:
                    sent, remaining_in_batch = self._flush_batch_isolating_poison(batch)
                except Exception as exc:  # noqa: BLE001
                    delay = self._defer(now_epoch)
                    self._rewrite_outbox(pending[idx:], seen_ids)
                    logger.warning(
                        "Telemetry outbox flush deferred (%d pending, retry in %.1fs): %s",
                        len(pending[idx:]),
                        delay,
                        exc,
                    )
                    return flushed_count

                flushed_count += sent
                if remaining_in_batch:
                    self._defer(now_epoch)
                    self._rewrite_outbox(remaining_in_batch + pending[idx + len(batch) :], seen_ids)
                    return flushed_count
                idx += len(batch)

            self._consecutive_failures = 0
            self._next_retry_epoch = 0.0
            self._rewrite_outbox([], seen_ids)
            return flushed_count
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unexpected error during telemetry outbox flush (non-fatal): %s", exc)
            return 0
