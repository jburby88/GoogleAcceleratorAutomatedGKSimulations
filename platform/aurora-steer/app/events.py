"""Telemetry event ingestion, structured Cloud Logging sink output, and DuckDB KPI views."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from app.observability import current_request_id, log_event
from app.schemas import LifecycleEvent


def _find_views_sql_path() -> Path:
    env_root = os.getenv("REPO_ROOT", "").strip()
    if env_root:
        cand = Path(env_root) / "platform" / "telemetry" / "02_views.sql"
        if cand.is_file():
            return cand
    return Path(__file__).resolve().parents[3] / "platform" / "telemetry" / "02_views.sql"


def _split_balanced_args(arg_str: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    in_str = False
    for ch in arg_str:
        if ch == "'":
            in_str = not in_str
            current.append(ch)
        elif not in_str and ch == "(":
            depth += 1
            current.append(ch)
        elif not in_str and ch == ")":
            depth -= 1
            current.append(ch)
        elif not in_str and depth == 0 and ch == ",":
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return parts


def _replace_func_calls(sql: str, func_name: str, replacer: Any) -> str:
    pattern = re.compile(rf"\b{func_name}\s*\(", re.IGNORECASE)
    pos = 0
    out: list[str] = []
    while True:
        match = pattern.search(sql, pos)
        if not match:
            out.append(sql[pos:])
            break
        out.append(sql[pos : match.start()])
        i = match.end()
        depth = 1
        in_str = False
        while i < len(sql) and depth > 0:
            ch = sql[i]
            if ch == "'":
                in_str = not in_str
            elif not in_str:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
            i += 1
        inner = sql[match.end() : i - 1]
        args = [
            _translate_expr_recursive(a, func_name, replacer) for a in _split_balanced_args(inner)
        ]
        out.append(replacer(args))
        pos = i
    return "".join(out)


def _translate_expr_recursive(expr: str, func_name: str, replacer: Any) -> str:
    return _replace_func_calls(expr, func_name, replacer)


def translate_bigquery_views_to_duckdb(bigquery_sql: str) -> str:
    """Deterministically translate `platform/telemetry/02_views.sql` from BigQuery to DuckDB SQL."""
    sql = re.sub(
        r"`[a-zA-Z0-9_${}-]+\.[a-zA-Z0-9_${}-]+\.([a-zA-Z0-9_]+)`",
        r"\1",
        bigquery_sql,
    )
    sql = sql.replace(
        "TIMESTAMP('2025-01-01T00:00:00Z')",
        "CAST('2025-01-01T00:00:00Z' AS TIMESTAMPTZ)",
    )
    sql = re.sub(r"\bFLOAT64\b", "DOUBLE", sql)
    sql = re.sub(r"\bINT64\b", "BIGINT", sql)

    sql = _replace_func_calls(
        sql,
        "SAFE_DIVIDE",
        lambda a: f"(CAST(({a[0]}) AS DOUBLE) / NULLIF(CAST(({a[1]}) AS DOUBLE), 0.0))",
    )
    sql = _replace_func_calls(
        sql,
        "JSON_VALUE",
        lambda a: f"json_extract_string({a[0]}, {a[1]})",
    )
    sql = _replace_func_calls(
        sql,
        "JSON_QUERY",
        lambda a: f"CAST(json_extract({a[0]}, {a[1]}) AS VARCHAR)",
    )
    sql = _replace_func_calls(
        sql,
        "TO_JSON_STRING",
        lambda a: f"CAST({a[0]} AS VARCHAR)",
    )
    sql = _replace_func_calls(
        sql,
        "TIMESTAMP_SUB",
        lambda a: f"({a[0]} - {a[1]})",
    )
    sql = _replace_func_calls(
        sql,
        "DATE_ADD",
        lambda a: _format_duckdb_date_add(a[0], a[1]),
    )
    sql = _replace_func_calls(
        sql,
        "DATE_DIFF",
        lambda a: f"date_diff('day', CAST({a[1]} AS DATE), CAST({a[0]} AS DATE))",
    )
    sql = re.sub(r"\bDATE\s*\(\s*s\.ref_ts_utc\s*\)", "CAST(s.ref_ts_utc AS DATE)", sql)
    return sql


def _format_duckdb_date_add(date_expr: str, interval_expr: str) -> str:
    m = re.match(r"^INTERVAL\s+(.+)\s+DAY$", interval_expr.strip(), flags=re.IGNORECASE | re.DOTALL)
    if m:
        days_expr = m.group(1).strip()
        return f"CAST(({date_expr} + ({days_expr}) * INTERVAL 1 DAY) AS DATE)"
    return f"CAST(({date_expr} + {interval_expr}) AS DATE)"


DEFAULT_ASSUMPTIONS: list[tuple[str, float, str, str, str, str, str]] = [
    (
        "minutes_per_manual_fix",
        45.0,
        "minutes",
        "Estimated physicist time per manual log inspection, parameter edit, and resubmit",
        "TODO(david): confirm baseline manual troubleshooting duration",
        "2026-11-05",
        "david",
    ),
    (
        "minutes_per_walltime_resubmit",
        10.0,
        "minutes",
        "Estimated operator overhead per manual checkpoint continuation",
        "TODO(david): confirm manual walltime continuation overhead",
        "2026-11-05",
        "david",
    ),
    (
        "allocated_walltime_hours_gene_low",
        4.0,
        "hours",
        "Default Slurm walltime allocation for low-fidelity GENE runs",
        "TODO(david): confirm Vista queue walltime defaults",
        "2026-11-05",
        "david",
    ),
    (
        "allocated_walltime_hours_gene_high",
        16.0,
        "hours",
        "Default Slurm walltime allocation for high-fidelity GENE runs",
        "TODO(david): confirm Vista queue walltime defaults",
        "2026-11-05",
        "david",
    ),
    (
        "allocated_walltime_hours_gx_low",
        2.0,
        "hours",
        "Default walltime allocation for low-fidelity GX runs",
        "TODO(david): confirm GX walltime defaults",
        "2026-11-05",
        "david",
    ),
    (
        "allocated_walltime_hours_gx_high",
        8.0,
        "hours",
        "Default walltime allocation for high-fidelity GX runs",
        "TODO(david): confirm GX walltime defaults",
        "2026-11-05",
        "david",
    ),
    (
        "gpus_per_run_gene_low",
        4.0,
        "gpus",
        "Default GPU count for low-fidelity GENE runs",
        "TODO(david): confirm GENE GPU count per fidelity",
        "2026-11-05",
        "david",
    ),
    (
        "gpus_per_run_gene_high",
        16.0,
        "gpus",
        "Default GPU count for high-fidelity GENE runs",
        "TODO(david): confirm GENE GPU count per fidelity",
        "2026-11-05",
        "david",
    ),
    (
        "gpus_per_run_gx_low",
        1.0,
        "gpus",
        "Default GPU count for low-fidelity GX runs (1x NVIDIA L4 on Cloud Batch)",
        "TODO(david): confirm GX GPU count per fidelity",
        "2026-11-05",
        "david",
    ),
    (
        "gpus_per_run_gx_high",
        4.0,
        "gpus",
        "Default GPU count for high-fidelity GX runs",
        "TODO(david): confirm GX GPU count per fidelity",
        "2026-11-05",
        "david",
    ),
]


def init_duckdb_telemetry_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the 5 canonical telemetry tables, seed assumptions, and register all 8 KPI views."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            event_id VARCHAR,
            schema_version VARCHAR,
            workstream VARCHAR,
            backend VARCHAR,
            run_id VARCHAR,
            campaign_id VARCHAR,
            solver VARCHAR,
            fidelity VARCHAR,
            event_type VARCHAR,
            ts_utc TIMESTAMPTZ,
            retry_count BIGINT,
            walltime_resubmits BIGINT,
            environment VARCHAR,
            operator VARCHAR,
            is_synthetic BOOLEAN,
            payload VARCHAR,
            ingested_at_utc TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS directives (
            directive_id VARCHAR,
            schema_version VARCHAR,
            workstream VARCHAR,
            backend VARCHAR,
            run_id VARCHAR,
            campaign_id VARCHAR,
            solver VARCHAR,
            fidelity VARCHAR,
            ts_utc TIMESTAMPTZ,
            retry_count BIGINT,
            walltime_resubmits BIGINT,
            environment VARCHAR,
            failure_type VARCHAR,
            action VARCHAR,
            reason VARCHAR,
            parameter_modifications_before VARCHAR,
            parameter_modifications_after VARCHAR,
            restart_from_checkpoint BOOLEAN,
            confidence VARCHAR,
            okf_concepts_used VARCHAR,
            guardrail_outcome VARCHAR,
            forced_escalation BOOLEAN,
            llm_called BOOLEAN,
            latency_ms DOUBLE,
            is_synthetic BOOLEAN
        );
        CREATE TABLE IF NOT EXISTS escalations (
            escalation_id VARCHAR,
            event_id VARCHAR,
            schema_version VARCHAR,
            workstream VARCHAR,
            run_id VARCHAR,
            campaign_id VARCHAR,
            solver VARCHAR,
            fidelity VARCHAR,
            ts_utc TIMESTAMPTZ,
            retry_count BIGINT,
            failure_type VARCHAR,
            issue_md_text VARCHAR,
            directives_tried VARCHAR,
            tacc_folder_path VARCHAR,
            environment VARCHAR,
            is_synthetic BOOLEAN
        );
        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id VARCHAR,
            workstream VARCHAR,
            target_runs BIGINT,
            start_date DATE,
            deadline_date DATE,
            notes VARCHAR,
            environment VARCHAR,
            is_synthetic BOOLEAN,
            updated_at_utc TIMESTAMPTZ
        );
        CREATE TABLE IF NOT EXISTS assumptions (
            assumption_key VARCHAR,
            numeric_value DOUBLE,
            unit VARCHAR,
            description VARCHAR,
            source VARCHAR,
            updated_date DATE,
            todo_owner VARCHAR
        );
        """
    )
    existing_count = conn.execute("SELECT COUNT(*) FROM assumptions").fetchone()
    if existing_count and existing_count[0] == 0:
        conn.executemany(
            "INSERT INTO assumptions VALUES (?, ?, ?, ?, ?, CAST(? AS DATE), ?)",
            DEFAULT_ASSUMPTIONS,
        )

    views_sql_path = _find_views_sql_path()
    duckdb_views_sql = translate_bigquery_views_to_duckdb(
        views_sql_path.read_text(encoding="utf-8")
    )
    conn.execute(duckdb_views_sql)


_cached_bq_creds = None
_bq_creds_lock = threading.Lock()
_bq_failure_lock = threading.Lock()
_bq_insert_failures = 0
_BQ_TRANSIENT_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})


def bq_insert_failure_count() -> int:
    """Number of BigQuery streaming batches that failed (rows not in the native tables)."""
    return _bq_insert_failures


def _record_bq_failure(table_target: str, n_rows: int, error: str) -> None:
    global _bq_insert_failures
    with _bq_failure_lock:
        _bq_insert_failures += 1
    log_event(
        "ERROR",
        "BigQuery streaming insert failed; rows are only in the Cloud Logging sink",
        table=table_target,
        rows=n_rows,
        error=error[:1000],
    )


def _get_bq_token() -> str | None:
    global _cached_bq_creds
    import google.auth
    import google.auth.transport.requests

    with _bq_creds_lock:
        if _cached_bq_creds is None:
            _cached_bq_creds, _ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
        if not _cached_bq_creds.valid or _cached_bq_creds.expired:
            _cached_bq_creds.refresh(google.auth.transport.requests.Request())
        return _cached_bq_creds.token


def _stream_rows_to_bigquery(table_target: str, rows: list[dict[str, Any]]) -> bool:
    """Batched streaming insert into BigQuery when `BQ_ENABLED=1` on Cloud Run.

    Returns True when every row was accepted (or streaming is disabled). Row-level
    `insertErrors` (returned with HTTP 200), HTTP errors and network failures are
    logged at ERROR severity and counted; transient failures are retried. Never
    raises: the structured stdout record (Cloud Logging sink) is the fallback copy.
    Callers must run this off the event loop (it blocks on network I/O).
    """
    if not rows:
        return True
    if os.getenv("BQ_ENABLED", "0").strip().lower() not in ("1", "true", "yes"):
        return True
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or os.getenv("GCP_PROJECT_ID", "").strip()
    )
    dataset = (
        os.getenv("BQ_DATASET", "").strip()
        or os.getenv("BQ_DATASET_ID", "").strip()
        or "sim_babysitter_telemetry"
    )
    if not project:
        _record_bq_failure(
            table_target, len(rows), "BQ_ENABLED=1 but GOOGLE_CLOUD_PROJECT is unset"
        )
        return False

    import time
    import urllib.error
    import urllib.request

    bq_payload_rows = []
    for row in rows:
        insert_id = str(
            row.get("event_id") or row.get("directive_id") or row.get("escalation_id") or ""
        )
        bq_row = dict(row)
        for json_col in (
            "parameter_modifications_before",
            "parameter_modifications_after",
            "okf_concepts_used",
            "directives_tried",
        ):
            if json_col in bq_row and not isinstance(bq_row[json_col], str):
                bq_row[json_col] = json.dumps(bq_row[json_col], separators=(",", ":"))
        bq_payload_rows.append({"insertId": insert_id, "json": bq_row})

    url = (
        f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}"
        f"/datasets/{dataset}/tables/{table_target}/insertAll"
    )
    body = json.dumps({"rows": bq_payload_rows}, default=str).encode("utf-8")

    last_error = "unknown error"
    for attempt in range(3):
        retryable = False
        try:
            token = _get_bq_token()
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8") or "{}")
            insert_errors = payload.get("insertErrors") or []
            if not insert_errors:
                return True
            # Row-level rejections (schema mismatch, invalid values) are not transient.
            _record_bq_failure(
                table_target,
                len(rows),
                f"insertErrors for {len(insert_errors)} row(s): "
                f"{json.dumps(insert_errors, default=str)}",
            )
            return False
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}: {exc.reason}"
            retryable = exc.code in _BQ_TRANSIENT_HTTP_CODES
        except Exception as exc:  # noqa: BLE001 - network/auth errors: retry, then log
            last_error = f"{type(exc).__name__}: {exc}"
            retryable = True
        if not retryable or attempt == 2:
            break
        time.sleep(0.5 * (attempt + 1))
    _record_bq_failure(table_target, len(rows), last_error)
    return False


def emit_structured_cloud_log(
    table_target: str, row: dict[str, Any], stream_bq: bool = True
) -> None:
    """Write single-line JSON to stdout for Cloud Logging sink and stream to BigQuery if enabled."""
    envelope = {
        "severity": "INFO",
        "log_stream": "sim_babysitter_telemetry",
        "table_target": table_target,
        "record": row,
    }
    request_id = current_request_id()
    if request_id:
        envelope["request_id"] = request_id
    sys.stdout.write(json.dumps(envelope, default=str) + "\n")
    sys.stdout.flush()
    if stream_bq:
        _stream_rows_to_bigquery(table_target, [row])


class _BoundedIdSet:
    """Insertion-ordered set of recently seen ids with a hard size cap (LRU eviction)."""

    def __init__(self, max_entries: int) -> None:
        self._max = max_entries
        self._ids: OrderedDict[str, None] = OrderedDict()

    def __contains__(self, item: object) -> bool:
        return item in self._ids

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, item: str) -> None:
        self._ids[item] = None
        self._ids.move_to_end(item)
        while len(self._ids) > self._max:
            self._ids.popitem(last=False)


MAX_SEEN_EVENT_IDS = 200_000

_EVENT_INSERT_SQL = """
    INSERT INTO events VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMPTZ),
        ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMPTZ)
    )
"""
_ESCALATION_INSERT_SQL = """
    INSERT INTO escalations VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMPTZ),
        ?, ?, ?, ?, ?, ?, ?
    )
"""
_EVENT_COLUMNS = (
    "event_id",
    "schema_version",
    "workstream",
    "backend",
    "run_id",
    "campaign_id",
    "solver",
    "fidelity",
    "event_type",
    "ts_utc",
    "retry_count",
    "walltime_resubmits",
    "environment",
    "operator",
    "is_synthetic",
    "payload",
    "ingested_at_utc",
)


class EventStore:
    """Thread-safe local DuckDB telemetry store + Cloud Logging structured sink emitter.

    On Cloud Run the DuckDB store is instance-local and ephemeral (it backs only the
    debug endpoint `/v1/events/summary`); BigQuery is the authoritative store.
    Methods block on I/O and must be called off the event loop.
    """

    def __init__(self, duckdb_path: str | None = None) -> None:
        resolved_path = (
            duckdb_path
            if duckdb_path is not None
            else os.getenv("LOCAL_EVENTS_DUCKDB_PATH", ":memory:")
        )
        if resolved_path != ":memory:":
            Path(resolved_path).parent.mkdir(parents=True, exist_ok=True)
        self.storage_backend = (
            "duckdb_local_instance_memory" if resolved_path == ":memory:" else "duckdb_local_file"
        )
        self._lock = threading.Lock()
        self._conn = duckdb.connect(resolved_path)
        self._seen_event_ids = _BoundedIdSet(MAX_SEEN_EVENT_IDS)
        init_duckdb_telemetry_schema(self._conn)
        rows = self._conn.execute(
            "SELECT event_id FROM events GROUP BY event_id ORDER BY MAX(ingested_at_utc) DESC LIMIT ?",
            [MAX_SEEN_EVENT_IDS],
        ).fetchall()
        for r in reversed(rows):
            if r[0]:
                self._seen_event_ids.add(str(r[0]))

    @property
    def bq_insert_failures(self) -> int:
        """Process-wide count of failed BigQuery streaming batches (exposed on /healthz)."""
        return bq_insert_failure_count()

    def ingest_events(self, events: list[LifecycleEvent]) -> tuple[int, int]:
        """Insert deduplicated events into DuckDB (one transaction) and emit structured logs."""
        deduplicated = 0
        now_iso = datetime.now(UTC).isoformat()
        event_rows: list[dict[str, Any]] = []
        escalation_rows: list[dict[str, Any]] = []
        batch_ids: set[str] = set()

        with self._lock:
            for ev in events:
                if ev.event_id in self._seen_event_ids or ev.event_id in batch_ids:
                    deduplicated += 1
                    continue
                batch_ids.add(ev.event_id)

                effective_env = ev.environment
                # Anything not explicitly prod (sim, synthetic, preflight) is synthetic.
                is_synth = bool(ev.is_synthetic or effective_env != "prod")
                payload_dict = dict(ev.payload)
                if ev.equilibrium_id and "equilibrium_id" not in payload_dict:
                    payload_dict["equilibrium_id"] = ev.equilibrium_id
                payload_str = json.dumps(payload_dict, separators=(",", ":"), allow_nan=False)

                row_dict = {
                    "event_id": ev.event_id,
                    "schema_version": ev.schema_version,
                    "workstream": ev.workstream,
                    "backend": ev.backend,
                    "run_id": ev.run_id,
                    "campaign_id": ev.campaign_id,
                    "solver": ev.solver,
                    "fidelity": ev.fidelity,
                    "event_type": ev.event_type,
                    "ts_utc": ev.ts_utc.astimezone(UTC).isoformat(),
                    "retry_count": ev.retry_count,
                    "walltime_resubmits": ev.walltime_resubmits,
                    "environment": effective_env,
                    "operator": ev.operator,
                    "is_synthetic": is_synth,
                    "payload": payload_str,
                    "ingested_at_utc": now_iso,
                }
                event_rows.append(row_dict)

                if ev.event_type == "escalated":
                    escalation_rows.append(
                        {
                            "escalation_id": f"esc-{ev.event_id}",
                            "event_id": ev.event_id,
                            "schema_version": ev.schema_version,
                            "workstream": ev.workstream,
                            "run_id": ev.run_id,
                            "campaign_id": ev.campaign_id,
                            "solver": ev.solver,
                            "fidelity": ev.fidelity,
                            "ts_utc": row_dict["ts_utc"],
                            "retry_count": ev.retry_count,
                            "failure_type": str(ev.payload.get("failure_type", "UNKNOWN")),
                            "issue_md_text": str(ev.payload.get("issue_md_text", "")),
                            "directives_tried": json.dumps(ev.payload.get("directives_tried", [])),
                            "tacc_folder_path": str(
                                ev.payload.get("tacc_folder_path", f"queue/needs_human/{ev.run_id}")
                            ),
                            "environment": effective_env,
                            "is_synthetic": is_synth,
                        }
                    )

            if event_rows:
                self._conn.execute("BEGIN TRANSACTION")
                try:
                    self._conn.executemany(
                        _EVENT_INSERT_SQL,
                        [[row[c] for c in _EVENT_COLUMNS] for row in event_rows],
                    )
                    if escalation_rows:
                        self._conn.executemany(
                            _ESCALATION_INSERT_SQL, [list(r.values()) for r in escalation_rows]
                        )
                    self._conn.execute("COMMIT")
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                # Mark ids seen only once they are durably inserted, so a failed batch
                # can be retried by the client instead of being "deduplicated" away.
                for row in event_rows:
                    self._seen_event_ids.add(row["event_id"])

        if event_rows or escalation_rows:
            lines = [
                json.dumps(
                    {
                        "severity": "INFO",
                        "log_stream": "sim_babysitter_telemetry",
                        "table_target": table,
                        "record": row,
                    },
                    default=str,
                )
                for table, rows in (("events", event_rows), ("escalations", escalation_rows))
                for row in rows
            ]
            sys.stdout.write("\n".join(lines) + "\n")
            sys.stdout.flush()

        _stream_rows_to_bigquery("events", event_rows)
        _stream_rows_to_bigquery("escalations", escalation_rows)

        return len(event_rows), deduplicated

    def record_directive(self, row: dict[str, Any]) -> None:
        """Persist a `/triage` directive outcome row into DuckDB and structured stdout."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO directives VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS TIMESTAMPTZ),
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                [
                    row["directive_id"],
                    row["schema_version"],
                    row["workstream"],
                    row["backend"],
                    row["run_id"],
                    row["campaign_id"],
                    row["solver"],
                    row["fidelity"],
                    row["ts_utc"],
                    row["retry_count"],
                    row["walltime_resubmits"],
                    row["environment"],
                    row["failure_type"],
                    row["action"],
                    row["reason"],
                    json.dumps(row["parameter_modifications_before"]),
                    json.dumps(row["parameter_modifications_after"]),
                    row["restart_from_checkpoint"],
                    row["confidence"],
                    json.dumps(row["okf_concepts_used"]),
                    row["guardrail_outcome"],
                    row["forced_escalation"],
                    row["llm_called"],
                    row["latency_ms"],
                    row["is_synthetic"],
                ],
            )
        emit_structured_cloud_log("directives", row)

    def query_summary(
        self, environment: str = "prod", workstream: str | None = None
    ) -> dict[str, Any]:
        """Return campaign KPIs and run statuses from the canonical SQL views (this instance only)."""
        with self._lock:
            params: list[Any] = [environment]
            ws_clause = ""
            if workstream:
                ws_clause = " AND workstream = ?"
                params.append(workstream)

            kpi_cur = self._conn.execute(
                f"SELECT * FROM v_kpis_campaign WHERE is_synthetic = FALSE AND environment = ?{ws_clause} ORDER BY campaign_id",
                params,
            )
            kpi_cols = [d[0] for d in kpi_cur.description]
            campaigns = [dict(zip(kpi_cols, r, strict=False)) for r in kpi_cur.fetchall()]

            runs_cur = self._conn.execute(
                f"SELECT workstream, campaign_id, run_id, backend, solver, fidelity, current_status, retry_count, is_agent_recovered, is_hands_free_completed FROM v_run_status WHERE is_synthetic = FALSE AND environment = ?{ws_clause} ORDER BY run_id",
                params,
            )
            run_cols = [d[0] for d in runs_cur.description]
            runs = [dict(zip(run_cols, r, strict=False)) for r in runs_cur.fetchall()]

            total_events = self._conn.execute(
                f"SELECT COUNT(*) FROM events WHERE is_synthetic = FALSE AND environment = ?{ws_clause}",
                params,
            ).fetchone()[0]

        return {
            "environment": environment,
            "workstream": workstream,
            # Instance-local view: on Cloud Run this is NOT the campaign-wide picture
            # (query the BigQuery KPI views / dashboard for that).
            "storage_backend": self.storage_backend,
            "total_events": int(total_events),
            "campaigns": campaigns,
            "runs": runs,
        }
