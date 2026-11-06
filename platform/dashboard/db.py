"""Read-only, views-only data access layer for the KR2 dashboard.

Every query reads exactly one allowlisted canonical view (`v_*`) and keeps only the
production rows of one workstream:

    is_synthetic = FALSE AND environment = 'prod' AND workstream = <bound parameter>

`DATA_MODE` selects the source; any other value fails closed in the constructor:

* `bigquery` (default, Cloud Run): views in `GOOGLE_CLOUD_PROJECT`.`BQ_DATASET`.
* `duckdb` (local inspection): the views inside a DuckDB file written by aurora-steer's
  `EventStore` (`LOCAL_EVENTS_DUCKDB_PATH`), opened read-only.
"""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_VIEWS = frozenset(
    {
        "v_run_status",
        "v_compute_savings",
        "v_kpis_campaign",
        "v_failure_taxonomy",
        "v_directive_outcomes",
        "v_runbook_effectiveness",
        "v_param_coverage",
        "v_timeline",
    }
)

DATA_MODES = frozenset({"bigquery", "duckdb"})
DEFAULT_DATA_MODE = "bigquery"
DEFAULT_WORKSTREAM = "kr2-gk-database"
DEFAULT_BQ_DATASET = "sim_babysitter_telemetry"
DEFAULT_BQ_MAX_BYTES_BILLED = 104_857_600  # 100 MiB per query
BQ_QUERY_TIMEOUT_S = 60.0
BQ_JOB_LABELS = {"app": "aurora-dashboard"}

# GCP project id: 6-30 chars, lowercase letter first, lowercase letters/digits/hyphens,
# no trailing hyphen; optionally domain-scoped ("example.com:my-project").
_PROJECT_ID_RE = re.compile(
    r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+:)?"
    r"[a-z][a-z0-9-]{4,28}[a-z0-9]"
)
_DATASET_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,1023}")
_DIGITS_RE = re.compile(r"[0-9]+")
_INT64_MAX = 2**63 - 1  # BigQuery's maximumBytesBilled is an int64

_BQ_CLIENTS: dict[str, Any] = {}
_BQ_CLIENTS_LOCK = threading.Lock()


class DashboardConfigError(ValueError):
    """Missing or invalid dashboard configuration. Raised by the constructor (fail closed)."""


def prod_view_sql(table_ref: str, workstream_param: str) -> str:
    """Build the only statement this module runs: one view, prod rows of one workstream."""
    return (
        f"SELECT * FROM {table_ref} "
        f"WHERE is_synthetic = FALSE AND environment = 'prod' AND workstream = {workstream_param}"
    )


def validate_project_id(project: str) -> str:
    """Return `project` if it is a valid GCP project id, else raise DashboardConfigError."""
    if not _PROJECT_ID_RE.fullmatch(project):
        raise DashboardConfigError(
            f"GOOGLE_CLOUD_PROJECT is not a valid GCP project id: {project!r}"
        )
    return project


def validate_dataset_id(dataset: str) -> str:
    """Return `dataset` if it is a valid BigQuery dataset id, else raise DashboardConfigError."""
    if not _DATASET_RE.fullmatch(dataset):
        raise DashboardConfigError(f"BQ_DATASET is not a valid BigQuery dataset id: {dataset!r}")
    return dataset


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _bigquery_project() -> str:
    project = _env("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise DashboardConfigError("GOOGLE_CLOUD_PROJECT must be set when DATA_MODE=bigquery")
    return validate_project_id(project)


def _max_bytes_billed() -> int:
    raw = _env("BQ_MAX_BYTES_BILLED")
    if not raw:
        return DEFAULT_BQ_MAX_BYTES_BILLED
    value = int(raw) if _DIGITS_RE.fullmatch(raw) else 0
    if not 0 < value <= _INT64_MAX:
        raise DashboardConfigError(f"BQ_MAX_BYTES_BILLED must be a positive integer; got {raw!r}")
    return value


def _duckdb_path() -> Path:
    raw = _env("LOCAL_EVENTS_DUCKDB_PATH")
    if not raw:
        raise DashboardConfigError(
            "LOCAL_EVENTS_DUCKDB_PATH must point to the DuckDB file written by aurora-steer's "
            "EventStore when DATA_MODE=duckdb"
        )
    path = Path(raw).expanduser()
    if not path.is_file():
        raise DashboardConfigError(
            f"LOCAL_EVENTS_DUCKDB_PATH={raw!r} does not exist or is not a file"
        )
    return path.resolve()


def _bigquery_client(project: str) -> Any:
    """Return the process-wide `bigquery.Client` for `project`, creating it once."""
    with _BQ_CLIENTS_LOCK:
        client = _BQ_CLIENTS.get(project)
        if client is None:
            from google.cloud import bigquery

            client = bigquery.Client(project=project)
            _BQ_CLIENTS[project] = client
        return client


class DashboardDataClient:
    """Queries ONLY the 8 canonical views (`v_*`), prod rows of one workstream."""

    def __init__(self, data_mode: str | None = None, workstream: str = DEFAULT_WORKSTREAM) -> None:
        mode = os.environ.get("DATA_MODE", DEFAULT_DATA_MODE) if data_mode is None else data_mode
        if mode not in DATA_MODES:
            raise DashboardConfigError(
                f"DATA_MODE must be one of {sorted(DATA_MODES)}; got {mode!r}"
            )
        if not isinstance(workstream, str) or not workstream.strip():
            raise DashboardConfigError(f"workstream must be a non-empty string; got {workstream!r}")
        self.data_mode: str = mode
        self.workstream: str = workstream
        self.project: str | None = None
        self.dataset: str | None = None
        self.max_bytes_billed: int | None = None
        self.duckdb_path: Path | None = None
        if mode == "bigquery":
            # Only these names are read; the GCP_PROJECT_ID / BQ_DATASET_ID aliases are not.
            self.project = _bigquery_project()
            self.dataset = validate_dataset_id(_env("BQ_DATASET") or DEFAULT_BQ_DATASET)
            self.max_bytes_billed = _max_bytes_billed()
        else:
            self.duckdb_path = _duckdb_path()

    @property
    def source_label(self) -> str:
        """Human-readable data source, shown in the dashboard caption."""
        if self.data_mode == "bigquery":
            return f"BigQuery `{self.project}.{self.dataset}`"
        return f"local DuckDB file `{self.duckdb_path}`"

    @property
    def cache_key(self) -> str:
        """Identifies the data behind this client; part of the dashboard's cache key."""
        return f"{self.data_mode}|{self.source_label}|{self.workstream}"

    def query_view(self, view_name: str) -> pd.DataFrame:
        """Return the prod rows of one allowlisted view. Empty results keep their columns."""
        if view_name not in ALLOWED_VIEWS:
            raise ValueError(f"Dashboard must query canonical views only; got {view_name!r}")
        if self.data_mode == "bigquery":
            return self._query_bigquery(view_name)
        if self.data_mode == "duckdb":
            return self._query_duckdb(view_name)
        raise DashboardConfigError(f"unsupported DATA_MODE {self.data_mode!r}")

    def _query_bigquery(self, view_name: str) -> pd.DataFrame:
        from google.cloud import bigquery

        sql = prod_view_sql(f"`{self.project}.{self.dataset}.{view_name}`", "@workstream")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("workstream", "STRING", self.workstream)
            ],
            maximum_bytes_billed=self.max_bytes_billed,
            labels=dict(BQ_JOB_LABELS),
            job_timeout_ms=int(BQ_QUERY_TIMEOUT_S * 1000),
            use_legacy_sql=False,
        )
        job = _bigquery_client(self.project).query(
            sql, job_config=job_config, timeout=BQ_QUERY_TIMEOUT_S
        )
        rows = job.result(timeout=BQ_QUERY_TIMEOUT_S)
        # Not QueryJob/RowIterator.to_dataframe(): it needs db-dtypes, which is not installed.
        columns = [field.name for field in rows.schema]
        return pd.DataFrame.from_records([dict(row.items()) for row in rows], columns=columns)

    def _query_duckdb(self, view_name: str) -> pd.DataFrame:
        import duckdb

        sql = prod_view_sql(view_name, "?")
        conn = duckdb.connect(str(self.duckdb_path), read_only=True)
        try:
            cursor = conn.execute(sql, [self.workstream])
            columns = [column[0] for column in cursor.description]
            # fetchnumpy(): fetchall() needs pytz (not installed) for TIMESTAMPTZ columns.
            data = cursor.fetchnumpy()
        finally:
            conn.close()
        return pd.DataFrame(data, columns=columns)
