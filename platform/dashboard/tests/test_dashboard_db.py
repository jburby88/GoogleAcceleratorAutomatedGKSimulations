import ast
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from google.cloud import bigquery
from google.cloud.bigquery.table import Row

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db
from db import (
    ALLOWED_VIEWS,
    DEFAULT_BQ_DATASET,
    DEFAULT_BQ_MAX_BYTES_BILLED,
    DashboardConfigError,
    DashboardDataClient,
)

PROD_FILTER = "WHERE is_synthetic = FALSE AND environment = 'prod' AND workstream = "
ENV_VARS = (
    "DATA_MODE",
    "GOOGLE_CLOUD_PROJECT",
    "GCP_PROJECT_ID",
    "BQ_DATASET",
    "BQ_DATASET_ID",
    "BQ_MAX_BYTES_BILLED",
    "LOCAL_EVENTS_DUCKDB_PATH",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(db, "_BQ_CLIENTS", {})


@pytest.fixture
def bq_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ut-austin-test")


# --- configuration (fail closed in the constructor) ---------------------------------------


def test_dashboard_data_client_init(bq_env):
    client = DashboardDataClient(data_mode="bigquery", workstream="test")
    assert client.data_mode == "bigquery"
    assert client.workstream == "test"
    assert client.project == "ut-austin-test"
    assert client.dataset == DEFAULT_BQ_DATASET
    assert client.max_bytes_billed == DEFAULT_BQ_MAX_BYTES_BILLED
    assert client.source_label == "BigQuery `ut-austin-test.sim_babysitter_telemetry`"


def test_default_data_mode_is_bigquery(bq_env):
    assert DashboardDataClient().data_mode == "bigquery"


@pytest.mark.parametrize(
    "mode", ["synthetic", "", " ", "sim", "prod", "BigQuery", "bigquery ", "duck_db", "local"]
)
def test_bad_data_mode_rejected(monkeypatch, bq_env, mode):
    with pytest.raises(ValueError, match="DATA_MODE must be one of"):
        DashboardDataClient(data_mode=mode)
    monkeypatch.setenv("DATA_MODE", mode)
    with pytest.raises(ValueError, match="DATA_MODE must be one of"):
        DashboardDataClient()


def test_blank_workstream_rejected(bq_env):
    with pytest.raises(DashboardConfigError, match="workstream must be a non-empty string"):
        DashboardDataClient(workstream=" ")


def test_dashboard_data_client_invalid_view(bq_env):
    client = DashboardDataClient()
    for view in ("invalid_view", "events", "v_run_status; DROP TABLE events"):
        with pytest.raises(ValueError, match="Dashboard must query canonical views only"):
            client.query_view(view)


def test_dashboard_data_client_bq_no_project():
    with pytest.raises(DashboardConfigError, match="GOOGLE_CLOUD_PROJECT must be set"):
        DashboardDataClient(data_mode="bigquery")


def test_legacy_env_aliases_are_ignored(monkeypatch):
    monkeypatch.setenv("GCP_PROJECT_ID", "legacy-project")
    with pytest.raises(DashboardConfigError, match="GOOGLE_CLOUD_PROJECT must be set"):
        DashboardDataClient(data_mode="bigquery")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "ut-austin-test")
    monkeypatch.setenv("BQ_DATASET_ID", "legacy_dataset")
    assert DashboardDataClient(data_mode="bigquery").dataset == DEFAULT_BQ_DATASET


@pytest.mark.parametrize(
    "project", ["ut-austin-bsitter", "abcdef", "a12345", "example.com:my-project", "a" * 30]
)
def test_valid_project_ids_accepted(monkeypatch, project):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    assert DashboardDataClient(data_mode="bigquery").project == project


@pytest.mark.parametrize(
    "project",
    [
        "short",
        "a" * 31,
        "1project",
        "project-",
        "My-Project",
        "my_project",
        "my project",
        "proj`.x.y` WHERE TRUE --",
        "proj.dataset",
        "example.com:",
        ":my-project",
        "example:my-project",
    ],
)
def test_invalid_project_ids_rejected(monkeypatch, project):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    with pytest.raises(DashboardConfigError, match="not a valid GCP project id"):
        DashboardDataClient(data_mode="bigquery")


def test_project_id_validator_rejects_trailing_newline():
    with pytest.raises(DashboardConfigError, match="not a valid GCP project id"):
        db.validate_project_id("my-project\n")


@pytest.mark.parametrize("dataset", ["sim_babysitter_telemetry", "_private", "A1", "a" * 1024])
def test_valid_datasets_accepted(monkeypatch, bq_env, dataset):
    monkeypatch.setenv("BQ_DATASET", dataset)
    assert DashboardDataClient(data_mode="bigquery").dataset == dataset


@pytest.mark.parametrize(
    "dataset", ["1abc", "my-dataset", "ds.name", "ds`; DROP TABLE x; --", "a" * 1025, "dåta"]
)
def test_invalid_datasets_rejected(monkeypatch, bq_env, dataset):
    monkeypatch.setenv("BQ_DATASET", dataset)
    with pytest.raises(DashboardConfigError, match="BQ_DATASET is not a valid"):
        DashboardDataClient(data_mode="bigquery")


@pytest.mark.parametrize(("raw", "expected"), [("1", 1), ("5000000000", 5_000_000_000)])
def test_max_bytes_billed_accepts_positive_ints(monkeypatch, bq_env, raw, expected):
    monkeypatch.setenv("BQ_MAX_BYTES_BILLED", raw)
    assert DashboardDataClient(data_mode="bigquery").max_bytes_billed == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "1e9", "1.5", "+5", "1_000", str(2**63)])
def test_max_bytes_billed_rejects_non_positive_ints(monkeypatch, bq_env, raw):
    monkeypatch.setenv("BQ_MAX_BYTES_BILLED", raw)
    with pytest.raises(DashboardConfigError, match="BQ_MAX_BYTES_BILLED must be a positive"):
        DashboardDataClient(data_mode="bigquery")


def test_db_never_imports_aurora_steer():
    tree = ast.parse(Path(db.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"app", "aurora_steer", "hpc_core", "tacc_exec", "gcp_exec"})


# --- BigQuery mode (fake client, no network) ------------------------------------------------

RUN_SCHEMA = [
    bigquery.SchemaField("run_id", "STRING"),
    bigquery.SchemaField("environment", "STRING"),
    bigquery.SchemaField("retry_count", "INTEGER"),
]
RUN_FIELDS = {"run_id": 0, "environment": 1, "retry_count": 2}


class _FakeRowIterator:
    def __init__(self, schema, rows):
        self.schema = schema
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def to_dataframe(self, *args, **kwargs):
        raise AssertionError("RowIterator.to_dataframe needs db-dtypes and must not be called")


class _FakeQueryJob:
    def __init__(self, rows):
        self._rows = rows
        self.result_calls = []

    def result(self, *args, **kwargs):
        self.result_calls.append((args, kwargs))
        return self._rows

    def to_dataframe(self, *args, **kwargs):
        raise AssertionError("QueryJob.to_dataframe needs db-dtypes and must not be called")


class _FakeBigQuery:
    """Stands in for google.cloud.bigquery.Client; records every client and query."""

    def __init__(self):
        self.clients = []
        self.queries = []
        self.rows = []

    def client_class(self):
        fake = self

        class FakeClient:
            def __init__(self, project=None, **kwargs):
                self.project = project
                fake.clients.append(self)

            def query(self, sql, job_config=None, **kwargs):
                job = _FakeQueryJob(_FakeRowIterator(RUN_SCHEMA, list(fake.rows)))
                fake.queries.append(
                    {"sql": sql, "job_config": job_config, "kwargs": kwargs, "job": job}
                )
                return job

        return FakeClient


@pytest.fixture
def fake_bq(monkeypatch, bq_env):
    fake = _FakeBigQuery()
    monkeypatch.setattr(bigquery, "Client", fake.client_class())
    return fake


def test_bigquery_query_is_prod_only_parameterized_and_capped(monkeypatch, fake_bq):
    fake_bq.rows = [Row(("run-1", "prod", 0), RUN_FIELDS), Row(("run-2", "prod", 2), RUN_FIELDS)]
    monkeypatch.setenv("BQ_MAX_BYTES_BILLED", "12345")

    df = DashboardDataClient(data_mode="bigquery").query_view("v_run_status")

    assert [c.project for c in fake_bq.clients] == ["ut-austin-test"]
    (call,) = fake_bq.queries
    assert call["sql"] == (
        "SELECT * FROM `ut-austin-test.sim_babysitter_telemetry.v_run_status` "
        f"{PROD_FILTER}@workstream"
    )
    cfg = call["job_config"]
    assert isinstance(cfg, bigquery.QueryJobConfig)
    (param,) = cfg.query_parameters
    assert (param.name, param.type_, param.value) == ("workstream", "STRING", "kr2-gk-database")
    assert cfg.maximum_bytes_billed == 12345
    assert cfg.labels == {"app": "aurora-dashboard"}
    assert cfg.use_legacy_sql is False
    assert int(cfg.job_timeout_ms) == int(db.BQ_QUERY_TIMEOUT_S * 1000)  # API repr is a str
    assert call["kwargs"]["timeout"] == db.BQ_QUERY_TIMEOUT_S
    assert call["job"].result_calls == [((), {"timeout": db.BQ_QUERY_TIMEOUT_S})]
    assert list(df.columns) == ["run_id", "environment", "retry_count"]
    assert df.to_dict("records") == [
        {"run_id": "run-1", "environment": "prod", "retry_count": 0},
        {"run_id": "run-2", "environment": "prod", "retry_count": 2},
    ]


@pytest.mark.parametrize("view", sorted(ALLOWED_VIEWS))
def test_every_bigquery_view_query_has_prod_filter(fake_bq, view):
    DashboardDataClient(data_mode="bigquery", workstream="kr2-gk-database").query_view(view)
    (call,) = fake_bq.queries
    assert call["sql"] == (
        f"SELECT * FROM `ut-austin-test.sim_babysitter_telemetry.{view}` {PROD_FILTER}@workstream"
    )
    assert [p.value for p in call["job_config"].query_parameters] == ["kr2-gk-database"]


def test_bigquery_client_is_reused_per_process(fake_bq):
    DashboardDataClient().query_view("v_run_status")
    DashboardDataClient().query_view("v_kpis_campaign")
    DashboardDataClient().query_view("v_param_coverage")
    assert len(fake_bq.clients) == 1
    assert len(fake_bq.queries) == 3


def test_bigquery_empty_result_keeps_columns(fake_bq):
    df = DashboardDataClient().query_view("v_run_status")
    assert df.empty
    assert list(df.columns) == ["run_id", "environment", "retry_count"]


# --- DuckDB mode ------------------------------------------------------------------------------

TS = "2026-11-01 12:00:00+00:00"
MIXED_ROWS = [
    ("prod-1", "kr2-gk-database", "prod", False, "completed", TS),
    ("prod-2", "kr2-gk-database", "prod", False, "escalated", TS),
    ("synthetic-1", "kr2-gk-database", "synthetic", True, "completed", TS),
    ("sim-1", "kr2-gk-database", "sim", False, "completed", TS),
    ("prod-but-synthetic-1", "kr2-gk-database", "prod", True, "completed", TS),
    ("prod-null-synthetic-1", "kr2-gk-database", "prod", None, "completed", TS),
    ("other-workstream-1", "kr1-optimizer", "prod", False, "completed", TS),
]
RUN_COLUMNS = [
    "run_id",
    "workstream",
    "environment",
    "is_synthetic",
    "current_status",
    "last_updated_ts_utc",
]


def _write_events_duckdb(path: Path, rows: list[tuple]) -> None:
    """Stand-in for aurora-steer's EventStore file: a table plus the 8 allowlisted views."""
    conn = duckdb.connect(str(path))
    try:
        conn.execute(
            "CREATE TABLE run_rows (run_id VARCHAR, workstream VARCHAR, environment VARCHAR, "
            "is_synthetic BOOLEAN, current_status VARCHAR, last_updated_ts_utc TIMESTAMPTZ)"
        )
        if rows:
            conn.executemany("INSERT INTO run_rows VALUES (?, ?, ?, ?, ?, ?)", rows)
        for view in sorted(ALLOWED_VIEWS):
            conn.execute(f"CREATE VIEW {view} AS SELECT * FROM run_rows")
    finally:
        conn.close()


@pytest.fixture
def duckdb_file(monkeypatch, tmp_path):
    path = tmp_path / "events.duckdb"
    _write_events_duckdb(path, MIXED_ROWS)
    monkeypatch.setenv("LOCAL_EVENTS_DUCKDB_PATH", str(path))
    return path


@pytest.mark.parametrize("view", sorted(ALLOWED_VIEWS))
def test_duckdb_mode_returns_only_prod_rows(duckdb_file, view):
    client = DashboardDataClient(data_mode="duckdb")
    df = client.query_view(view)

    assert sorted(df["run_id"]) == ["prod-1", "prod-2"]
    assert set(df["environment"]) == {"prod"}
    assert not df["is_synthetic"].astype(bool).any()
    assert list(df.columns) == RUN_COLUMNS
    # TIMESTAMPTZ columns load without pytz and keep the UTC instant.
    assert pd.to_datetime(df["last_updated_ts_utc"], utc=True).iloc[0] == pd.Timestamp(TS)
    assert client.source_label == f"local DuckDB file `{duckdb_file.resolve()}`"


def test_query_view_refuses_mode_tampered_after_construction(duckdb_file):
    client = DashboardDataClient(data_mode="duckdb")
    client.data_mode = "synthetic"
    with pytest.raises(DashboardConfigError, match="unsupported DATA_MODE"):
        client.query_view("v_run_status")


def test_duckdb_mode_honours_workstream_parameter(duckdb_file):
    client = DashboardDataClient(data_mode="duckdb", workstream="kr1-optimizer")
    assert list(client.query_view("v_run_status")["run_id"]) == ["other-workstream-1"]


def test_duckdb_empty_result_keeps_columns(monkeypatch, tmp_path):
    path = tmp_path / "events.duckdb"
    _write_events_duckdb(path, [r for r in MIXED_ROWS if not r[0].startswith("prod-")])
    monkeypatch.setenv("LOCAL_EVENTS_DUCKDB_PATH", str(path))
    df = DashboardDataClient(data_mode="duckdb").query_view("v_run_status")
    assert df.empty
    assert list(df.columns) == RUN_COLUMNS


def test_duckdb_opened_read_only_with_bound_parameter(monkeypatch, duckdb_file):
    real_connect = duckdb.connect
    calls = []

    class SpyConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=None):
            calls.append(("execute", sql, params))
            return self._conn.execute(sql, params)

        def close(self):
            calls.append(("close",))
            self._conn.close()

    def spy_connect(database, *args, **kwargs):
        calls.append(("connect", database, kwargs))
        return SpyConnection(real_connect(database, *args, **kwargs))

    monkeypatch.setattr(duckdb, "connect", spy_connect)
    DashboardDataClient(data_mode="duckdb").query_view("v_run_status")

    assert calls == [
        ("connect", str(duckdb_file.resolve()), {"read_only": True}),
        ("execute", f"SELECT * FROM v_run_status {PROD_FILTER}?", ["kr2-gk-database"]),
        ("close",),
    ]


def test_duckdb_mode_requires_existing_file(monkeypatch, tmp_path):
    with pytest.raises(DashboardConfigError, match="LOCAL_EVENTS_DUCKDB_PATH must point"):
        DashboardDataClient(data_mode="duckdb")
    for bad in (tmp_path / "missing.duckdb", tmp_path, ":memory:"):
        monkeypatch.setenv("LOCAL_EVENTS_DUCKDB_PATH", str(bad))
        with pytest.raises(DashboardConfigError, match="does not exist or is not a file"):
            DashboardDataClient(data_mode="duckdb")


def test_duckdb_mode_ignores_bigquery_settings(monkeypatch, duckdb_file):
    monkeypatch.setenv("BQ_DATASET", "not a dataset")
    client = DashboardDataClient(data_mode="duckdb")
    assert client.project is None and client.dataset is None
