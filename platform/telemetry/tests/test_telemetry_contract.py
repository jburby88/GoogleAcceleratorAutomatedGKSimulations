"""Contract tests for platform/telemetry: Terraform view parsing, DDL parity, prod-only KPIs.

The SQL in 02_views.sql is consumed twice: Terraform splits it into one BigQuery view per
statement (platform/infra/terraform/main.tf, locals.view_queries), and aurora-steer translates
it to DuckDB for its local store (app/events.py). These tests pin both contracts and the ML-06
rule that aggregate KPI views count production rows only.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import duckdb
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLES_SQL = REPO_ROOT / "platform" / "telemetry" / "01_tables.sql"
VIEWS_SQL = REPO_ROOT / "platform" / "telemetry" / "02_views.sql"
MAIN_TF = REPO_ROOT / "platform" / "infra" / "terraform" / "main.tf"

KPI_VIEWS = (
    "v_compute_savings",
    "v_kpis_campaign",
    "v_failure_taxonomy",
    "v_runbook_effectiveness",
)
ROW_LEVEL_VIEWS = ("v_run_status", "v_directive_outcomes", "v_param_coverage", "v_timeline")
VIEW_HEADER = re.compile(r"CREATE OR REPLACE VIEW `[^`]+\.([a-z0-9_]+)` AS")


def _terraform_view_queries(sql: str) -> dict[str, str]:
    """Mirror main.tf locals.view_statements / view_queries exactly."""
    statements = [s.strip() for s in sql.split(";") if VIEW_HEADER.search(s)]
    queries: dict[str, str] = {}
    for stmt in statements:
        name = VIEW_HEADER.search(stmt).group(1)
        queries[name] = re.sub(r"(?s)^.*CREATE OR REPLACE VIEW `[^`]+` AS\s*", "", stmt).strip()
    return queries


def _tf_set(name: str, tf_text: str) -> set[str]:
    match = re.search(rf"{name}\s*=\s*toset\(\[(.*?)\]\)", tf_text, flags=re.S)
    assert match, f"{name} not found in main.tf"
    return set(re.findall(r'"([a-z0-9_]+)"', match.group(1)))


def test_terraform_parses_exactly_the_eight_views() -> None:
    sql = VIEWS_SQL.read_text(encoding="utf-8")
    queries = _terraform_view_queries(sql)
    assert set(queries) == set(KPI_VIEWS) | set(ROW_LEVEL_VIEWS)
    for name, body in queries.items():
        assert body.startswith(("WITH", "SELECT")), f"{name} body does not start with WITH/SELECT"
    # One terminator per view, so no semicolon can hide inside a comment or a view body.
    assert sql.count(";") == len(queries)

    tf_text = MAIN_TF.read_text(encoding="utf-8")
    tiers = _tf_set("views_tier1", tf_text) | _tf_set("views_tier2", tf_text)
    tiers |= _tf_set("views_tier3", tf_text)
    assert tiers == set(queries)


def test_views_use_only_the_project_placeholder() -> None:
    sql = VIEWS_SQL.read_text(encoding="utf-8")
    refs = re.findall(r"`([^`\s]+\.[^`\s]+\.[^`\s]+)`", sql)
    assert refs, "no fully qualified references found"
    assert all(r.startswith("${project_id}.sim_babysitter_telemetry.") for r in refs), refs
    # Terraform must rewrite that exact placeholder ("$${" is HCL's escape for a literal "${").
    assert '"$${project_id}.sim_babysitter_telemetry"' in MAIN_TF.read_text(encoding="utf-8")


def _sql_table_columns() -> dict[str, list[tuple[str, str, bool]]]:
    text = TABLES_SQL.read_text(encoding="utf-8")
    tables: dict[str, list[tuple[str, str, bool]]] = {}
    for match in re.finditer(
        r"CREATE TABLE IF NOT EXISTS `[^`]+\.([a-z_]+)` \((.*?)\n\)", text, re.S
    ):
        cols = []
        for line in match.group(2).splitlines():
            col = re.match(
                r"^\s+([a-z_]+)\s+(STRING|INT64|BOOL|TIMESTAMP|DATE|FLOAT64|JSON)\b(.*)$", line
            )
            if col:
                head = col.group(3).split("OPTIONS", 1)[0]
                cols.append((col.group(1), col.group(2), "NOT NULL" in head))
        tables[match.group(1)] = cols
    return tables


def _tf_table_columns() -> dict[str, list[tuple[str, str, bool]]]:
    text = MAIN_TF.read_text(encoding="utf-8")
    tables: dict[str, list[tuple[str, str, bool]]] = {}
    pattern = r'resource "google_bigquery_table" "[a-z_]+" \{\s*dataset_id\s*=[^\n]*\n\s*table_id\s*=\s*"([a-z_]+)"(.*?)\n\}'
    for match in re.finditer(pattern, text, re.S):
        fields = re.findall(
            r'\{ name = "([a-z_]+)", type = "([A-Z0-9]+)", mode = "(REQUIRED|NULLABLE)" \}',
            match.group(2),
        )
        if fields:
            tables[match.group(1)] = [(n, t, m == "REQUIRED") for n, t, m in fields]
    return tables


def test_01_tables_sql_matches_terraform_schemas() -> None:
    sql_tables = _sql_table_columns()
    tf_tables = _tf_table_columns()
    assert set(sql_tables) == {"events", "directives", "escalations", "campaigns", "assumptions"}
    assert sql_tables == tf_tables


# --- ML-06: aggregate KPI views count prod rows only -------------------------------------------


@pytest.fixture()
def conn(monkeypatch: pytest.MonkeyPatch) -> duckdb.DuckDBPyConnection:
    events = pytest.importorskip(
        "app.events", reason="aurora-steer (DuckDB view translator) not installed"
    )
    monkeypatch.setenv("REPO_ROOT", str(REPO_ROOT))
    connection = duckdb.connect(":memory:")
    events.init_duckdb_telemetry_schema(connection)
    return connection


def _insert_run(
    conn: duckdb.DuckDBPyConnection, run_id: str, environment: str | None, is_synthetic: bool
) -> None:
    """A run that fails (STALL), is patched by the agent, and completes: agent-recovered."""
    t0 = datetime(2026, 10, 1, tzinfo=UTC)
    steps = [
        ("submitted", {}),
        ("failed", {"failure_type": "STALL", "early_cancelled": "true", "walltime_elapsed_s": 600}),
        ("patched", {"action": "warm_restart"}),
        ("completed", {"metrics": {"heat_flux_es": 1.0}}),
    ]
    for i, (event_type, payload) in enumerate(steps):
        ts = (t0 + timedelta(minutes=10 * i)).isoformat()
        conn.execute(
            "INSERT INTO events (event_id, schema_version, workstream, backend, run_id, campaign_id,"
            " solver, fidelity, event_type, ts_utc, retry_count, walltime_resubmits, environment,"
            " operator, is_synthetic, payload, ingested_at_utc)"
            " VALUES (?, '1.0', 'kr2-gk-database', 'tacc_slurm', ?, 'pilot_50', 'GX', 'low', ?,"
            " CAST(? AS TIMESTAMPTZ), ?, 0, ?, NULL, ?, ?, CAST(? AS TIMESTAMPTZ))",
            [
                f"{run_id}-{i}",
                run_id,
                event_type,
                ts,
                min(i, 1),
                environment,
                is_synthetic,
                json.dumps(payload),
                ts,
            ],
        )
    conn.execute(
        "INSERT INTO directives (directive_id, schema_version, workstream, backend, run_id, campaign_id,"
        " solver, fidelity, ts_utc, retry_count, walltime_resubmits, environment, failure_type, action,"
        " reason, parameter_modifications_before, parameter_modifications_after, restart_from_checkpoint,"
        " confidence, okf_concepts_used, guardrail_outcome, forced_escalation, llm_called, latency_ms,"
        " is_synthetic) VALUES (?, '1.0', 'kr2-gk-database', 'tacc_slurm', ?, 'pilot_50', 'GX', 'low',"
        " CAST(? AS TIMESTAMPTZ), 0, 0, ?, 'STALL', 'warm_restart', 'halve dt_max', '{}', '{}', TRUE,"
        " 'high', '[\"runbook-stalled\"]', 'accepted', FALSE, TRUE, 12.5, ?)",
        [
            f"dir-{run_id}",
            run_id,
            (t0 + timedelta(minutes=15)).isoformat(),
            environment,
            is_synthetic,
        ],
    )


@pytest.fixture()
def mixed_conn(conn: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    _insert_run(conn, "run-prod", "prod", False)
    _insert_run(conn, "run-sim", "sim", False)
    _insert_run(conn, "run-synthetic", "synthetic", True)
    _insert_run(conn, "run-prod-flagged-synthetic", "prod", True)
    _insert_run(conn, "run-no-env", None, False)
    for target, updated in ((50, "2026-09-01T00:00:00Z"), (60, "2026-09-15T00:00:00Z")):
        conn.execute(
            "INSERT INTO campaigns VALUES ('pilot_50', 'kr2-gk-database', ?, DATE '2026-09-01',"
            " DATE '2026-12-31', NULL, 'prod', FALSE, CAST(? AS TIMESTAMPTZ))",
            [target, updated],
        )
    return conn


def _rows(conn: duckdb.DuckDBPyConnection, view: str, columns: str = "*") -> list[dict]:
    cur = conn.execute(f"SELECT {columns} FROM {view}")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]


@pytest.mark.parametrize("view", KPI_VIEWS)
def test_kpi_views_count_prod_rows_only(mixed_conn: duckdb.DuckDBPyConnection, view: str) -> None:
    rows = _rows(mixed_conn, view)
    assert rows, f"{view} lost the prod rows"
    assert {(r["environment"], r["is_synthetic"]) for r in rows} == {("prod", False)}


def test_kpis_campaign_values_and_latest_campaign_target(
    mixed_conn: duckdb.DuckDBPyConnection,
) -> None:
    rows = _rows(mixed_conn, "v_kpis_campaign")
    assert len(rows) == 1, "duplicate campaign rows or non-prod rows leaked into the KPI view"
    kpi = rows[0]
    assert kpi["total_runs"] == 1
    assert kpi["completed_runs"] == 1
    assert kpi["agent_recovered_runs"] == 1
    assert kpi["target_runs"] == 60  # newest campaigns row wins
    assert kpi["llm_calls_count"] == 1


def test_failure_taxonomy_and_runbook_counts_are_prod_only(
    mixed_conn: duckdb.DuckDBPyConnection,
) -> None:
    (tax,) = _rows(mixed_conn, "v_failure_taxonomy")
    assert tax["affected_runs_count"] == 1
    (runbook,) = _rows(mixed_conn, "v_runbook_effectiveness")
    assert runbook["directives_count"] == 1
    (savings,) = _rows(mixed_conn, "v_compute_savings")
    assert savings["early_cancelled_failures"] == 1


def test_row_level_views_keep_environments_separated(mixed_conn: duckdb.DuckDBPyConnection) -> None:
    rows = _rows(mixed_conn, "v_run_status", "run_id, environment, is_synthetic")
    status = {r["run_id"]: (r["environment"], r["is_synthetic"]) for r in rows}
    assert status["run-prod"] == ("prod", False)
    assert status["run-sim"] == ("sim", False)
    assert status["run-synthetic"] == ("synthetic", True)
    # A missing environment is never promoted to prod.
    assert status["run-no-env"] == ("unknown", False)
