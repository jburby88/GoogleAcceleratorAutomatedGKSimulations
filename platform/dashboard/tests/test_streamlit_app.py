"""Headless UI tests (QA-07) for streamlit_app.py via streamlit.testing.v1.AppTest."""

import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit_app
from theme import COVERAGE_COLORS, PALETTE, STATUS_COLORS

TIMEOUT_S = 60

KPI_COLUMNS = [
    "workstream",
    "campaign_id",
    "is_synthetic",
    "environment",
    "completed_runs",
    "target_runs",
    "hands_free_rate",
    "agent_recovery_rate",
    "estimated_gpu_hours_saved",
    "estimated_human_hours_avoided",
    "walltime_resubmit_events",
    "llm_calls_per_failed_run",
]
RUN_COLUMNS = [
    "workstream",
    "campaign_id",
    "run_id",
    "environment",
    "is_synthetic",
    "backend",
    "solver",
    "geometry_id",
    "current_status",
    "last_updated_ts_utc",
    "last_failure_type",
    "escalation_reason",
    "tacc_folder_path",
]
TAX_COLUMNS = ["campaign_id", "solver", "failure_type", "affected_runs_count"]
RB_COLUMNS = ["campaign_id", "okf_concept_id", "solver", "directives_count"]
COV_COLUMNS = [
    "campaign_id",
    "run_id",
    "solver",
    "fidelity",
    "geometry_id",
    "coverage_status",
    "retry_count",
    "nx",
    "heat_flux_es",
]


def _kpi(campaign, completed, target, hands_free, gpu_hours):
    return {
        "workstream": "kr2-gk-database",
        "campaign_id": campaign,
        "is_synthetic": False,
        "environment": "prod",
        "completed_runs": completed,
        "target_runs": target,
        "hands_free_rate": hands_free,
        "agent_recovery_rate": None,  # SAFE_DIVIDE by zero -> NULL
        "estimated_gpu_hours_saved": gpu_hours,
        "estimated_human_hours_avoided": 1.25,
        "walltime_resubmit_events": 3,
        "llm_calls_per_failed_run": float("nan"),
    }


def _run(campaign, run_id, status, updated, backend="tacc_slurm", solver="GX"):
    return {
        "workstream": "kr2-gk-database",
        "campaign_id": campaign,
        "run_id": run_id,
        "environment": "prod",
        "is_synthetic": False,
        "backend": backend,
        "solver": solver,
        "geometry_id": "w7x_standard",
        "current_status": status,
        "last_updated_ts_utc": pd.Timestamp(updated, tz="UTC"),
        "last_failure_type": "STALL" if status == "escalated" else None,
        "escalation_reason": "retry_count >= 2" if status == "escalated" else None,
        "tacc_folder_path": f"queue/needs_human/{run_id}" if status == "escalated" else None,
    }


def populated_views():
    """Two campaigns; camp-b is the most recently updated one."""
    kpis = pd.DataFrame(
        [_kpi("camp-b", 4, 20, 0.5, 12.0), _kpi("camp-a", 1, 10, 0.25, 2.0)],
        columns=KPI_COLUMNS,
    )
    runs = pd.DataFrame(
        [
            _run("camp-a", "a-1", "completed", "2026-10-01"),
            _run("camp-a", "a-2", "escalated", "2026-10-02", solver="GENE"),
            _run("camp-a", "a-3", "escalated", "2026-10-03", backend="gcp_batch"),
            _run("camp-b", "b-1", "failed", "2026-11-01", backend="gcp_batch"),
            _run("camp-b", "b-2", "completed", "2026-11-02"),
            _run("camp-b", "b-3", "escalated", "2026-11-03", solver="GENE"),
        ],
        columns=RUN_COLUMNS,
    )
    tax = pd.DataFrame(
        [
            {
                "campaign_id": "camp-a",
                "solver": "GX",
                "failure_type": "STALL",
                "affected_runs_count": 2,
            },
            {
                "campaign_id": "camp-b",
                "solver": "GENE",
                "failure_type": "CRASH",
                "affected_runs_count": 1,
            },
        ],
        columns=TAX_COLUMNS,
    )
    rb = pd.DataFrame(
        [
            {
                "campaign_id": "camp-a",
                "okf_concept_id": "runbook-stalled",
                "solver": "GX",
                "directives_count": 2,
            },
            {
                "campaign_id": "camp-b",
                "okf_concept_id": "runbook-divergence",
                "solver": "GENE",
                "directives_count": 1,
            },
        ],
        columns=RB_COLUMNS,
    )
    cov = pd.DataFrame(
        [
            {
                "campaign_id": campaign,
                "run_id": run_id,
                "solver": "GX",
                "fidelity": "low",
                "geometry_id": "w7x_standard",
                "coverage_status": status,
                "retry_count": 0,
                "nx": 64.0,
                "heat_flux_es": flux,
            }
            for campaign, run_id, status, flux in [
                ("camp-a", "a-1", "converged", 1.5),
                ("camp-a", "a-2", "escalated", None),
                ("camp-b", "b-1", "failed", None),
                ("camp-b", "b-2", "converged", 2.5),
            ]
        ],
        columns=COV_COLUMNS,
    )
    return {
        "v_kpis_campaign": kpis,
        "v_run_status": runs,
        "v_failure_taxonomy": tax,
        "v_runbook_effectiveness": rb,
        "v_param_coverage": cov,
    }


class FakeDashboardClient:
    """Duck-types DashboardDataClient: source_label, cache_key, workstream, query_view."""

    def __init__(self, views, *, error=None):
        self._views = views
        self._error = error
        self.workstream = "kr2-gk-database"
        self.source_label = "BigQuery `ut-austin-test.sim_babysitter_telemetry`"
        self.cache_key = f"fake|{uuid.uuid4()}"
        self.calls = []

    def query_view(self, view_name):
        self.calls.append(view_name)
        if self._error is not None:
            raise self._error
        return self._views[view_name].copy()


def _render(client):
    from streamlit_app import main

    main(client=client)


def _render_from_env():
    from streamlit_app import main

    main()


def run_app(client) -> AppTest:
    return AppTest.from_function(
        _render, kwargs={"client": client}, default_timeout=TIMEOUT_S
    ).run()


@pytest.fixture(autouse=True)
def _fresh_view_cache():
    streamlit_app._cached_view.clear()
    yield
    streamlit_app._cached_view.clear()


def _plotly_charts(at):
    return at.get("plotly_chart")


def _escalation_header(at):
    (header,) = [s.value for s in at.subheader if s.value.startswith("Physicist Escalation Queue")]
    return header


# (a) populated views


def test_populated_views_render_without_exceptions():
    client = FakeDashboardClient(populated_views())
    at = run_app(client)

    assert not at.exception
    assert not at.error
    assert "BigQuery `ut-austin-test.sim_babysitter_telemetry`" in at.caption[0].value
    assert "is_synthetic = FALSE" in at.caption[0].value
    assert [m.label for m in at.metric] == [
        "Completed Runs",
        "Hands-Free Rate",
        "Agent Recovery Rate",
        "GPU Hours Saved",
        "Human Hours Avoided",
    ]
    assert len(at.tabs) == 3
    assert len(_plotly_charts(at)) == 3
    # Default campaign = most recently updated (camp-b); only its runs are shown.
    assert at.selectbox(key="campaign_id").options == ["camp-a", "camp-b"]
    assert at.selectbox(key="campaign_id").value == "camp-b"
    assert [m.value for m in at.metric] == ["4 / 20", "50.0%", "0.0%", "12.0 h", "1.2 h"]
    assert _escalation_header(at).endswith("— 1 runs)")


def test_campaign_selector_switches_kpis_and_filters_frames():
    at = run_app(FakeDashboardClient(populated_views()))
    assert at.metric[0].value == "4 / 20"

    at.selectbox(key="campaign_id").select("camp-a").run()

    assert not at.exception
    assert at.metric[0].value == "1 / 10"
    assert at.metric[1].value == "25.0%"
    assert at.metric[3].value == "2.0 h"
    assert _escalation_header(at).endswith("— 2 runs)")
    escalated = [df.value for df in at.dataframe if "tacc_folder_path" in df.value.columns][0]
    assert sorted(escalated["run_id"]) == ["a-2", "a-3"]
    assert list(escalated.columns) == streamlit_app.ESCALATION_COLUMNS


def test_views_are_cached_across_reruns():
    client = FakeDashboardClient(populated_views())
    at = run_app(client)
    at.run()
    assert not at.exception
    assert sorted(client.calls) == sorted(
        [
            "v_kpis_campaign",
            "v_run_status",
            "v_failure_taxonomy",
            "v_runbook_effectiveness",
            "v_param_coverage",
        ]
    )


# (b) empty KPIs


@pytest.mark.parametrize("empty_kpis", [pd.DataFrame(columns=KPI_COLUMNS), pd.DataFrame()])
def test_empty_kpis_show_info_message(empty_kpis):
    views = populated_views()
    views["v_kpis_campaign"] = empty_kpis
    at = run_app(FakeDashboardClient(views))

    assert not at.exception
    assert [i.value for i in at.info] == [streamlit_app.NO_TELEMETRY_MESSAGE]
    assert not at.metric
    assert not at.selectbox


# (c) empty runs / coverage


@pytest.mark.parametrize(
    "empty_runs, empty_cov",
    [
        (pd.DataFrame(columns=RUN_COLUMNS), pd.DataFrame(columns=COV_COLUMNS)),
        (pd.DataFrame(), pd.DataFrame()),
    ],
    ids=["columns-kept", "no-columns"],
)
def test_empty_runs_and_coverage_render_without_exceptions(empty_runs, empty_cov):
    views = populated_views()
    views["v_run_status"] = empty_runs
    views["v_param_coverage"] = empty_cov
    at = run_app(FakeDashboardClient(views))

    assert not at.exception
    assert len(at.metric) == 5
    assert len(_plotly_charts(at)) == 1  # only the failure taxonomy chart has data
    assert _escalation_header(at).endswith("— 0 runs)")
    # Without run timestamps the default campaign is the first sorted one.
    assert at.selectbox(key="campaign_id").value == "camp-a"


# failure paths


def test_view_query_failure_shows_error_not_traceback():
    client = FakeDashboardClient(populated_views(), error=RuntimeError("403 Access Denied"))
    at = run_app(client)

    assert not at.exception
    assert len(at.error) == 1
    assert "Could not load telemetry views" in at.error[0].value
    assert "403" not in at.error[0].value
    assert not at.metric


def test_invalid_data_mode_fails_closed(monkeypatch):
    monkeypatch.setenv("DATA_MODE", "synthetic")
    at = AppTest.from_function(_render_from_env, default_timeout=TIMEOUT_S).run()

    assert not at.exception
    assert len(at.error) == 1
    assert "DATA_MODE must be one of" in at.error[0].value
    assert not at.metric


# helpers


def test_status_colours_are_fixed_per_status():
    runs = populated_views()["v_run_status"]
    fig = streamlit_app.runs_by_backend_figure(runs)
    seen = {trace.name: trace.marker.color for trace in fig.data}
    assert seen == {status: STATUS_COLORS[status] for status in seen}
    assert seen["failed"] == PALETTE["crimson"]
    assert seen["escalated"] == PALETTE["crimson"]
    assert seen["completed"] == PALETTE["emerald"]
    assert PALETTE["emerald"] not in {STATUS_COLORS["failed"], COVERAGE_COLORS["failed"]}


def test_escalation_queue_tolerates_missing_columns():
    partial = pd.DataFrame([{"run_id": "r-1", "current_status": "escalated"}])
    queue = streamlit_app.escalation_queue(partial)
    assert list(queue.columns) == streamlit_app.ESCALATION_COLUMNS
    assert list(queue["run_id"]) == ["r-1"]
    assert streamlit_app.escalation_queue(pd.DataFrame()).empty


def test_for_campaign_never_mixes_campaigns():
    df = pd.DataFrame({"campaign_id": ["a", None, "b"], "x": [1, 2, 3]})
    assert list(streamlit_app.for_campaign(df, "a")["x"]) == [1]
    assert streamlit_app.for_campaign(pd.DataFrame({"x": [1]}), "a").empty
