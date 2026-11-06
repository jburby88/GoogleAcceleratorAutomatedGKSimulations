"""Streamlit Telemetry & Operational Audit Dashboard (`platform/dashboard`)."""

from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from db import DashboardConfigError, DashboardDataClient
from theme import COVERAGE_COLORS, DASHBOARD_CSS, STATUS_COLORS

LOGGER = logging.getLogger("aurora_dashboard")

CACHE_TTL_S = 300
NO_TELEMETRY_MESSAGE = "No campaign telemetry rows available yet for the selected environment."
ESCALATION_COLUMNS = [
    "run_id",
    "backend",
    "solver",
    "geometry_id",
    "last_failure_type",
    "escalation_reason",
    "tacc_folder_path",
]
COVERAGE_HOVER_COLUMNS = ["run_id", "geometry_id", "fidelity", "retry_count"]
_OBSERVED_BADGE = '<span class="kpi-badge-observed">Observed</span>'
_ESTIMATED_BADGE = '<span class="kpi-badge-estimated">Estimated (TODO(david))</span>'


@st.cache_data(ttl=CACHE_TTL_S, show_spinner="Loading telemetry views...")
def _cached_view(_client: DashboardDataClient, source_key: str, view_name: str) -> pd.DataFrame:
    """Query one view. Cached per (source_key, view_name); `_client` itself is not hashed."""
    return _client.query_view(view_name)


def load_view(client: DashboardDataClient, view_name: str) -> pd.DataFrame:
    """Return one view's prod rows, cached for CACHE_TTL_S seconds per data source."""
    return _cached_view(client, client.cache_key, view_name)


def _has_columns(df: pd.DataFrame, *columns: str) -> bool:
    return not df.empty and all(column in df.columns for column in columns)


def _as_float(value: object, default: float = 0.0) -> float:
    """`value` as a float; `default` for NULL/NaN or anything non-numeric."""
    try:
        return default if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object) -> int:
    return int(_as_float(value))


def campaign_options(kpis_df: pd.DataFrame) -> list[str]:
    """Sorted campaign ids present in v_kpis_campaign."""
    if "campaign_id" not in kpis_df.columns:
        return []
    return sorted({str(campaign) for campaign in kpis_df["campaign_id"].dropna()})


def default_campaign_index(options: list[str], runs_df: pd.DataFrame) -> int:
    """Index of the most recently updated campaign (per v_run_status), else 0 (first sorted)."""
    if not options or not _has_columns(runs_df, "campaign_id", "last_updated_ts_utc"):
        return 0
    known = runs_df["campaign_id"].notna()
    recent = pd.DataFrame(
        {
            "campaign_id": runs_df.loc[known, "campaign_id"].astype(str),
            "ts": pd.to_datetime(
                runs_df.loc[known, "last_updated_ts_utc"], errors="coerce", utc=True
            ),
        }
    ).dropna()
    recent = recent[recent["campaign_id"].isin(options)]
    if recent.empty:
        return 0
    newest = recent.sort_values(["ts", "campaign_id"]).iloc[-1]["campaign_id"]
    return options.index(newest)


def for_campaign(df: pd.DataFrame, campaign_id: str) -> pd.DataFrame:
    """Rows of one campaign. A frame without campaign_id yields no rows (never mixes campaigns)."""
    if "campaign_id" not in df.columns:
        return df.iloc[0:0]
    ids = df["campaign_id"]
    return df.loc[ids.notna() & (ids.astype(str) == campaign_id)]


def escalation_queue(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Escalated runs with the queue columns; empty or partial frames never raise."""
    if "current_status" not in runs_df.columns:
        return pd.DataFrame(columns=ESCALATION_COLUMNS)
    escalated = runs_df.loc[runs_df["current_status"] == "escalated"]
    return escalated.reindex(columns=ESCALATION_COLUMNS)


def runs_by_backend_figure(runs_df: pd.DataFrame):
    """Runs per backend and solver, one fixed colour per current_status."""
    return px.histogram(
        runs_df,
        x="backend",
        color="current_status",
        barmode="group",
        facet_col="solver",
        color_discrete_map=STATUS_COLORS,
        category_orders={"current_status": list(STATUS_COLORS)},
    )


def _kpi_card(column, badge: str, label: str, value: str) -> None:
    with column:
        st.markdown(badge, unsafe_allow_html=True)
        st.metric(label, value)


def main(client: DashboardDataClient | None = None) -> None:
    """Render the Streamlit campaign telemetry and operational audit dashboard."""
    st.set_page_config(
        page_title="KR2 Gyrokinetic Database — aurora-steer",
        layout="wide",
    )
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)
    st.title("UT Austin × Google.org — KR2 Automated Gyrokinetic Database (`sim-babysitter`)")

    if client is None:
        try:
            client = DashboardDataClient()
        except DashboardConfigError as exc:
            LOGGER.error("Dashboard configuration error: %s", exc)
            st.error(f"Dashboard configuration error: {exc}")
            return

    st.caption(
        f"Data Source: **{client.source_label}** (prod only: `environment = 'prod'`, "
        f"`is_synthetic = FALSE`) | Workstream: `{client.workstream}` | "
        "Backends: **TACC Vista Slurm** (`GENE` + `GX`) & **GCP Cloud Batch** (`GX` Spot L4 / A100)"
    )

    try:
        kpis_df = load_view(client, "v_kpis_campaign")
        runs_df = load_view(client, "v_run_status")
        tax_df = load_view(client, "v_failure_taxonomy")
        rb_df = load_view(client, "v_runbook_effectiveness")
        cov_df = load_view(client, "v_param_coverage")
    except Exception:
        LOGGER.exception("Failed to load telemetry views from %s", client.source_label)
        st.error(
            f"Could not load telemetry views from {client.source_label}. "
            "The error details are in the server log."
        )
        return

    campaigns = campaign_options(kpis_df)
    if not campaigns:
        st.info(NO_TELEMETRY_MESSAGE)
        return

    campaign = st.selectbox(
        "Campaign",
        campaigns,
        index=default_campaign_index(campaigns, runs_df),
        key="campaign_id",
    )
    kpis_df = for_campaign(kpis_df, str(campaign))
    if kpis_df.empty:
        st.info(NO_TELEMETRY_MESSAGE)
        return
    runs_df = for_campaign(runs_df, str(campaign))
    tax_df = for_campaign(tax_df, str(campaign))
    rb_df = for_campaign(rb_df, str(campaign))
    cov_df = for_campaign(cov_df, str(campaign))

    row = kpis_df.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    _kpi_card(
        c1,
        _OBSERVED_BADGE,
        "Completed Runs",
        f"{_as_int(row.get('completed_runs'))} / {_as_int(row.get('target_runs'))}",
    )
    _kpi_card(
        c2,
        _OBSERVED_BADGE,
        "Hands-Free Rate",
        f"{_as_float(row.get('hands_free_rate')) * 100:.1f}%",
    )
    _kpi_card(
        c3,
        _OBSERVED_BADGE,
        "Agent Recovery Rate",
        f"{_as_float(row.get('agent_recovery_rate')) * 100:.1f}%",
    )
    _kpi_card(
        c4,
        _ESTIMATED_BADGE,
        "GPU Hours Saved",
        f"{_as_float(row.get('estimated_gpu_hours_saved')):.1f} h",
    )
    _kpi_card(
        c5,
        _ESTIMATED_BADGE,
        "Human Hours Avoided",
        f"{_as_float(row.get('estimated_human_hours_avoided')):.1f} h",
    )

    tab_exec, tab_ops, tab_audit = st.tabs(
        [
            "1. Dual-Backend Campaign Overview",
            "2. Failure Taxonomy & OKF Runbooks",
            "3. Parameter Space & Escalation Queue",
        ]
    )

    with tab_exec:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Runs by Execution Backend & Solver")
            if _has_columns(runs_df, "backend", "current_status", "solver"):
                st.plotly_chart(runs_by_backend_figure(runs_df), width="stretch")
        with col_b:
            st.subheader("Checkpoint Continuations (0 LLM Calls)")
            st.write(
                f"- **Total Slurm Walltime + GCP Spot Preemption Continuations:** `{_as_int(row.get('walltime_resubmit_events'))}`\n"
                f"- **LLM Calls / Failure Event:** `{_as_float(row.get('llm_calls_per_failed_run')):.2f}` (Hard cap `<= 1.0`)"
            )
            st.dataframe(kpis_df, width="stretch")

    with tab_ops:
        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Failure Taxonomy & Recovery Rate")
            if _has_columns(tax_df, "failure_type", "affected_runs_count", "solver"):
                fig_tax = px.bar(
                    tax_df,
                    x="failure_type",
                    y="affected_runs_count",
                    color="solver",
                    barmode="group",
                )
                st.plotly_chart(fig_tax, width="stretch")
        with col_d:
            st.subheader("OKF Runbook Effectiveness")
            st.dataframe(rb_df, width="stretch")

    with tab_audit:
        st.subheader(
            "Numerical Grid & Transport Flux Coverage (`ToBlick/mrx` Equilibria + Benchmark Geometries)"
        )
        if _has_columns(cov_df, "nx", "heat_flux_es", "coverage_status", "solver"):
            fig_cov = px.scatter(
                cov_df,
                x="nx",
                y="heat_flux_es",
                color="coverage_status",
                symbol="solver",
                color_discrete_map=COVERAGE_COLORS,
                hover_data=[c for c in COVERAGE_HOVER_COLUMNS if c in cov_df.columns],
            )
            st.plotly_chart(fig_cov, width="stretch")

        esc_df = escalation_queue(runs_df)
        st.subheader(f"Physicist Escalation Queue (`needs_human/` — {len(esc_df)} runs)")
        st.dataframe(esc_df, width="stretch")


if __name__ == "__main__":
    main()
