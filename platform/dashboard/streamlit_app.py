"""Streamlit Telemetry & Operational Audit Dashboard (`platform/dashboard`)."""

from __future__ import annotations

import html
import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from chat import ask_telemetry, default_suggestions
from db import DashboardConfigError, DashboardDataClient
from theme import (
    COVERAGE_COLORS,
    DASHBOARD_CSS,
    PALETTE,
    STATUS_COLORS,
    build_dashboard_css,
    plot_layout,
)

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

# TODO(david): confirm baseline manual troubleshooting duration and GPU walltime assumptions
_OBSERVED_BADGE = '<span class="kpi-badge-observed">● Observed Telemetry</span>'
_ESTIMATED_BADGE = '<span class="kpi-badge-estimated">◆ Modelled Savings</span>'


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


def runs_by_backend_figure(runs_df: pd.DataFrame, *, is_dark: bool = True):
    """Runs per backend and solver, one fixed colour per current_status."""
    fig = px.histogram(
        runs_df,
        x="backend",
        color="current_status",
        barmode="group",
        facet_col="solver",
        color_discrete_map=STATUS_COLORS,
        category_orders={"current_status": list(STATUS_COLORS)},
        labels={
            "backend": "Execution Backend",
            "current_status": "Status",
            "solver": "Solver",
        },
    )
    fig.update_layout(**plot_layout(is_dark=is_dark))
    fig.update_yaxes(title_text="Runs")
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("solver=", "Solver: ")))
    return fig


def _status_badge_html(status: object) -> str:
    val = str(status or "unknown").strip().lower()
    cls = {
        "completed": "badge-green",
        "converged": "badge-green",
        "running": "badge-blue",
        "submitted": "badge-blue",
        "human_resolved": "badge-amber",
        "escalated": "badge-red",
        "failed": "badge-red",
    }.get(val, "badge-blue")
    return f'<span class="badge {cls}">{html.escape(val)}</span>'


def _render_runs_html_table(runs_df: pd.DataFrame) -> None:
    """Render a full-width, horizontally scrollable HTML table of campaign runs."""
    if runs_df.empty:
        return
    cols = [
        c
        for c in (
            "run_id",
            "backend",
            "solver",
            "geometry_id",
            "current_status",
            "last_failure_type",
            "escalation_reason",
            "tacc_folder_path",
        )
        if c in runs_df.columns
    ]
    if not cols:
        return
    headers = {
        "run_id": "Run ID",
        "backend": "Backend",
        "solver": "Solver",
        "geometry_id": "Equilibrium / Geometry",
        "current_status": "Status",
        "last_failure_type": "Last Failure Mode",
        "escalation_reason": "Escalation / Recovery Note",
        "tacc_folder_path": "Queue Artifact Path",
    }
    th_html = "".join(f"<th>{headers.get(c, c)}</th>" for c in cols)
    rows_html: list[str] = []
    for _, r in runs_df.head(25).iterrows():
        cells: list[str] = []
        for c in cols:
            raw = r.get(c)
            if pd.isna(raw) or raw is None or str(raw) == "":
                cells.append('<td style="color:var(--text-dim);">—</td>')
            elif c == "current_status":
                cells.append(f"<td>{_status_badge_html(raw)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(raw))}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f'<div class="data-table-wrap"><table class="data-table">'
        f"<thead><tr>{th_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _render_chat_turn(msg: dict[str, object]) -> None:
    """Render one compact, cohesive chat message card without vertical clutter."""
    role = str(msg.get("role", "assistant"))
    content = str(msg.get("content", ""))
    with st.chat_message(role):
        if role == "assistant":
            st.markdown(
                f'<div class="aurora-msg-role"><span>✦ AURORA</span></div>\n\n{content}',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="aurora-msg-role user"><span>YOU</span></div>\n\n{html.escape(content)}',
                unsafe_allow_html=True,
            )


def _render_telemetry_chat(
    *,
    client: DashboardDataClient,
    campaign_id: str,
    kpis_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    all_kpis_df: pd.DataFrame,
    all_runs_df: pd.DataFrame,
    all_tax_df: pd.DataFrame,
    all_rb_df: pd.DataFrame,
    all_cov_df: pd.DataFrame,
) -> None:
    """Render the top-right '✦ Ask Aurora' copilot drawer with instant Thinking feedback."""
    state_key = "aurora_chat_messages"
    pending_key = "aurora_chat_pending_prompt"
    if state_key not in st.session_state:
        st.session_state[state_key] = []

    with st.popover("✦ Ask Aurora", width="stretch"):

        @st.fragment
        def _chat_drawer() -> None:
            messages: list[dict[str, object]] = st.session_state[state_key]

            top_l, top_r = st.columns([7.8, 2.2])
            with top_l:
                st.markdown(
                    f"""
                    <div class="aurora-drawer-title">
                        <span>✦ Ask Aurora</span>
                        <span class="chat-engine-pill">{html.escape(campaign_id)}</span>
                    </div>
                    <div class="aurora-drawer-sub">
                        Simulation &amp; telemetry assistant
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with top_r:
                if messages and st.button("Clear", key="clear_aurora_chat_btn", width="stretch"):
                    st.session_state[state_key] = []
                    messages = st.session_state[state_key]

            history_box = st.container(height=400, border=False)
            user_prompt = st.chat_input(
                "Ask Aurora about campaigns, runs, OKF directives, or flux...",
                key="telemetry_chat_input",
            )
            active_prompt = user_prompt or st.session_state.pop(pending_key, None)

            with history_box:
                if not messages and not active_prompt:
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.5rem;">'
                        f"Ask anything across your campaigns or active campaign "
                        f"<strong>{html.escape(campaign_id)}</strong>:"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    for idx, prompt_text in enumerate(default_suggestions(campaign_id)):
                        if st.button(
                            f"✦ {prompt_text}",
                            key=f"chat_starter_{idx}",
                            width="stretch",
                        ):
                            st.session_state[pending_key] = prompt_text
                            st.rerun(scope="fragment")

                for msg in messages:
                    _render_chat_turn(msg)

                if active_prompt:
                    user_msg: dict[str, object] = {"role": "user", "content": str(active_prompt)}
                    messages.append(user_msg)
                    _render_chat_turn(user_msg)

                    with st.chat_message("assistant"):
                        reply_slot = st.empty()

                        def _show_thinking(status_text: str = "Thinking...") -> None:
                            reply_slot.markdown(
                                '<div class="aurora-msg-role"><span>✦ AURORA</span></div>'
                                '<div class="aurora-thinking">'
                                '<span class="aurora-thinking-dots"><span></span><span></span><span></span></span>'
                                f"<span>{html.escape(status_text)}</span>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                        _show_thinking("Thinking...")
                        turn = ask_telemetry(
                            question=str(active_prompt),
                            campaign_id=campaign_id,
                            messages=messages,
                            project_id=getattr(client, "project", None),
                            dataset_id=getattr(client, "dataset", None),
                            data_mode=getattr(client, "data_mode", "bigquery"),
                            kpis_df=kpis_df,
                            runs_df=runs_df,
                            tax_df=tax_df,
                            rb_df=rb_df,
                            cov_df=cov_df,
                            client=client,
                            all_kpis_df=all_kpis_df,
                            all_runs_df=all_runs_df,
                            all_tax_df=all_tax_df,
                            all_rb_df=all_rb_df,
                            all_cov_df=all_cov_df,
                            on_status=_show_thinking,
                        )
                        reply_slot.markdown(
                            f'<div class="aurora-msg-role"><span>✦ AURORA</span></div>\n\n{turn.answer}',
                            unsafe_allow_html=True,
                        )

                    asst_msg: dict[str, object] = {
                        "role": "assistant",
                        "content": turn.answer,
                        "thoughts": turn.thoughts,
                        "suggestions": turn.suggestions,
                        "engine": turn.engine,
                    }
                    messages.append(asst_msg)

        _chat_drawer()


def _kpi_card(column, badge: str, label: str, value: str) -> None:
    with column:
        st.markdown(badge, unsafe_allow_html=True)
        st.metric(label, value)


def _toggle_theme() -> None:
    st.session_state[" aurora_theme"] = (
        "light" if st.session_state.get(" aurora_theme", "dark") == "dark" else "dark"
    )


def main(client: DashboardDataClient | None = None) -> None:
    """Render the Streamlit campaign telemetry and operational audit dashboard."""
    st.set_page_config(
        page_title="KR2 Gyrokinetic Database — aurora-steer",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if " aurora_theme" not in st.session_state:
        st.session_state[" aurora_theme"] = "dark"
    is_dark = st.session_state[" aurora_theme"] == "dark"
    st.markdown(
        build_dashboard_css(is_dark=is_dark) if is_dark is not True else DASHBOARD_CSS,
        unsafe_allow_html=True,
    )

    head_left, head_chat, head_right = st.columns([7.1, 1.7, 1.2])
    with head_left:
        st.title("UT Austin × Google.org — KR2 Automated Gyrokinetic Database (`sim-babysitter`)")
    with head_right:
        st.button(
            "☀️ Light" if is_dark else "🌙 Dark",
            key="theme_toggle_btn",
            on_click=_toggle_theme,
            width="stretch",
        )

    if client is None:
        try:
            client = DashboardDataClient()
        except DashboardConfigError as exc:
            LOGGER.error("Dashboard configuration error: %s", exc)
            st.error(f"Dashboard configuration error: {exc}")
            return

    st.markdown(
        """
        <div class="hero-banner">
            <div>
                <div class="hero-eyebrow"><span class="hero-live-dot"></span> Gyrokinetic Simulation Watchdog &amp; Steering</div>
                <div style="font-size:0.86rem;color:var(--text-muted);">
                    Transport-consistent high-&beta; stellarator optimization telemetry across <strong>TACC Vista Slurm</strong> and <strong>GCP Cloud Batch</strong>.
                </div>
            </div>
            <div class="hero-meta-pills">
                <span class="meta-pill">Solvers: GENE + GX</span>
                <span class="meta-pill">TACC: Slurm Driver</span>
                <span class="meta-pill">GCP: Spot L4 / A100</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

    if "campaign_id" not in st.session_state:
        st.session_state["campaign_id"] = campaigns[default_campaign_index(campaigns, runs_df)]

    campaign = st.selectbox(
        "Campaign",
        campaigns,
        key="campaign_id",
    )
    all_kpis_df = kpis_df
    all_runs_df = runs_df
    all_tax_df = tax_df
    all_rb_df = rb_df
    all_cov_df = cov_df

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
            st.markdown(
                '<div class="panel-subtitle">Distribution of simulation outcomes across TACC Slurm and GCP Cloud Batch</div>',
                unsafe_allow_html=True,
            )
            if _has_columns(runs_df, "backend", "current_status", "solver"):
                st.plotly_chart(
                    runs_by_backend_figure(runs_df, is_dark=is_dark),
                    width="stretch",
                    config={"displayModeBar": False},
                )
        with col_b:
            st.subheader("Checkpoint Continuations & Operational Guardrails")
            st.markdown(
                '<div class="panel-subtitle">Automated walltime/preemption recovery and LLM invocation bounds</div>',
                unsafe_allow_html=True,
            )
            walltime_cnt = _as_int(row.get("walltime_resubmit_events"))
            llm_ratio = _as_float(row.get("llm_calls_per_failed_run"))
            solvers_active = (
                " · ".join(sorted({str(s) for s in runs_df["solver"].dropna()}))
                if "solver" in runs_df.columns and not runs_df.empty
                else "—"
            )
            esc_count = len(escalation_queue(runs_df))
            st.markdown(
                f"""
                <div class="stat-grid">
                    <div class="stat-box">
                        <div class="stat-box-label">Zero-LLM Preemption / Walltime Continuations</div>
                        <div class="stat-box-value">{walltime_cnt}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-box-label">LLM Triage Calls per Failure (Cap &le; 1.0)</div>
                        <div class="stat-box-value">{llm_ratio:.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-box-label">Active Solvers in Campaign</div>
                        <div class="stat-box-value">{html.escape(solvers_active)}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-box-label">Escalated to Physicist (needs_human/)</div>
                        <div class="stat-box-value">{esc_count}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Campaign Run Ledger")
        st.markdown(
            '<div class="panel-subtitle">Per-run execution status, equilibrium geometry, failure mode, and recovery notes (slide horizontally on narrow screens)</div>',
            unsafe_allow_html=True,
        )
        _render_runs_html_table(runs_df)

    with tab_ops:
        col_c, col_d = st.columns(2)
        with col_c:
            st.subheader("Failure Taxonomy & Recovery Rate")
            st.markdown(
                '<div class="panel-subtitle">Deterministic failure modes classified prior to OKF steering</div>',
                unsafe_allow_html=True,
            )
            if _has_columns(tax_df, "failure_type", "affected_runs_count", "solver"):
                fig_tax = px.bar(
                    tax_df,
                    x="failure_type",
                    y="affected_runs_count",
                    color="solver",
                    barmode="group",
                    color_discrete_map={
                        "GX": PALETTE["google_blue"],
                        "GENE": PALETTE["ut_burnt_orange"],
                    },
                    labels={
                        "failure_type": "Failure Mode",
                        "affected_runs_count": "Affected Runs",
                        "solver": "Solver",
                    },
                )
                fig_tax.update_layout(**plot_layout(is_dark=is_dark))
                st.plotly_chart(fig_tax, width="stretch", config={"displayModeBar": False})
        with col_d:
            st.subheader("OKF Runbook Effectiveness")
            st.markdown(
                '<div class="panel-subtitle">Recovery rate by Ontology Knowledge File (OKF) runbook concept</div>',
                unsafe_allow_html=True,
            )
            st.dataframe(rb_df, width="stretch", hide_index=True)

    with tab_audit:
        st.subheader(
            "Numerical Grid & Transport Flux Coverage (`ToBlick/mrx` Equilibria + Benchmark Geometries)"
        )
        st.markdown(
            '<div class="panel-subtitle">Stationary electrostatic heat flux (Q_es) vs. radial grid resolution (nx) across converged and escalated runs</div>',
            unsafe_allow_html=True,
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
                labels={
                    "nx": "Radial Grid Resolution (nx)",
                    "heat_flux_es": "Electrostatic Heat Flux (Q_es)",
                    "coverage_status": "Outcome",
                    "solver": "Solver",
                },
            )
            fig_cov.update_traces(marker={"size": 13, "line": {"width": 1.5, "color": "#09090b"}})
            fig_cov.update_layout(**plot_layout(is_dark=is_dark))
            st.plotly_chart(fig_cov, width="stretch", config={"displayModeBar": False})

        esc_df = escalation_queue(runs_df)
        st.subheader(f"Physicist Escalation Queue (`needs_human/` — {len(esc_df)} runs)")
        st.dataframe(esc_df, width="stretch", hide_index=True)

    with head_chat:
        _render_telemetry_chat(
            client=client,
            campaign_id=str(campaign),
            kpis_df=kpis_df,
            runs_df=runs_df,
            tax_df=tax_df,
            rb_df=rb_df,
            cov_df=cov_df,
            all_kpis_df=all_kpis_df,
            all_runs_df=all_runs_df,
            all_tax_df=all_tax_df,
            all_rb_df=all_rb_df,
            all_cov_df=all_cov_df,
        )


if __name__ == "__main__":
    main()
