"""Gemini Data Analytics conversational assistant for KR2 campaign telemetry."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("aurora_dashboard.chat")

CHAT_VIEWS = (
    "v_kpis_campaign",
    "v_run_status",
    "v_failure_taxonomy",
    "v_runbook_effectiveness",
    "v_param_coverage",
)

SYSTEM_INSTRUCTION = (
    "You are the UT Austin × Google.org KR2 Gyrokinetic Simulation Telemetry Analyst "
    "(`aurora-steer` / `sim-babysitter`). Answer questions concisely using exact numbers, "
    "run_ids, solvers (GX, GENE), execution backends (tacc_slurm, gcp_batch), failure modes "
    "(TIMESTEP_COLLAPSE, SPECTRAL_PILEUP, DIVERGENCE, STALL, CRASH), and OKF runbook concepts "
    "from the production telemetry views (`environment = 'prod'`, `is_synthetic = FALSE`). "
    "Distinguish clearly between observed telemetry metrics and modelled compute/human savings."
)


@dataclass
class ChatTurnResult:
    """Structured response from one conversational analytics turn."""

    answer: str
    thoughts: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    engine: str = "gemini-data-analytics"


def default_suggestions(campaign_id: str) -> list[str]:
    """Return starter prompt chips scoped to the active campaign."""
    return [
        f"Summarize KPIs and recovery rate for {campaign_id}",
        f"Which runs in {campaign_id} had failures and what were the outcomes?",
        f"Compare TACC Slurm vs GCP Cloud Batch runs in {campaign_id}",
        f"Why were runs escalated to needs_human/ in {campaign_id}?",
    ]


def _query_gemini_data_analytics(
    *,
    project_id: str,
    dataset_id: str,
    messages: list[dict[str, Any]],
    campaign_id: str,
) -> ChatTurnResult | None:
    """Stream a stateless chat turn against BigQuery views via Gemini Data Analytics API."""
    if os.environ.get("AURORA_DISABLE_GDA_API", "").strip() == "1":
        return None
    try:
        from google.cloud import geminidataanalytics
    except ImportError:
        return None

    try:
        client = geminidataanalytics.DataChatServiceClient()
        bq_tables = [
            geminidataanalytics.BigQueryTableReference(
                project_id=project_id,
                dataset_id=dataset_id,
                table_id=view_name,
            )
            for view_name in CHAT_VIEWS
        ]
        datasource_references = geminidataanalytics.DatasourceReferences(
            bq=geminidataanalytics.BigQueryTableReferences(table_references=bq_tables)
        )
        inline_context = geminidataanalytics.Context(
            system_instruction=(
                f"{SYSTEM_INSTRUCTION} The user currently has campaign '{campaign_id}' selected."
            ),
            datasource_references=datasource_references,
            options=geminidataanalytics.ConversationOptions(
                analysis=geminidataanalytics.AnalysisOptions(
                    python=geminidataanalytics.Python(enabled=False)
                )
            ),
        )

        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if role == "user":
                api_messages.append(
                    geminidataanalytics.Message(
                        user_message=geminidataanalytics.UserMessage(text=content)
                    )
                )
            else:
                api_messages.append(
                    geminidataanalytics.Message(
                        system_message=geminidataanalytics.SystemMessage(
                            text=geminidataanalytics.TextMessage(parts=[content])
                        )
                    )
                )

        request = geminidataanalytics.ChatRequest(
            parent=f"projects/{project_id}/locations/global",
            messages=api_messages,
            inline_context=inline_context,
        )

        thoughts: list[str] = []
        answer_parts: list[str] = []
        suggestions: list[str] = []

        stream = client.chat(request=request, timeout=25.0)
        for packet in stream:
            if not hasattr(packet, "system_message"):
                continue
            sys_msg = packet.system_message
            if hasattr(sys_msg, "text") and getattr(sys_msg.text, "parts", None):
                part_text = "".join(sys_msg.text.parts).strip()
                if not part_text:
                    continue
                text_type = getattr(sys_msg.text, "text_type", None)
                type_name = getattr(text_type, "name", str(text_type or ""))
                if "THOUGHT" in type_name:
                    thoughts.append(part_text)
                elif "SUGGESTION" in type_name:
                    suggestions.append(part_text)
                else:
                    answer_parts.append(part_text)

        final_answer = "\n\n".join(p for p in answer_parts if p).strip()
        if not final_answer:
            return None
        if not suggestions:
            suggestions = default_suggestions(campaign_id)[:3]
        return ChatTurnResult(
            answer=final_answer,
            thoughts=thoughts,
            suggestions=suggestions[:3],
            engine="Gemini Data Analytics API (`BigQuery`)",
        )
    except Exception as exc:
        LOGGER.info("Gemini Data Analytics API fallback triggered: %s", exc)
        return None


def _grounded_telemetry_answer(
    *,
    question: str,
    campaign_id: str,
    kpis_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    cov_df: pd.DataFrame,
) -> ChatTurnResult:
    """Synthesize a deterministic, citation-backed answer from the canonical telemetry views."""
    q = question.lower().strip()
    thoughts = [
        f"Scoped telemetry context to `campaign_id = '{campaign_id}'` (`environment = 'prod'`, `is_synthetic = FALSE`).",
        f"Inspected canonical views: `v_kpis_campaign` ({len(kpis_df)} row), `v_run_status` ({len(runs_df)} runs), "
        f"`v_failure_taxonomy` ({len(tax_df)} failure buckets), `v_runbook_effectiveness` ({len(rb_df)} OKF rules), "
        f"`v_param_coverage` ({len(cov_df)} grid/flux points).",
    ]

    if kpis_df.empty:
        return ChatTurnResult(
            answer=f"No telemetry rows are available yet for campaign `{campaign_id}`.",
            thoughts=thoughts,
            suggestions=default_suggestions(campaign_id)[:3],
            engine="Canonical View Telemetry Engine",
        )

    kpi = kpis_df.iloc[0]
    completed = int(float(kpi.get("completed_runs") or 0))
    target = int(float(kpi.get("target_runs") or 0))
    hands_free = float(kpi.get("hands_free_rate") or 0.0) * 100.0
    recovery = (
        0.0
        if pd.isna(kpi.get("agent_recovery_rate"))
        else float(kpi.get("agent_recovery_rate") or 0.0) * 100.0
    )
    gpu_saved = float(kpi.get("estimated_gpu_hours_saved") or 0.0)
    human_saved = float(kpi.get("estimated_human_hours_avoided") or 0.0)
    walltime_cnt = int(float(kpi.get("walltime_resubmit_events") or 0))
    llm_ratio = (
        0.0
        if pd.isna(kpi.get("llm_calls_per_failed_run"))
        else float(kpi.get("llm_calls_per_failed_run") or 0.0)
    )

    # Check specific question intents
    if any(k in q for k in ("escalat", "needs_human", "human", "div_005", "crash")):
        esc = (
            runs_df[runs_df["current_status"] == "escalated"]
            if "current_status" in runs_df.columns
            else pd.DataFrame()
        )
        if esc.empty:
            answer = (
                f"Campaign `{campaign_id}` currently has **0 escalated runs** in `needs_human/`."
            )
        else:
            bullets = []
            for _, r in esc.iterrows():
                bullets.append(
                    f"- **`{r.get('run_id')}`** (`{r.get('solver')}` on `{r.get('backend')}`, "
                    f"geometry `{r.get('geometry_id')}`): failure mode **`{r.get('last_failure_type') or 'UNKNOWN'}`** "
                    f"— reason: *{r.get('escalation_reason') or 'Physics review required'}* "
                    f"(`{r.get('tacc_folder_path') or 'queue/needs_human/'}`)"
                )
            answer = (
                f"### Physicist Escalation Queue (`{campaign_id}`)\n\n"
                f"Found **{len(esc)} run(s)** escalated to `needs_human/`:\n\n"
                + "\n".join(bullets)
                + "\n\nPer the OKF safety guardrails (`runbook-divergence`), macro-scale physical "
                "instabilities or unrecoverable crashes bypass blind retries and generate an `ISSUE.md` "
                "handoff for physicist inspection."
            )
        suggestions = [
            f"Which runs in {campaign_id} had failures and what were the outcomes?",
            f"Compare TACC Slurm vs GCP Cloud Batch runs in {campaign_id}",
            f"Summarize KPIs and recovery rate for {campaign_id}",
        ]
        return ChatTurnResult(
            answer=answer,
            thoughts=thoughts,
            suggestions=suggestions,
            engine="Canonical View Telemetry Engine",
        )

    if any(k in q for k in ("backend", "tacc", "gcp", "slurm", "batch", "compare", "solver")):
        lines = [f"### Dual-Backend & Solver Breakdown (`{campaign_id}`)\n"]
        if not runs_df.empty and {"backend", "solver", "current_status"}.issubset(runs_df.columns):
            grouped = (
                runs_df.groupby(["backend", "solver", "current_status"], dropna=False)
                .size()
                .reset_index(name="runs")
            )
            for _, row in grouped.iterrows():
                lines.append(
                    f"- **`{row['backend']}`** · **`{row['solver']}`** · `{row['current_status']}`: **{int(row['runs'])} run(s)**"
                )
        lines.append(
            f"\n- **Zero-LLM Checkpoint / Preemption Continuations:** `{walltime_cnt}`\n"
            f"- **LLM Triage Calls per Failed Run:** `{llm_ratio:.2f}` (cap $\\le 1.0$)"
        )
        suggestions = [
            f"Why were runs escalated to needs_human/ in {campaign_id}?",
            f"Which runs in {campaign_id} had failures and what were the outcomes?",
            f"What are the heat flux and grid coverage metrics in {campaign_id}?",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions,
            engine="Canonical View Telemetry Engine",
        )

    if any(k in q for k in ("flux", "grid", "nx", "coverage", "geometry", "w7x", "mrx")):
        lines = [f"### Numerical Grid & Transport Flux Coverage (`{campaign_id}`)\n"]
        if not cov_df.empty:
            for _, r in cov_df.iterrows():
                flux_val = r.get("heat_flux_es")
                flux_str = (
                    f"{float(flux_val):.3f}"
                    if flux_val is not None and not pd.isna(flux_val)
                    else "—"
                )
                nx_val = r.get("nx")
                nx_str = (
                    f"{int(float(nx_val))}" if nx_val is not None and not pd.isna(nx_val) else "—"
                )
                lines.append(
                    f"- **`{r.get('run_id')}`** (`{r.get('solver')}`, `{r.get('geometry_id')}`): "
                    f"status **`{r.get('coverage_status')}`**, radial grid `nx = {nx_str}`, "
                    f"electrostatic heat flux $Q_{{\\text{{es}}}} = {flux_str}$, retries `{r.get('retry_count')}`"
                )
        else:
            lines.append("No parameter coverage rows recorded yet for this campaign.")
        suggestions = [
            f"Summarize KPIs and recovery rate for {campaign_id}",
            f"Which runs in {campaign_id} had failures and what were the outcomes?",
            f"Compare TACC Slurm vs GCP Cloud Batch runs in {campaign_id}",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions,
            engine="Canonical View Telemetry Engine",
        )

    if any(
        k in q
        for k in (
            "fail",
            "recover",
            "okf",
            "runbook",
            "timestep",
            "pileup",
            "divergence",
            "directive",
        )
    ):
        lines = [f"### Failure Taxonomy & OKF Steering Outcomes (`{campaign_id}`)\n"]
        if not runs_df.empty and "last_failure_type" in runs_df.columns:
            failed_runs = runs_df[runs_df["last_failure_type"].notna()]
            if not failed_runs.empty:
                for _, r in failed_runs.iterrows():
                    lines.append(
                        f"- **`{r.get('run_id')}`** (`{r.get('solver')}` on `{r.get('backend')}`): "
                        f"encountered **`{r.get('last_failure_type')}`** $\\rightarrow$ current status **`{r.get('current_status')}`**"
                    )
            else:
                lines.append("- No failure events recorded in `v_run_status` for this campaign.")
        if not rb_df.empty and "okf_concept_id" in rb_df.columns:
            lines.append("\n**OKF Runbooks Invoked (`v_runbook_effectiveness`):**")
            for _, r in rb_df.iterrows():
                rec_rate = r.get("recovery_rate")
                rec_str = (
                    f" — recovery rate **{float(rec_rate) * 100:.0f}%**"
                    if rec_rate is not None and not pd.isna(rec_rate)
                    else ""
                )
                lines.append(
                    f"- `{r.get('okf_concept_id')}` (`{r.get('solver')}`): "
                    f"`{int(float(r.get('directives_count') or 0))}` directive(s){rec_str}"
                )
        suggestions = [
            f"Why were runs escalated to needs_human/ in {campaign_id}?",
            f"What are the heat flux and grid coverage metrics in {campaign_id}?",
            f"Summarize KPIs and recovery rate for {campaign_id}",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions,
            engine="Canonical View Telemetry Engine",
        )

    # Default comprehensive campaign summary
    run_summaries = []
    if not runs_df.empty and "run_id" in runs_df.columns:
        for _, r in runs_df.head(8).iterrows():
            fail_tag = (
                f" (recovered from `{r.get('last_failure_type')}`)"
                if pd.notna(r.get("last_failure_type")) and r.get("current_status") == "completed"
                else (
                    f" (`{r.get('last_failure_type')}`)"
                    if pd.notna(r.get("last_failure_type"))
                    else ""
                )
            )
            run_summaries.append(
                f"- **`{r.get('run_id')}`** (`{r.get('solver')}` · `{r.get('backend')}`): "
                f"**`{r.get('current_status')}`**{fail_tag}"
            )

    answer = (
        f"### Campaign Telemetry Summary (`{campaign_id}`)\n\n"
        f"- **Completed Runs (Observed):** `{completed} / {target}`\n"
        f"- **Hands-Free Rate (Observed):** `{hands_free:.1f}%`\n"
        f"- **Agent Recovery Rate (Observed):** `{recovery:.1f}%`\n"
        f"- **Modelled GPU Hours Saved:** `{gpu_saved:.1f} h` (early termination + checkpoint recovery)\n"
        f"- **Modelled Human Hours Avoided:** `{human_saved:.1f} h`\n\n"
        + ("**Run Ledger:**\n" + "\n".join(run_summaries) if run_summaries else "")
    )
    return ChatTurnResult(
        answer=answer,
        thoughts=thoughts,
        suggestions=default_suggestions(campaign_id)[1:4],
        engine="Canonical View Telemetry Engine",
    )


def ask_telemetry(
    *,
    question: str,
    campaign_id: str,
    messages: list[dict[str, Any]],
    project_id: str | None,
    dataset_id: str | None,
    data_mode: str,
    kpis_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    cov_df: pd.DataFrame,
) -> ChatTurnResult:
    """Answer a natural-language telemetry question using Gemini Data Analytics with view fallback."""
    if data_mode == "bigquery" and project_id and dataset_id:
        gda_result = _query_gemini_data_analytics(
            project_id=project_id,
            dataset_id=dataset_id,
            messages=messages,
            campaign_id=campaign_id,
        )
        if gda_result is not None:
            return gda_result

    return _grounded_telemetry_answer(
        question=question,
        campaign_id=campaign_id,
        kpis_df=kpis_df,
        runs_df=runs_df,
        tax_df=tax_df,
        rb_df=rb_df,
        cov_df=cov_df,
    )
