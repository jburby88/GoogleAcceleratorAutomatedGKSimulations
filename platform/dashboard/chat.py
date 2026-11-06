"""Gemini Data Analytics conversational assistant for KR2 campaign telemetry."""

from __future__ import annotations

import json
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
    "Distinguish clearly between observed telemetry metrics and modelled compute/human savings.\n\n"
    "CRITICAL FORMATTING RULES:\n"
    "1. NEVER wrap every single value, status, solver, or number in backticks! Use bold (e.g., **w7x_ke_s064_001**) or clean Markdown tables.\n"
    "2. NEVER use inline LaTeX ($Q_{{\\text{{es}}}}$). Use clean Unicode instead (e.g., Q_es, →, ≤).\n"
    "3. NEVER put spaces inside *italics* that break Markdown parsing.\n"
    "4. Provide answers in concise executive bullets or clean Markdown tables."
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
        "Summarize KPIs & recovery rate",
        "Which runs failed & how did OKF recover them?",
        "Compare TACC Slurm vs GCP Cloud Batch",
        "Why were runs escalated to needs_human/?",
    ]


def _query_vertex_gemini(
    *,
    project_id: str,
    messages: list[dict[str, Any]],
    campaign_id: str,
    kpis_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    cov_df: pd.DataFrame,
) -> ChatTurnResult | None:
    """Stream a multi-turn chat using Vertex AI Gemini grounded on canonical views."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        LOGGER.warning("google-genai not installed, falling back.")
        return None

    try:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or "global"
        primary_model = os.environ.get("AURORA_CHAT_MODEL_ID", "").strip() or "gemini-3.8-flash"
        fallback_model = "gemini-4-argon"
        client = genai.Client(vertexai=True, project=project_id, location=location)

        context_parts = [
            f"Active Campaign: {campaign_id}\n",
            "Here is the canonical telemetry view data:\n",
            "v_kpis_campaign:\n" + kpis_df.to_csv(index=False),
            "\nv_run_status:\n" + runs_df.to_csv(index=False),
            "\nv_failure_taxonomy:\n" + tax_df.to_csv(index=False),
            "\nv_runbook_effectiveness:\n" + rb_df.to_csv(index=False),
            "\nv_param_coverage:\n" + cov_df.to_csv(index=False),
        ]

        full_system_instruction = SYSTEM_INSTRUCTION + "\n\n" + "".join(context_parts)

        api_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            api_messages.append(
                types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=content)],
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=full_system_instruction,
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type="OBJECT",
                properties={
                    "answer": types.Schema(
                        type="STRING",
                        description="The formatted markdown answer. NO backtick spam, NO latex.",
                    ),
                    "thoughts": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                    "suggestions": types.Schema(type="ARRAY", items=types.Schema(type="STRING")),
                },
                required=["answer", "thoughts", "suggestions"],
            ),
        )

        used_model = primary_model
        try:
            response = client.models.generate_content(
                model=primary_model,
                contents=api_messages,
                config=config,
            )
        except Exception as e:
            LOGGER.warning("%s failed (%s); falling back to %s.", primary_model, e, fallback_model)
            used_model = fallback_model
            response = client.models.generate_content(
                model=fallback_model,
                contents=api_messages,
                config=config,
            )

        if not response.text:
            return None

        data = json.loads(response.text)

        return ChatTurnResult(
            answer=data.get("answer", ""),
            thoughts=data.get("thoughts", [])[:2],
            suggestions=data.get("suggestions", default_suggestions(campaign_id))[:3],
            engine=f"Vertex AI · {used_model}",
        )
    except Exception as exc:
        LOGGER.info("Vertex AI Gemini fallback triggered: %s", exc)
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
    messages: list[dict[str, Any]] = None,
) -> ChatTurnResult:
    """Synthesize a deterministic, citation-backed answer from the canonical telemetry views."""
    q = question.lower().strip()
    thoughts = [
        f"Scoped telemetry context to campaign_id = {campaign_id}.",
        "Inspected canonical views.",
    ]

    messages_context = ""
    if messages and len(messages) > 1:
        prev_asst = [m for m in messages[:-1] if m.get("role") == "assistant"]
        if prev_asst:
            messages_context = str(prev_asst[-1].get("content", "")).lower()

    q_context = q + " " + messages_context

    if kpis_df.empty:
        return ChatTurnResult(
            answer=f"No telemetry rows are available yet for campaign **{campaign_id}**.",
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

    if any(
        k in q_context
        for k in ("escalat", "needs_human", "human", "div_005", "crash", "happen", "why")
    ):
        if any(k in q for k in ("escalat", "needs_human", "human", "div_005", "crash")) or (
            "escalat" in messages_context
        ):
            esc = (
                runs_df[runs_df["current_status"] == "escalated"]
                if "current_status" in runs_df.columns
                else pd.DataFrame()
            )
            if esc.empty:
                answer = f"Campaign **{campaign_id}** currently has **0 escalated runs** in needs_human/."
            else:
                bullets = []
                for _, r in esc.iterrows():
                    bullets.append(
                        f"- **{r.get('run_id')}** ({r.get('solver')} on {r.get('backend')}, "
                        f"geometry {r.get('geometry_id')}): failure mode **{r.get('last_failure_type') or 'UNKNOWN'}** "
                        f"— reason: *{r.get('escalation_reason') or 'Physics review required'}* "
                        f"({r.get('tacc_folder_path') or 'queue/needs_human/'})"
                    )
                answer = (
                    f"### Physicist Escalation Queue ({campaign_id})\n\n"
                    f"Found **{len(esc)} run(s)** escalated to needs_human/:\n\n"
                    + "\n".join(bullets)
                    + "\n\nPer the OKF safety guardrails (runbook-divergence), macro-scale physical "
                    "instabilities or unrecoverable crashes bypass blind retries and generate an ISSUE.md "
                    "handoff for physicist inspection."
                )
            suggestions = [
                "Which runs failed & how did OKF recover them?",
                "Compare TACC Slurm vs GCP Cloud Batch",
                "Summarize KPIs & recovery rate",
            ]
            return ChatTurnResult(
                answer=answer,
                thoughts=thoughts,
                suggestions=suggestions,
                engine="Canonical View Telemetry Engine",
            )

    if any(k in q for k in ("backend", "tacc", "gcp", "slurm", "batch", "compare", "solver")):
        lines = [f"### Dual-Backend & Solver Breakdown ({campaign_id})\n"]
        lines.append("| Backend | Solver | Status | Runs |")
        lines.append("|---|---|---|---|")
        if not runs_df.empty and {"backend", "solver", "current_status"}.issubset(runs_df.columns):
            grouped = (
                runs_df.groupby(["backend", "solver", "current_status"], dropna=False)
                .size()
                .reset_index(name="runs")
            )
            for _, row in grouped.iterrows():
                lines.append(
                    f"| **{row['backend']}** | **{row['solver']}** | {row['current_status']} | **{int(row['runs'])}** |"
                )
        lines.append("\n")
        lines.append(
            f"- **Zero-LLM Checkpoint / Preemption Continuations:** {walltime_cnt}\n"
            f"- **LLM Triage Calls per Failed Run:** {llm_ratio:.2f} (cap ≤ 1.0)"
        )
        suggestions = [
            "Why were runs escalated to needs_human/?",
            "Which runs failed & how did OKF recover them?",
            "What are the heat flux and grid coverage metrics?",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions[:3],
            engine="Canonical View Telemetry Engine",
        )

    if any(k in q for k in ("flux", "grid", "nx", "coverage", "geometry", "w7x", "mrx")):
        lines = [f"### Numerical Grid & Transport Flux Coverage ({campaign_id})\n"]
        lines.append("| Run ID | Solver | Geometry | Status | nx | Q_es | Retries |")
        lines.append("|---|---|---|---|---|---|---|")
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
                    f"| **{r.get('run_id')}** | {r.get('solver')} | {r.get('geometry_id')} | "
                    f"**{r.get('coverage_status')}** | {nx_str} | {flux_str} | {r.get('retry_count')} |"
                )
        else:
            lines.append("No parameter coverage rows recorded yet for this campaign.")
        suggestions = [
            "Summarize KPIs & recovery rate",
            "Which runs failed & how did OKF recover them?",
            "Compare TACC Slurm vs GCP Cloud Batch",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions[:3],
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
            "stall",
        )
    ):
        lines = [f"### Failure Taxonomy & OKF Steering Outcomes ({campaign_id})\n"]
        if not runs_df.empty and "last_failure_type" in runs_df.columns:
            failed_runs = runs_df[runs_df["last_failure_type"].notna()]
            if not failed_runs.empty:
                lines.append("| Run ID | Solver | Backend | Failure | Status |")
                lines.append("|---|---|---|---|---|")
                for _, r in failed_runs.iterrows():
                    lines.append(
                        f"| **{r.get('run_id')}** | {r.get('solver')} | {r.get('backend')} | "
                        f"**{r.get('last_failure_type')}** | {r.get('current_status')} |"
                    )
            else:
                lines.append("- No failure events recorded in v_run_status for this campaign.")
        if not rb_df.empty and "okf_concept_id" in rb_df.columns:
            lines.append("\n**OKF Runbooks Invoked (v_runbook_effectiveness):**")
            for _, r in rb_df.iterrows():
                rec_rate = r.get("recovery_rate")
                rec_str = (
                    f" — recovery rate **{float(rec_rate) * 100:.0f}%**"
                    if rec_rate is not None and not pd.isna(rec_rate)
                    else ""
                )
                lines.append(
                    f"- {r.get('okf_concept_id')} ({r.get('solver')}): "
                    f"{int(float(r.get('directives_count') or 0))} directive(s){rec_str}"
                )
        suggestions = [
            "Why were runs escalated to needs_human/?",
            "What are the heat flux and grid coverage metrics?",
            "Summarize KPIs & recovery rate",
        ]
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=thoughts,
            suggestions=suggestions[:3],
            engine="Canonical View Telemetry Engine",
        )

    run_summaries = []
    if not runs_df.empty and "run_id" in runs_df.columns:
        for _, r in runs_df.head(8).iterrows():
            fail_tag = (
                f" (recovered from {r.get('last_failure_type')})"
                if pd.notna(r.get("last_failure_type")) and r.get("current_status") == "completed"
                else (
                    f" ({r.get('last_failure_type')})"
                    if pd.notna(r.get("last_failure_type"))
                    else ""
                )
            )
            run_summaries.append(
                f"- **{r.get('run_id')}** ({r.get('solver')} · {r.get('backend')}): "
                f"**{r.get('current_status')}**{fail_tag}"
            )

    answer = (
        f"### Campaign Telemetry Summary ({campaign_id})\n\n"
        f"- **Completed Runs (Observed):** {completed} / {target}\n"
        f"- **Hands-Free Rate (Observed):** {hands_free:.1f}%\n"
        f"- **Agent Recovery Rate (Observed):** {recovery:.1f}%\n"
        f"- **Modelled GPU Hours Saved:** {gpu_saved:.1f} h (early termination + checkpoint recovery)\n"
        f"- **Modelled Human Hours Avoided:** {human_saved:.1f} h\n\n"
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
    if data_mode == "bigquery" and project_id:
        gda_result = _query_vertex_gemini(
            project_id=project_id,
            messages=messages,
            campaign_id=campaign_id,
            kpis_df=kpis_df,
            runs_df=runs_df,
            tax_df=tax_df,
            rb_df=rb_df,
            cov_df=cov_df,
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
        messages=messages,
    )
