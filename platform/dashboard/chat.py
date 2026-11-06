"""Agent runner and fallback orchestrator for Ask Aurora (`platform/dashboard`)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from agent import (
    FALLBACK_CHAT_MODEL_ID,
    configure_vertex_env,
    create_telemetry_analyst_agent,
    resolve_chat_model_id,
)
from prompts import STARTER_SUGGESTIONS
from tools import TelemetryToolkit, _filter_campaign

LOGGER = logging.getLogger("aurora_dashboard.chat")
APP_NAME = "aurora_dashboard_chat"


@dataclass
class ChatTurnResult:
    """Structured response from one conversational analytics turn."""

    answer: str
    thoughts: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    engine: str = "gemini-3.8-flash"
    tools_called: list[str] = field(default_factory=list)


def default_suggestions(campaign_id: str) -> list[str]:
    """Return concise starter prompt chips."""
    del campaign_id
    return list(STARTER_SUGGESTIONS)


def _final_response_text(event: Any) -> str | None:
    """Extract final response text from an agent event, ignoring internal thought parts."""
    is_final = getattr(event, "is_final_response", None)
    if not callable(is_final) or not is_final():
        return None
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)]
    return "".join(texts).strip() if texts else None


def _format_history_prompt(messages: list[dict[str, Any]]) -> str:
    """Combine prior conversation turns with the latest user question for agent execution."""
    if not messages:
        return ""
    latest = str(messages[-1].get("content", "")).strip()
    prior = messages[:-1][-6:]
    if not prior:
        return latest
    history_lines = ["<conversation_history>"]
    for msg in prior:
        role = "User" if msg.get("role") == "user" else "Aurora"
        content = str(msg.get("content", "")).strip()
        if content:
            history_lines.append(f"{role}: {content}")
    history_lines.append("</conversation_history>\n")
    history_lines.append(f"Current User Question: {latest}")
    return "\n".join(history_lines)


async def _run_adk_turn_async(
    *,
    project_id: str,
    campaign_id: str,
    messages: list[dict[str, Any]],
    toolkit: TelemetryToolkit,
    model_id: str,
) -> str | None:
    """Execute one `InMemoryRunner` turn with bound telemetry tools."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    configure_vertex_env(project_id)
    agent = create_telemetry_analyst_agent(
        tools=toolkit.as_adk_tools(),
        active_campaign_id=campaign_id,
        model=model_id,
    )
    runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
    session_service = runner.session_service
    user_id = "dashboard_operator"
    session_id = f"chat-{uuid.uuid4().hex}"

    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    prompt_text = _format_history_prompt(messages)
    final_text: str | None = None
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=prompt_text)]),
        ):
            text = _final_response_text(event)
            if text:
                final_text = text
    finally:
        try:
            await session_service.delete_session(
                app_name=APP_NAME, user_id=user_id, session_id=session_id
            )
        except Exception:  # noqa: BLE001
            pass
    return final_text


def _run_adk_agent(
    *,
    project_id: str,
    campaign_id: str,
    messages: list[dict[str, Any]],
    toolkit: TelemetryToolkit,
) -> ChatTurnResult | None:
    """Run the telemetry analyst agent on a dedicated event loop thread."""
    if os.environ.get("AURORA_DISABLE_ADK_CHAT", "").strip() == "1":
        return None
    primary_model = resolve_chat_model_id()
    for model_id in (primary_model, FALLBACK_CHAT_MODEL_ID):
        try:
            coro = asyncio.wait_for(
                _run_adk_turn_async(
                    project_id=project_id,
                    campaign_id=campaign_id,
                    messages=messages,
                    toolkit=toolkit,
                    model_id=model_id,
                ),
                timeout=35.0,
            )
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is None:
                answer_text = asyncio.run(coro)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    answer_text = pool.submit(asyncio.run, coro).result(timeout=36.0)
            if answer_text:
                tools_list = list(toolkit.called_tools)
                return ChatTurnResult(
                    answer=answer_text,
                    thoughts=[f"Checked telemetry tools: {', '.join(tools_list) or 'none'}"],
                    suggestions=default_suggestions(campaign_id)[:3],
                    engine=model_id,
                    tools_called=tools_list,
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Model %s failed (%s); trying next fallback.", model_id, exc)
    return None


def _grounded_telemetry_answer(
    *,
    question: str,
    campaign_id: str,
    toolkit: TelemetryToolkit,
    kpis_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    rb_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    messages: list[dict[str, Any]] | None = None,
) -> ChatTurnResult:
    """Deterministic tool-backed answer when offline or in headless unit tests."""
    del tax_df
    q = question.lower().strip()
    q_words = set(q.replace("!", " ").replace("?", " ").replace(",", " ").split())
    messages_context = ""
    if messages and len(messages) > 1:
        prev_asst = [m for m in messages[:-1] if m.get("role") == "assistant"]
        if prev_asst:
            messages_context = str(prev_asst[-1].get("content", "")).lower()
    q_context = f"{q} {messages_context}"

    # 0. Conversational greetings / "what can you do"
    if q_words.issubset({"hello", "hi", "hey", "greetings", "morning", "afternoon", "aurora"}) or (
        any(g in q for g in ("what can you do", "who are you", "help me"))
    ):
        info = toolkit.list_campaigns()
        camps = ", ".join(f"**{c}**" for c in info["campaigns"])
        return ChatTurnResult(
            answer=(
                f"Hello! I'm **Aurora**, your simulation and telemetry assistant across "
                f"**TACC Vista Slurm** and **GCP Cloud Batch**.\n\n"
                f"You currently have **{info['total_campaigns']} campaign(s)** available "
                f"({camps}), with **{campaign_id}** selected in your view. "
                f"Feel free to ask about campaign KPIs, solver runs, automated recoveries, "
                f"or why specific runs were escalated."
            ),
            thoughts=["Checked available campaigns."],
            suggestions=default_suggestions(campaign_id)[:3],
            engine="telemetry",
            tools_called=list(toolkit.called_tools),
        )

    # 1. Cross-campaign / "how many campaigns" queries
    if any(k in q for k in ("how many campaign", "list campaign", "all campaign", "campaigns")):
        info = toolkit.list_campaigns()
        camp_names = info["campaigns"]
        rows = info["campaign_kpis"]
        lines = [
            f"You have **{info['total_campaigns']} campaign(s)** in the system:\n",
            "| Campaign ID | Completed Runs | Hands-Free Rate | Recovery Rate | GPU Hours Saved |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            cid = r.get("campaign_id")
            comp = int(float(r.get("completed_runs") or 0))
            targ = int(float(r.get("target_runs") or 0))
            hf = float(r.get("hands_free_rate") or 0.0) * 100.0
            rec = (
                "—"
                if r.get("agent_recovery_rate") is None
                else f"{float(r.get('agent_recovery_rate') or 0.0) * 100.0:.1f}%"
            )
            gpu = float(r.get("estimated_gpu_hours_saved") or 0.0)
            active_tag = " *(active)*" if cid == campaign_id else ""
            lines.append(
                f"| **{cid}**{active_tag} | {comp} / {targ} | {hf:.1f}% | {rec} | {gpu:.1f} h |"
            )
        if not rows:
            lines.append(f"| **{', '.join(camp_names)}** | — | — | — | — |")
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=["Called `list_campaigns` across all production rows."],
            suggestions=default_suggestions(campaign_id)[:3],
            engine="telemetry",
            tools_called=list(toolkit.called_tools),
        )

    if kpis_df.empty:
        return ChatTurnResult(
            answer=f"No telemetry rows are available yet for campaign **{campaign_id}**.",
            thoughts=[],
            suggestions=default_suggestions(campaign_id)[:3],
            engine="telemetry",
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

    # 2. Escalation queries
    if any(k in q_context for k in ("escalat", "needs_human", "human", "div_005", "crash", "why")):
        if any(k in q for k in ("escalat", "needs_human", "human", "div_005", "crash")) or (
            "escalat" in messages_context
        ):
            toolkit.query_campaign_runs(campaign_id=campaign_id, status_filter="escalated")
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
                        f"— *{str(r.get('escalation_reason') or 'Physics review required').strip()}*"
                    )
                answer = (
                    f"**Physicist Escalation Queue ({campaign_id})**\n\n"
                    f"Found **{len(esc)} run(s)** escalated to needs_human/:\n\n"
                    + "\n".join(bullets)
                    + "\n\nPer the OKF safety guardrails (runbook-divergence), macro-scale physical "
                    "instabilities or unrecoverable crashes bypass blind retries and generate an ISSUE.md "
                    "handoff for physicist inspection."
                )
            return ChatTurnResult(
                answer=answer,
                thoughts=["Called `query_campaign_runs(status_filter='escalated')`."],
                suggestions=[
                    "Which runs failed & how did OKF recover them?",
                    "Compare TACC Slurm vs GCP Cloud Batch",
                    "Summarize KPIs & recovery rate",
                ],
                engine="telemetry",
                tools_called=list(toolkit.called_tools),
            )

    # 3. Backend / Solver comparison
    if any(k in q for k in ("backend", "tacc", "gcp", "slurm", "batch", "compare", "solver")):
        toolkit.query_campaign_runs(campaign_id=campaign_id)
        lines = [
            f"**Dual-Backend & Solver Breakdown ({campaign_id})**\n",
            "| Backend | Solver | Status | Runs |",
            "|---|---|---|---|",
        ]
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
        lines.append(
            f"\n- **Zero-LLM Checkpoint Continuations:** {walltime_cnt}\n"
            f"- **LLM Triage Calls per Failed Run:** {llm_ratio:.2f} (cap ≤ 1.0)"
        )
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=["Called `query_campaign_runs`."],
            suggestions=[
                "Why were runs escalated to needs_human/?",
                "Which runs failed & how did OKF recover them?",
                "How many campaigns do I have?",
            ],
            engine="telemetry",
            tools_called=list(toolkit.called_tools),
        )

    # 4. Flux / Grid coverage
    if any(k in q for k in ("flux", "grid", "nx", "coverage", "geometry", "w7x", "mrx")):
        toolkit.get_parameter_and_flux_coverage(campaign_id=campaign_id)
        lines = [
            f"**Numerical Grid & Transport Flux Coverage ({campaign_id})**\n",
            "| Run ID | Solver | Geometry | Status | nx | Q_es | Retries |",
            "|---|---|---|---|---|---|---|",
        ]
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
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=["Called `get_parameter_and_flux_coverage`."],
            suggestions=default_suggestions(campaign_id)[:3],
            engine="telemetry",
            tools_called=list(toolkit.called_tools),
        )

    # 5. Failures & OKF runbooks
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
        toolkit.get_failure_and_runbook_outcomes(campaign_id=campaign_id)
        lines = [f"**Failure Taxonomy & OKF Steering Outcomes ({campaign_id})**\n"]
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
        if not rb_df.empty and "okf_concept_id" in rb_df.columns:
            lines.append("\n**OKF Runbooks Invoked:**")
            for _, r in rb_df.iterrows():
                rec_rate = r.get("recovery_rate")
                rec_str = (
                    f" — recovery rate **{float(rec_rate) * 100:.0f}%**"
                    if rec_rate is not None and not pd.isna(rec_rate)
                    else ""
                )
                lines.append(
                    f"- **{r.get('okf_concept_id')}** ({r.get('solver')}): "
                    f"{int(float(r.get('directives_count') or 0))} directive(s){rec_str}"
                )
        return ChatTurnResult(
            answer="\n".join(lines),
            thoughts=["Called `get_failure_and_runbook_outcomes`."],
            suggestions=[
                "Why were runs escalated to needs_human/?",
                "Compare TACC Slurm vs GCP Cloud Batch",
                "Summarize KPIs & recovery rate",
            ],
            engine="telemetry",
            tools_called=list(toolkit.called_tools),
        )

    # 6. Default Campaign Summary
    toolkit.get_campaign_kpis(campaign_id=campaign_id)
    answer = (
        f"**Campaign Telemetry Summary ({campaign_id})**\n\n"
        f"- **Completed Runs (Observed):** {completed} / {target}\n"
        f"- **Hands-Free Rate (Observed):** {hands_free:.1f}%\n"
        f"- **Agent Recovery Rate (Observed):** {recovery:.1f}%\n"
        f"- **Modelled GPU Hours Saved:** {gpu_saved:.1f} h\n"
        f"- **Modelled Human Hours Avoided:** {human_saved:.1f} h"
    )
    return ChatTurnResult(
        answer=answer,
        thoughts=["Called `get_campaign_kpis`."],
        suggestions=default_suggestions(campaign_id)[1:4],
        engine="telemetry",
        tools_called=list(toolkit.called_tools),
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
    client: Any = None,
    all_kpis_df: pd.DataFrame | None = None,
    all_runs_df: pd.DataFrame | None = None,
    all_tax_df: pd.DataFrame | None = None,
    all_rb_df: pd.DataFrame | None = None,
    all_cov_df: pd.DataFrame | None = None,
    on_status: Callable[[str], None] | None = None,
) -> ChatTurnResult:
    """Answer a telemetry question using the Gemini agent + read-only tools with view fallback."""
    del dataset_id
    toolkit = TelemetryToolkit(
        client=client,
        active_campaign_id=campaign_id,
        all_kpis_df=all_kpis_df if all_kpis_df is not None else kpis_df,
        all_runs_df=all_runs_df if all_runs_df is not None else runs_df,
        all_tax_df=all_tax_df if all_tax_df is not None else tax_df,
        all_rb_df=all_rb_df if all_rb_df is not None else rb_df,
        all_cov_df=all_cov_df if all_cov_df is not None else cov_df,
        on_status=on_status,
    )

    if data_mode == "bigquery" and project_id:
        adk_result = _run_adk_agent(
            project_id=project_id,
            campaign_id=campaign_id,
            messages=messages,
            toolkit=toolkit,
        )
        if adk_result is not None:
            return adk_result

    scoped_kpis = _filter_campaign(kpis_df, campaign_id) if not kpis_df.empty else kpis_df
    scoped_runs = _filter_campaign(runs_df, campaign_id) if not runs_df.empty else runs_df
    scoped_tax = _filter_campaign(tax_df, campaign_id) if not tax_df.empty else tax_df
    scoped_rb = _filter_campaign(rb_df, campaign_id) if not rb_df.empty else rb_df
    scoped_cov = _filter_campaign(cov_df, campaign_id) if not cov_df.empty else cov_df

    return _grounded_telemetry_answer(
        question=question,
        campaign_id=campaign_id,
        toolkit=toolkit,
        kpis_df=scoped_kpis,
        runs_df=scoped_runs,
        tax_df=scoped_tax,
        rb_df=scoped_rb,
        cov_df=scoped_cov,
        messages=messages,
    )
