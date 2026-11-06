"""Google ADK 2.0 runner for Cloud Run `/triage` requests.

Because Cloud Run scales to zero between simulation failures, `SteeringEngine`
executes a self-contained single-turn ADK session per `/triage` request:
- Persistent simulation state (`retry_count`, `current_parameters`, patch history)
  is supplied in the `<2 KB` `DiagnosticSummary` request payload from TACC or GCS.
- Every resulting `ActionDirective` is recorded (BigQuery + structured logs) by
  `app.main` before the HTTP response returns.

`diagnose()` is a coroutine that runs on the server's event loop: it uses ADK's
async API (`await session_service.create_session(...)`, `runner.run_async(...)`)
and never calls `asyncio.run()` or the sync `Runner.run()` wrapper, which would
fail or block inside FastAPI. Deadlines, transient-error retries and the
fail-closed mapping of every error to `escalate_to_human` live in `app.main`.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.config import configure_vertex_ai_env, resolve_model_id
from app.okf_loader import LoadedSkill
from app.prompts import build_triage_user_prompt
from app.schemas import ActionDirective, DiagnosticSummary

APP_NAME = "aurora_steer"
_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


class MalformedModelOutputError(ValueError):
    """The model returned no final text, or text that is not a valid ActionDirective."""


def parse_model_output(raw_text: str | None) -> ActionDirective:
    """Parse the model's final text into an `ActionDirective` (fail closed)."""
    if not raw_text or not raw_text.strip():
        raise MalformedModelOutputError("model returned no final response text")
    match = _CODE_FENCE_RE.match(raw_text)
    text = match.group(1) if match else raw_text
    try:
        return ActionDirective.model_validate_json(text)
    except ValueError as exc:
        raise MalformedModelOutputError(
            f"model output is not a valid ActionDirective: {exc}"
        ) from exc


def _final_response_text(event: Any) -> str | None:
    """Text of a final-response event, skipping thought parts; None for other events."""
    is_final = getattr(event, "is_final_response", None)
    if not callable(is_final) or not is_final():
        return None
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False)]
    return "".join(texts) if texts else None


class SteeringEngine:
    """Executes single-turn simulation triage with Google ADK 2.0's `InMemoryRunner`."""

    def __init__(self, model: Any = None) -> None:
        # `model`: Vertex AI model id (default: validated `MODEL_ID`) or an ADK `BaseLlm`.
        self._model = model
        self._runner: Any = None

    def _get_runner(self) -> Any:
        # Lazy: requires google-adk, and is only built on the first real LLM call.
        if self._runner is None:
            from google.adk.runners import InMemoryRunner

            from app.agent import create_aurora_steer_agent

            configure_vertex_ai_env()
            agent = create_aurora_steer_agent(self._model or resolve_model_id())
            self._runner = InMemoryRunner(agent=agent, app_name=APP_NAME)
        return self._runner

    async def diagnose(
        self,
        diagnostic: DiagnosticSummary,
        skill: LoadedSkill,
        *,
        allowed_actions: frozenset[str],
    ) -> ActionDirective:
        """Run the ADK `aurora_steer` agent on a `<2 KB` simulation failure summary.

        Args:
            diagnostic: Validated failure telemetry from TACC Slurm or GCP Cloud Batch.
            skill: Dynamically loaded Open Knowledge Framework (OKF) skill containing
                parameter bounds and failure-mode runbooks.
            allowed_actions: Actions the runbooks permit for this failure type
                (shown to the model; enforced again by `app.guardrails`).

        Returns:
            Parsed `ActionDirective` returned by the ADK agent (prior to server-side
            guardrail verification in `app.guardrails.validate_directive`).

        Raises:
            MalformedModelOutputError: no final text, or invalid ActionDirective JSON.
            Exception: any ADK / google-genai error (mapped to escalation by the caller).
        """
        from google.genai import types

        prompt_context, _ = skill.format_prompt_context(diagnostic.failure_type)
        user_prompt = build_triage_user_prompt(
            prompt_context,
            diagnostic,
            allowed_actions=allowed_actions,
            parameter_rules=skill.parameter_rules,
        )
        runner = self._get_runner()
        session_service = runner.session_service
        user_id = diagnostic.run_id
        session_id = f"triage-{uuid.uuid4().hex}"

        await session_service.create_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
        final_text: str | None = None
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(role="user", parts=[types.Part(text=user_prompt)]),
            ):
                text = _final_response_text(event)
                if text is not None:
                    final_text = text
        finally:
            # The runner is reused across requests: drop the per-request session so
            # InMemorySessionService does not grow without bound.
            try:
                await session_service.delete_session(
                    app_name=APP_NAME, user_id=user_id, session_id=session_id
                )
            except Exception:  # noqa: BLE001 - cleanup must never mask the triage outcome
                pass
        return parse_model_output(final_text)
