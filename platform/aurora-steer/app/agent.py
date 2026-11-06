"""Google ADK 2.0 (`google-adk`) agent definition for `aurora-steer`.

Follows the standard Google ADK 2.0 convention:
- Exposes `root_agent` (lazily, via module `__getattr__`) so `adk run` and
  `adk web` can discover and inspect the agent without import-time side effects.
- Uses `output_schema=ActionDirective` so Gemini returns structured JSON that is
  then validated by the server-side guardrails (`app.guardrails`).

`google-adk` is imported inside the factory so the rest of the service (schemas,
guardrails, OKF loader, API) imports and tests without it.
"""

from __future__ import annotations

import os
from typing import Any

from app.config import resolve_model_id
from app.prompts import SYSTEM_INSTRUCTION
from app.schemas import ActionDirective

AGENT_NAME = "aurora_steer"
DEFAULT_THINKING_LEVEL = "MEDIUM"


def _use_json_schema(_callback_context: Any, llm_request: Any) -> Any:
    """Use full JSON Schema (`response_json_schema`) instead of OpenAPI `response_schema`.

    Vertex AI's legacy OpenAPI `response_schema` treats `dict[str, StrictNumber]`
    (`parameter_modifications`) as an `OBJECT` with no `properties` and constrains
    the output to `{}`. `response_json_schema` preserves `additionalProperties`.
    """
    cfg = getattr(llm_request, "config", None)
    if cfg is not None:
        cfg.response_json_schema = ActionDirective.model_json_schema()
        cfg.response_schema = None
    return None


def create_aurora_steer_agent(model: Any = None) -> Any:
    """Create the single-turn Google ADK 2.0 triage agent.

    Args:
        model: Optional explicit Vertex AI model identifier (or an ADK `BaseLlm`
            instance, e.g. in tests). When omitted, resolves (and validates)
            `MODEL_ID` via `app.config.resolve_model_id()`.

    Returns:
        Configured `google.adk.agents.LlmAgent` instance.
    """
    from google.adk.agents import LlmAgent
    from google.genai import types

    level_name = os.getenv("THINKING_LEVEL", DEFAULT_THINKING_LEVEL).strip().upper()
    thinking_level = getattr(types.ThinkingLevel, level_name, types.ThinkingLevel.MEDIUM)

    return LlmAgent(
        name=AGENT_NAME,
        model=model or resolve_model_id(),
        description=(
            "Automated numerical stability triage agent for GENE and GX "
            "gyrokinetic plasma turbulence simulations."
        ),
        instruction=SYSTEM_INSTRUCTION,
        # Enforce strict Pydantic JSON output matching ActionDirective.
        output_schema=ActionDirective,
        before_model_callback=_use_json_schema,
        # Disable multi-agent delegation to guarantee <= 1 LLM call per failure.
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        # Low temperature + MEDIUM thinking for fast, runbook-faithful numerical patching.
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        ),
    )


def __getattr__(name: str) -> Any:
    # Standard ADK 2.0 entrypoint discovered by `adk run` / `adk web`.
    if name == "root_agent":
        return create_aurora_steer_agent()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
