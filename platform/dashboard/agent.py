"""Google ADK 2.0 (`google-adk`) agent definition for `aurora_telemetry_analyst`.

Follows the same Google ADK 2.0 convention as `platform/aurora-steer/app/agent.py`:
- Exposes `root_agent` lazily via `__getattr__` so `adk run` and `adk web` can
  discover and inspect the agent without import-time side effects.
- Binds the 6 read-only telemetry tool functions from `tools.TelemetryToolkit`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any

from prompts import build_system_instruction

AGENT_NAME = "aurora_telemetry_analyst"
DEFAULT_CHAT_MODEL_ID = "gemini-3.8-flash"
FALLBACK_CHAT_MODEL_ID = "gemini-4-argon"
DEFAULT_LOCATION = "global"


def configure_vertex_env(project_id: str | None = None) -> str:
    """Configure environment variables so `google-adk` / `google-genai` use Vertex AI `global`."""
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "true"
    if project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "").strip() or DEFAULT_LOCATION
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    return location


def resolve_chat_model_id() -> str:
    """Return the configured Vertex AI Gemini model for Ask Aurora."""
    return os.environ.get("AURORA_CHAT_MODEL_ID", "").strip() or DEFAULT_CHAT_MODEL_ID


def create_telemetry_analyst_agent(
    *,
    tools: Sequence[Callable[..., Any]],
    active_campaign_id: str = "w7x_backtest_demo",
    model: Any = None,
) -> Any:
    """Create the Google ADK 2.0 `LlmAgent` for conversational telemetry analysis."""
    from google.adk.agents import LlmAgent
    from google.genai import types

    instruction_text = build_system_instruction(active_campaign_id)
    return LlmAgent(
        name=AGENT_NAME,
        model=model or resolve_chat_model_id(),
        description=(
            "Conversational telemetry analyst for UT Austin × Google.org KR2 "
            "automated gyrokinetic simulations across TACC Slurm and GCP Cloud Batch."
        ),
        instruction=lambda _ctx: instruction_text,
        tools=list(tools),
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
        generate_content_config=types.GenerateContentConfig(
            temperature=0.1,
        ),
    )


def __getattr__(name: str) -> Any:
    if name == "root_agent":
        from tools import TelemetryToolkit

        toolkit = TelemetryToolkit(client=None, active_campaign_id="w7x_backtest_demo")
        return create_telemetry_analyst_agent(tools=toolkit.as_adk_tools())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
