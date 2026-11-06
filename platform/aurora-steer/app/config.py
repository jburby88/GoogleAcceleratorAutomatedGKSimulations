"""Centralized configuration and environment resolution for `aurora-steer`.

Every value is read from the environment at call time and validated, so a bad
deployment fails fast at startup (`app.main` lifespan) instead of escalating
every `/triage` request at runtime.
"""

from __future__ import annotations

import math
import os
import re

# Default Vertex AI model: Gemini 4 Argon, generally available on Vertex AI and served
# on the `global` endpoint. Override with `MODEL_ID` (e.g. `gemini-3.8-flash` for a
# lower-cost Flash model).
DEFAULT_MODEL_ID = "gemini-4-argon"
DEFAULT_LOCATION = "global"
DEFAULT_OKF_SKILL = "sim-babysitter"

# Server-side deadline for one LLM triage. It must stay below the HPC client's read
# timeout (`hpc_core.client.AuroraSteerClient.timeout_s = 100`) minus the BigQuery
# audit write that follows the LLM call (worst case ~17 s), so the server always
# answers (with `escalate_to_human` on timeout) before the client gives up.
DEFAULT_VERTEX_TIMEOUT_S = 60.0
MAX_VERTEX_TIMEOUT_S = 80.0
DEFAULT_VERTEX_MAX_ATTEMPTS = 2

# `ENVIRONMENT` values under which `ALLOW_UNAUTH_LOCAL=1` may disable authentication.
LOCAL_ENVIRONMENTS = frozenset({"dev", "local", "test"})

# Vertex AI Gemini model identifiers accepted by `MODEL_ID`:
#   gemini-3.8-flash
#   publishers/google/models/gemini-3.8-flash
#   projects/<project>/locations/<location>/publishers/google/models/<model>
#   projects/<project>/locations/<location>/endpoints/<numeric id>  (tuned model)
_MODEL_NAME = r"gemini-[0-9][a-z0-9.\-]*[a-z0-9]"
_PROJECT = r"[a-z][a-z0-9\-]{4,28}[a-z0-9]"
_LOCATION = r"[a-z][a-z0-9\-]*[a-z0-9]"
_MODEL_ID_RE = re.compile(
    rf"^(?:{_MODEL_NAME}"
    rf"|publishers/google/models/{_MODEL_NAME}"
    rf"|projects/{_PROJECT}/locations/{_LOCATION}/publishers/google/models/{_MODEL_NAME}"
    rf"|projects/{_PROJECT}/locations/{_LOCATION}/endpoints/[0-9]+)$"
)
_LOCATION_RE = re.compile(rf"^{_LOCATION}$")


class ConfigError(RuntimeError):
    """Raised when an environment variable holds an invalid value."""


def validate_model_id(model_id: str) -> str:
    """Return `model_id` if it is a well-formed Vertex AI Gemini model identifier."""
    if not isinstance(model_id, str) or not _MODEL_ID_RE.fullmatch(model_id):
        raise ConfigError(
            f"Invalid MODEL_ID {model_id!r}: expected a Vertex AI Gemini model id such as "
            f"{DEFAULT_MODEL_ID!r}, 'publishers/google/models/<model>', or a full "
            "'projects/<p>/locations/<l>/(publishers/google/models/<model>|endpoints/<id>)' name."
        )
    return model_id


def resolve_model_id() -> str:
    """Resolve and validate the Vertex AI Gemini model id from `MODEL_ID`."""
    return validate_model_id(os.getenv("MODEL_ID", "").strip() or DEFAULT_MODEL_ID)


def resolve_location() -> str:
    """Resolve and validate the Vertex AI endpoint location, defaulting to `global`."""
    location = (
        os.getenv("GOOGLE_CLOUD_LOCATION", "").strip()
        or os.getenv("GCP_LOCATION", "").strip()
        or DEFAULT_LOCATION
    )
    if not _LOCATION_RE.fullmatch(location):
        raise ConfigError(f"Invalid GOOGLE_CLOUD_LOCATION {location!r}")
    return location


def _env_number(name: str, default: float, *, low: float, high: float, integer: bool) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name}={raw!r} is not a valid number") from exc
    if not math.isfinite(value) or not low <= value <= high:
        raise ConfigError(f"{name}={raw!r} must be within [{low}, {high}]")
    return value


def resolve_vertex_timeout_s() -> float:
    """Server-side deadline (seconds) for one LLM triage (`VERTEX_TIMEOUT_S`)."""
    return _env_number(
        "VERTEX_TIMEOUT_S",
        DEFAULT_VERTEX_TIMEOUT_S,
        low=1.0,
        high=MAX_VERTEX_TIMEOUT_S,
        integer=False,
    )


def resolve_vertex_max_attempts() -> int:
    """Attempts per triage on transient Vertex errors (429/5xx), within the deadline."""
    return int(
        _env_number("VERTEX_MAX_ATTEMPTS", DEFAULT_VERTEX_MAX_ATTEMPTS, low=1, high=4, integer=True)
    )


def resolve_int_setting(name: str, default: int, *, low: int, high: int) -> int:
    """Read a bounded integer setting (rate limits, budgets) from the environment."""
    return int(_env_number(name, default, low=low, high=high, integer=True))


def resolve_float_setting(name: str, default: float, *, low: float, high: float) -> float:
    """Read a bounded float setting from the environment."""
    return _env_number(name, default, low=low, high=high, integer=False)


def configure_vertex_ai_env() -> None:
    """Point `google-adk` / `google-genai` at Vertex AI (never the API-key Gemini API).

    `google-genai` gives `GOOGLE_GENAI_USE_ENTERPRISE` precedence over
    `GOOGLE_GENAI_USE_VERTEXAI`, so both are forced on: this service only ever
    authenticates with the Cloud Run service account (ADC).
    """
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "true"
    project = (
        os.getenv("GOOGLE_CLOUD_PROJECT", "").strip() or os.getenv("GCP_PROJECT_ID", "").strip()
    )
    if project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = project
    os.environ["GOOGLE_CLOUD_LOCATION"] = resolve_location()


def resolve_bearer_tokens() -> list[str]:
    """Return active Secret Manager bearer tokens for `/triage` and `/v1/events`.

    `AURORA_STEER_BEARER_TOKEN` is canonical and may hold comma-separated tokens
    for zero-downtime rotation while long-running TACC Slurm jobs are active.
    `AURORA_API_TOKEN` is a deprecated fallback.
    """
    raw = os.environ.get("AURORA_STEER_BEARER_TOKEN") or os.environ.get("AURORA_API_TOKEN") or ""
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


def is_cloud_run() -> bool:
    """True when running on Cloud Run (`K_SERVICE` is set by the platform)."""
    return bool(os.environ.get("K_SERVICE", "").strip())


def is_unauth_local_allowed() -> bool:
    """Allow unauthenticated requests only for explicit local development.

    Requires `ALLOW_UNAUTH_LOCAL=1` AND `ENVIRONMENT` explicitly set to a local value
    (`dev`, `local` or `test`) AND not running on Cloud Run. Anything else (including
    an unset `ENVIRONMENT`, or `ENVIRONMENT=prod` on GKE/VMs) fails closed.
    """
    return (
        os.getenv("ALLOW_UNAUTH_LOCAL", "0").strip() == "1"
        and os.getenv("ENVIRONMENT", "").strip().lower() in LOCAL_ENVIRONMENTS
        and not is_cloud_run()
    )


def api_docs_enabled() -> bool:
    """Serve `/docs`, `/redoc` and `/openapi.json` locally; on Cloud Run only if opted in."""
    return os.getenv("ENABLE_API_DOCS", "").strip() == "1" or not is_cloud_run()


def validate_runtime_config() -> dict[str, object]:
    """Validate every runtime setting; raises `ConfigError` on the first invalid value."""
    return {
        "model": resolve_model_id(),
        "location": resolve_location(),
        "vertex_timeout_s": resolve_vertex_timeout_s(),
        "vertex_max_attempts": resolve_vertex_max_attempts(),
    }
