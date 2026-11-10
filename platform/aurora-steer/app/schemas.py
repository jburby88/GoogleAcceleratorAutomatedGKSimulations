"""Pydantic v2 request/response contracts for `/triage` and `/v1/events`.

Wire contract shared with `platform/hpc-core/hpc_core/client.py` and
`simulations/hpc/tacc_exec/watchdog.py` (change only by agreement).
`ActionDirective` doubles as the Gemini structured-output schema, so its JSON
schema must contain only keywords that `google-genai` / Vertex AI accept.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StringConstraints,
    field_validator,
    model_validator,
)

FailureType = Literal[
    "STALL",
    "TIMESTEP_COLLAPSE",
    "SPECTRAL_PILEUP",
    "DIVERGENCE",
    "CRASH",
    "UNKNOWN",
]

ActionType = Literal[
    "warm_restart",
    "cold_restart",
    "resubmit_walltime",
    "escalate_to_human",
]

ACTION_TYPES: frozenset[str] = frozenset(
    {"warm_restart", "cold_restart", "resubmit_walltime", "escalate_to_human"}
)

EnvironmentType = Literal["prod", "sim", "synthetic", "preflight"]

BackendType = Literal["tacc_slurm", "gcp_batch", "gcp_k8s"]

EventType = Literal[
    "submitted",
    "running",
    "progress",
    "completed",
    "failed",
    "triage_requested",
    "triage",
    "directive_received",
    "patched",
    "resubmitted",
    "walltime_resubmitted",
    "resubmit_walltime",
    "escalated",
    "human_resolved",
]

MAX_TRIAGE_BYTES = 2048
MAX_EVENT_BYTES = 1024

# Identifier grammar shared with hpc-core (`hpc_core/fsutil.py` SAFE_ID_RE and
# `tacc_exec/driver.py` SAFE_RUN_ID_RE): no whitespace, quotes or markup.
RunId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")]
CampaignId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")]
WorkstreamId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")]
SkillName = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")]
SchemaVersion = Annotated[str, StringConstraints(pattern=r"^[0-9]{1,3}\.[0-9]{1,3}$")]

StrictNumber = StrictInt | StrictFloat

_PARAM_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_METRIC_KEY_RE = re.compile(r"^[A-Za-z0-9_.:/()\[\]+-]{1,64}$")
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MAX_LOG_LINES = 30
MAX_LOG_LINE_CHARS = 240


def _ensure_finite_number(val: Any, field_label: str) -> int | float:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"{field_label} must be an int or float, got {type(val).__name__}")
    if not math.isfinite(float(val)):
        raise ValueError(f"{field_label} must be finite (NaN/Inf forbidden)")
    return val


def sanitize_text_line(line: str, max_chars: int = MAX_LOG_LINE_CHARS) -> str:
    """Strip ANSI escapes and control characters (untrusted solver output) and truncate."""
    cleaned = _ANSI_ESCAPE_RE.sub("", line)
    cleaned = _CONTROL_CHARS_RE.sub(" ", cleaned.replace("\t", " "))
    return cleaned[:max_chars]


class PreviousDirective(BaseModel):
    """Compact record of an earlier directive, as trimmed by the HPC watchdog."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    action: ActionType
    mods: dict[str, StrictNumber] = Field(default_factory=dict, max_length=16)

    @field_validator("mods")
    @classmethod
    def _validate_mods(cls, mods: dict[str, int | float]) -> dict[str, int | float]:
        for k, v in mods.items():
            if not _PARAM_KEY_RE.fullmatch(k):
                raise ValueError(f"previous_directives.mods key {k!r} is not a parameter name")
            _ensure_finite_number(v, f"previous_directives.mods.{k}")
        return mods


class DiagnosticSummary(BaseModel):
    """Outbound `<2 KB` diagnostic summary sent from TACC or Cloud Batch to `/triage`.

    Everything in this payload is untrusted input (solver logs in particular): it
    is validated strictly here and presented to the LLM as data, never as
    instructions (`app.prompts`).
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: SchemaVersion = "1.0"
    workstream: WorkstreamId = "kr2-gk-database"
    backend: BackendType = "tacc_slurm"
    skill_name: SkillName = "sim-babysitter"
    run_id: RunId
    campaign_id: CampaignId = "default_campaign"
    solver: Literal["GENE", "GX", "MRX", "SFINCS"]
    fidelity: Literal["low", "high"] = "low"
    equilibrium_id: RunId | None = None
    environment: EnvironmentType = "prod"
    failure_type: FailureType
    retry_count: StrictInt = Field(ge=0, le=20)
    walltime_resubmits: StrictInt = Field(default=0, ge=0, le=100)
    sim_time_reached: StrictFloat = Field(ge=0.0)
    sim_time_target: StrictFloat = Field(gt=0.0)
    current_parameters: dict[str, StrictNumber] = Field(default_factory=dict, max_length=32)
    scalar_metrics: dict[str, StrictFloat] = Field(default_factory=dict, max_length=48)
    log_tail: list[str] = Field(default_factory=list, max_length=MAX_LOG_LINES)
    previous_directives: list[PreviousDirective] = Field(default_factory=list, max_length=10)

    @field_validator("sim_time_reached", "sim_time_target")
    @classmethod
    def _validate_sim_times(cls, v: float, info: Any) -> float:
        _ensure_finite_number(v, info.field_name)
        return float(v)

    @field_validator("current_parameters")
    @classmethod
    def _validate_current_parameters(cls, params: dict[str, int | float]) -> dict[str, int | float]:
        for k, v in params.items():
            if not _PARAM_KEY_RE.fullmatch(k):
                raise ValueError(f"current_parameters key {k!r} is not a parameter name")
            _ensure_finite_number(v, f"current_parameters.{k}")
        return params

    @field_validator("scalar_metrics")
    @classmethod
    def _validate_scalar_metrics(cls, metrics: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for k, v in metrics.items():
            if not _METRIC_KEY_RE.fullmatch(k):
                raise ValueError(f"scalar_metrics key {k!r} contains forbidden characters")
            _ensure_finite_number(v, f"scalar_metrics.{k}")
            cleaned[k] = float(v)
        return cleaned

    @field_validator("log_tail")
    @classmethod
    def _validate_log_tail(cls, lines: list[str]) -> list[str]:
        if len(lines) > MAX_LOG_LINES:
            raise ValueError(f"log_tail cannot exceed {MAX_LOG_LINES} lines")
        return [sanitize_text_line(line) for line in lines]


class ActionDirective(BaseModel):
    """Structured directive returned by `/triage` (and constrained via Gemini JSON schema)."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    action: ActionType
    reason: str = Field(min_length=5, max_length=300)
    parameter_modifications: dict[str, StrictNumber] = Field(default_factory=dict, max_length=16)
    restart_from_checkpoint: bool = False
    confidence: Literal["low", "medium", "high"] = "medium"
    okf_concepts_used: list[Annotated[str, StringConstraints(max_length=64)]] = Field(
        default_factory=list, max_length=12
    )

    @field_validator("reason")
    @classmethod
    def _clean_reason(cls, reason: str) -> str:
        return sanitize_text_line(reason, max_chars=300)

    @field_validator("parameter_modifications")
    @classmethod
    def _validate_modifications_finite(cls, mods: dict[str, int | float]) -> dict[str, int | float]:
        for k, v in mods.items():
            _ensure_finite_number(v, f"parameter_modifications.{k}")
        return mods


class LifecycleEvent(BaseModel):
    """Single `<1 KB` telemetry event posted to `/v1/events`."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    event_id: str = Field(min_length=8, max_length=64)
    schema_version: str = Field(default="1.0", max_length=16)
    workstream: str = Field(default="kr2-gk-database", min_length=2, max_length=64)
    backend: str = Field(default="tacc_slurm", min_length=2, max_length=32)
    run_id: str = Field(min_length=1, max_length=128)
    campaign_id: str = Field(default="default_campaign", min_length=1, max_length=64)
    solver: Literal["GENE", "GX", "MRX", "SFINCS"]
    fidelity: Literal["low", "high"] = "low"
    equilibrium_id: str | None = None
    event_type: EventType
    ts_utc: datetime
    ts: datetime | None = None
    retry_count: int = Field(default=0, ge=0, le=50)
    walltime_resubmits: int = Field(default=0, ge=0, le=100)
    environment: EnvironmentType = "prod"
    is_synthetic: bool = False
    operator: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_serialized_size(self) -> LifecycleEvent:
        encoded = json.dumps(
            self.model_dump(mode="json"), separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(encoded) >= MAX_EVENT_BYTES:
            raise ValueError(
                f"Serialized LifecycleEvent is {len(encoded)} bytes, exceeding <{MAX_EVENT_BYTES}B limit"
            )
        return self


class EventsBatchRequest(BaseModel):
    """Batch wrapper for `POST /v1/events`."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = Field(default="1.0", max_length=16)
    events: list[LifecycleEvent] = Field(min_length=1, max_length=500)


class EventsBatchResponse(BaseModel):
    """Response from `POST /v1/events`."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    accepted: int
    deduplicated: int
