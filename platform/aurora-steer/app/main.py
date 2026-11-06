"""FastAPI entrypoint for `aurora-steer` (`/healthz`, `/triage`, `/v1/events`).

Responsibilities:
1. Enforce `<2 KB` payload size (bounded streaming read) and strict, finite-JSON
   `DiagnosticSummary` validation on `/triage` requests.
2. Enforce the deterministic 3-strike circuit breaker (`retry_count >= 2` escalates
   to a human physicist with zero LLM calls) and a per-run LLM budget.
3. Invoke the Google ADK 2.0 `SteeringEngine` (`app.runner`) with dynamically loaded
   Open Knowledge Framework (OKF) runbooks (`app.okf_loader`), under a server-side
   deadline with bounded retries on transient Vertex AI errors.
4. Validate every LLM `ActionDirective` through server-side reject-only guardrails
   (`app.guardrails`: runbook action policy, allowlist, bounds, restart semantics)
   before returning or recording it. Every failure path fails closed to
   `escalate_to_human`.
5. Deduplicate retries by `Idempotency-Key` (in-flight sharing, request
   fingerprinting, bounded LRU), and keep blocking I/O (DuckDB, BigQuery) off the
   event loop.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import inspect
import json
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.auth import verify_bearer_token
from app.config import (
    DEFAULT_OKF_SKILL,
    ConfigError,
    api_docs_enabled,
    resolve_float_setting,
    resolve_int_setting,
    resolve_location,
    resolve_model_id,
    resolve_vertex_max_attempts,
    resolve_vertex_timeout_s,
    validate_runtime_config,
)
from app.events import EventStore
from app.guardrails import validate_directive
from app.observability import log_event, new_request_context
from app.okf_loader import LoadedSkill, OKFLoadError, SkillNotFoundError, load_skill
from app.runner import SteeringEngine
from app.schemas import (
    MAX_TRIAGE_BYTES,
    ActionDirective,
    DiagnosticSummary,
    EventsBatchRequest,
    EventsBatchResponse,
)

MAX_IDEMPOTENCY_ENTRIES = 2048
MAX_EVENTS_BODY_BYTES = 524_288
CIRCUIT_BREAKER_RETRY_COUNT = 2
TRANSIENT_VERTEX_HTTP_CODES = frozenset({408, 429, 500, 502, 503, 504})
_IDEMPOTENCY_KEY_RE = re.compile(r"^[\x21-\x7e]{1,200}$")

# `X-Aurora-Decision` response header values (the ActionDirective body is unchanged).
DECISION_LLM = "llm"
DECISION_CACHED = "cached"
DECISION_CIRCUIT_BREAKER = "circuit_breaker"
DECISION_BUDGET = "budget_exhausted"
DECISION_GUARDRAIL = "guardrail_rejected"
DECISION_LLM_ERROR = "llm_error"
DECISION_LLM_TIMEOUT = "llm_timeout"
# Outcomes caused by a transient failure are not cached, so a retry re-attempts triage.
_UNCACHED_DECISIONS = frozenset({DECISION_LLM_ERROR, DECISION_LLM_TIMEOUT})


class RateLimiter:
    """Thread-safe sliding-window rate limiter (monotonic clock)."""

    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()

    def allow_request(self) -> bool:
        if self.max_requests <= 0:
            return True
        now = time.monotonic()
        with self.lock:
            while self.requests and self.requests[0] < now - self.window_seconds:
                self.requests.popleft()
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False


class RunLLMBudget:
    """Per-run cap on LLM triage calls within a sliding window (server-side runaway guard).

    The 3-strike breaker relies on the client-reported `retry_count`; this budget does
    not, so a client bug that never increments it cannot loop LLM calls (and spend).
    """

    def __init__(self, max_calls: int, window_seconds: float, max_runs: int = 10_000) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.max_runs = max_runs
        self._calls: OrderedDict[str, collections.deque[float]] = OrderedDict()
        self._lock = threading.Lock()

    def try_consume(self, run_key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            calls = self._calls.setdefault(run_key, collections.deque())
            self._calls.move_to_end(run_key)
            while calls and calls[0] < now - self.window_seconds:
                calls.popleft()
            if len(calls) >= self.max_calls:
                return False
            calls.append(now)
            while len(self._calls) > self.max_runs:
                self._calls.popitem(last=False)
            return True


@dataclass(frozen=True)
class TriageOutcome:
    """A completed triage decision, as cached for Idempotency-Key replays."""

    fingerprint: str
    directive: ActionDirective
    decision: str


class IdempotencyCache:
    """Bounded LRU cache mapping `workstream:run_id:Idempotency-Key` -> `TriageOutcome`."""

    def __init__(self, max_entries: int = MAX_IDEMPOTENCY_ENTRIES) -> None:
        self._max = max_entries
        self._lock = threading.Lock()
        self._store: OrderedDict[str, TriageOutcome] = OrderedDict()

    def get(self, key: str) -> TriageOutcome | None:
        with self._lock:
            val = self._store.get(key)
            if val is not None:
                self._store.move_to_end(key)
            return val

    def put(self, key: str, outcome: TriageOutcome) -> None:
        with self._lock:
            self._store[key] = outcome
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)


def request_fingerprint(diagnostic: DiagnosticSummary) -> str:
    """Hash of the fields that define *which* failure a triage request is about."""
    subset = {
        "workstream": diagnostic.workstream,
        "run_id": diagnostic.run_id,
        "skill_name": diagnostic.skill_name,
        "solver": diagnostic.solver,
        "failure_type": diagnostic.failure_type,
        "retry_count": diagnostic.retry_count,
        "current_parameters": diagnostic.current_parameters,
    }
    canonical = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_transient_vertex_error(exc: BaseException) -> bool:
    try:
        from google.genai import errors as genai_errors
    except ImportError:  # pragma: no cover - google-genai is a hard dependency
        return False
    return isinstance(exc, genai_errors.APIError) and exc.code in TRANSIENT_VERTEX_HTTP_CODES


def _escalation(reason: str, confidence: str = "high") -> ActionDirective:
    return ActionDirective(
        action="escalate_to_human",
        reason=reason[:300],
        parameter_modifications={},
        restart_from_checkpoint=False,
        confidence=confidence,
        okf_concepts_used=["escalation-policy"],
    )


async def _read_body_bounded(request: Request, max_bytes: int) -> bytes:
    """Read at most `max_bytes` of the body; 413 as soon as the limit is exceeded."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_len = int(declared)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid Content-Length") from exc
        if declared_len > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Request body ({declared_len} bytes) exceeds the {max_bytes}-byte limit.",
            )
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"Request body exceeds the {max_bytes}-byte limit.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_finite_json(raw_body: bytes) -> Any:
    def _reject_constant(token: str) -> Any:
        raise ValueError(f"Non-finite JSON constant: {token}")

    return json.loads(raw_body.decode("utf-8"), parse_constant=_reject_constant)


def create_app(
    event_store: EventStore | None = None,
    steering_engine: SteeringEngine | None = None,
) -> FastAPI:
    """Create a configured `aurora-steer` FastAPI instance."""
    store = event_store or EventStore()
    engine = steering_engine or SteeringEngine()
    idem_cache = IdempotencyCache()
    inflight: dict[str, tuple[str, asyncio.Future[TriageOutcome]]] = {}
    skill_cache: dict[str, LoadedSkill] = {}
    skill_lock = threading.Lock()
    metrics_lock = threading.Lock()
    llm_call_counter = {"count": 0}

    def _get_skill(skill_name: str) -> LoadedSkill:
        with skill_lock:
            if skill_name not in skill_cache:
                skill_cache[skill_name] = load_skill(skill_name=skill_name)
            return skill_cache[skill_name]

    def _default_skill_name() -> str:
        return os.getenv("DEFAULT_OKF_SKILL", DEFAULT_OKF_SKILL)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Fail fast: a bad MODEL_ID / location / timeout or a broken OKF skill must stop
        # the revision from starting (Cloud Run keeps serving the previous one) instead
        # of escalating every /triage at runtime.
        settings = validate_runtime_config()
        skill = _get_skill(_default_skill_name())
        log_event(
            "INFO",
            "aurora-steer started",
            **settings,
            default_skill=skill.skill_name,
            allowlisted_parameters=sorted(skill.parameter_rules),
        )
        yield

    docs = api_docs_enabled()
    application = FastAPI(
        title="aurora-steer",
        version="0.3.0",
        description="Shared Cloud Run LLM steering & telemetry service for UT Austin x Google.org",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    application.state.event_store = store
    application.state.steering_engine = engine
    application.state.llm_call_counter = llm_call_counter

    triage_limiter = RateLimiter(
        resolve_int_setting("RATE_LIMIT_TRIAGE", 120, low=0, high=100_000),
        resolve_float_setting("RATE_LIMIT_TRIAGE_WINDOW", 60.0, low=1.0, high=86_400.0),
    )
    events_limiter = RateLimiter(
        resolve_int_setting("RATE_LIMIT_EVENTS", 5000, low=0, high=1_000_000),
        resolve_float_setting("RATE_LIMIT_EVENTS_WINDOW", 60.0, low=1.0, high=86_400.0),
    )
    run_budget = RunLLMBudget(
        resolve_int_setting("MAX_LLM_TRIAGES_PER_RUN", 6, low=1, high=1000),
        resolve_float_setting("LLM_RUN_BUDGET_WINDOW_S", 86_400.0, low=60.0, high=30 * 86_400.0),
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next: Any) -> Response:
        rid = new_request_context(
            request.headers.get("x-request-id"), request.headers.get("x-cloud-trace-context")
        )
        response = await call_next(request)
        response.headers["X-Request-Id"] = rid
        return response

    @application.get("/health")
    @application.get("/healthz")
    @application.get("/v1/healthz")
    def healthz(response: Response) -> dict[str, Any]:
        """Service readiness on `/health`, `/healthz` (internal probe), and `/v1/healthz`."""
        problems: list[str] = []
        try:
            model: str | None = resolve_model_id()
            location: str | None = resolve_location()
        except ConfigError as exc:
            model, location = None, None
            problems.append(str(exc))
        skill_loaded = False
        try:
            _get_skill(_default_skill_name())
            skill_loaded = True
        except OKFLoadError as exc:
            problems.append(str(exc))
        if problems:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            log_event("ERROR", "healthz degraded", problems=problems)
        return {
            "status": "ok" if not problems else "degraded",
            "service": "aurora-steer",
            "model": model,
            "location": location,
            "default_skill_loaded": skill_loaded,
            "llm_calls_total": llm_call_counter["count"],
            "bq_insert_failures_total": getattr(store, "bq_insert_failures", 0),
        }

    async def _call_engine(
        diagnostic: DiagnosticSummary, skill: LoadedSkill, allowed_actions: frozenset[str]
    ) -> ActionDirective:
        kwargs = {"diagnostic": diagnostic, "skill": skill, "allowed_actions": allowed_actions}
        if inspect.iscoroutinefunction(engine.diagnose):
            return await engine.diagnose(**kwargs)
        # Synchronous engines (test doubles) must never block the event loop.
        return await run_in_threadpool(engine.diagnose, **kwargs)

    async def _diagnose_with_deadline(
        diagnostic: DiagnosticSummary, skill: LoadedSkill, allowed_actions: frozenset[str]
    ) -> ActionDirective:
        """One triage under the VERTEX_TIMEOUT_S deadline, retrying transient Vertex errors."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + resolve_vertex_timeout_s()
        max_attempts = resolve_vertex_max_attempts()
        attempt = 0
        while True:
            attempt += 1
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("LLM deadline exhausted")
            try:
                return await asyncio.wait_for(
                    _call_engine(diagnostic, skill, allowed_actions), timeout=remaining
                )
            except Exception as exc:
                if attempt >= max_attempts or not _is_transient_vertex_error(exc):
                    raise
                backoff = min(0.5 * 2 ** (attempt - 1), max(0.0, (deadline - loop.time()) / 2))
                log_event(
                    "WARNING",
                    "Transient Vertex AI error; retrying triage",
                    run_id=diagnostic.run_id,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    error=str(exc)[:500],
                )
                await asyncio.sleep(backoff)

    def _audit(outcome: str, forced: bool, llm_called: bool) -> dict[str, Any]:
        return {"guardrail_outcome": outcome, "forced_escalation": forced, "llm_called": llm_called}

    async def _decide(
        diagnostic: DiagnosticSummary, skill: LoadedSkill
    ) -> tuple[ActionDirective, str, dict[str, Any]]:
        """Return (directive, decision, audit fields) for a validated request."""
        run_key = f"{diagnostic.workstream}:{diagnostic.run_id}"

        if diagnostic.retry_count >= CIRCUIT_BREAKER_RETRY_COUNT:
            directive = _escalation(
                f"3-strike circuit breaker reached (retry_count={diagnostic.retry_count}); "
                "escalating to human physicist with zero additional LLM calls."
            )
            return directive, DECISION_CIRCUIT_BREAKER, _audit("accepted", True, False)

        if not run_budget.try_consume(run_key):
            directive = _escalation(
                f"Per-run LLM budget exhausted ({run_budget.max_calls} triage calls per "
                f"{int(run_budget.window_seconds)} s for {diagnostic.run_id}); escalating with "
                "zero additional LLM calls."
            )
            log_event("WARNING", "Per-run LLM budget exhausted", run_id=diagnostic.run_id)
            return directive, DECISION_BUDGET, _audit("rejected", True, False)

        allowed_actions = skill.allowed_actions_for(diagnostic.failure_type)
        try:
            raw_directive = await _diagnose_with_deadline(diagnostic, skill, allowed_actions)
        except TimeoutError:
            log_event(
                "ERROR",
                "LLM triage timed out; escalating",
                run_id=diagnostic.run_id,
                timeout_s=resolve_vertex_timeout_s(),
            )
            return (
                _escalation("LLM steering timed out; escalating safely to human review.", "low"),
                DECISION_LLM_TIMEOUT,
                _audit("rejected", True, True),
            )
        except Exception as exc:  # noqa: BLE001 - every LLM failure must fail closed
            log_event(
                "ERROR",
                "LLM triage failed; escalating",
                run_id=diagnostic.run_id,
                error_type=type(exc).__name__,
                error=str(exc)[:1000],
            )
            return (
                _escalation(
                    f"LLM steering error ({type(exc).__name__}); escalating safely to human review.",
                    "low",
                ),
                DECISION_LLM_ERROR,
                _audit("rejected", True, True),
            )

        with metrics_lock:
            llm_call_counter["count"] += 1
        gr = validate_directive(
            raw_directive,
            skill.parameter_rules,
            allowed_actions=allowed_actions,
            current_parameters=diagnostic.current_parameters,
        )
        if not gr.accepted:
            log_event(
                "WARNING",
                "Guardrail rejected LLM directive",
                run_id=diagnostic.run_id,
                failure_type=diagnostic.failure_type,
                proposed_action=raw_directive.action,
                proposed_modifications=raw_directive.parameter_modifications,
                rejection_reason=gr.rejection_reason,
            )
            return gr.directive, DECISION_GUARDRAIL, _audit("rejected", True, True)
        return gr.directive, DECISION_LLM, _audit("accepted", gr.forced_escalation, True)

    async def _handle_triage(
        request: Request,
        response: Response,
        idempotency_key: str | None,
        skill_override: str | None = None,
    ) -> ActionDirective:
        """Process a single simulation failure summary through ADK 2.0 and server guardrails."""
        # 1. Rate limiting & strict < 2 KB payload boundary (never buffer more).
        if not triage_limiter.allow_request():
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many triage requests")
        t0 = time.perf_counter()
        raw_body = await _read_body_bounded(request, MAX_TRIAGE_BYTES - 1)

        # 2. Parse JSON (rejecting NaN/Infinity constants) and validate DiagnosticSummary.
        try:
            parsed_json = _parse_finite_json(raw_body)
            if skill_override is not None:
                if not isinstance(parsed_json, dict):
                    raise ValueError("DiagnosticSummary must be a JSON object")
                parsed_json["skill_name"] = skill_override
            diagnostic = DiagnosticSummary.model_validate(parsed_json)
        except (UnicodeDecodeError, ValueError, ValidationError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

        # 3. Idempotency: replay completed outcomes, share in-flight ones, 409 on key reuse.
        fingerprint = request_fingerprint(diagnostic)
        scoped_key: str | None = None
        if idempotency_key is not None:
            if not _IDEMPOTENCY_KEY_RE.fullmatch(idempotency_key):
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "Malformed Idempotency-Key")
            scoped_key = f"{diagnostic.workstream}:{diagnostic.run_id}:{idempotency_key}"
            cached = idem_cache.get(scoped_key)
            if cached is None and scoped_key in inflight:
                inflight_fp, future = inflight[scoped_key]
                if inflight_fp != fingerprint:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        "Idempotency-Key is already in use for a different triage request.",
                    )
                cached = await asyncio.shield(future)
            if cached is not None:
                if cached.fingerprint != fingerprint:
                    raise HTTPException(
                        status.HTTP_409_CONFLICT,
                        "Idempotency-Key was already used for a different triage request.",
                    )
                response.headers["X-Aurora-Decision"] = DECISION_CACHED
                return cached.directive

        try:
            skill = _get_skill(diagnostic.skill_name)
        except SkillNotFoundError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
        except OKFLoadError as exc:
            log_event(
                "ERROR", "OKF skill failed to load", skill=diagnostic.skill_name, error=str(exc)
            )
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

        future: asyncio.Future[TriageOutcome] | None = None
        if scoped_key is not None:
            future = asyncio.get_running_loop().create_future()
            inflight[scoped_key] = (fingerprint, future)
        try:
            directive, decision, audit = await _decide(diagnostic, skill)

            # 4. Persist the audit record (DuckDB + structured log + BigQuery) off the loop.
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            await run_in_threadpool(
                store.record_directive,
                {
                    "directive_id": str(uuid.uuid4()),
                    "schema_version": diagnostic.schema_version,
                    "workstream": diagnostic.workstream,
                    "backend": diagnostic.backend,
                    "run_id": diagnostic.run_id,
                    "campaign_id": diagnostic.campaign_id,
                    "solver": diagnostic.solver,
                    "fidelity": diagnostic.fidelity,
                    "ts_utc": datetime.now(UTC).isoformat(),
                    "retry_count": diagnostic.retry_count,
                    "walltime_resubmits": diagnostic.walltime_resubmits,
                    "environment": diagnostic.environment,
                    "failure_type": diagnostic.failure_type,
                    "action": directive.action,
                    "reason": directive.reason,
                    "parameter_modifications_before": diagnostic.current_parameters,
                    "parameter_modifications_after": directive.parameter_modifications,
                    "restart_from_checkpoint": directive.restart_from_checkpoint,
                    "confidence": directive.confidence,
                    "okf_concepts_used": directive.okf_concepts_used,
                    **audit,
                    "latency_ms": latency_ms,
                    "is_synthetic": diagnostic.environment != "prod",
                },
            )
            log_event(
                "INFO",
                "triage decided",
                run_id=diagnostic.run_id,
                failure_type=diagnostic.failure_type,
                retry_count=diagnostic.retry_count,
                decision=decision,
                action=directive.action,
                latency_ms=latency_ms,
            )
            outcome = TriageOutcome(fingerprint=fingerprint, directive=directive, decision=decision)
            if scoped_key is not None and decision not in _UNCACHED_DECISIONS:
                idem_cache.put(scoped_key, outcome)
            if future is not None:
                future.set_result(outcome)
        except BaseException as exc:
            if future is not None and not future.done():
                future.set_exception(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
                future.exception()  # mark retrieved: waiters re-raise it, no "never retrieved" noise
            raise
        finally:
            if scoped_key is not None and future is not None:
                if inflight.get(scoped_key, (None, None))[1] is future:
                    inflight.pop(scoped_key, None)

        response.headers["X-Aurora-Decision"] = decision
        return directive

    @application.post(
        "/triage",
        response_model=ActionDirective,
        dependencies=[Depends(verify_bearer_token)],
    )
    async def triage(
        request: Request,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ActionDirective:
        """Diagnose a `<2 KB` simulation failure summary and return a guardrail-verified `ActionDirective`."""
        return await _handle_triage(request, response, idempotency_key)

    @application.post(
        "/v1/skills/{skill_name}/triage",
        response_model=ActionDirective,
        dependencies=[Depends(verify_bearer_token)],
    )
    async def triage_by_skill(
        skill_name: str,
        request: Request,
        response: Response,
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    ) -> ActionDirective:
        """Diagnose a simulation failure using an explicit OKF skill (the path wins over the body)."""
        return await _handle_triage(request, response, idempotency_key, skill_override=skill_name)

    @application.post(
        "/v1/events",
        response_model=EventsBatchResponse,
        dependencies=[Depends(verify_bearer_token)],
    )
    async def post_events(request: Request) -> EventsBatchResponse:
        """Ingest a batch of lifecycle telemetry events (`<= 500` events) into BigQuery."""
        if not events_limiter.allow_request():
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many events requests")
        raw_body = await _read_body_bounded(request, MAX_EVENTS_BODY_BYTES)
        try:
            batch = EventsBatchRequest.model_validate(_parse_finite_json(raw_body))
        except (UnicodeDecodeError, ValueError, ValidationError) as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

        accepted, deduplicated = await run_in_threadpool(store.ingest_events, batch.events)
        return EventsBatchResponse(accepted=accepted, deduplicated=deduplicated)

    @application.get(
        "/v1/events/summary",
        dependencies=[Depends(verify_bearer_token)],
    )
    async def events_summary(
        environment: str = Query(default="prod", max_length=16),
        workstream: str | None = Query(default=None, max_length=64),
    ) -> dict[str, Any]:
        """KPIs from this instance's local DuckDB store (debug aid; BigQuery is authoritative)."""
        return await run_in_threadpool(
            store.query_summary, environment=environment, workstream=workstream
        )

    return application


app = create_app()
