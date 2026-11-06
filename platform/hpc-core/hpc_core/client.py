"""Outbound HTTPS client for posting DiagnosticSummary and LifecycleEvent payloads to aurora-steer.

Error contract (the watchdog relies on it):

- `TriageTransientError` (a `TriageClientError`): nothing was decided. aurora-steer could not
  be reached or could not answer right now: connection errors, timeouts, HTTP 429, HTTP 5xx
  (429/502/503/504 are retried in-call with backoff first), or an HTTP 200 whose
  `X-Aurora-Decision` header says the LLM itself failed (`llm_error`, `llm_timeout`; the
  server's fallback directive is ignored). Retry later with the same Idempotency-Key.
- Any other `TriageClientError`: the request or the answer is unusable (HTTP 4xx incl. 409,
  redirects, oversize payload, malformed or unsafe directive, missing token). Retrying cannot
  help: escalate to a human.

Payloads are never truncated: a DiagnosticSummary of 2048 bytes or more is refused before
sending (trim it upstream). The bearer token is never included in exception messages, and
server-provided text is truncated to 300 characters.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

import requests

MAX_PAYLOAD_BYTES = 2048
MAX_RESPONSE_BYTES = 64 * 1024
MAX_ERROR_BODY_BYTES = 4096
MAX_ERROR_TEXT_CHARS = 300
MAX_REASON_CHARS = 300
DEFAULT_CONNECT_TIMEOUT_S = 10.0
VALID_ACTIONS = {
    "warm_restart",
    "cold_restart",
    "resubmit_walltime",
    "escalate_to_human",
}
VALID_CONFIDENCE = frozenset({"low", "medium", "high"})
# Precedence order. A comma-separated value (server-side rotation list) sends its first token.
TOKEN_ENV_VARS = ("AURORA_STEER_BEARER_TOKEN", "AURORA_API_TOKEN", "AURORA_STEER_API_TOKEN")
DECISION_HEADER = "X-Aurora-Decision"
TRANSIENT_DECISIONS = frozenset({"llm_error", "llm_timeout"})
_RETRYABLE_STATUS = frozenset({429, 502, 503, 504})
_IDEMPOTENCY_KEY_RE = re.compile(r"^[\x21-\x7e]{1,200}$")  # mirrors aurora-steer app/main.py
_SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_LOCALHOST_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_TRANSIENT_REQUEST_ERRORS: tuple[type[BaseException], ...] = (
    requests.ConnectionError,
    requests.Timeout,
    requests.exceptions.ChunkedEncodingError,
    requests.exceptions.ContentDecodingError,
)


class TriageClientError(RuntimeError):
    """Raised when aurora-steer returns an error or invalid directive (do not retry)."""


class TriageTransientError(TriageClientError):
    """aurora-steer was unreachable or temporarily unable to decide; retry later (same key)."""


class EventsValidationError(TriageClientError):
    """Raised when POST /v1/events returns HTTP 400/413/422 due to a malformed/poison event."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _reject_nonfinite_constant(token: str) -> Any:
    raise TriageClientError(f"Non-finite numeric constant {token!r} is not allowed in JSON.")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def resolve_api_token(explicit: str | None = None) -> str:
    """Return the bearer token: explicit value, else the first set variable in TOKEN_ENV_VARS."""
    for raw in (explicit, *(os.environ.get(name) for name in TOKEN_ENV_VARS)):
        if not raw:
            continue
        for part in str(raw).split(","):
            token = part.strip()
            if token:
                return token
    return ""


def validate_aurora_steer_url(url: str) -> str:
    """Validate that aurora_steer_url uses https:// (or http:// only for localhost/127.0.0.1)."""
    cleaned = (url or "").strip().rstrip("/")
    if not cleaned:
        raise ValueError("aurora_steer_url must not be empty.")
    parsed = urlparse(cleaned)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Invalid aurora_steer_url: credentials, query strings and fragments are not allowed "
            "(the bearer token comes from AURORA_STEER_BEARER_TOKEN)."
        )
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and hostname:
        return cleaned
    if parsed.scheme == "http" and hostname in _LOCALHOST_HOSTS:
        return cleaned
    raise ValueError(
        f"Invalid aurora_steer_url {url!r}: must use 'https://' "
        "(or 'http://' only for localhost / 127.0.0.1)."
    )


def resolve_aurora_steer_url(
    config: Mapping[str, Any] | None = None,
    explicit_url: str | None = None,
) -> str:
    """Resolve aurora-steer base URL with AURORA_STEER_URL env override and scheme validation."""
    env_url = os.environ.get("AURORA_STEER_URL", "").strip()
    cfg_url = str((config or {}).get("aurora_steer_url", "")).strip()
    raw = (env_url or explicit_url or cfg_url).strip()
    return validate_aurora_steer_url(raw)


class AuroraSteerClient:
    """Stateless HTTPS client with timeouts, bounded retry/backoff and strict response checks."""

    def __init__(
        self,
        base_url: str | None = None,
        skill_name: str = "sim-babysitter",
        api_token: str | None = None,
        timeout_s: float = 100.0,
        max_attempts: int = 3,
        backoff_base_s: float = 0.5,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ) -> None:
        self.base_url = resolve_aurora_steer_url(explicit_url=base_url)
        if not _SKILL_NAME_RE.fullmatch(skill_name or ""):
            raise ValueError(f"Invalid skill_name {skill_name!r}")
        self.skill_name = skill_name
        self.api_token = resolve_api_token(api_token)
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.max_attempts = max(1, int(max_attempts))
        self.backoff_base_s = max(0.0, float(backoff_base_s))
        # X-Aurora-Decision of the last triage answer (llm, cached, circuit_breaker, ...).
        self.last_decision: str | None = None
        self._session = requests.Session()
        parsed = urlparse(self.base_url)
        if (parsed.hostname or "").lower() in _LOCALHOST_HOSTS:
            # Never route loopback requests through corporate/HPC HTTP proxies
            self._session.trust_env = False

    # -- helpers ---------------------------------------------------------------------------
    def _sanitize_text(self, text: object) -> str:
        """One-line, token-free, length-bounded rendering of server/library text."""
        flat = " ".join(str(text).split())
        if self.api_token:
            flat = flat.replace(self.api_token, "[REDACTED]")
        if len(flat) > MAX_ERROR_TEXT_CHARS:
            flat = flat[:MAX_ERROR_TEXT_CHARS] + "..."
        return flat

    def _backoff(self, attempt: int) -> None:
        time.sleep(self.backoff_base_s * (2 ** (attempt - 1)))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @staticmethod
    def _read_capped(resp: Any, cap: int = MAX_RESPONSE_BYTES) -> bytes:
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                if not chunk:
                    continue
                total += len(chunk)
                if total > cap:
                    raise TriageClientError(f"aurora-steer response exceeds {cap} bytes")
                chunks.append(chunk)
        except _TRANSIENT_REQUEST_ERRORS as exc:
            raise TriageTransientError(
                f"Connection lost while reading the aurora-steer response ({type(exc).__name__})"
            ) from exc
        return b"".join(chunks)

    def _read_error_text(self, resp: Any) -> str:
        chunks: list[bytes] = []
        total = 0
        try:
            for chunk in resp.iter_content(chunk_size=1024):
                chunks.append(chunk)
                total += len(chunk)
                if total >= MAX_ERROR_BODY_BYTES:
                    break
        except (requests.RequestException, OSError):
            pass
        return self._sanitize_text(
            b"".join(chunks)[:MAX_ERROR_BODY_BYTES].decode("utf-8", "replace")
        )

    @staticmethod
    def _parse_json(body: bytes) -> Any:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TriageClientError("aurora-steer response is not valid UTF-8") from exc
        try:
            return json.loads(text, parse_constant=_reject_nonfinite_constant)
        except ValueError as exc:
            raise TriageClientError(f"Invalid JSON in aurora-steer response: {exc}") from exc

    # -- API -------------------------------------------------------------------------------
    def triage(
        self,
        diagnostic_summary: dict[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """POST a DiagnosticSummary (<2 KB) and return the validated ActionDirective dict."""
        self.last_decision = None
        if not self.api_token:
            raise TriageClientError(
                "No aurora-steer bearer token configured (set AURORA_STEER_BEARER_TOKEN)."
            )
        if not isinstance(diagnostic_summary, Mapping):
            raise TriageClientError("DiagnosticSummary must be a JSON object.")
        try:
            raw_bytes = json.dumps(
                diagnostic_summary, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise TriageClientError(
                f"Diagnostic payload is not strict JSON (NaN/Inf or unserializable value): "
                f"{self._sanitize_text(exc)}"
            ) from exc

        if len(raw_bytes) >= MAX_PAYLOAD_BYTES:
            raise TriageClientError(
                f"Refusing to send a {len(raw_bytes)}-byte DiagnosticSummary (limit: < "
                f"{MAX_PAYLOAD_BYTES} bytes). Payloads are never truncated; trim it upstream."
            )

        idem_key = str(
            idempotency_key
            or (
                f"{diagnostic_summary.get('run_id', 'run')}:"
                f"{diagnostic_summary.get('retry_count', 0)}:"
                f"{diagnostic_summary.get('failure_type', 'unknown')}"
            )
        )
        if not _IDEMPOTENCY_KEY_RE.fullmatch(idem_key):
            raise TriageClientError(
                "Idempotency-Key must be 1-200 printable ASCII characters without spaces."
            )

        url = f"{self.base_url}/v1/skills/{self.skill_name}/triage"
        headers = {**self._headers(), "Idempotency-Key": idem_key}

        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = self._session.post(
                    url,
                    data=raw_bytes,
                    headers=headers,
                    timeout=(self.connect_timeout_s, self.timeout_s),
                    allow_redirects=False,
                    stream=True,
                )
            except requests.ReadTimeout as exc:
                # Never retry after a read timeout to guarantee <= 1 Vertex LLM call per failure.
                raise TriageTransientError(
                    f"HTTPS read timeout after {self.timeout_s:g}s calling {url}; not retried in "
                    "this call. Retry later with the same Idempotency-Key."
                ) from exc
            except _TRANSIENT_REQUEST_ERRORS as exc:
                if attempt < self.max_attempts:
                    self._backoff(attempt)
                    continue
                raise TriageTransientError(
                    f"aurora-steer unreachable after {attempt} attempt(s) "
                    f"({type(exc).__name__}): {self._sanitize_text(exc)}"
                ) from exc
            except requests.RequestException as exc:
                raise TriageClientError(
                    f"Invalid HTTPS request to {url} ({type(exc).__name__}): "
                    f"{self._sanitize_text(exc)}"
                ) from exc

            try:
                status = resp.status_code
                if status == 200:
                    body = self._read_capped(resp)
                    decision = (resp.headers.get(DECISION_HEADER) or "").strip() or None
                    self.last_decision = decision
                    if decision in TRANSIENT_DECISIONS:
                        raise TriageTransientError(
                            f"aurora-steer could not consult its LLM ({DECISION_HEADER}="
                            f"{decision}); its fallback directive is ignored. Retry later."
                        )
                    return self._validate_directive_shape(self._parse_json(body))
                error_text = self._read_error_text(resp)
            finally:
                resp.close()

            if status in _RETRYABLE_STATUS and attempt < self.max_attempts:
                self._backoff(attempt)
                continue
            message = f"aurora-steer returned HTTP {status}: {error_text}"
            if status in _RETRYABLE_STATUS or 500 <= status <= 599:
                raise TriageTransientError(message)
            if 300 <= status <= 399:
                raise TriageClientError(
                    f"aurora-steer answered with a redirect (HTTP {status}); check "
                    "aurora_steer_url. Redirects are never followed with the bearer token."
                )
            raise TriageClientError(message)

        raise TriageTransientError(f"Exhausted retries calling {url}")

    @staticmethod
    def _validate_directive_shape(directive: Any) -> dict[str, Any]:
        """Strictly validate an ActionDirective; return a copy with only the known keys."""
        if not isinstance(directive, dict):
            raise TriageClientError("ActionDirective response must be a JSON object.")
        action = directive.get("action")
        if not isinstance(action, str) or action not in VALID_ACTIONS:
            raise TriageClientError(f"Invalid action in directive: {str(action)[:64]!r}")
        reason = directive.get("reason")
        if not isinstance(reason, str):
            raise TriageClientError("Directive 'reason' must be a string.")
        if len(reason) > MAX_REASON_CHARS:
            raise TriageClientError(f"Directive 'reason' exceeds {MAX_REASON_CHARS} characters.")
        mods = directive.get("parameter_modifications")
        if not isinstance(mods, dict):
            raise TriageClientError("parameter_modifications must be a dictionary.")
        clean_mods: dict[str, int | float] = {}
        for key, value in mods.items():
            if not isinstance(key, str) or not key:
                raise TriageClientError("parameter_modifications keys must be non-empty strings.")
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TriageClientError(
                    f"Modified parameter {key[:64]!r} must be numeric, got {type(value).__name__}"
                )
            if not _is_finite_number(value):
                raise TriageClientError(
                    f"Modified parameter {key[:64]!r} must be finite (no NaN/Inf/overflow)."
                )
            clean_mods[key] = value
        restart = directive.get("restart_from_checkpoint", False)
        if not isinstance(restart, bool):
            raise TriageClientError("restart_from_checkpoint must be a JSON boolean.")

        clean: dict[str, Any] = {
            "action": action,
            "reason": reason,
            "parameter_modifications": clean_mods,
            "restart_from_checkpoint": restart,
        }
        confidence = directive.get("confidence")
        if isinstance(confidence, str) and confidence in VALID_CONFIDENCE:
            clean["confidence"] = confidence
        concepts = directive.get("okf_concepts_used")
        if isinstance(concepts, list) and all(isinstance(c, str) for c in concepts):
            clean["okf_concepts_used"] = [c[:64] for c in concepts[:20]]
        return clean

    def post_events(self, events: list[dict[str, Any]], timeout_s: float = 5.0) -> dict[str, Any]:
        """POST a batch of lifecycle events (up to 500) to /v1/events."""
        if not self.api_token:
            raise TriageClientError(
                "No aurora-steer bearer token configured (set AURORA_STEER_BEARER_TOKEN)."
            )
        try:
            body = json.dumps(
                {"schema_version": "1.0", "events": events},
                allow_nan=False,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            # Not strict JSON (NaN/Inf, unserializable value): a poison batch, never sendable.
            raise EventsValidationError(
                f"Telemetry batch is not strict JSON: {self._sanitize_text(exc)}", status_code=422
            ) from exc

        url = f"{self.base_url}/v1/events"
        read_timeout = float(timeout_s)
        try:
            resp = self._session.post(
                url,
                data=body,
                headers=self._headers(),
                timeout=(min(self.connect_timeout_s, read_timeout), read_timeout),
                allow_redirects=False,
                stream=True,
            )
        except _TRANSIENT_REQUEST_ERRORS as exc:
            raise TriageTransientError(
                f"POST /v1/events failed ({type(exc).__name__}): {self._sanitize_text(exc)}"
            ) from exc
        except requests.RequestException as exc:
            raise TriageClientError(
                f"POST /v1/events failed ({type(exc).__name__}): {self._sanitize_text(exc)}"
            ) from exc

        try:
            status = resp.status_code
            if status == 200:
                data = self._parse_json(self._read_capped(resp))
                return data if isinstance(data, dict) else {"status": "ok"}
            error_text = self._read_error_text(resp)
        finally:
            resp.close()

        message = f"POST /v1/events returned HTTP {status}: {error_text}"
        if status in (400, 413, 422):
            raise EventsValidationError(
                f"POST /v1/events rejected payload with HTTP {status}: {error_text}",
                status_code=status,
            )
        if status == 429 or 500 <= status <= 599:
            raise TriageTransientError(message)
        raise TriageClientError(message)
