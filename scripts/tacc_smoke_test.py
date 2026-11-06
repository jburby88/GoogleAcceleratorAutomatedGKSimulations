#!/usr/bin/env python3
"""Zero-dependency TACC Login Node -> Cloud Run (`aurora-steer`) Outbound Smoke Test.

Uses ONLY the Python 3 standard library (`urllib.request`, `json`, `socket`) and runs on the
system `python3` (3.8 or newer) of a TACC Vista / Lonestar6 / Frontera login node before
creating any virtual environment:

    export AURORA_STEER_URL="https://aurora-steer-xxxx-uc.a.run.app"
    export AURORA_STEER_BEARER_TOKEN="<your-token>"   # AURORA_STEER_TOKEN is read as a fallback
    python3 scripts/tacc_smoke_test.py

The bearer token is read from the environment and is never printed. Avoid `--token` on shared
login nodes: command-line arguments are visible to every user through `ps`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotations only: keeps the script importable on Python 3.8/3.9
    from collections.abc import Callable, Mapping

    HttpResult = tuple[int, dict[str, Any]] | tuple[int, dict[str, Any], Mapping[str, str]]
    HttpTransport = Callable[[str, str, str | None, dict[str, Any] | None, float], HttpResult]

SKILL_NAME = "sim-babysitter"
# Same route as hpc_core.client.AuroraSteerClient.triage (platform/aurora-steer/app/main.py).
TRIAGE_PATH = f"/v1/skills/{SKILL_NAME}/triage"
# The server's LLM deadline is 60 s (VERTEX_TIMEOUT_S); leave room for queueing and TLS.
TRIAGE_TIMEOUT_S = 90.0
DEFAULT_TIMEOUT_S = 30.0
TOKEN_ENV_VARS = ("AURORA_STEER_BEARER_TOKEN", "AURORA_STEER_TOKEN")
DECISION_HEADER = "X-Aurora-Decision"
MAX_RESPONSE_BYTES = 64 * 1024
MAX_PRINTED_CHARS = 300
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def resolve_token(cli_token: str | None = None) -> str:
    """Bearer token: --token, else AURORA_STEER_BEARER_TOKEN, else AURORA_STEER_TOKEN."""
    for raw in (cli_token, *(os.environ.get(name) for name in TOKEN_ENV_VARS)):
        if not raw:
            continue
        for part in str(raw).split(","):  # rotation lists: send the first token
            token = part.strip()
            if token:
                return token
    return ""


def validate_base_url(url: str) -> str:
    """https:// only (http:// for loopback); no credentials, query or fragment in the URL."""
    cleaned = (url or "").strip().rstrip("/")
    parsed = urllib.parse.urlparse(cleaned)
    host = (parsed.hostname or "").lower()
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("the URL must not contain credentials, a query string or a fragment")
    if parsed.scheme == "https" and host:
        return cleaned
    if parsed.scheme == "http" and host in _LOOPBACK_HOSTS:
        return cleaned
    raise ValueError("the URL must use https:// (http:// only for localhost)")


def _parse_json_body(raw: bytes) -> dict[str, Any]:
    text = raw[:MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        return {"raw_body": text[:MAX_PRINTED_CHARS]}
    return parsed if isinstance(parsed, dict) else {"raw_body": text[:MAX_PRINTED_CHARS]}


def _default_http_json(
    method: str,
    url: str,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = {"Accept": "application/json"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    if token:
        # Unredirected header: urllib never forwards it to a redirect target.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = _parse_json_body(resp.read(MAX_RESPONSE_BYTES + 1))
            return resp.status, body, dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        body = _parse_json_body(exc.read(MAX_RESPONSE_BYTES + 1))
        return exc.code, body, dict(exc.headers.items()) if exc.headers else {}


def _call(
    transport: HttpTransport,
    method: str,
    url: str,
    token: str | None,
    payload: dict[str, Any] | None,
    timeout_s: float,
) -> tuple[int, dict[str, Any], dict[str, str] | None]:
    """Normalize transports returning (status, body) or (status, body, headers)."""
    result = transport(method, url, token, payload, timeout_s)
    if len(result) == 3:
        status, body, headers = result  # type: ignore[misc]
        return int(status), body, {str(k).lower(): str(v) for k, v in dict(headers).items()}
    status, body = result  # type: ignore[misc]
    return int(status), body, None


def _preview(obj: Any, token: str | None) -> str:
    text = json.dumps(obj, default=str) if not isinstance(obj, str) else obj
    if token:
        text = text.replace(token, "[REDACTED]")
    return text if len(text) <= MAX_PRINTED_CHARS else text[:MAX_PRINTED_CHARS] + "..."


def run_smoke_test(
    base_url: str,
    token: str | None = None,
    *,
    skip_triage: bool = False,
    transport: HttpTransport | None = None,
) -> int:
    """Execute the 3-step TACC -> Cloud Run smoke check (/healthz, /v1/events, triage)."""
    http_call = transport or _default_http_json
    try:
        base_url = validate_base_url(base_url)
    except ValueError as exc:
        print(f"[FAIL] Invalid aurora-steer URL: {exc}", file=sys.stderr)
        return 2
    hostname = socket.gethostname()
    host_slug = re.sub(r"[^A-Za-z0-9_-]", "_", hostname.split(".")[0])[:40] or "host"
    smoke_run_id = f"tacc_smoke_{host_slug}_{uuid.uuid4().hex[:6]}"

    print("==================================================================")
    print(" TACC Login Node -> GCP Cloud Run (aurora-steer) Outbound Smoke Test")
    print(f" Hostname      : {hostname}")
    print(f" Target URL    : {base_url}")
    print(f" Smoke Run ID  : {smoke_run_id}")
    print(f" Bearer token  : {'set' if token else 'NOT SET'}")
    print("==================================================================")

    # Check 1: Outbound HTTPS & /health (/healthz fallback; unauthenticated: token not sent)
    t0 = time.monotonic()
    try:
        status, health, _ = _call(
            http_call, "GET", f"{base_url}/health", None, None, DEFAULT_TIMEOUT_S
        )
        if status == 404:
            status, health, _ = _call(
                http_call, "GET", f"{base_url}/healthz", None, None, DEFAULT_TIMEOUT_S
            )
    except Exception as exc:
        print(
            f"\n[FAIL] Step 1 (/health): Outbound HTTPS connection failed: "
            f"{_preview(str(exc), token)}",
            file=sys.stderr,
        )
        print(
            "Hint: Check if proxy env vars (HTTPS_PROXY) or firewall rules block *.run.app:443 on this node.",
            file=sys.stderr,
        )
        return 1
    dt_ms = (time.monotonic() - t0) * 1000.0
    if status != 200:
        print(
            f"\n[FAIL] Step 1 (/health): Expected HTTP 200, got {status}: {_preview(health, token)}",
            file=sys.stderr,
        )
        return 1
    print(f"[PASS] Step 1 (/health in {dt_ms:.0f} ms): {_preview(health, token)}")

    # Check 2: Authenticated Telemetry Write (POST /v1/events -> Cloud Logging & BigQuery)
    now_iso = datetime.now(timezone.utc).isoformat()  # noqa: UP017 - datetime.UTC needs 3.11
    events_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "events": [
            {
                "event_id": f"evt-{uuid.uuid4().hex[:16]}",
                "schema_version": "1.0",
                "workstream": "kr2-gk-database",
                "backend": "tacc_slurm",
                "run_id": smoke_run_id,
                "campaign_id": "pilot_50",
                "solver": "GX",
                "fidelity": "low",
                "event_type": "running",
                "ts_utc": now_iso,
                "retry_count": 0,
                "walltime_resubmits": 0,
                "environment": "preflight",
                "is_synthetic": True,
                "payload": {
                    "smoke_test": True,
                    "source_hostname": host_slug,
                    "note": "TACC login node outbound HTTPS smoke test to Cloud Run + BigQuery",
                },
            }
        ],
    }
    t0 = time.monotonic()
    try:
        status, ev_resp, _ = _call(
            http_call, "POST", f"{base_url}/v1/events", token, events_payload, DEFAULT_TIMEOUT_S
        )
    except Exception as exc:
        print(f"\n[FAIL] Step 2 (POST /v1/events): {_preview(str(exc), token)}", file=sys.stderr)
        return 1
    dt_ms = (time.monotonic() - t0) * 1000.0
    try:
        accepted = int(ev_resp.get("accepted", 0))
    except (TypeError, ValueError):
        accepted = 0
    if status != 200 or accepted < 1:
        print(
            f"\n[FAIL] Step 2 (POST /v1/events): Expected HTTP 200 with accepted=1, got {status}: "
            f"{_preview(ev_resp, token)}\n"
            "Hint: Verify AURORA_STEER_BEARER_TOKEN matches the token deployed to Cloud Run.",
            file=sys.stderr,
        )
        return 1
    print(
        f"[PASS] Step 2 (POST /v1/events in {dt_ms:.0f} ms): Telemetry event accepted "
        f"({_preview(ev_resp, token)}) -> visible in BigQuery & Dashboard as run_id='{smoke_run_id}'"
    )

    if skip_triage:
        print("\n[SUCCESS] Outbound HTTPS & Telemetry write verified (--skip-triage set).")
        return 0

    # Check 3: Authenticated triage round-trip (Gemini + guardrails), same route as tacc-driver
    triage_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "workstream": "kr2-gk-database",
        "backend": "tacc_slurm",
        "skill_name": SKILL_NAME,
        "run_id": smoke_run_id,
        "campaign_id": "pilot_50",
        "solver": "GX",
        "fidelity": "low",
        "environment": "preflight",
        "failure_type": "TIMESTEP_COLLAPSE",
        "retry_count": 0,
        "walltime_resubmits": 0,
        "sim_time_reached": 14.2,
        "sim_time_target": 200.0,
        "current_parameters": {
            "nx": 64,
            "ny": 32,
            "nz": 48,
            "dt_max": 0.05,
            "cfl_fraction": 0.9,
            "hyperdiffusion_coeff": 0.1,
        },
        "scalar_metrics": {"cfl_max": 1.45, "dt_last": 0.0001},
        "log_tail": [
            f"[{host_slug}] Step 142: CFL violation / timestep collapse detected, dt=0.05 exceeds limit"
        ],
    }
    t0 = time.monotonic()
    try:
        status, directive, headers = _call(
            http_call, "POST", f"{base_url}{TRIAGE_PATH}", token, triage_payload, TRIAGE_TIMEOUT_S
        )
    except Exception as exc:
        print(f"\n[FAIL] Step 3 (POST {TRIAGE_PATH}): {_preview(str(exc), token)}", file=sys.stderr)
        return 1
    dt_ms = (time.monotonic() - t0) * 1000.0
    if status != 200 or "action" not in directive:
        print(
            f"\n[FAIL] Step 3 (POST {TRIAGE_PATH}): Expected HTTP 200 directive, got {status}: "
            f"{_preview(directive, token)}",
            file=sys.stderr,
        )
        return 1
    if headers is None:
        print(
            f"[WARN] Step 3: this transport returns no headers, so {DECISION_HEADER} (whether the "
            "LLM actually answered) cannot be verified.",
            file=sys.stderr,
        )
    else:
        decision = headers.get(DECISION_HEADER.lower())
        if decision != "llm":
            print(
                f"\n[FAIL] Step 3 (POST {TRIAGE_PATH}): {DECISION_HEADER}={decision!r}, expected "
                f"'llm'. The directive came from a fallback path, not the model: "
                f"{_preview(directive, token)}",
                file=sys.stderr,
            )
            return 1
    print(
        f"[PASS] Step 3 (POST {TRIAGE_PATH} in {dt_ms:.0f} ms): Received guarded directive:\n"
        f"       action={directive.get('action')!r}, "
        f"parameter_modifications={directive.get('parameter_modifications')}, "
        f"confidence={directive.get('confidence')!r}"
    )
    print("\n==================================================================")
    print(" ALL 3 TACC -> CLOUD RUN SMOKE CHECKS PASSED!")
    print(f" Check your Cloud Run Dashboard for run_id: {smoke_run_id}")
    print(
        " (Use GET /v1/events/summary or view BigQuery `events` table directly if filtering by environment='preflight')"
    )
    print("==================================================================")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify outbound HTTPS connectivity, auth, BigQuery telemetry write, and Gemini triage from a TACC login node to Cloud Run."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("AURORA_STEER_URL", "").strip(),
        help="Cloud Run aurora-steer base URL (or set AURORA_STEER_URL)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Bearer token. Discouraged: visible to other users via ps. Prefer "
            "AURORA_STEER_BEARER_TOKEN (AURORA_STEER_TOKEN is read as a fallback)."
        ),
    )
    parser.add_argument(
        "--skip-triage",
        action="store_true",
        help="Only test /healthz and /v1/events (skip the triage Gemini call)",
    )
    args = parser.parse_args(argv)

    if not args.url.strip():
        print(
            "ERROR: Missing --url or AURORA_STEER_URL environment variable.\n"
            "Example:\n"
            "  export AURORA_STEER_URL='https://aurora-steer-xxxx-uc.a.run.app'\n"
            "  export AURORA_STEER_BEARER_TOKEN='<token-from-env>'\n"
            "  python3 scripts/tacc_smoke_test.py",
            file=sys.stderr,
        )
        return 2

    if args.token:
        print(
            "[WARN] --token is visible to other users of this login node (ps); prefer "
            "AURORA_STEER_BEARER_TOKEN.",
            file=sys.stderr,
        )
    token = resolve_token(args.token)
    if not token:
        print(
            "[WARN] No bearer token found (AURORA_STEER_BEARER_TOKEN / AURORA_STEER_TOKEN); "
            "authenticated steps will fail.",
            file=sys.stderr,
        )
    return run_smoke_test(args.url, token or None, skip_triage=args.skip_triage)


if __name__ == "__main__":
    raise SystemExit(main())
