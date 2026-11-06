"""Structured logging and request correlation for Cloud Run / Cloud Logging.

Cloud Run forwards each JSON line on stdout to Cloud Logging and maps the special
fields `severity`, `message` and `logging.googleapis.com/trace`. Application logs
written here never carry `log_stream`, so the BigQuery telemetry sink (which
filters on `jsonPayload.log_stream`) does not pick them up.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import sys
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_trace: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace", default=None)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_write_lock = threading.Lock()


def new_request_context(request_id_header: str | None, cloud_trace_header: str | None) -> str:
    """Bind a request id (client-supplied if well-formed) and Cloud Trace id; return the id."""
    rid = (
        request_id_header
        if request_id_header and _REQUEST_ID_RE.fullmatch(request_id_header)
        else None
    )
    rid = rid or uuid.uuid4().hex
    _request_id.set(rid)
    trace_id = (cloud_trace_header or "").split("/", 1)[0].strip()
    project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    _trace.set(
        f"projects/{project}/traces/{trace_id}"
        if project and re.fullmatch(r"[0-9a-f]{32}", trace_id)
        else None
    )
    return rid


def current_request_id() -> str | None:
    """Request id bound to the current context (None outside a request)."""
    return _request_id.get()


def log_event(severity: str, message: str, **fields: Any) -> None:
    """Write one structured JSON log line (Cloud Logging special fields included)."""
    entry: dict[str, Any] = {
        "severity": severity,
        "message": message,
        "component": "aurora-steer",
        "time": datetime.now(UTC).isoformat(),
    }
    rid = _request_id.get()
    if rid:
        entry["request_id"] = rid
    trace = _trace.get()
    if trace:
        entry["logging.googleapis.com/trace"] = trace
    entry.update(fields)
    line = json.dumps(entry, default=str)
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
