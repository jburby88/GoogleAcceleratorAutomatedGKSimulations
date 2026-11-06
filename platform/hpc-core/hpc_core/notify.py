"""Optional Slack/webhook notifier for human escalations.

The webhook URL is a credential (for Slack the path *is* the secret), so it is never logged or
raised: `requests` exceptions embed the URL in their message, so only the exception class and
the HTTP status are logged. Only https:// endpoints are used (http:// is accepted for loopback
test servers only) and redirects are never followed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 500
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _webhook_url_allowed(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    if not host or parsed.username or parsed.password:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and host in _LOOPBACK_HOSTS


def _escape_slack(text: object, limit: int | None = None) -> str:
    """Escape Slack control characters so model- or log-derived text cannot ping @channel."""
    value = str(text)
    if limit is not None and len(value) > limit:
        value = value[:limit] + "..."
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_escalation_notification(
    run_id: str,
    solver: str,
    reason: str,
    needs_human_dir: Path,
    diagnostic: dict[str, Any] | None = None,
    webhook_url: str | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """Send a short notification to Slack/webhook when a run is escalated to needs_human.

    Never raises an exception so the watchdog/driver queue is never blocked by notification failures.
    """
    try:
        target_url = (webhook_url or os.environ.get("TACC_NOTIFY_WEBHOOK_URL", "")).strip()
        if not target_url:
            return False
        if not _webhook_url_allowed(target_url):
            logger.warning(
                "Escalation webhook for run %s skipped: the URL must use https:// "
                "(value not logged)",
                run_id,
            )
            return False

        diag = diagnostic if isinstance(diagnostic, dict) else {}
        failure_type = diag.get("failure_type", "UNKNOWN")
        retry_count = diag.get("retry_count", 0)
        payload = {
            "text": (
                f"[sim-babysitter] Run `{_escape_slack(run_id, 128)}` ({_escape_slack(solver, 16)}) "
                f"escalated to `needs_human` (failure_type={_escape_slack(failure_type, 32)}, "
                f"retry_count={_escape_slack(retry_count, 8)}).\n"
                f"Reason: {_escape_slack(reason, MAX_REASON_CHARS)}\n"
                f"Folder: `{_escape_slack(needs_human_dir, 400)}`"
            )
        }
        resp = requests.post(
            target_url,
            json=payload,
            timeout=(timeout_s, timeout_s),
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        logger.warning(
            "Escalation webhook for run %s failed: %s (URL not logged)", run_id, type(exc).__name__
        )
        return False
    except Exception as exc:  # noqa: BLE001 - notifications must never break the queue
        logger.warning(
            "Escalation webhook for run %s failed unexpectedly: %s", run_id, type(exc).__name__
        )
        return False

    if 200 <= resp.status_code < 300:
        return True
    logger.warning("Escalation webhook for run %s returned HTTP %s", run_id, resp.status_code)
    return False
