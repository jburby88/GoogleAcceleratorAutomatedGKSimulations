"""Constant-time Bearer token authentication for the TACC <-> Cloud Run boundary.

Architectural Context:
- TACC (`Vista` / `LoneStar6`) login-node watchdogs run strictly with Python's
  standard library (`urllib.request`) and do not have a machine OIDC Identity
  Provider for Workload Identity Federation or `google-auth` C-extensions.
- Rather than placing a privileged GCP Service Account private key on a shared
  university HPC filesystem, `aurora-steer` authenticates `/triage` and
  `/v1/events` calls using a Secret Manager Bearer token verified in constant
  time (`hmac.compare_digest` over SHA-256 digests, so neither the token value
  nor its length leaks through timing).
- Supports comma-separated active tokens in `AURORA_STEER_BEARER_TOKEN` for
  zero-downtime secret rotation during multi-day simulation campaigns.
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Header, HTTPException, status

from app.config import is_unauth_local_allowed, resolve_bearer_tokens
from app.observability import log_event


def _digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def verify_bearer_token(authorization: str | None = Header(default=None)) -> None:
    """Validate `Authorization: Bearer <token>` against configured Secret Manager token(s)."""
    active_tokens = resolve_bearer_tokens()

    if not active_tokens:
        if is_unauth_local_allowed():
            return
        log_event("ERROR", "Bearer token is not configured on the server; refusing request")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AURORA_STEER_BEARER_TOKEN is not configured on server.",
        )

    scheme, _, credentials = (authorization or "").partition(" ")
    provided = credentials.strip()
    if scheme.lower() != "bearer" or not provided:
        log_event("WARNING", "Rejected request with missing or malformed Authorization header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization Bearer header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Evaluate every configured token (no short-circuit) to support zero-downtime key
    # rotation without leaking which token matched.
    provided_digest = _digest(provided)
    matched = False
    for expected in active_tokens:
        if hmac.compare_digest(provided_digest, _digest(expected)):
            matched = True

    if not matched:
        log_event("WARNING", "Rejected request with an invalid bearer token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
