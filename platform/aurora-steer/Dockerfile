# Build from repository root:
#   docker build -f platform/aurora-steer/Dockerfile -t aurora-steer:<git-sha> .
# Supply chain: the base image is pinned to a Debian release line here; CI should
# additionally pin it by digest (python:3.12-slim-bookworm@sha256:...) and rebuild
# on security updates. Python dependencies are bounded to tested major versions
# in pyproject.toml.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    PORT=8080 \
    REPO_ROOT=/srv/ut-austin

WORKDIR /srv/ut-austin

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser

# Copy only what the service needs at runtime (no tests, caches or local artifacts).
COPY platform/aurora-steer/pyproject.toml platform/aurora-steer/README.md /srv/ut-austin/platform/aurora-steer/
COPY platform/aurora-steer/app /srv/ut-austin/platform/aurora-steer/app
COPY platform/telemetry/02_views.sql /srv/ut-austin/platform/telemetry/02_views.sql
COPY simulations/skill /srv/ut-austin/simulations/skill

# Application code and the OKF guardrail files stay root-owned and read-only for the
# runtime user (no `chown`): a compromised process cannot rewrite its own allowlist.
RUN pip install --no-cache-dir /srv/ut-austin/platform/aurora-steer && \
    find /srv/ut-austin -name '__pycache__' -prune -exec rm -rf {} + && \
    chmod -R a+rX,go-w /srv/ut-austin

WORKDIR /srv/ut-austin/platform/aurora-steer
USER 10001:10001

EXPOSE 8080
# Cloud Run ignores HEALTHCHECK (it uses the service's startup probe on /healthz);
# this serves local / compose runs. /healthz returns 503 when the OKF skill or the
# model configuration is broken.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import os, sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%s/healthz' % os.environ.get('PORT', '8080'), timeout=4).status == 200 else 1)"]

# One worker per instance: Idempotency-Key dedup, in-flight sharing and the per-run LLM
# budget live in process memory (Cloud Run max_instance_count = 1 makes them global).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers 1 --no-server-header --timeout-graceful-shutdown 8"]
