# Build from repository root:
#   docker build -f platform/dashboard/Dockerfile -t aurora-dashboard:<tag> .
# Only platform/dashboard is copied and installed: the image carries no aurora-steer code.
FROM python:3.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/tmp \
    PORT=8080

RUN groupadd --gid 10001 appgroup && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /srv/dashboard

# App files only (no tests or local caches); they stay root-owned and read-only for appuser.
COPY platform/dashboard/pyproject.toml \
     platform/dashboard/README.md \
     platform/dashboard/db.py \
     platform/dashboard/theme.py \
     platform/dashboard/prompts.py \
     platform/dashboard/tools.py \
     platform/dashboard/agent.py \
     platform/dashboard/chat.py \
     platform/dashboard/streamlit_app.py \
     /srv/dashboard/
COPY platform/dashboard/.streamlit/config.toml /srv/dashboard/.streamlit/config.toml

RUN pip install --no-cache-dir /srv/dashboard

USER 10001:10001

EXPOSE 8080

# Cloud Run ignores HEALTHCHECK; it is for local `docker run` and other runtimes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/_stcore/health', timeout=4)"]

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false", "--server.fileWatcherType=none"]
