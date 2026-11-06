# `platform/aurora-steer` — Shared Cloud Run Steering & Telemetry Brain

`aurora-steer` is a stateless FastAPI + **Google ADK 2.0 (`google-adk`)** service deployed on Google Cloud Run that serves all workstreams (`kr1-optimizer-orchestration`, `kr2-gk-database`, `kr3-transport-consistent-design`) by dynamically loading Open Knowledge Framework (OKF 2.0) skills by `skill_name`.

## Endpoints

- `GET /healthz` (unauthenticated): readiness, active Vertex AI model (`MODEL_ID`), OKF skill health. Returns **HTTP 503** (`"status": "degraded"`) when the OKF skill or the model configuration is invalid. The service also validates both at startup and refuses to start, so a broken revision never takes traffic.
- `POST /triage` and `POST /v1/skills/{skill_name}/triage` (the path's skill wins over the body's `skill_name`): accept a `<2048`-byte `DiagnosticSummary` (`Bearer` auth, optional `Idempotency-Key`). The flow is:
  1. Bounded body read (413 beyond the limit) and strict schema validation (422): strict JSON numbers, safe identifiers, no NaN/Inf, typed `previous_directives`, control characters stripped from `log_tail`.
  2. `retry_count >= 2` → `escalate_to_human` with 0 LLM calls (3-strike circuit breaker). A per-run LLM budget (`MAX_LLM_TRIAGES_PER_RUN`, default 6 per `LLM_RUN_BUDGET_WINDOW_S` = 24 h) does not trust the client's `retry_count`.
  3. One Google ADK `InMemoryRunner` session (async API) against Vertex AI with structured `ActionDirective` output. It runs under a `VERTEX_TIMEOUT_S` deadline (default 60 s, below the HPC client's 100 s) and retries transient Vertex errors (429/5xx) up to `VERTEX_MAX_ATTEMPTS` (default 2) within that deadline. The telemetry is presented to the model as untrusted data.
  4. **Reject-only guardrails** on the model output:
     - the failure type's runbook action policy (`okf/manifest.yaml` `allowed_actions`; `resubmit_walltime` is never accepted);
     - allowlisted parameters within `[min, max]`;
     - only `warm_restart_allowed` parameters (`dt_max`, `cfl_fraction`) on a checkpoint restart;
     - no no-op patches.
     Any violation, timeout, error or malformed output becomes `escalate_to_human`.
  5. The decision is recorded (DuckDB + structured log + BigQuery) and returned. The `X-Aurora-Decision` response header is one of `llm`, `cached`, `circuit_breaker`, `budget_exhausted`, `guardrail_rejected`, `llm_error`, `llm_timeout`.
  - Idempotency: a repeated `Idempotency-Key` replays the stored decision, and concurrent duplicates share one LLM call. Reusing a key for a different failure returns 409. Transient LLM failures are not cached.
- `POST /v1/events`: batches of `1..500` lifecycle events (each `<1024` bytes, body `<=512 KB`), deduplicated by `event_id`, written to DuckDB, structured logs and BigQuery (`insertAll`; row errors are logged at `ERROR` and counted in `/healthz` `bq_insert_failures_total`). Events whose `environment` is not `prod` are always `is_synthetic = true`.
- `GET /v1/events/summary`: the canonical KPI views (`platform/telemetry/02_views.sql`) over **this instance's local DuckDB store only** (debug aid; BigQuery is authoritative).
- `/docs`, `/redoc`, `/openapi.json` are served locally, and on Cloud Run only with `ENABLE_API_DOCS=1`.

Every response carries `X-Request-Id`: the client's value if well-formed, otherwise a generated one. Application logs are one JSON object per line with `severity`, `message`, `request_id` and Cloud Trace correlation.

## Configuration (environment)

| Variable | Default | Notes |
| :--- | :--- | :--- |
| `MODEL_ID` | `gemini-4-argon` | Vertex AI Gemini model id (GA on Vertex AI, `global` endpoint; `gemini-3.8-flash` is an optional lower-cost override), validated at startup: bare `gemini-*` id, `publishers/google/models/<id>`, or a full `projects/.../locations/.../(publishers/google/models/<id>\|endpoints/<n>)` name. |
| `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` | project from ADC, `global` | The service always uses Vertex AI (ADC); it never uses API keys. |
| `AURORA_STEER_BEARER_TOKEN` | (required) | Secret Manager token(s), comma-separated for rotation. Without it every protected endpoint returns 503. |
| `VERTEX_TIMEOUT_S`, `VERTEX_MAX_ATTEMPTS` | `60`, `2` | Deadline must stay `<= 80` s: the HPC client's read timeout (100 s) minus the BigQuery audit write that follows. A timeout escalates to a human. |
| `RATE_LIMIT_TRIAGE`, `RATE_LIMIT_EVENTS` | `120`, `5000` per 60 s | Per instance. |
| `MAX_LLM_TRIAGES_PER_RUN`, `LLM_RUN_BUDGET_WINDOW_S` | `6`, `86400` | Server-side runaway-spend guard. |
| `BQ_ENABLED`, `BQ_DATASET` | `0`, `sim_babysitter_telemetry` | BigQuery streaming. |
| `ALLOW_UNAUTH_LOCAL` | `0` | Disables auth only if also `ENVIRONMENT` ∈ {`dev`, `local`, `test`} and not on Cloud Run. |

Deploy with Cloud Run `max_instance_count = 1` (Idempotency-Key dedup, in-flight sharing and the per-run budget are per process) and a startup probe on `/healthz`.

## Development

```bash
.venv/bin/python -m ruff check platform/aurora-steer && .venv/bin/python -m ruff format --check platform/aurora-steer
NO_PROXY=127.0.0.1,localhost .venv/bin/python -m pytest --import-mode=importlib platform/aurora-steer/tests tests/e2e
```

`google-adk` is imported lazily, so the API, schemas, guardrails and OKF loader can be tested without it. When `google-adk` is installed, `test_real_google_adk_pipeline_with_fake_llm` also exercises the real ADK runner with a fake `BaseLlm`.
