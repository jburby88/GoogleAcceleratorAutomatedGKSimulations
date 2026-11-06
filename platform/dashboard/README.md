# `platform/dashboard`: KR2 Telemetry & Operational Audit Dashboard

A read-only Streamlit app for the `kr2-gk-database` workstream. It shows campaign KPIs, runs by backend and
solver, the failure taxonomy, OKF runbook effectiveness, parameter-space coverage and the physicist
escalation queue. It reads only the canonical SQL views (`v_*`) defined in
[`platform/telemetry/02_views.sql`](../telemetry/02_views.sql). It never writes anything.

## Data modes (`DATA_MODE`)

| `DATA_MODE` | Source | Use |
|---|---|---|
| `bigquery` (default when unset) | Views in `` `GOOGLE_CLOUD_PROJECT.BQ_DATASET` `` | Cloud Run (production) |
| `duckdb` | Views inside the DuckDB file at `LOCAL_EVENTS_DUCKDB_PATH`, opened **read-only** | Local inspection of a file written by aurora-steer's `EventStore` |

Any other value fails closed: the app shows a configuration error and loads no data. This includes
`synthetic`, an empty string, different casing (`BigQuery`) and typos. There is no demo or synthetic
data path.

## Prod-only filter

Both modes issue only one kind of statement:

```sql
SELECT * FROM <allowlisted view>
WHERE is_synthetic = FALSE AND environment = 'prod' AND workstream = <bound parameter>
```

- The view must be one of the 8 allowlisted `v_*` views. The workstream (default `kr2-gk-database`)
  is always a bound query parameter, never interpolated. Rows where `is_synthetic` is NULL are excluded.
- BigQuery: the project id and dataset are validated before they are put in the SQL. The project id
  must use the GCP format (6–30 characters, optionally with a `domain.com:` prefix). The dataset must
  match `^[A-Za-z_][A-Za-z0-9_]{0,1023}$`.
- Each BigQuery job sets `maximum_bytes_billed` (`BQ_MAX_BYTES_BILLED`), the label
  `app=aurora-dashboard` (for cost attribution), a 60 s server-side `job_timeout_ms` and a 60 s
  client-side `result()` timeout. Each process creates one `bigquery.Client` and reuses it.
- The DataFrame is built from the result rows. `to_dataframe()` is not used, because it needs
  `db-dtypes`, which is not a dependency. Empty results keep their columns in both modes.
- DuckDB mode never imports aurora-steer. It opens the file read-only for each query. DuckDB does not
  let another process open a file that a writer holds open, so stop aurora-steer first or point
  `LOCAL_EVENTS_DUCKDB_PATH` at a copy of the file.

## What the page shows

- **Campaign selector:** lists the campaign ids found in `v_kpis_campaign`, sorted. The default is
  the campaign most recently updated in `v_run_status`. The KPI cards, the runs, failure taxonomy,
  runbook and coverage panels, and the escalation queue all show only the selected campaign.
- Each `current_status` and `coverage_status` has a fixed colour (`theme.py`), so failed runs are
  always red and never green.
- If no campaign has data yet, the page shows an info message. If the views cannot be loaded (for
  example permission denied, a timeout or the bytes-billed cap), it shows an error banner and writes
  the details to the server log. The browser never receives a traceback.

## Caching

Each view result is cached in the process for 5 minutes (`st.cache_data(ttl=300)`). The cache key is
the data source (mode, project.dataset or DuckDB file, workstream) plus the view name. Page reloads
and widget changes therefore do not re-query BigQuery, and new telemetry appears within 5 minutes.

## Access control (IAP)

The app has no authentication of its own, so never expose it directly. In production, access is
handled by Terraform ([`platform/infra/terraform`](../infra/terraform)): the
`aurora-steer-dashboard` Cloud Run service with direct IAP (`enable_dashboard_iap`), access limited
to `dashboard_authorized_members`, and a dedicated read-only service account `aurora-dashboard-sa`
(BigQuery job user plus data viewer on the telemetry dataset). Terraform also sets the environment
variables below.

## Environment variables

| Variable | Mode | Default | Meaning |
|---|---|---|---|
| `DATA_MODE` | both | `bigquery` | `bigquery` or `duckdb`. Anything else is rejected. |
| `GOOGLE_CLOUD_PROJECT` | bigquery | **required** | Project that holds the telemetry dataset and runs the query jobs. |
| `BQ_DATASET` | bigquery | `sim_babysitter_telemetry` | Dataset that holds the `v_*` views. |
| `BQ_MAX_BYTES_BILLED` | bigquery | `104857600` (100 MiB) | Per-query cap on bytes billed. Must be a positive integer. |
| `LOCAL_EVENTS_DUCKDB_PATH` | duckdb | **required** | Existing DuckDB file written by aurora-steer's `EventStore`. |

Only these names are read. The legacy aliases `GCP_PROJECT_ID` and `BQ_DATASET_ID` are ignored.
Credentials come from Application Default Credentials: the service account on Cloud Run, or
`gcloud auth application-default login` locally.

## Run locally

```bash
# BigQuery (needs ADC with BigQuery job user + data viewer on the dataset)
DATA_MODE=bigquery GOOGLE_CLOUD_PROJECT=<project-id> \
  .venv/bin/python -m streamlit run platform/dashboard/streamlit_app.py

# DuckDB file written by aurora-steer (LOCAL_EVENTS_DUCKDB_PATH set when aurora-steer ran)
DATA_MODE=duckdb LOCAL_EVENTS_DUCKDB_PATH=/path/to/events.duckdb \
  .venv/bin/python -m streamlit run platform/dashboard/streamlit_app.py
```

## Container

Build the image from the repository root with
`docker build -f platform/dashboard/Dockerfile -t aurora-dashboard:<tag> .`

The image is based on `python:3.12-slim-bookworm` and installs only this package. It runs as the
non-root UID 10001 and serves Streamlit headless on port 8080, with usage statistics and the file
watcher disabled.

## Tests

```bash
NO_PROXY=127.0.0.1,localhost .venv/bin/python -m pytest --import-mode=importlib platform/dashboard/tests
```

The tests need no network. `test_dashboard_db.py` covers configuration validation, BigQuery against
a fake client (SQL, parameters, byte cap, label, timeouts, no `to_dataframe()`) and DuckDB against a
temporary file. `test_streamlit_app.py` renders the app headlessly with `streamlit.testing.v1.AppTest`
and a fake client.
