"""Read-only tool functions for querying KR2 canonical telemetry views.

Every tool reads exclusively from `DashboardDataClient` (`db.py`) or its preloaded
all-campaign DataFrames, enforcing `ALLOWED_VIEWS` and `is_synthetic = FALSE AND environment = 'prod'`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import pandas as pd

LOGGER = logging.getLogger("aurora_dashboard.tools")

_TOOL_STATUS_LABELS: dict[str, str] = {
    "list_campaigns": "Thinking · Checking available campaigns...",
    "get_campaign_kpis": "Thinking · Reading campaign KPIs & savings...",
    "query_campaign_runs": "Thinking · Inspecting simulation runs...",
    "get_failure_and_runbook_outcomes": "Thinking · Reviewing failure modes & recoveries...",
    "get_parameter_and_flux_coverage": "Thinking · Checking grid resolution & heat flux...",
    "get_run_timeline": "Thinking · Tracing run event timeline...",
}


def _clean_value(val: Any) -> Any:
    """Convert pandas/numpy scalars into JSON-safe Python primitives."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if hasattr(val, "item") and callable(val.item):
        try:
            return val.item()
        except Exception:  # noqa: BLE001
            return str(val)
    return val


def _df_to_records(df: pd.DataFrame, *, limit: int = 50) -> list[dict[str, Any]]:
    """Serialize up to `limit` DataFrame rows into JSON-clean dictionaries."""
    if df is None or df.empty:
        return []
    records: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        records.append({str(col): _clean_value(row[col]) for col in df.columns})
    return records


def _filter_campaign(df: pd.DataFrame, campaign_id: str) -> pd.DataFrame:
    if df is None or df.empty or "campaign_id" not in df.columns:
        return pd.DataFrame()
    target = str(campaign_id).strip()
    if not target or target.lower() in {"all", "*"}:
        return df
    mask = df["campaign_id"].notna() & (df["campaign_id"].astype(str) == target)
    return df.loc[mask]


class TelemetryToolkit:
    """Fast, read-only toolkit backed by all-campaign telemetry views in memory."""

    def __init__(
        self,
        *,
        client: Any,
        active_campaign_id: str,
        all_kpis_df: pd.DataFrame | None = None,
        all_runs_df: pd.DataFrame | None = None,
        all_tax_df: pd.DataFrame | None = None,
        all_rb_df: pd.DataFrame | None = None,
        all_cov_df: pd.DataFrame | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
        self.active_campaign_id = str(active_campaign_id)
        self.called_tools: list[str] = []
        self._on_status = on_status
        self._full_view_cache: dict[str, pd.DataFrame] = {}
        if all_kpis_df is not None:
            self._full_view_cache["v_kpis_campaign"] = all_kpis_df
        if all_runs_df is not None:
            self._full_view_cache["v_run_status"] = all_runs_df
        if all_tax_df is not None:
            self._full_view_cache["v_failure_taxonomy"] = all_tax_df
        if all_rb_df is not None:
            self._full_view_cache["v_runbook_effectiveness"] = all_rb_df
        if all_cov_df is not None:
            self._full_view_cache["v_param_coverage"] = all_cov_df

    def _record_call(self, name: str) -> None:
        if name not in self.called_tools:
            self.called_tools.append(name)
        if self._on_status is not None:
            try:
                self._on_status(_TOOL_STATUS_LABELS.get(name, "Thinking..."))
            except Exception:  # noqa: BLE001
                pass

    def _get_full_view(self, view_name: str) -> pd.DataFrame:
        if view_name in self._full_view_cache:
            return self._full_view_cache[view_name]
        if self._client is not None and hasattr(self._client, "query_view"):
            try:
                df = self._client.query_view(view_name)
                self._full_view_cache[view_name] = df
                return df
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Tool query_view(%s) failed: %s", view_name, exc)
        return pd.DataFrame()

    def _get_view_for_campaign(self, view_name: str, campaign_id: str) -> pd.DataFrame:
        target = str(campaign_id or self.active_campaign_id).strip()
        full_df = self._get_full_view(view_name)
        return _filter_campaign(full_df, target)

    def list_campaigns(self) -> dict[str, Any]:
        """List all production campaigns in `v_kpis_campaign`, total count, and KPI summaries."""
        self._record_call("list_campaigns")
        full_kpis = self._get_full_view("v_kpis_campaign")
        full_runs = self._get_full_view("v_run_status")
        campaigns_set: set[str] = set()
        if "campaign_id" in full_kpis.columns:
            campaigns_set.update(str(c) for c in full_kpis["campaign_id"].dropna())
        if "campaign_id" in full_runs.columns:
            campaigns_set.update(str(c) for c in full_runs["campaign_id"].dropna())
        campaigns = sorted(campaigns_set) if campaigns_set else [self.active_campaign_id]
        return {
            "total_campaigns": len(campaigns),
            "campaigns": campaigns,
            "active_dashboard_campaign": self.active_campaign_id,
            "campaign_kpis": _df_to_records(full_kpis, limit=25),
        }

    def get_campaign_kpis(self, campaign_id: str = "") -> dict[str, Any]:
        """Return observed KPIs (`v_kpis_campaign`) and modelled compute savings (`v_compute_savings`) for a campaign."""
        self._record_call("get_campaign_kpis")
        target = str(campaign_id or self.active_campaign_id).strip()
        kpis = self._get_view_for_campaign("v_kpis_campaign", target)
        savings = self._get_view_for_campaign("v_compute_savings", target)
        return {
            "campaign_id": target,
            "kpis": _df_to_records(kpis, limit=5),
            "compute_savings": _df_to_records(savings, limit=5),
        }

    def query_campaign_runs(
        self,
        campaign_id: str = "",
        status_filter: str = "",
        solver_filter: str = "",
        backend_filter: str = "",
    ) -> dict[str, Any]:
        """Query `v_run_status` for a campaign (or `all`) with optional status, solver (`GX`/`GENE`), or backend (`tacc_slurm`/`gcp_batch`) filters."""
        self._record_call("query_campaign_runs")
        target = str(campaign_id or self.active_campaign_id).strip()
        df = self._get_view_for_campaign("v_run_status", target)
        if not df.empty and status_filter and "current_status" in df.columns:
            df = df[df["current_status"].astype(str).str.lower() == status_filter.strip().lower()]
        if not df.empty and solver_filter and "solver" in df.columns:
            df = df[df["solver"].astype(str).str.upper() == solver_filter.strip().upper()]
        if not df.empty and backend_filter and "backend" in df.columns:
            df = df[df["backend"].astype(str).str.lower() == backend_filter.strip().lower()]
        return {
            "campaign_id": target,
            "matched_runs_count": len(df),
            "runs": _df_to_records(df, limit=50),
        }

    def get_failure_and_runbook_outcomes(self, campaign_id: str = "") -> dict[str, Any]:
        """Return failure taxonomy (`v_failure_taxonomy`), OKF runbook effectiveness (`v_runbook_effectiveness`), and directive outcomes (`v_directive_outcomes`)."""
        self._record_call("get_failure_and_runbook_outcomes")
        target = str(campaign_id or self.active_campaign_id).strip()
        tax = self._get_view_for_campaign("v_failure_taxonomy", target)
        rb = self._get_view_for_campaign("v_runbook_effectiveness", target)
        directives = self._get_view_for_campaign("v_directive_outcomes", target)
        return {
            "campaign_id": target,
            "failure_taxonomy": _df_to_records(tax, limit=25),
            "runbook_effectiveness": _df_to_records(rb, limit=25),
            "directive_outcomes": _df_to_records(directives, limit=25),
        }

    def get_parameter_and_flux_coverage(self, campaign_id: str = "") -> dict[str, Any]:
        """Return numerical grid resolution (`nx`) and electrostatic heat flux (`heat_flux_es`) coverage from `v_param_coverage`."""
        self._record_call("get_parameter_and_flux_coverage")
        target = str(campaign_id or self.active_campaign_id).strip()
        cov = self._get_view_for_campaign("v_param_coverage", target)
        return {
            "campaign_id": target,
            "coverage_rows": _df_to_records(cov, limit=50),
        }

    def get_run_timeline(self, run_id: str = "", campaign_id: str = "") -> dict[str, Any]:
        """Query `v_timeline` for chronological execution, failure, and OKF steering events of a specific `run_id` or campaign."""
        self._record_call("get_run_timeline")
        target = str(campaign_id or self.active_campaign_id).strip()
        tl = self._get_view_for_campaign("v_timeline", target)
        if tl.empty and run_id:
            tl = self._get_full_view("v_timeline")
        if not tl.empty and run_id and "run_id" in tl.columns:
            tl = tl[tl["run_id"].astype(str) == run_id.strip()]
        return {
            "campaign_id": target,
            "run_id": run_id or "all",
            "events": _df_to_records(tl, limit=50),
        }

    def as_adk_tools(self) -> list[Callable[..., Any]]:
        """Return the 6 bound tool methods for `google.adk.agents.LlmAgent(tools=...)`."""
        return [
            self.list_campaigns,
            self.get_campaign_kpis,
            self.query_campaign_runs,
            self.get_failure_and_runbook_outcomes,
            self.get_parameter_and_flux_coverage,
            self.get_run_timeline,
        ]
