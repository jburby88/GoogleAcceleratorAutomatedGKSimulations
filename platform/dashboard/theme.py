"""Visual styling & Observed-vs-Estimated badge helpers for the KR2 dashboard."""

from __future__ import annotations

PALETTE = {
    "ut_burnt_orange": "#BF5700",
    "google_blue": "#1A73E8",
    "emerald": "#0F9D58",
    "amber": "#F4B400",
    "crimson": "#D93025",
    "slate": "#3C4043",
}

# One fixed colour per `v_run_status.current_status`, so a failed run can never render green.
STATUS_COLORS = {
    "completed": PALETTE["emerald"],
    "running": PALETTE["google_blue"],
    "submitted": PALETTE["slate"],
    "human_resolved": PALETTE["amber"],
    "failed": PALETTE["crimson"],
    "escalated": PALETTE["crimson"],
}

# Same idea for `v_param_coverage.coverage_status`.
COVERAGE_COLORS = {
    "converged": PALETTE["emerald"],
    "escalated": PALETTE["amber"],
    "failed": PALETTE["crimson"],
}

DASHBOARD_CSS = """
<style>
.kpi-badge-observed {
    display: inline-block;
    background: #E6F4EA;
    color: #137333;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
.kpi-badge-estimated {
    display: inline-block;
    background: #FEF7E0;
    color: #B06000;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
"""
