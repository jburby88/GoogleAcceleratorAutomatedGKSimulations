"""Visual design system, Zinc theme variables, and Plotly layout for the KR2 dashboard."""

from __future__ import annotations

from typing import Any

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


def build_dashboard_css(*, is_dark: bool = True) -> str:
    """Return theme-aware SaaS CSS following the Unified Data App Design System."""
    bg = "#09090b" if is_dark else "#f8fafc"
    bg_subtle = "#111115" if is_dark else "#f1f5f9"
    card = "#121217" if is_dark else "#ffffff"
    card_hover = "#18181f" if is_dark else "#f8fafc"
    border = "#23232b" if is_dark else "#e2e8f0"
    border_subtle = "#1a1a22" if is_dark else "#f1f5f9"
    text = "#fafafa" if is_dark else "#09090b"
    text_muted = "#a1a1aa" if is_dark else "#64748b"
    text_dim = "#71717a" if is_dark else "#94a3b8"
    green = "#22c55e" if is_dark else "#15803d"
    green_bg = "rgba(34, 197, 94, 0.14)" if is_dark else "rgba(22, 163, 74, 0.10)"
    amber = "#f59e0b" if is_dark else "#b45309"
    amber_bg = "rgba(245, 158, 11, 0.14)" if is_dark else "rgba(217, 119, 6, 0.10)"
    red = "#ef4444" if is_dark else "#b91c1c"
    red_bg = "rgba(239, 68, 68, 0.14)" if is_dark else "rgba(220, 38, 38, 0.10)"
    blue = "#3b82f6" if is_dark else "#1d4ed8"
    blue_bg = "rgba(59, 130, 246, 0.14)" if is_dark else "rgba(37, 99, 235, 0.10)"
    shadow = (
        "0 4px 20px rgba(0, 0, 0, 0.35)"
        if is_dark
        else "0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04)"
    )

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400..700;1,9..40,400..700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: #2563eb;
    --ut-orange: #BF5700;
    --green: {green};
    --green-muted: {green_bg};
    --amber: {amber};
    --amber-muted: {amber_bg};
    --red: {red};
    --red-muted: {red_bg};
    --blue: {blue};
    --blue-muted: {blue_bg};
    --shadow: {shadow};
    --radius: 12px;
}}

/* Hide default Streamlit chrome */
header[data-testid="stHeader"], #MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton,
div[data-testid="stSidebarCollapsedControl"] {{
    display: none !important;
}}

/* Global app container */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main,
.block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    padding: 1.75rem 2.5rem 3.5rem !important;
    max-width: 1400px !important;
}}

/* Hide duplicate hidden st.title used for accessibility/test parity */
h1 {{
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.025em !important;
    color: var(--text) !important;
    margin-bottom: 0.2rem !important;
}}

h2, h3, [data-testid="stSubheader"] {{
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    color: var(--text) !important;
}}

/* Refined inline code pills instead of harsh default grey boxes */
code {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.84em !important;
    background: var(--bg-subtle) !important;
    color: var(--text) !important;
    border: none !important;
    padding: 1px 4px !important;
}}
pre code {{
    background: transparent !important;
    padding: 0 !important;
}}

/* Brand banner */
.hero-banner {{
    background: var(--card);
    border: 1px solid var(--border);
    border-top: 3px solid var(--ut-orange);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.85rem;
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}}
.hero-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.73rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ut-orange);
    margin-bottom: 0.25rem;
}}
.hero-live-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 3px var(--green-muted);
    display: inline-block;
}}
.hero-meta-pills {{
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}}
.meta-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-muted);
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 999px;
}}

/* Selectbox z-index & polish */
div[data-baseweb="select"] {{
    position: relative;
    z-index: 30;
}}
div[data-baseweb="select"] > div {{
    background-color: var(--card) !important;
    border-color: var(--border) !important;
    border-radius: 9px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}}

/* KPI Metric Cards */
div[data-testid="stColumn"]:has([data-testid="stMetric"]) {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.25rem 1rem !important;
    box-shadow: var(--shadow);
    transition: transform 0.16s ease, border-color 0.16s ease, background-color 0.16s ease;
}}
div[data-testid="stColumn"]:has([data-testid="stMetric"]):hover {{
    transform: translateY(-2px);
    border-color: var(--accent);
    background: var(--card-hover);
}}
[data-testid="stMetricLabel"] {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    margin-top: 0.25rem !important;
}}
[data-testid="stMetricValue"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    letter-spacing: -0.03em !important;
}}

/* Observed vs Modelled Badges */
.kpi-badge-observed {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--green-muted);
    color: var(--green);
    border: 1px solid rgba(34, 197, 94, 0.25);
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}
.kpi-badge-estimated {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--amber-muted);
    color: var(--amber);
    border: 1px solid rgba(245, 158, 11, 0.25);
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}}

/* Segmented Pill Tabs */
[data-baseweb="tab-list"] {{
    gap: 6px !important;
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-top: 0.75rem !important;
    margin-bottom: 1rem !important;
}}
button[data-baseweb="tab"] {{
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.15rem !important;
    border: 1px solid transparent !important;
    border-radius: 7px !important;
    transition: all 0.15s ease !important;
}}
button[data-baseweb="tab"]:hover {{
    color: var(--text) !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--text) !important;
    background: var(--card) !important;
    border-color: var(--border) !important;
    box-shadow: var(--shadow) !important;
}}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{
    display: none !important;
}}

/* Card headers & summary panels */
.panel-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.15rem 1.35rem;
    box-shadow: var(--shadow);
    margin-bottom: 0.9rem;
}}
.panel-subtitle {{
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: -0.35rem;
    margin-bottom: 0.75rem;
}}
.stat-grid {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 0.5rem;
    margin-bottom: 0.75rem;
}}
.stat-box {{
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 0.75rem 0.95rem;
}}
.stat-box-label {{
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
.stat-box-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin-top: 0.2rem;
}}

/* HTML Data Table — full horizontal scroll support on all viewports */
.data-table-wrap {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
    width: 100%;
    max-width: 100%;
    scrollbar-width: thin;
    scrollbar-color: var(--border) var(--bg-subtle);
    box-shadow: var(--shadow);
    margin-bottom: 0.85rem;
}}
.data-table-wrap::-webkit-scrollbar {{
    height: 8px;
}}
.data-table-wrap::-webkit-scrollbar-track {{
    background: var(--bg-subtle);
    border-radius: 0 0 var(--radius) var(--radius);
}}
.data-table-wrap::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 999px;
}}
.data-table-wrap::-webkit-scrollbar-thumb:hover {{
    background: var(--text-dim);
}}
.data-table {{
    width: 100%;
    min-width: 760px;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.81rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.65rem 1rem;
    color: var(--text-muted);
    background: var(--bg-subtle);
    font-weight: 600;
    font-size: 0.71rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
    white-space: nowrap;
}}
.data-table td {{
    padding: 0.65rem 1rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    white-space: nowrap;
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}
.data-table tbody tr:hover td {{
    background: var(--card-hover);
}}

/* Status badges */
.badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--blue); background: var(--blue-muted); }}

/* Ask Aurora Top-Right Popover Drawer — zero red focus outlines */
div[data-testid="stPopover"] > button,
div[data-testid="stPopover"] > button:focus,
div[data-testid="stPopover"] > button:focus-visible,
div[data-testid="stPopover"] > button:active,
div[data-testid="stPopover"] > button[aria-expanded="true"] {{
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 9px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 0.9rem !important;
    box-shadow: var(--shadow) !important;
    outline: none !important;
    transition: border-color 0.15s ease, background-color 0.15s ease !important;
}}
div[data-testid="stPopover"] > button:hover,
div[data-testid="stPopover"] > button[aria-expanded="true"] {{
    border-color: var(--accent) !important;
    background: var(--card-hover) !important;
    color: var(--text) !important;
}}
div[data-testid="stPopoverBody"] {{
    width: min(560px, 94vw) !important;
    min-width: min(520px, 92vw) !important;
    max-width: 560px !important;
    background: var(--card) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.95rem 1.05rem !important;
    box-shadow: 0 24px 56px rgba(0, 0, 0, 0.6) !important;
}}

/* Override default red focus border on chat input & buttons inside Ask Aurora */
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInputContainer"] {{
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}}
div[data-testid="stChatInput"]:focus-within,
div[data-testid="stChatInput"] > div:focus-within,
div[data-testid="stChatInputContainer"]:focus-within {{
    border-color: var(--accent) !important;
    outline: none !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}}
div[data-testid="stChatInput"] textarea {{
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
}}
div[data-testid="stPopoverBody"] div[data-testid="stButton"] > button {{
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 2.1rem !important;
    padding: 0.4rem 0.75rem !important;
    font-size: 0.79rem !important;
    line-height: 1.35 !important;
    background: var(--bg-subtle) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}}
div[data-testid="stPopoverBody"] div[data-testid="stButton"] > button:hover {{
    color: var(--text) !important;
    border-color: var(--accent) !important;
    background: var(--card-hover) !important;
}}
.aurora-drawer-head {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    padding-bottom: 0.55rem;
    margin-bottom: 0.45rem;
    border-bottom: 1px solid var(--border);
}}
.aurora-drawer-title {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text);
}}
.aurora-drawer-sub {{
    font-size: 0.73rem;
    color: var(--text-muted);
    margin-top: 1px;
}}
.chat-engine-pill {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    font-weight: 600;
    color: var(--blue);
    background: var(--blue-muted);
    border: 1px solid rgba(59, 130, 246, 0.25);
    padding: 2px 8px;
    border-radius: 999px;
    white-space: nowrap;
}}

/* Sleek chat message cards — hide default cartoon avatars & tighten vertical rhythm */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessage"] > div:first-child:has(svg) {{
    display: none !important;
}}
[data-testid="stChatMessage"] {{
    background: var(--bg-subtle) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.7rem 0.9rem !important;
    margin-bottom: 0.55rem !important;
    font-size: 0.84rem !important;
    line-height: 1.5 !important;
}}
[data-testid="stChatMessage"] p {{
    margin-bottom: 0.4rem !important;
    font-size: 0.84rem !important;
}}
[data-testid="stChatMessage"] ul, [data-testid="stChatMessage"] ol {{
    margin: 0.3rem 0 0.45rem 1.1rem !important;
    padding-left: 0.2rem !important;
}}
[data-testid="stChatMessage"] li {{
    margin-bottom: 0.2rem !important;
}}
.aurora-msg-role {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--blue);
    margin-bottom: 0.25rem;
}}
.aurora-msg-role.user {{
    color: var(--ut-orange);
}}
.aurora-msg-engine {{
    font-weight: 500;
    color: var(--text-dim);
    font-size: 0.64rem;
    letter-spacing: 0.01em;
}}

/* Animated Thinking indicator inside assistant bubble */
@keyframes auroraPulse {{
    0%, 80%, 100% {{ opacity: 0.25; transform: scale(0.85); }}
    40% {{ opacity: 1; transform: scale(1.1); }}
}}
.aurora-thinking {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.2rem 0;
    color: var(--text-muted);
    font-size: 0.83rem;
    font-weight: 500;
}}
.aurora-thinking-dots {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
}}
.aurora-thinking-dots span {{
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--blue);
    display: inline-block;
    animation: auroraPulse 1.3s infinite ease-in-out both;
}}
.aurora-thinking-dots span:nth-child(1) {{ animation-delay: -0.26s; }}
.aurora-thinking-dots span:nth-child(2) {{ animation-delay: -0.13s; }}
.aurora-thinking-dots span:nth-child(3) {{ animation-delay: 0s; }}

[data-testid="stChatMessage"] table {{
    width: 100% !important;
    border-collapse: collapse !important;
    margin: 0.45rem 0 !important;
}}

[data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {{
    border: 1px solid var(--border) !important;
    padding: 0.32rem 0.55rem !important;
    font-size: 0.78rem !important;
}}

[data-testid="stChatMessage"] th {{
    background: var(--card) !important;
    font-weight: 600 !important;
}}

[data-testid="stHorizontalBlock"] {{
    gap: 1.1rem !important;
}}
</style>
"""


# Default dark CSS exported for callers/tests importing `DASHBOARD_CSS`.
DASHBOARD_CSS = build_dashboard_css(is_dark=True)


def plot_layout(*, is_dark: bool = True) -> dict[str, Any]:
    """Return consistent Plotly styling for dark and light modes."""
    axis_color = "#a1a1aa" if is_dark else "#475569"
    grid_color = "rgba(255, 255, 255, 0.06)" if is_dark else "rgba(15, 23, 42, 0.06)"
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "DM Sans, sans-serif", "color": axis_color, "size": 12},
        "margin": {"l": 12, "r": 12, "t": 36, "b": 12},
        "height": 320,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0.0,
            "title": {"text": ""},
        },
        "xaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 11, "color": axis_color},
        },
        "yaxis": {
            "gridcolor": grid_color,
            "zerolinecolor": grid_color,
            "tickfont": {"size": 11, "color": axis_color},
        },
    }
