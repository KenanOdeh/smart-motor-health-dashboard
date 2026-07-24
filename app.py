"""Smart Motor & Sensor Health Dashboard.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import timedelta
from html import escape
from io import BytesIO
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from health_logic import (
    SENSOR_LIMITS,
    build_maintenance_queue,
    latest_by_machine,
    normalized_sensor_profile,
    prepare_dataset,
    sensor_condition,
    trend_direction,
    validate_dataset,
)


st.set_page_config(
    page_title="Smart Motor Health Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data" / "motor_sensor_data.csv"
STATUS_COLORS = {
    "Healthy": "#22C55E",
    "Warning": "#F59E0B",
    "Critical": "#EF4444",
}
PRIORITY_COLORS = {
    "Urgent": "#EF4444",
    "Schedule Soon": "#F59E0B",
    "Monitor": "#38BDF8",
    "Routine": "#22C55E",
}
SEVERITY_ORDER = ["None", "Low", "Medium", "High", "Critical"]
SENSOR_OPTIONS = {
    "Temperature": "temperature_c",
    "Vibration": "vibration_mm_s",
    "Current": "current_a",
    "Voltage": "voltage_v",
    "RPM": "rpm",
    "Efficiency": "efficiency_pct",
    "Noise": "noise_db",
    "Power": "power_kw",
    "Torque": "torque_nm",
}
SENSOR_UNITS = {
    "temperature_c": "°C",
    "vibration_mm_s": "mm/s",
    "current_a": "A",
    "voltage_v": "V",
    "rpm": "RPM",
    "efficiency_pct": "%",
    "noise_db": "dB",
    "power_kw": "kW",
    "torque_nm": "N·m",
}
DATA_DICTIONARY = pd.DataFrame(
    [
        ("timestamp", "Date and time of the simulated reading", "datetime"),
        ("machine_id", "Unique motor identifier", "text"),
        ("motor_type", "Motor technology", "text"),
        ("location", "Factory area", "text"),
        ("rpm", "Rotational speed", "revolutions/min"),
        ("voltage_v", "Line voltage", "V"),
        ("current_a", "Electrical current", "A"),
        ("power_kw", "Estimated three-phase active power", "kW"),
        ("torque_nm", "Shaft torque", "N·m"),
        ("temperature_c", "Motor temperature", "°C"),
        ("vibration_mm_s", "Vibration velocity", "mm/s"),
        ("noise_db", "Sound pressure level", "dB"),
        ("efficiency_pct", "Estimated motor efficiency", "%"),
        ("anomaly_type", "Simulated detected fault type", "category"),
        ("severity", "Simulated event severity", "category"),
        ("health_index", "Transparent rule-based condition score", "0–100"),
        ("risk_score", "Rule-based maintenance risk score", "0–100"),
        ("health_status", "Healthy, Warning, or Critical", "category"),
        ("energy_kwh", "Estimated energy represented by each sample", "kWh"),
    ],
    columns=["Column", "Meaning", "Unit / format"],
)


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame:
    return pd.read_csv(DEFAULT_DATA)


@st.cache_data(show_spinner=False)
def load_uploaded_data(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


def render_html(markup: str) -> None:
    """Render HTML as one line so Markdown never turns nested tags into code."""
    clean_markup = " ".join(
        line.strip() for line in dedent(markup).splitlines() if line.strip()
    )
    st.markdown(clean_markup, unsafe_allow_html=True)


def inject_css(dark: bool) -> None:
    if dark:
        palette = {
            "bg": "#07111F",
            "surface": "rgba(14, 27, 45, 0.82)",
            "surface2": "rgba(20, 38, 61, 0.74)",
            "text": "#E8F1FA",
            "muted": "#93A9BE",
            "border": "rgba(125, 211, 252, 0.17)",
            "grid": "rgba(56, 189, 248, 0.06)",
            "shadow": "rgba(0, 0, 0, 0.32)",
            "accent": "#38BDF8",
            "sidebar": "#091524",
        }
    else:
        palette = {
            "bg": "#EEF5FA",
            "surface": "rgba(255, 255, 255, 0.88)",
            "surface2": "rgba(242, 248, 252, 0.92)",
            "text": "#102033",
            "muted": "#52677A",
            "border": "rgba(15, 94, 140, 0.16)",
            "grid": "rgba(15, 94, 140, 0.06)",
            "shadow": "rgba(30, 64, 90, 0.16)",
            "accent": "#0284C7",
            "sidebar": "#F7FBFE",
        }

    render_html(
        f"""
        <style>
        :root {{
            --bg: {palette["bg"]};
            --surface: {palette["surface"]};
            --surface-2: {palette["surface2"]};
            --text: {palette["text"]};
            --muted: {palette["muted"]};
            --border: {palette["border"]};
            --accent: {palette["accent"]};
            --shadow: {palette["shadow"]};
        }}
        html, body, [class*="css"], .stApp {{
            color: var(--text);
        }}
        .stApp {{
            background: var(--bg);
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient({palette["grid"]} 1px, transparent 1px),
                linear-gradient(90deg, {palette["grid"]} 1px, transparent 1px);
            background-size: 42px 42px;
            animation: engineeringGrid 24s linear infinite;
            mask-image: linear-gradient(to bottom, black, transparent 88%);
        }}
        @keyframes engineeringGrid {{
            from {{ background-position: 0 0, 0 0; }}
            to {{ background-position: 84px 42px, 84px 42px; }}
        }}
        [data-testid="stSidebar"] {{
            background: {palette["sidebar"]};
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text);
        }}
        header[data-testid="stHeader"] {{
            background: color-mix(in srgb, {palette["bg"]} 90%, transparent);
        }}
        input, textarea, [data-baseweb="select"] > div,
        [data-baseweb="input"] > div, [data-baseweb="base-input"] {{
            background-color: var(--surface-2) !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }}
        [data-baseweb="popover"], [data-baseweb="menu"] {{
            background-color: {palette["sidebar"]} !important;
            color: var(--text) !important;
        }}
        .block-container {{
            max-width: 1320px;
            padding-top: .7rem;
            padding-bottom: 1.5rem;
        }}
        .hero {{
            position: relative;
            overflow: hidden;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: .7rem;
            padding: .72rem 1rem;
            margin-bottom: .45rem;
            border: 1px solid var(--border);
            border-radius: 15px;
            background: linear-gradient(120deg, var(--surface), var(--surface-2));
            box-shadow: 0 9px 24px var(--shadow);
            animation: enterUp .55s ease both;
        }}
        .hero::after {{
            content: "";
            position: absolute;
            width: 170px;
            height: 170px;
            right: -55px;
            top: -88px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(56,189,248,.24), transparent 68%);
        }}
        .hero h1 {{
            margin: 0;
            color: var(--text);
            font-size: clamp(1.2rem, 2.2vw, 1.75rem);
            letter-spacing: -0.04em;
        }}
        .hero p {{
            color: var(--muted);
            margin: .18rem 0 0 0;
            font-size: .78rem;
        }}
        .hero-right {{
            display: flex;
            align-items: center;
            gap: .7rem;
            z-index: 1;
            white-space: nowrap;
        }}
        .gear {{
            display: inline-block;
            font-size: 1.55rem;
            animation: gearSpin 9s linear infinite;
            filter: drop-shadow(0 0 12px rgba(56,189,248,.45));
        }}
        @keyframes gearSpin {{ to {{ transform: rotate(360deg); }} }}
        .live-pill {{
            padding: .3rem .55rem;
            border-radius: 999px;
            border: 1px solid rgba(34,197,94,.35);
            background: rgba(34,197,94,.10);
            color: #22C55E;
            font-size: .65rem;
            font-weight: 700;
            letter-spacing: .04em;
        }}
        .live-dot {{
            display: inline-block;
            width: 8px;
            height: 8px;
            margin-right: .4rem;
            border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 0 rgba(34,197,94,.5);
            animation: livePulse 1.8s infinite;
        }}
        @keyframes livePulse {{
            70% {{ box-shadow: 0 0 0 9px rgba(34,197,94,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0); }}
        }}
        .section-title {{
            margin: .7rem 0 .38rem;
            font-weight: 800;
            color: var(--text);
            font-size: .98rem;
        }}
        .section-subtitle {{
            margin: -.22rem 0 .5rem;
            color: var(--muted);
            font-size: .76rem;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
            grid-auto-rows: max-content;
            align-items: start;
            gap: .55rem;
            margin: .3rem 0 .55rem;
        }}
        .kpi-card, .machine-card, .maintenance-card, .sensor-card {{
            position: relative;
            overflow: hidden;
            background: var(--surface);
            border: 1px solid var(--border);
            box-shadow: 0 12px 28px var(--shadow);
            backdrop-filter: blur(14px);
            transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
            animation: enterUp .5s ease both;
        }}
        .kpi-card {{
            padding: .65rem .72rem;
            height: auto;
            min-height: 72px;
            border-radius: 12px;
        }}
        .kpi-card::after {{
            content: "";
            position: absolute;
            inset: 0;
            transform: translateX(-120%);
            background: linear-gradient(100deg, transparent, rgba(255,255,255,.08), transparent);
            animation: shimmer 5s ease-in-out infinite;
        }}
        .kpi-card:hover, .machine-card:hover, .maintenance-card:hover, .sensor-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 13px 26px var(--shadow);
            border-color: rgba(56,189,248,.42);
        }}
        @keyframes shimmer {{
            0%, 68% {{ transform: translateX(-120%); }}
            88%, 100% {{ transform: translateX(120%); }}
        }}
        @keyframes enterUp {{
            from {{ opacity: 0; transform: translateY(12px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .kpi-label {{
            color: var(--muted);
            font-size: .64rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .07em;
        }}
        .kpi-value {{
            color: var(--text);
            margin-top: .22rem;
            font-size: 1.26rem;
            line-height: 1;
            font-weight: 850;
        }}
        .kpi-note {{
            color: var(--muted);
            margin-top: .25rem;
            font-size: .66rem;
        }}
        .factory-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
            gap: .5rem;
            margin-bottom: .45rem;
        }}
        .machine-card {{
            padding: .62rem .7rem;
            border-radius: 11px;
            border-left: 3px solid var(--status);
        }}
        .machine-name {{
            display: flex;
            justify-content: space-between;
            font-weight: 800;
            color: var(--text);
        }}
        .machine-status {{
            color: var(--status);
            font-size: .65rem;
        }}
        .machine-details {{
            color: var(--muted);
            margin-top: .38rem;
            font-size: .69rem;
            line-height: 1.4;
        }}
        .sensor-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(125px, 1fr));
            gap: .45rem;
            margin-bottom: .5rem;
        }}
        .sensor-card {{
            border-radius: 10px;
            padding: .55rem .62rem;
            border-top: 2px solid var(--status);
        }}
        .sensor-value {{
            color: var(--text);
            font-size: 1rem;
            font-weight: 800;
            margin-top: .14rem;
        }}
        .maintenance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: .5rem;
            margin-bottom: .55rem;
        }}
        .maintenance-card {{
            border-radius: 11px;
            padding: .65rem .72rem;
            border-left: 3px solid var(--priority);
        }}
        .maintenance-card h4 {{
            margin: 0 0 .3rem;
            color: var(--text);
            font-size: .9rem;
        }}
        .priority-pill {{
            color: var(--priority);
            border: 1px solid var(--priority);
            border-radius: 999px;
            padding: .12rem .4rem;
            font-size: .61rem;
            float: right;
        }}
        .maintenance-card p {{
            color: var(--muted);
            margin: .16rem 0;
            font-size: .69rem;
            line-height: 1.35;
        }}
        [data-testid="stPlotlyChart"] {{
            padding: .18rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--surface);
            box-shadow: 0 7px 18px var(--shadow);
            transition: transform .24s ease, box-shadow .24s ease, filter .24s ease;
            transform-style: preserve-3d;
            animation: enterUp .55s ease both;
        }}
        [data-testid="stPlotlyChart"]:hover {{
            transform: perspective(1000px) rotateX(.6deg) rotateY(-.6deg) translateY(-2px);
            box-shadow: 0 12px 24px var(--shadow);
            filter: brightness(1.04);
        }}
        [data-testid="stDataFrame"] {{
            border: 1px solid var(--border);
            border-radius: 14px;
            overflow: hidden;
        }}
        div[data-testid="stMetric"] {{
            background: var(--surface);
            border: 1px solid var(--border);
            padding: .55rem;
            border-radius: 10px;
        }}
        .disclaimer {{
            padding: .5rem .65rem;
            border: 1px solid var(--border);
            border-radius: 12px;
            color: var(--muted);
            background: var(--surface-2);
            font-size: .68rem;
        }}
        .footer {{
            color: var(--muted);
            font-size: .65rem;
            text-align: center;
            padding-top: 1rem;
        }}
        .stButton > button, .stDownloadButton > button {{
            background: var(--surface-2);
            color: var(--text);
            border-radius: 9px;
            border: 1px solid var(--border);
            transition: transform .18s ease, box-shadow .18s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 18px var(--shadow);
        }}
        @media (max-width: 700px) {{
            .hero {{ align-items: flex-start; }}
            .hero-right {{ flex-direction: column; align-items: flex-end; }}
            .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
        @media (prefers-reduced-motion: reduce) {{
            *, *::before, *::after {{
                animation-duration: .01ms !important;
                animation-iteration-count: 1 !important;
                transition-duration: .01ms !important;
            }}
        }}
        </style>
        """
    )


def chart_layout(figure: go.Figure, dark: bool, height: int = 315) -> go.Figure:
    text = "#E8F1FA" if dark else "#102033"
    muted = "#93A9BE" if dark else "#52677A"
    grid = "rgba(147,169,190,.13)" if dark else "rgba(82,103,122,.13)"
    figure.update_layout(
        height=height,
        margin=dict(l=28, r=18, t=48, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text, family="Inter, Segoe UI, sans-serif"),
        title_font=dict(size=15, color=text),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=muted, size=11),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hoverlabel=dict(bgcolor="#102033" if dark else "#FFFFFF", font_color=text),
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    figure.update_xaxes(gridcolor=grid, zerolinecolor=grid, title_font_color=muted)
    figure.update_yaxes(gridcolor=grid, zerolinecolor=grid, title_font_color=muted)
    return figure


def hero(filtered: pd.DataFrame) -> None:
    start = filtered["timestamp"].min().strftime("%d %b %Y")
    end = filtered["timestamp"].max().strftime("%d %b %Y")
    render_html(
        f"""
        <div class="hero">
          <div>
            <h1>Smart Motor &amp; Sensor Health</h1>
            <p>Condition monitoring • {filtered["machine_id"].nunique()} motors • {start} — {end}</p>
          </div>
          <div class="hero-right">
            <span class="live-pill"><span class="live-dot"></span>SIMULATION ACTIVE</span>
            <span class="gear">⚙️</span>
          </div>
        </div>
        """
    )


def section(title: str, subtitle: str = "") -> None:
    render_html(f'<div class="section-title">{escape(title)}</div>')
    if subtitle:
        render_html(f'<div class="section-subtitle">{escape(subtitle)}</div>')


def kpi_cards(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(
        f"""
        <div class="kpi-card" style="animation-delay:{index * 55}ms">
          <div class="kpi-label">{escape(label)}</div>
          <div class="kpi-value">{escape(value)}</div>
          <div class="kpi-note">{escape(note)}</div>
        </div>
        """
        for index, (label, value, note) in enumerate(items)
    )
    render_html(f'<div class="kpi-grid">{cards}</div>')


def factory_cards(latest: pd.DataFrame) -> None:
    cards = []
    for _, row in latest.sort_values("risk_score", ascending=False).iterrows():
        status = row["health_status"]
        cards.append(
            f"""
            <div class="machine-card" style="--status:{STATUS_COLORS[status]}">
              <div class="machine-name">
                <span>{escape(str(row["machine_id"]))}</span>
                <span class="machine-status">● {escape(status)}</span>
              </div>
              <div class="machine-details">
                {escape(str(row["motor_type"]))} • {escape(str(row["location"]))}<br>
                Health <b>{row["health_index"]:.0f}</b> &nbsp; Risk <b>{row["risk_score"]:.0f}</b><br>
                {row["temperature_c"]:.1f} °C &nbsp; {row["vibration_mm_s"]:.2f} mm/s
              </div>
            </div>
            """
        )
    render_html(f'<div class="factory-grid">{"".join(cards)}</div>')


def sensor_cards(row: pd.Series) -> None:
    sensor_specs = [
        ("Temperature", "temperature_c", "°C"),
        ("Vibration", "vibration_mm_s", "mm/s"),
        ("Current", "current_a", "A"),
        ("Voltage", "voltage_v", "V"),
        ("Speed", "rpm", "RPM"),
        ("Efficiency", "efficiency_pct", "%"),
        ("Noise", "noise_db", "dB"),
    ]
    cards = []
    for label, column, unit in sensor_specs:
        if column in SENSOR_LIMITS:
            condition = sensor_condition(column, float(row[column]))
        else:
            condition = "Healthy"
        cards.append(
            f"""
            <div class="sensor-card" style="--status:{STATUS_COLORS[condition]}">
              <div class="kpi-label">{escape(label)}</div>
              <div class="sensor-value">{row[column]:.1f} <small>{escape(unit)}</small></div>
              <div class="kpi-note">{escape(condition)}</div>
            </div>
            """
        )
    render_html(f'<div class="sensor-grid">{"".join(cards)}</div>')


def condition_summary_html(
    machine_id: str, machine_data: pd.DataFrame, queue_row: pd.Series
) -> str:
    latest = machine_data.sort_values("timestamp").iloc[-1]
    rows = "".join(
        f"<tr><td>{escape(label)}</td><td>{latest[column]:.2f} {escape(unit)}</td></tr>"
        for label, column, unit in [
            ("Temperature", "temperature_c", "°C"),
            ("Vibration", "vibration_mm_s", "mm/s"),
            ("Current", "current_a", "A"),
            ("Voltage", "voltage_v", "V"),
            ("RPM", "rpm", "RPM"),
            ("Efficiency", "efficiency_pct", "%"),
            ("Noise", "noise_db", "dB"),
        ]
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(machine_id)} Condition Summary</title>
<style>
body{{font-family:Arial,sans-serif;max-width:800px;margin:35px auto;color:#142437}}
h1{{border-bottom:4px solid #0284c7;padding-bottom:12px}}
.score{{display:inline-block;padding:14px;margin-right:10px;background:#eef6fb;border-radius:10px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}td{{border:1px solid #dce7ef;padding:9px}}
.note{{font-size:12px;color:#5a6c7d;background:#f4f7f9;padding:10px;border-radius:8px}}
</style></head><body>
<h1>Smart Motor Condition Summary — {escape(machine_id)}</h1>
<p>Generated from the latest simulated reading at {latest["timestamp"]:%d %b %Y %H:%M}.</p>
<div class="score"><b>Health Index</b><br>{latest["health_index"]:.0f}/100</div>
<div class="score"><b>Risk Score</b><br>{latest["risk_score"]:.0f}/100</div>
<div class="score"><b>Condition</b><br>{escape(str(latest["health_status"]))}</div>
<h2>Latest readings</h2><table>{rows}</table>
<h2>Maintenance decision</h2>
<p><b>Priority:</b> {escape(str(queue_row["priority"]))}</p>
<p><b>Detected:</b> {escape(str(queue_row["detected"]))}</p>
<p><b>Recommended action:</b> {escape(str(queue_row["recommended_action"]))}</p>
<p><b>Inspection window:</b> {escape(str(queue_row["inspection_window"]))}</p>
<p><b>Simulated RUL estimate:</b> {int(queue_row["simulated_rul_hours"])} hours</p>
<p class="note">Educational simulation only. Scores and RUL are transparent rule-based
estimates using artificial data, not certified industrial predictions.</p>
</body></html>"""


def reset_filters() -> None:
    for key in [
        "filter_locations",
        "filter_types",
        "filter_health",
        "filter_dates",
        "filter_period",
        "filter_scope",
        "filter_machines",
    ]:
        st.session_state.pop(key, None)


def apply_global_filters(data: pd.DataFrame) -> pd.DataFrame:
    min_date = data["timestamp"].min().date()
    max_date = data["timestamp"].max().date()
    location_options = sorted(data["location"].unique())

    with st.sidebar.expander("Filters (optional)", expanded=False):
        st.caption("Leave these unchanged to view the complete fleet.")
        period = st.selectbox(
            "Time period",
            ["All data", "Last 7 days", "Last 14 days", "Last 30 days", "Custom"],
            key="filter_period",
        )
        if period == "Custom":
            if "filter_dates" not in st.session_state:
                st.session_state["filter_dates"] = (min_date, max_date)
            selected_dates = st.date_input(
                "Date range",
                min_value=min_date,
                max_value=max_date,
                key="filter_dates",
            )
            if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
                start_date, end_date = selected_dates
            else:
                start_date = end_date = selected_dates
        else:
            days = {"Last 7 days": 7, "Last 14 days": 14, "Last 30 days": 30}.get(
                period
            )
            start_date = (
                min_date
                if days is None
                else max(min_date, max_date - timedelta(days=days - 1))
            )
            end_date = max_date

        scope = st.selectbox(
            "Motors",
            ["All motors", "Choose motors"],
            key="filter_scope",
        )

        if "filter_locations" not in st.session_state:
            st.session_state["filter_locations"] = location_options
        else:
            st.session_state["filter_locations"] = [
                item for item in st.session_state["filter_locations"] if item in location_options
            ]
        selected_locations = st.multiselect(
            "Locations",
            location_options,
            key="filter_locations",
        )
        location_data = data[data["location"].isin(selected_locations)]

        available_types = sorted(location_data["motor_type"].unique())
        if "filter_types" not in st.session_state:
            st.session_state["filter_types"] = available_types
        else:
            st.session_state["filter_types"] = [
                item for item in st.session_state["filter_types"] if item in available_types
            ]
        selected_types = st.multiselect(
            "Motor types",
            available_types,
            key="filter_types",
        )
        dependent_data = location_data[location_data["motor_type"].isin(selected_types)]
        available_machines = sorted(dependent_data["machine_id"].unique())

        if scope == "All motors":
            selected_machines = available_machines
            st.caption(f"{len(selected_machines)} motors included")
        else:
            if "filter_machines" not in st.session_state:
                st.session_state["filter_machines"] = available_machines[:1]
            else:
                st.session_state["filter_machines"] = [
                    machine
                    for machine in st.session_state["filter_machines"]
                    if machine in available_machines
                ]
            selected_machines = st.multiselect(
                "Choose motors",
                available_machines,
                key="filter_machines",
            )

        if "filter_health" not in st.session_state:
            st.session_state["filter_health"] = ["Healthy", "Warning", "Critical"]
        selected_health = st.multiselect(
            "Health status",
            ["Healthy", "Warning", "Critical"],
            key="filter_health",
        )
        st.button("↺ Reset filters", width="stretch", on_click=reset_filters)

    mask = (
        data["location"].isin(selected_locations)
        & data["motor_type"].isin(selected_types)
        & data["machine_id"].isin(selected_machines)
        & data["health_status"].isin(selected_health)
        & data["timestamp"].dt.date.between(start_date, end_date)
    )
    return data.loc[mask].copy()


def render_overview(data: pd.DataFrame, dark: bool) -> None:
    latest = latest_by_machine(data)
    anomalies = data[data["anomaly_type"].ne("None")]
    critical = int(latest["health_status"].eq("Critical").sum())
    top_risk = latest.sort_values("risk_score", ascending=False).iloc[0]
    kpi_cards(
        [
            ("Motors monitored", f"{len(latest)}", "Latest available condition"),
            ("Average Health Index", f"{latest['health_index'].mean():.0f}/100", "Across selected motors"),
            ("Critical motors", str(critical), "Require the closest attention"),
            ("Anomaly events", f"{len(anomalies):,}", "Within the selected period"),
            ("Estimated energy", f"{data['energy_kwh'].sum():,.0f} kWh", "Based on sample intervals"),
        ]
    )

    left, right = st.columns([1.35, 1])
    with left:
        section("Interactive factory view", "Hover over a motor to inspect its condition.")
        factory_cards(latest)
    with right:
        section("Current operating decision")
        status = top_risk["health_status"]
        render_html(
            f"""
            <div class="maintenance-card" style="--priority:{STATUS_COLORS[status]}">
              <span class="priority-pill">{escape(status)}</span>
              <h4>{escape(str(top_risk["machine_id"]))} has the highest current risk</h4>
              <p>Risk Score <b>{top_risk["risk_score"]:.0f}/100</b> and Health Index
              <b>{top_risk["health_index"]:.0f}/100</b>.</p>
              <p>Latest condition: {top_risk["temperature_c"]:.1f} °C,
              {top_risk["vibration_mm_s"]:.2f} mm/s vibration,
              {top_risk["current_a"]:.1f} A current.</p>
            </div>
            """
        )

    health_tab, trend_tab = st.tabs(["Fleet health", "Health trend"])
    with health_tab:
        health_counts = (
            latest["health_status"]
            .value_counts()
            .reindex(["Healthy", "Warning", "Critical"], fill_value=0)
            .reset_index()
        )
        health_counts.columns = ["Status", "Motors"]
        fig = px.pie(
            health_counts,
            names="Status",
            values="Motors",
            hole=0.64,
            color="Status",
            color_discrete_map=STATUS_COLORS,
            title="Latest fleet health",
        )
        fig.update_traces(textinfo="label+value", pull=[0, 0.02, 0.05])
        chart_layout(fig, dark, 290)
        st.plotly_chart(fig, width="stretch", key="overview_health")
    with trend_tab:
        daily = (
            data.set_index("timestamp")
            .groupby("machine_id")["health_index"]
            .resample("1D")
            .mean()
            .reset_index()
        )
        fig = px.line(
            daily,
            x="timestamp",
            y="health_index",
            color="machine_id",
            title="Daily Health Index trend",
            labels={"timestamp": "", "health_index": "Health Index", "machine_id": "Motor"},
        )
        fig.add_hline(y=80, line_dash="dot", line_color="#22C55E")
        fig.add_hline(y=55, line_dash="dot", line_color="#EF4444")
        fig.update_yaxes(range=[0, 105])
        chart_layout(fig, dark, 290)
        st.plotly_chart(fig, width="stretch", key="overview_trend")

    with st.expander("Latest priority events", expanded=False):
        if anomalies.empty:
            st.success("No anomaly events match the current filters.")
        else:
            display = anomalies.sort_values("timestamp", ascending=False).head(12)[
                [
                    "timestamp",
                    "machine_id",
                    "anomaly_type",
                    "severity",
                    "health_index",
                    "risk_score",
                ]
            ]
            st.dataframe(display, width="stretch", hide_index=True)


def render_machine_health(data: pd.DataFrame, dark: bool) -> None:
    selector, _ = st.columns([1, 2])
    with selector:
        machine_id = st.selectbox(
            "Choose a motor", sorted(data["machine_id"].unique()), key="machine_profile"
        )
    machine = data[data["machine_id"].eq(machine_id)].sort_values("timestamp")
    latest = machine.iloc[-1]
    queue_row = build_maintenance_queue(machine).iloc[0]
    trend = trend_direction(machine)

    kpi_cards(
        [
            ("Health Index", f"{latest['health_index']:.0f}/100", latest["health_status"]),
            ("Risk Score", f"{latest['risk_score']:.0f}/100", f"{trend} trend"),
            ("Operating condition", latest["health_status"], f"Latest: {latest['timestamp']:%d %b %H:%M}"),
            ("Simulated RUL", f"{queue_row['simulated_rul_hours']:.0f} h", "Educational estimate only"),
            ("Maintenance", queue_row["priority"], queue_row["inspection_window"]),
        ]
    )

    section("Latest sensor readings", "Green is normal, amber needs attention, and red is critical.")
    sensor_cards(latest)

    gauge_col, decision_col = st.columns([.9, 1.1])
    with gauge_col:
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=float(latest["health_index"]),
                title={"text": f"{machine_id} Health Index"},
                number={"suffix": "/100"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": STATUS_COLORS[latest["health_status"]], "thickness": 0.28},
                    "steps": [
                        {"range": [0, 55], "color": "rgba(239,68,68,.24)"},
                        {"range": [55, 80], "color": "rgba(245,158,11,.24)"},
                        {"range": [80, 100], "color": "rgba(34,197,94,.24)"},
                    ],
                    "threshold": {
                        "line": {"color": "#38BDF8", "width": 4},
                        "thickness": 0.75,
                        "value": float(latest["health_index"]),
                    },
                },
            )
        )
        chart_layout(fig, dark, 285)
        st.plotly_chart(fig, width="stretch", key="machine_gauge")
    with decision_col:
        section("What should be done?")
        render_html(
            f"""
            <div class="maintenance-card" style="--priority:{PRIORITY_COLORS[queue_row["priority"]]}">
              <span class="priority-pill">{escape(queue_row["priority"])}</span>
              <h4>{escape(machine_id)}</h4>
              <p><b>Detected:</b> {escape(queue_row["detected"])}</p>
              <p><b>Recommended action:</b> {escape(queue_row["recommended_action"])}</p>
              <p><b>Inspection window:</b> {escape(queue_row["inspection_window"])}</p>
            </div>
            """
        )
        summary = condition_summary_html(machine_id, machine, queue_row)
        st.download_button(
            "Download condition summary",
            summary,
            file_name=f"{machine_id}_condition_summary.html",
            mime="text/html",
            width="stretch",
        )

    trend_tab, technical_tab = st.tabs(["Recent trend", "Technical profile"])
    with trend_tab:
        recent = machine.tail(100)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent["temperature_c"],
                name="Temperature °C",
                line=dict(color="#F97316", width=2.5),
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=recent["timestamp"],
                y=recent["vibration_mm_s"],
                name="Vibration mm/s",
                line=dict(color="#A78BFA", width=2.5),
            ),
            secondary_y=True,
        )
        fig.update_layout(title="Recent thermal and vibration trend")
        fig.update_yaxes(title_text="Temperature °C", secondary_y=False)
        fig.update_yaxes(title_text="Vibration mm/s", secondary_y=True)
        chart_layout(fig, dark, 305)
        st.plotly_chart(fig, width="stretch", key="machine_recent")
    with technical_tab:
        labels, scores = normalized_sensor_profile(latest)
        fig = go.Figure(
            go.Scatterpolar(
                r=scores + [scores[0]],
                theta=labels + [labels[0]],
                fill="toself",
                line_color="#38BDF8",
                fillcolor="rgba(56,189,248,.22)",
                name=machine_id,
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=[100] * (len(labels) + 1),
                theta=labels + [labels[0]],
                line=dict(color="#22C55E", dash="dot"),
                name="Ideal safe profile",
            )
        )
        fig.update_polars(radialaxis=dict(range=[0, 100], showticklabels=True))
        fig.update_layout(title="Sensor condition profile")
        chart_layout(fig, dark, 305)
        st.plotly_chart(fig, width="stretch", key="machine_radar")

    machine_anomalies = machine[machine["anomaly_type"].ne("None")].tail(10)
    with st.expander("Recent anomaly history", expanded=False):
        if machine_anomalies.empty:
            st.success("No anomaly events were recorded for this motor in the selected period.")
        else:
            st.dataframe(
                machine_anomalies[
                    ["timestamp", "anomaly_type", "severity", "health_index", "risk_score"]
                ].sort_values("timestamp", ascending=False),
                width="stretch",
                hide_index=True,
            )
    render_html(
        '<div class="disclaimer">The safe ranges and simulated RUL shown here are '
        "educational rules. A real installation must use manufacturer limits, "
        "calibrated sensors, load context, and verified failure history.</div>"
    )


def render_sensor_trends(data: pd.DataFrame, dark: bool) -> None:
    controls = st.columns([1, 1])
    with controls[0]:
        label = st.selectbox(
            "Choose a sensor",
            list(SENSOR_OPTIONS),
            key="trend_sensor",
        )
    with controls[1]:
        view_mode = st.selectbox(
            "Motor view",
            ["All filtered motors", "One motor", "Compare two motors"],
            key="trend_view_mode",
        )

    available_machines = sorted(data["machine_id"].unique())
    if view_mode == "One motor":
        machines = [
            st.selectbox("Motor", available_machines, key="trend_single_machine")
        ]
    elif view_mode == "Compare two motors":
        compare = st.multiselect(
            "Choose two motors",
            available_machines,
            default=available_machines[:2],
            max_selections=2,
            key="trend_compare_machines",
        )
        if len(compare) < 2:
            st.info("Choose two motors to compare.")
            return
        machines = compare
    else:
        machines = available_machines

    selected = data[data["machine_id"].isin(machines)].sort_values("timestamp")
    option_col1, option_col2, _ = st.columns([1, 1, 2])
    with option_col1:
        moving_average = st.toggle("Smooth trend", value=False)
    with option_col2:
        show_thresholds = st.toggle("Show safe limits", value=True)

    column = SENSOR_OPTIONS[label]
    fig = go.Figure()
    colors = px.colors.qualitative.Safe
    for index, machine_id in enumerate(machines):
        motor = selected[selected["machine_id"].eq(machine_id)]
        values = (
            motor[column].rolling(8, min_periods=1).mean()
            if moving_average
            else motor[column]
        )
        fig.add_trace(
            go.Scatter(
                x=motor["timestamp"],
                y=values,
                name=machine_id,
                line=dict(color=colors[index % len(colors)], width=2),
                hovertemplate=(
                    f"{escape(machine_id)}<br>%{{x|%d %b %H:%M}}"
                    f"<br>{escape(label)}: %{{y:.2f}} {SENSOR_UNITS[column]}<extra></extra>"
                ),
            )
        )
    if show_thresholds and column in SENSOR_LIMITS:
        limit = SENSOR_LIMITS[column]
        normal_edge = (
            limit["normal"][1]
            if limit["direction"] == "high"
            else limit["normal"][0]
        )
        fig.add_hline(y=normal_edge, line_dash="dot", line_color="#F59E0B")
        fig.add_hline(y=limit["critical"], line_dash="dash", line_color="#EF4444")

    fig.update_layout(
        title=f"{label} history" + (" — smoothed" if moving_average else ""),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text=SENSOR_UNITS[column])
    chart_layout(fig, dark, 350)
    st.plotly_chart(fig, width="stretch", key="sensor_trend_chart")

    section("Latest comparison")
    latest = latest_by_machine(selected)
    st.dataframe(
        latest[["machine_id", column, "health_status", "health_index"]],
        width="stretch",
        hide_index=True,
    )


def render_anomalies(data: pd.DataFrame, dark: bool) -> None:
    anomalies = data[data["anomaly_type"].ne("None")].copy()
    if anomalies.empty:
        st.success("No anomaly events match the current filters.")
        return

    most_affected = anomalies["machine_id"].value_counts().idxmax()
    highest = anomalies["severity"].isin(["High", "Critical"]).sum()
    kpi_cards(
        [
            ("Anomaly events", f"{len(anomalies):,}", "Selected period"),
            ("Most affected motor", most_affected, f"{(anomalies['machine_id'] == most_affected).sum()} events"),
            ("High-severity events", str(int(highest)), "High or Critical"),
            ("Anomaly rate", f"{len(anomalies) / len(data) * 100:.1f}%", "Share of selected readings"),
        ]
    )

    summary_tab, timeline_tab, heatmap_tab, records_tab = st.tabs(
        ["Summary", "Timeline", "Heatmap", "Event records"]
    )
    with summary_tab:
        col1, col2 = st.columns(2)
        with col1:
            counts = anomalies["anomaly_type"].value_counts().reset_index()
            counts.columns = ["Anomaly", "Events"]
            fig = px.bar(
                counts,
                x="Events",
                y="Anomaly",
                orientation="h",
                color="Events",
                color_continuous_scale="Turbo",
                title="Anomalies by type",
            )
            fig.update_layout(coloraxis_showscale=False)
            chart_layout(fig, dark, 300)
            st.plotly_chart(fig, width="stretch", key="anomaly_types")
        with col2:
            severity = (
                anomalies["severity"]
                .value_counts()
                .reindex(SEVERITY_ORDER[1:], fill_value=0)
                .reset_index()
            )
            severity.columns = ["Severity", "Events"]
            fig = px.pie(
                severity,
                names="Severity",
                values="Events",
                hole=0.58,
                color="Severity",
                color_discrete_map={
                    "Low": "#38BDF8",
                    "Medium": "#F59E0B",
                    "High": "#F97316",
                    "Critical": "#EF4444",
                },
                title="Severity distribution",
            )
            chart_layout(fig, dark, 300)
            st.plotly_chart(fig, width="stretch", key="anomaly_severity")
    with timeline_tab:
        daily = (
            anomalies.set_index("timestamp")
            .groupby([pd.Grouper(freq="1D"), "machine_id"])
            .size()
            .rename("Events")
            .reset_index()
        )
        fig = px.area(
            daily,
            x="timestamp",
            y="Events",
            color="machine_id",
            title="Anomaly events over time",
            labels={"machine_id": "Motor", "timestamp": ""},
        )
        chart_layout(fig, dark, 330)
        st.plotly_chart(fig, width="stretch", key="anomaly_timeline")
    with heatmap_tab:
        heatmap = pd.crosstab(anomalies["machine_id"], anomalies["anomaly_type"])
        fig = px.imshow(
            heatmap,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Motor × anomaly heatmap",
            labels=dict(x="Anomaly type", y="Motor", color="Events"),
        )
        chart_layout(fig, dark, 330)
        st.plotly_chart(fig, width="stretch", key="anomaly_heatmap")
    with records_tab:
        search = st.text_input(
            "Search records", placeholder="Motor, type, or severity…"
        )
        event_table = anomalies[
            [
                "timestamp",
                "machine_id",
                "location",
                "anomaly_type",
                "severity",
                "temperature_c",
                "vibration_mm_s",
                "health_index",
                "risk_score",
            ]
        ].sort_values("timestamp", ascending=False)
        if search:
            search_mask = event_table.astype(str).apply(
                lambda column: column.str.contains(search, case=False, na=False)
            ).any(axis=1)
            event_table = event_table[search_mask]
        st.dataframe(event_table, width="stretch", hide_index=True, height=360)
        st.download_button(
            "Export anomaly report",
            event_table.to_csv(index=False).encode("utf-8"),
            file_name="anomaly_report.csv",
            mime="text/csv",
        )


def render_maintenance(data: pd.DataFrame) -> None:
    queue = build_maintenance_queue(data)
    urgent = int(queue["priority"].eq("Urgent").sum())
    schedule = int(queue["priority"].eq("Schedule Soon").sum())
    kpi_cards(
        [
            ("Urgent", str(urgent), "Inspect within 24 hours"),
            ("Schedule soon", str(schedule), "Inspect within 7 days"),
            ("Average risk", f"{queue['risk_score'].mean():.0f}/100", "Across selected motors"),
            ("Queue size", str(len(queue)), "One decision per motor"),
        ]
    )

    display = queue.rename(
        columns={
            "machine_id": "Motor",
            "priority": "Priority",
            "health_index": "Health Index",
            "risk_score": "Risk Score",
            "trend": "Trend",
            "detected": "Detected",
            "recommended_action": "Recommended Action",
            "inspection_window": "Inspection Window",
            "simulated_rul_hours": "Simulated RUL (h)",
        }
    )
    queue_tab, action_tab = st.tabs(["Priority queue", "Recommended action"])
    with queue_tab:
        st.dataframe(display, width="stretch", hide_index=True, height=300)
        st.download_button(
            "Download maintenance summary",
            display.to_csv(index=False).encode("utf-8"),
            file_name="maintenance_summary.csv",
            mime="text/csv",
        )
    with action_tab:
        selected_motor = st.selectbox(
            "Choose a motor",
            queue["machine_id"].tolist(),
            key="maintenance_motor",
        )
        row = queue[queue["machine_id"].eq(selected_motor)].iloc[0]
        render_html(
            f"""
            <div class="maintenance-card" style="--priority:{PRIORITY_COLORS[row["priority"]]}">
              <span class="priority-pill">{escape(row["priority"])}</span>
              <h4>{escape(row["machine_id"])}</h4>
              <p><b>Health:</b> {row["health_index"]:.0f}/100 &nbsp;
              <b>Risk:</b> {row["risk_score"]:.0f}/100 &nbsp;
              <b>Trend:</b> {escape(row["trend"])}</p>
              <p><b>Detected:</b> {escape(row["detected"])}</p>
              <p><b>Recommended action:</b> {escape(row["recommended_action"])}</p>
              <p><b>Inspection window:</b> {escape(row["inspection_window"])}</p>
            </div>
            """
        )
    render_html(
        '<div class="disclaimer">Priority is calculated from the latest condition, '
        "recent anomalies, limit violations, and Health Index direction. Simulated "
        "RUL is included for education and must not be treated as a real failure forecast.</div>"
    )


def render_data_explorer(data: pd.DataFrame) -> None:
    kpi_cards(
        [
            ("Filtered rows", f"{len(data):,}", "Ready to explore or export"),
            ("Motors", str(data["machine_id"].nunique()), "Unique IDs"),
            ("Columns", str(len(data.columns)), "Original and derived fields"),
            ("Missing values", f"{int(data.isna().sum().sum()):,}", "Across filtered data"),
        ]
    )
    data_tab, dictionary_tab = st.tabs(["Explore data", "Column guide"])
    with data_tab:
        search = st.text_input(
            "Search all columns", placeholder="Example: MTR-04, Bearing Wear, Critical…"
        )
        explored = data.copy()
        if search:
            mask = explored.astype(str).apply(
                lambda column: column.str.contains(search, case=False, na=False)
            ).any(axis=1)
            explored = explored[mask]
        st.caption(f"Showing {len(explored):,} matching rows")
        st.dataframe(explored, width="stretch", hide_index=True, height=390)
        st.download_button(
            "Export filtered data",
            explored.to_csv(index=False).encode("utf-8"),
            file_name="filtered_motor_sensor_data.csv",
            mime="text/csv",
        )
    with dictionary_tab:
        st.caption("Plain-language meaning and unit for every field.")
        st.dataframe(DATA_DICTIONARY, width="stretch", hide_index=True, height=390)


def main() -> None:
    st.sidebar.markdown("## ⚙️ Motor Health")
    st.sidebar.caption("Simple condition monitoring for six simulated motors")
    dark = st.sidebar.toggle("Dark mode", value=True, key="dark_mode")
    inject_css(dark)

    st.sidebar.markdown("#### 1. Choose a page")
    page_icons = {
        "Overview": "🏠 Overview",
        "Machine Health": "⚙️ Machine Health",
        "Sensor Trends": "📈 Sensor Trends",
        "Anomaly Analysis": "⚠️ Anomaly Analysis",
        "Maintenance": "🛠️ Maintenance",
        "Data Explorer": "📋 Data Explorer",
    }
    page = st.sidebar.radio(
        "Navigation",
        list(page_icons),
        format_func=page_icons.get,
        label_visibility="collapsed",
        key="page_navigation",
    )

    with st.sidebar.expander("Advanced options", expanded=False):
        st.caption("Only use this if you want to replace the included demo data.")
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    source_label = "Included simulated dataset"
    raw = load_default_data()
    if uploaded is not None:
        try:
            candidate = load_uploaded_data(uploaded.getvalue())
            result = validate_dataset(candidate)
            if result.valid:
                raw = candidate
                source_label = uploaded.name
                st.sidebar.success("Uploaded dataset is valid.")
                for warning in result.warnings:
                    st.sidebar.warning(warning)
            else:
                st.sidebar.error("The upload could not be used.")
                for error in result.errors:
                    st.sidebar.caption(f"• {error}")
                st.sidebar.info("Using the included dataset instead.")
        except Exception as exc:
            st.sidebar.error(f"Could not read the upload: {exc}")
            st.sidebar.info("Using the included dataset instead.")

    default_validation = validate_dataset(raw)
    if not default_validation.valid:
        st.error("The included dataset is invalid: " + " ".join(default_validation.errors))
        st.stop()
    data = prepare_dataset(raw)
    st.sidebar.caption(
        f"Ready: {data['machine_id'].nunique()} motors · {len(data):,} records"
    )

    filtered = apply_global_filters(data)
    if filtered.empty:
        st.warning("No records match the current filters. Reset or widen the filters.")
        st.stop()

    hero(filtered)
    page_help = {
        "Overview": "Start here for the fleet condition and the motor that needs attention first.",
        "Machine Health": "Choose one motor to see its condition, sensor readings, and recommended action.",
        "Sensor Trends": "Choose one sensor, then decide whether to view one motor or compare the fleet.",
        "Anomaly Analysis": "Use the tabs to move from a simple summary to detailed event records.",
        "Maintenance": "Review the ranked queue, then choose a motor for its exact maintenance action.",
        "Data Explorer": "Search the readings or open the Column guide to understand the data.",
    }
    st.caption(page_help[page])
    if page == "Overview":
        render_overview(filtered, dark)
    elif page == "Machine Health":
        render_machine_health(filtered, dark)
    elif page == "Sensor Trends":
        render_sensor_trends(filtered, dark)
    elif page == "Anomaly Analysis":
        render_anomalies(filtered, dark)
    elif page == "Maintenance":
        render_maintenance(filtered)
    else:
        render_data_explorer(filtered)

    render_html(
        '<div class="footer">Smart Motor &amp; Sensor Health Dashboard • '
        "Rule-based educational simulation • No live hardware connected</div>"
    )


if __name__ == "__main__":
    main()
