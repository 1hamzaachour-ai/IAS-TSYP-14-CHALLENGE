"""Interactive Streamlit dashboard for the waste-heat-recovery (WHR) + PCM thermal storage plant.

Launch:  streamlit run app.py
On the first launch the dataset and the forecaster are generated automatically if
factory_sensor_data.csv / heat_forecaster.pkl are missing.

Everything on the page is recomputed from the sidebar: the engine-load override re-derives the
exhaust conditions and re-runs the ML forecast, and both dispatch strategies are re-simulated
hour by hour on the selected window with the chosen storage size and energy prices.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import dataset_builder
import train_model
from dataset_builder import PRICE_RANGE, apply_load_override
from optimizer import (MPCDispatcher, PlantConfig, RuleBasedDispatcher, build_hourly_inputs,
                       no_storage_baseline, simulate, summarize)
from storage_simulator import NITRATE_SALT_PCM, ThermalStorageBuffer
from train_model import load_forecaster, load_sensor_data, predict_horizons, regression_metrics

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / dataset_builder.OUTPUT_CSV
MODEL_PATH = ROOT / train_model.MODEL_PATH

MPC_NAME, RULE_NAME, BASE_NAME = MPCDispatcher.name, RuleBasedDispatcher.name, "No storage (direct + ORC)"
HOUR = pd.Timedelta(hours=1)

FONT = '"Source Sans Pro", "Source Sans 3", system-ui, -apple-system, "Segoe UI", sans-serif'
# Validated categorical slots (light / dark steps) + chart chrome. Colour follows the entity:
# facility = blue, ORC = orange, storage = aqua, everywhere on the page.
PALETTE = {
    "light": {
        "facility": "#2a78d6", "orc": "#eb6834", "storage": "#1baf7a", "storage_to_facility": "#eda100",
        "storage_to_orc": "#e87ba4", "bypass": "#898781", "boiler": "#c3c2b7", "actual": "#0b0b0b",
        "forecast_1h": "#2a78d6", "forecast_4h": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#ffffff",
        "wash": "rgba(11, 11, 11, 0.05)",
    },
    "dark": {
        "facility": "#3987e5", "orc": "#d95926", "storage": "#199e70", "storage_to_facility": "#c98500",
        "storage_to_orc": "#d55181", "bypass": "#898781", "boiler": "#52514e", "actual": "#f0efec",
        "forecast_1h": "#3987e5", "forecast_4h": "#d95926", "ink": "#f0efec", "ink2": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835", "surface": "#0e1117",
        "wash": "rgba(255, 255, 255, 0.06)",
    },
}
# (results column, legend label, palette key, +1 above / -1 below the zero line)
ROUTES = [
    ("to_facility_kw", "Facility heating", "facility", 1),
    ("to_orc_kw", "ORC generator", "orc", 1),
    ("to_storage_kw", "Into PCM storage", "storage", 1),
    ("bypass_kw", "Bypass to stack", "bypass", 1),
    ("storage_to_facility_kw", "Storage → facility", "storage_to_facility", -1),
    ("storage_to_orc_kw", "Storage → ORC", "storage_to_orc", -1),
]
PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

st.set_page_config(page_title="WHR Storage Optimizer", page_icon="🔥", layout="wide")


# --------------------------------------------------------------------------------------
# Data, model and simulation (cached)
# --------------------------------------------------------------------------------------
def ensure_artifacts() -> tuple[float, float]:
    """Build the dataset / train the model on first launch; returns a cache key from file mtimes."""
    if not DATA_PATH.exists():
        with st.spinner("First launch: generating 30 days of engine sensor data..."):
            df, _ = dataset_builder.build_dataset()
            df.to_csv(DATA_PATH, index=False)
    if not MODEL_PATH.exists():
        with st.spinner("First launch: training the LightGBM heat forecaster..."):
            train_model.train(DATA_PATH, MODEL_PATH)
    return DATA_PATH.stat().st_mtime, MODEL_PATH.stat().st_mtime


@st.cache_resource(show_spinner=False)
def load_artifacts(key: tuple[float, float]):
    return load_sensor_data(DATA_PATH), load_forecaster(MODEL_PATH)


@st.cache_data(show_spinner=False)
def scenario_frames(load_factor: float, key: tuple[float, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """15-minute telemetry + forecasts and the hourly dispatcher inputs for a load scenario."""
    df, bundle = load_artifacts(key)
    scenario = apply_load_override(df, load_factor)
    forecasts = predict_horizons(bundle, scenario)
    hourly = build_hourly_inputs(scenario, forecasts, profile_until=bundle["test_start"])
    return scenario.join(forecasts), hourly


def rescale_tariff(price, low: float, high: float):
    """Map the recorded tariff shape onto a new off-peak..peak range."""
    lo0, hi0 = PRICE_RANGE
    return low + (price - lo0) / (hi0 - lo0) * (high - low)


@st.cache_data(show_spinner=False)
def run_strategies(key, load_factor, start, end, capacity, initial_soc, tariff, gas_price,
                   orc_capacity, orc_efficiency, horizon_h, grid_co2) -> tuple[dict[str, pd.DataFrame], PlantConfig]:
    _, hourly = scenario_frames(load_factor, key)
    window = hourly.loc[start:end].copy()
    window["price"] = rescale_tariff(window["price"], *tariff)
    cfg = replace(PlantConfig(), gas_price_kwh=gas_price, orc_thermal_capacity_kw=orc_capacity,
                  orc_efficiency=orc_efficiency, grid_co2_kg_kwh=grid_co2,
                  peak_price_threshold=tariff[0] + 0.7 * (tariff[1] - tariff[0]))
    results = {}
    for name in (MPC_NAME, RULE_NAME):
        tank = ThermalStorageBuffer(capacity_kwh=capacity, initial_soc=initial_soc)
        dispatcher = (MPCDispatcher(tank, cfg, horizon_h=horizon_h) if name == MPC_NAME
                      else RuleBasedDispatcher(tank, cfg))
        results[name] = simulate(window, dispatcher, cfg)
    results[BASE_NAME] = no_storage_baseline(window, cfg)
    return results, cfg


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def theme_tokens() -> dict[str, str]:
    return PALETTE["dark" if st.context.theme.type == "dark" else "light"]


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


def style_figure(fig: go.Figure, t: dict, height: int, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=16, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=t["ink2"]),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", traceorder="normal",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=t["ink2"])),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["axis"], font=dict(family=FONT, color=t["ink"])),
    )
    fig.update_xaxes(automargin=True, showgrid=False, showline=True, linecolor=t["axis"], ticks="outside",
                     tickcolor=t["axis"], tickfont=dict(color=t["muted"]), showspikes=True, spikemode="across",
                     spikesnap="cursor", spikethickness=1, spikecolor=t["muted"], spikedash="solid")
    fig.update_yaxes(automargin=True, showgrid=True, gridcolor=t["grid"], gridwidth=1, zeroline=False,
                     showline=False, tickfont=dict(color=t["muted"]),
                     title=dict(text=y_title, font=dict(color=t["muted"], size=12)))
    return fig


def forecast_figure(frame: pd.DataFrame, t: dict) -> go.Figure:
    fig = go.Figure()
    for col, name, key in (("q_in_kw", "Actual Q_in", "actual"),
                           ("q_pred_1h_kw", "Forecast 1 h ahead", "forecast_1h"),
                           ("q_pred_4h_kw", "Forecast 4 h ahead", "forecast_4h")):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[col], name=name, mode="lines",
                                 line=dict(color=t[key], width=2), hovertemplate="%{y:.0f} kW"))
    style_figure(fig, t, height=380, y_title="Recoverable heat Q_in (kW)")
    fig.update_xaxes(hoverformat="%a %d %b, %H:%M")
    return fig


def peak_windows(day: pd.DataFrame, threshold: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous hours whose (rescaled) price is above the peak threshold."""
    windows, start = [], None
    for ts, is_peak in (day["price"] > threshold).items():
        if is_peak and start is None:
            start = ts
        if not is_peak and start is not None:
            windows.append((start, ts))
            start = None
    if start is not None:
        windows.append((start, day.index[-1] + HOUR))
    return windows


def soc_figure(days: dict[str, pd.DataFrame], selected: str, other: str, threshold: float,
               tank: ThermalStorageBuffer, t: dict) -> go.Figure:
    fig = go.Figure()
    for i, (x0, x1) in enumerate(peak_windows(days[selected], threshold)):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=t["wash"], line_width=0, layer="below",
                      annotation_text="Peak tariff" if i == 0 else None, annotation_position="top left",
                      annotation_font=dict(color=t["muted"], size=11))
    for level, label, position in ((tank.soc_max, "SoC max", "bottom right"), (tank.soc_min, "SoC min", "top right")):
        fig.add_hline(y=level * 100, line=dict(color=t["axis"], width=1), annotation_text=f"{label} {level:.0%}",
                      annotation_position=position, annotation_font=dict(color=t["muted"], size=11))
    # Selected strategy in the storage colour; the alternative as a dotted grey line drawn on top,
    # so it stays visible where the two trajectories coincide.
    for name, line in ((selected, dict(color=t["storage"], width=2.5)),
                       (other, dict(color=t["muted"], width=1.5, dash="dot"))):
        res = days[name]
        x = [res.index[0]] + list(res.index + HOUR)
        y = np.concatenate(([res["soc_start"].iloc[0]], res["soc"].to_numpy())) * 100
        fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=line, hovertemplate="%{y:.1f} %"))
        if name == selected:
            fig.add_trace(go.Scatter(x=[x[-1]], y=[y[-1]], mode="markers+text", showlegend=False, hoverinfo="skip",
                                     marker=dict(color=line["color"], size=9, line=dict(color=t["surface"], width=2)),
                                     text=[f"{y[-1]:.0f}%  "], textposition="middle left",
                                     textfont=dict(color=t["ink"], size=12)))
    style_figure(fig, t, height=370, y_title="State of charge")
    fig.update_yaxes(range=[0, 100], ticksuffix="%", dtick=20)
    fig.update_xaxes(tickformat="%H:%M", dtick=3 * 3600 * 1000, hoverformat="%H:%M")
    return fig


def routing_figure(day: pd.DataFrame, t: dict) -> go.Figure:
    fig = go.Figure()
    x = day.index + HOUR / 2
    for col, label, key, sign in ROUTES:
        fig.add_trace(go.Bar(x=x, y=day[col] * sign, name=label, width=0.78 * 3600 * 1000,
                             marker=dict(color=t[key], line=dict(color=t["surface"], width=1)),
                             customdata=day[col], hovertemplate="%{customdata:.0f} kW"))
    fig.add_hline(y=0, line=dict(color=t["axis"], width=1))
    style_figure(fig, t, height=370, y_title="Heat flow (kW)")
    fig.update_layout(barmode="relative", barcornerradius=4)
    fig.update_xaxes(tickformat="%H:%M", dtick=3 * 3600 * 1000, hoverformat="%H:%M")
    return fig


def sankey_figure(res: pd.DataFrame, cfg: PlantConfig, tank: ThermalStorageBuffer, t: dict) -> go.Figure:
    def kwh(col: str) -> float:
        return float(res[col].sum())

    charged, sto_fac, sto_orc = kwh("to_storage_kw"), kwh("storage_to_facility_kw"), kwh("storage_to_orc_kw")
    orc_heat = kwh("to_orc_kw") + sto_orc
    electricity = cfg.orc_efficiency * orc_heat
    storage_losses = (charged * (1 - tank.charge_efficiency)
                      + (sto_fac + sto_orc) * (1 / tank.discharge_efficiency - 1) + kwh("standing_loss_kw"))
    stock_change = (res["soc"].iloc[-1] - res["soc_start"].iloc[0]) * tank.capacity_kwh

    # (label, colour, column x, preferred y): sources | storage | ORC | sinks
    nodes = [("Engine exhaust", t["ink2"], 0.001, 0.30), ("Backup boiler (gas)", t["boiler"], 0.001, 0.82),
             ("Stored at start", t["storage"], 0.001, 0.97), ("PCM storage", t["storage"], 0.30, 0.70),
             ("Facility heating", t["facility"], 0.999, 0.35), ("ORC", t["orc"], 0.55, 0.88),
             ("Electricity", t["orc"], 0.999, 0.84), ("Bypass to stack", t["bypass"], 0.999, 0.91),
             ("Losses", t["boiler"], 0.999, 0.76), ("Left in storage", t["storage"], 0.999, 0.98)]
    links = [(0, 4, kwh("to_facility_kw")), (0, 5, kwh("to_orc_kw")), (0, 3, charged), (0, 7, kwh("bypass_kw")),
             (1, 4, kwh("boiler_kw")), (3, 4, sto_fac), (3, 5, sto_orc), (3, 8, storage_losses),
             (5, 6, electricity), (5, 8, orc_heat - electricity),
             (3, 9, stock_change) if stock_change > 0 else (2, 3, -stock_change)]
    links = [(s, d, v) for s, d, v in links if v > 1.0]
    # Plotly drops nodes without links, which would shift the x/y positions: keep only used nodes
    used = sorted({i for s, d, _ in links for i in (s, d)})
    remap = {old: new for new, old in enumerate(used)}
    nodes = [nodes[i] for i in used]
    links = [(remap[s], remap[d], v) for s, d, v in links]
    throughput = [max(sum(v for s, _, v in links if s == i), sum(v for _, d, v in links if d == i))
                  for i in range(len(nodes))]
    labels = [f"{node[0]}  {throughput[i] / 1000:,.1f} MWh" for i, node in enumerate(nodes)]

    fig = go.Figure(go.Sankey(
        arrangement="snap", valueformat=",.0f", valuesuffix=" kWh",
        node=dict(label=labels, color=[n[1] for n in nodes], x=[n[2] for n in nodes], y=[n[3] for n in nodes],
                  pad=14, thickness=14, line=dict(width=0), hovertemplate="%{label}<extra></extra>"),
        link=dict(source=[s for s, _, _ in links], target=[d for _, d, _ in links], value=[v for *_, v in links],
                  color=[rgba(nodes[d][1], 0.35) for _, d, _ in links]),
        textfont=dict(family=FONT, size=12, color=t["ink"]),
    ))
    fig.update_layout(height=470, margin=dict(l=8, r=8, t=12, b=12), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, color=t["ink"]))
    return fig


def importance_figure(importance: dict[str, float], t: dict, top: int = 10) -> go.Figure:
    items = list(importance.items())[:top][::-1]
    fig = go.Figure(go.Bar(x=[v * 100 for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=t["facility"]), width=0.6, hovertemplate="%{x:.1f}% of gain<extra></extra>"))
    style_figure(fig, t, height=320)
    fig.update_layout(hovermode="closest", barcornerradius=4, margin=dict(l=8, r=16, t=8, b=8))
    fig.update_xaxes(ticksuffix="%", showgrid=True, gridcolor=t["grid"], showspikes=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(color=t["ink2"]))
    return fig


# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
def fmt_delta(value: float, unit: str) -> str:
    if unit == "$":
        return f"{'-' if value < 0 else '+'}${abs(value):,.0f}"
    return f"{value:+,.0f} {unit}"


def daily(series: pd.Series) -> list[float] | None:
    """Daily totals for a KPI sparkline (only when the window spans at least 3 days)."""
    totals = series.resample("1D").sum()
    return totals.round(1).tolist() if len(totals) >= 3 else None


def esc(text: str) -> str:
    """Escape dollar signs so Streamlit markdown does not render them as LaTeX math."""
    return text.replace("$", "\\$")


SHORT_NAMES = {MPC_NAME: "MPC optimizer", RULE_NAME: "Rule-based", BASE_NAME: "No storage"}


def comparison_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """KPIs as rows, strategies as columns (pre-formatted, since units differ per row)."""
    rows = [
        ("Heat recaptured", lambda s: f"{s['heat_recaptured_kwh'] / 1000:,.1f} MWh"),
        ("Recapture rate", lambda s: f"{s['recapture_rate']:.1%}"),
        ("Gas savings", lambda s: f"${s['gas_savings_usd']:,.0f}"),
        ("ORC revenue", lambda s: f"${s['orc_revenue_usd']:,.0f}"),
        ("Total savings", lambda s: f"${s['total_savings_usd']:,.0f}"),
        ("CO2 avoided", lambda s: f"{s['co2_avoided_kg'] / 1000:,.2f} t"),
        ("Boiler heat still needed", lambda s: f"{s['boiler_heat_kwh'] / 1000:,.1f} MWh"),
        ("Heat bypassed", lambda s: f"{s['bypass_kwh'] / 1000:,.2f} MWh"),
        ("Storage SoC at the end", lambda s: "-" if np.isnan(s["final_soc"]) else f"{s['final_soc']:.0%}"),
    ]
    table = {"KPI": [label for label, _ in rows]}
    for name, s in summaries.items():
        table[SHORT_NAMES[name]] = [fmt(s) for _, fmt in rows]
    return pd.DataFrame(table)


def main() -> None:
    t = theme_tokens()
    key = ensure_artifacts()
    df, bundle = load_artifacts(key)
    test_start = pd.Timestamp(bundle["test_start"])
    data_start, data_end = df.index[0], df.index[-1]

    # ---------------------------------------------------------------- sidebar
    with st.sidebar:
        st.header("Control panel")
        st.subheader("Simulation window")
        dates = st.date_input("Dates", value=(test_start.date(), data_end.date()),
                              min_value=data_start.date(), max_value=data_end.date(),
                              help="Default: the hold-out week the forecaster never saw during training.")
        if not isinstance(dates, tuple) or len(dates) != 2:
            st.info("Pick an end date to run the simulation.")
            st.stop()
        start = pd.Timestamp(dates[0])
        end = pd.Timestamp(dates[1]) + pd.Timedelta(hours=23)
        day = st.selectbox("Day for the 24 h views", pd.date_range(start, end.normalize(), freq="D"),
                           format_func=lambda d: d.strftime("%a %d %b %Y"))
        strategy = st.radio("Dispatch strategy", [MPC_NAME, RULE_NAME],
                            help="MPC re-plans every hour with a linear programme over the forecast horizon; "
                                 "the heuristic applies fixed priority rules.")

        st.subheader("Thermal storage")
        capacity = st.slider("Storage capacity (kWh)", 250, 6000, 2000, step=250)
        initial_soc = st.slider("Initial state of charge (%)", 15, 95, 50, step=5) / 100

        st.subheader("Engine")
        load_pct = st.slider("Engine load override (% of schedule)", 60, 130, 100, step=5,
                             help="Scales the recorded load. Rpm, exhaust flow, exhaust temperature and Q_in "
                                  "are re-derived from the engine map, then re-forecast by the ML model.")

        st.subheader("Energy prices")
        tariff = st.slider(esc("Grid tariff, off-peak to peak ($/kWh)"), 0.04, 0.60, (0.08, 0.28), step=0.01,
                           format="%.2f", help="Rescales the time-of-use tariff of the dataset.")
        gas_price = st.slider(esc("Natural gas price ($/kWh)"), 0.020, 0.120, 0.045, step=0.005, format="%.3f",
                              help="Fuel price of the backup boiler that recovered heat displaces.")

        with st.expander("Advanced plant parameters"):
            orc_capacity = st.slider("ORC thermal capacity (kW_th)", 50, 400, 150, step=25)
            orc_efficiency = st.slider("ORC net electrical efficiency (%)", 8, 22, 15) / 100
            horizon_h = st.slider("MPC look-ahead horizon (h)", 4, 48, 24, step=4)
            grid_co2 = st.slider("Grid CO2 factor (kg/kWh)", 0.05, 0.90, 0.40, step=0.05, format="%.2f")

    # ---------------------------------------------------------------- compute
    with st.spinner("Forecasting heat and optimising the dispatch..."):
        frame15, _ = scenario_frames(load_pct / 100, key)
        results, cfg = run_strategies(key, load_pct / 100, start, end, capacity, initial_soc, tuple(tariff),
                                      gas_price, orc_capacity, orc_efficiency, horizon_h, grid_co2)
    other = RULE_NAME if strategy == MPC_NAME else MPC_NAME
    res = results[strategy]
    summaries = {name: summarize(r) for name, r in results.items()}
    s, base = summaries[strategy], summaries[BASE_NAME]
    tank = ThermalStorageBuffer(capacity_kwh=capacity, initial_soc=initial_soc)
    window15 = frame15.loc[start:end + pd.Timedelta(minutes=45)]

    # ---------------------------------------------------------------- header + KPIs
    st.title("Waste Heat Recovery & Thermal Storage Optimizer")
    st.caption(f"Engine exhaust → thermal-oil loop → {capacity:,} kWh {NITRATE_SALT_PCM.name} storage + "
               f"{orc_capacity} kW_th ORC · **{strategy}** · {start:%d %b} – {end:%d %b %Y} "
               f"({len(res)} h) · engine load {load_pct}% of schedule")
    if start < test_start:
        st.info("The window overlaps the forecaster's training period: forecasts there are in-sample. "
                f"The hold-out test week starts on {test_start:%a %d %b}.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total heat recaptured", f"{s['heat_recaptured_kwh']:,.0f} kWh",
              fmt_delta(s["heat_recaptured_kwh"] - base["heat_recaptured_kwh"], "kWh"),
              delta_description="vs no storage", border=True, chart_data=daily(res["q_in_kw"] - res["bypass_kw"]),
              chart_type="bar", help=f"Exhaust heat put to use instead of bypassed: {s['recapture_rate']:.1%} of "
                                     f"the {s['heat_available_kwh']:,.0f} kWh available.")
    c2.metric("Financial savings", f"${s['total_savings_usd']:,.0f}",
              fmt_delta(s["total_savings_usd"] - base["total_savings_usd"], "$"),
              delta_description="vs no storage", border=True, chart_data=daily(res["savings_usd"]), chart_type="bar",
              help=esc(f"Boiler gas avoided ${s['gas_savings_usd']:,.0f} + ORC electricity "
                       f"${s['orc_revenue_usd']:,.0f}, compared with a plant without heat recovery."))
    c3.metric("CO2 emissions avoided", f"{s['co2_avoided_kg']:,.0f} kg",
              fmt_delta(s["co2_avoided_kg"] - base["co2_avoided_kg"], "kg"),
              delta_description="vs no storage", border=True, chart_data=daily(res["co2_avoided_kg"]),
              chart_type="bar", help="Gas not burnt in the boiler plus grid electricity displaced by the ORC.")
    c4.metric("ORC electricity", f"{s['orc_electricity_kwh']:,.0f} kWh",
              fmt_delta(s["orc_electricity_kwh"] - base["orc_electricity_kwh"], "kWh"), delta_color="off",
              delta_description="vs no storage", border=True, chart_data=daily(res["orc_power_kw"]),
              chart_type="bar", help=f"Net output of the {orc_capacity} kW_th ORC at {orc_efficiency:.0%} efficiency. "
                                     "Storage shifts surplus heat to the facility, where it is worth more, so this "
                                     "can fall while savings rise.")

    # ---------------------------------------------------------------- forecast
    st.subheader("Real vs. predicted heat generation")
    scored = window15.dropna(subset=["q_pred_1h_kw", "q_pred_4h_kw"])
    m1 = regression_metrics(scored["q_in_kw"], scored["q_pred_1h_kw"])
    m4 = regression_metrics(scored["q_in_kw"], scored["q_pred_4h_kw"])
    st.caption(f"LightGBM forecasts issued 1 h and 4 h before each 15-minute reading. On this window: "
               f"**1 h** RMSE {m1['rmse']:.1f} kW · MAE {m1['mae']:.1f} kW · R² {m1['r2']:.3f}  |  "
               f"**4 h** RMSE {m4['rmse']:.1f} kW · MAE {m4['mae']:.1f} kW · R² {m4['r2']:.3f}")
    st.plotly_chart(forecast_figure(window15, t), width="stretch", theme=None, config=PLOTLY_CONFIG)

    # ---------------------------------------------------------------- 24 h views
    day_slice = slice(day, day + pd.Timedelta(hours=23))
    days = {name: results[name].loc[day_slice] for name in (MPC_NAME, RULE_NAME)}
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(f"Storage state of charge · {day:%a %d %b}")
        st.plotly_chart(soc_figure(days, strategy, other, cfg.peak_price_threshold, tank, t),
                        width="stretch", theme=None, config=PLOTLY_CONFIG)
    with right:
        st.subheader(f"Heat routing · {day:%a %d %b}")
        st.caption("Above zero: where the exhaust heat went. Below zero: heat drawn back out of the PCM storage.")
        st.plotly_chart(routing_figure(days[strategy], t), width="stretch", theme=None, config=PLOTLY_CONFIG)

    # ---------------------------------------------------------------- balance + comparison
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.subheader("Energy balance over the window")
        st.plotly_chart(sankey_figure(res, cfg, tank, t), width="stretch", theme=None, config=PLOTLY_CONFIG)
    with right:
        st.subheader("Strategy comparison")
        st.dataframe(comparison_table(summaries), hide_index=True, width="stretch")
        mpc, rule = summaries[MPC_NAME], summaries[RULE_NAME]
        edge = mpc["total_savings_usd"] - rule["total_savings_usd"]
        st.caption(esc(
            "Savings are measured against a plant without heat recovery. The no-storage case is dispatched "
            "optimally hour by hour, so it is the bar the storage has to clear: with the selected strategy the "
            f"storage adds **{fmt_delta(s['total_savings_usd'] - base['total_savings_usd'], '$')}**. "
            f"MPC vs rule-based: **{fmt_delta(edge, '$')}** ({edge / rule['total_savings_usd']:+.1%}). Both start "
            f"at {initial_soc:.0%} SoC but can end at different levels, so check the last row too."
        ))

    # ---------------------------------------------------------------- decision log
    st.subheader(f"Hourly dispatch decisions · {day:%a %d %b}")
    st.caption("Hour-average flows in kW. The decision for each hour was taken at its start, on the forecast; "
               "the valves then routed the heat that actually arrived.")
    log = days[strategy][["action", "soc", "q_forecast_kw", "q_in_kw", "demand_kw", "price", "to_facility_kw",
                          "to_orc_kw", "to_storage_kw", "storage_to_facility_kw", "storage_to_orc_kw",
                          "bypass_kw", "boiler_kw"]].copy()
    log["soc"] *= 100
    log["action"] = log["action"].str.replace("->", "→", regex=False)
    log.insert(0, "hour", log.index.strftime("%H:%M"))
    kw = lambda label: st.column_config.NumberColumn(label, format="%.0f", width="small")
    st.dataframe(log, hide_index=True, width="stretch", height=35 * (len(log) + 1) + 3, column_config={
        "hour": st.column_config.TextColumn("Hour", width="small"),
        "action": st.column_config.TextColumn("Decision", width="medium"),
        "soc": st.column_config.ProgressColumn("SoC (end)", min_value=0, max_value=100, format="%.0f%%",
                                               width="small"),
        "q_forecast_kw": kw("Q_in forecast"), "q_in_kw": kw("Q_in actual"), "demand_kw": kw("Demand"),
        "price": st.column_config.NumberColumn("$/kWh", format="%.3f", width="small"),
        "to_facility_kw": kw("→ Facility"), "to_orc_kw": kw("→ ORC"), "to_storage_kw": kw("→ Storage"),
        "storage_to_facility_kw": kw("Storage → fac."), "storage_to_orc_kw": kw("Storage → ORC"),
        "bypass_kw": kw("Bypass"), "boiler_kw": kw("Boiler"),
    })
    st.download_button("Download the full window as CSV", data=res.to_csv().encode("utf-8"),
                       file_name=f"whr_dispatch_{strategy.split()[0].lower()}_{start:%Y%m%d}_{end:%Y%m%d}.csv",
                       mime="text/csv", icon=":material/download:")

    # ---------------------------------------------------------------- model card + assumptions
    with st.expander("Forecast model card"):
        metric_rows = []
        for horizon, m in bundle["metrics"].items():
            for label, k in (("LightGBM", "model"), ("Persistence", "persistence"),
                             ("Seasonal naive (24 h)", "seasonal_naive_24h")):
                metric_rows.append({"Horizon": horizon, "Forecaster": label, "RMSE (kW)": m[k]["rmse"],
                                    "MAE (kW)": m[k]["mae"], "R²": m[k]["r2"]})
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch", column_config={
            "RMSE (kW)": st.column_config.NumberColumn(format="%.2f"),
            "MAE (kW)": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
        })
        st.caption(f"Hold-out test targets from {test_start:%d %b %Y}; trained on {bundle['data_start'][:10]} → "
                   f"{(test_start - pd.Timedelta(days=1)):%Y-%m-%d} with {bundle['library']}. Walk-forward CV RMSE: "
                   + ", ".join(f"{h} {m['cv_rmse_mean']:.1f} ± {m['cv_rmse_std']:.1f} kW"
                               for h, m in bundle["metrics"].items()) + ".")
        cols = st.columns(len(bundle["feature_importance"]))
        for col, (horizon, importance) in zip(cols, bundle["feature_importance"].items()):
            with col:
                st.markdown(f"**Top features, {horizon} model** (share of split gain)")
                if importance:
                    st.plotly_chart(importance_figure(importance, t), width="stretch", theme=None,
                                    config=PLOTLY_CONFIG)

    with st.expander("Plant model and assumptions"):
        spec = tank.describe()
        st.markdown(esc(
            f"""
- **Waste heat**: Q_in = m_dot · Cp · (T_exhaust − T_ambient), Cp = {bundle['cp_exhaust_kj_kgk']:.2f} kJ/(kg·K). It is referenced to ambient as specified, so it is an upper bound; acid-dew-point limits would put the reference near 120–150 °C.
- **PCM storage**: {spec['medium']}, melting at {NITRATE_SALT_PCM.t_melt_c:.0f} °C, 70 % latent share; {spec['capacity_kwh']:,.0f} kWh nominal, SoC window {spec['soc_window'][0]:.0%}–{spec['soc_window'][1]:.0%} ({spec['usable_kwh']:,.0f} kWh usable); 90 % charge and 90 % discharge efficiency; ±{spec['max_charge_kw']:,.0f} kW heat-exchanger rating; U·A = {spec['ua_w_per_k']:.1f} W/K ({spec['surface_area_m2']:.0f} m², scaled with capacity^(2/3)).
- **ORC**: {cfg.orc_thermal_capacity_kw:.0f} kW_th evaporator, {cfg.orc_efficiency:.0%} net electrical efficiency, electricity valued at the grid price.
- **Backup boiler**: {cfg.boiler_efficiency:.0%} efficient on natural gas at ${cfg.gas_price_kwh:.3f}/kWh, so every kWh of recovered heat delivered to the facility saves ${cfg.heat_value_kwh:.3f}.
- **CO2**: gas {cfg.gas_co2_kg_kwh} kg/kWh_fuel, grid {cfg.grid_co2_kg_kwh} kg/kWh_e.
- **Dispatch**: hourly. MPC = LP (SciPy HiGHS) over {horizon_h} h using the 1 h forecast for the next hour, the 4 h forecast for hours 2–4 and a day-type profile beyond. Rule-based = facility first (ORC first when power is worth more than the gas it displaces), surplus to the ORC above ${cfg.peak_price_threshold:.3f}/kWh and to storage otherwise, storage covers deficits and feeds the ORC at peak above {cfg.peak_discharge_reserve_soc:.0%} SoC.
"""
        ))


main()
