"""Synthetic 15-minute sensor dataset for an industrial engine waste-heat-recovery (WHR) plant.

The site runs one variable-speed engine (compressor / process drive) on a two-shift weekday
schedule, a short Saturday shift and a Sunday standby with a weekly test run. The script
simulates the engine, the weather, a time-of-use grid tariff and the facility's process-heat
demand, then computes the ground-truth exhaust heat available to the WHR heat exchanger:

    Q_in [kW] = m_dot [kg/s] * Cp [kJ/(kg K)] * (T_exhaust - T_ambient) [K]

Columns written to factory_sensor_data.csv
    timestamp                    15-minute timestamps
    engine_rpm                   800 - 3000 rpm
    engine_load_pct              20 - 100 %
    exhaust_temp_c               300 - 520 C
    exhaust_flow_kg_h            500 - 2500 kg/h
    ambient_temp_c               10 - 35 C
    grid_electricity_price_kwh   0.08 - 0.28 $/kWh (time-of-use tariff)
    facility_heat_demand_kw      100 - 800 kW
    q_in_kw                      ground-truth waste heat output (kW)

Usage
    python dataset_builder.py
    python dataset_builder.py --days 60 --start 2026-08-03 --seed 7 --out other.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_CSV = "factory_sensor_data.csv"
STEP_MINUTES = 15
STEPS_PER_HOUR = 60 // STEP_MINUTES
STEPS_PER_DAY = 24 * STEPS_PER_HOUR

# Exhaust-gas property and the engine's operating envelope
CP_EXHAUST_KJ_KGK = 1.10            # mean Cp of engine exhaust between 300 and 520 C
EXHAUST_TIME_CONSTANT_MIN = 20.0    # thermal inertia of exhaust manifold + turbine housing
LOAD_RANGE = (20.0, 100.0)          # %
RPM_RANGE = (800.0, 3000.0)         # rpm
FLOW_RANGE = (500.0, 2500.0)        # kg/h
EXHAUST_TEMP_RANGE = (300.0, 520.0) # C
AMBIENT_RANGE = (10.0, 35.0)        # C
PRICE_RANGE = (0.08, 0.28)          # $/kWh
DEMAND_RANGE = (100.0, 800.0)       # kW

ENGINE_COLUMNS = ["engine_rpm", "exhaust_flow_kg_h", "exhaust_temp_c"]

# Production plan: engine load (%) at given hours of the day, linearly interpolated.
# Day types: 0 = Monday-Friday, 1 = Saturday, 2 = Sunday.
_LOAD_KNOTS = {
    0: ([0.0, 5.5, 6.5, 12.0, 12.25, 12.75, 13.0, 14.0, 14.25, 14.5, 21.5, 22.5, 24.0],
        [44, 44, 88, 88, 70, 70, 88, 88, 74, 92, 92, 44, 44]),        # 2 shifts, lunch dip, night baseload
    1: ([0.0, 6.5, 7.5, 15.0, 16.0, 24.0],
        [36, 36, 66, 66, 36, 36]),                                      # single Saturday shift
    2: ([0.0, 10.0, 10.5, 12.0, 12.5, 24.0],
        [27, 27, 45, 45, 27, 27]),                                      # Sunday standby + weekly test run
}

# Process heat (kW) on top of the base load: morning start-up of ovens / CIP, day production,
# light second shift. Same day types as above.
_PROCESS_HEAT_KNOTS = {
    0: ([0.0, 4.5, 5.0, 5.5, 6.5, 8.0, 9.0, 13.0, 13.5, 14.25, 21.5, 22.0, 24.0],
        [12, 12, 90, 240, 260, 210, 150, 150, 90, 28, 25, 12, 12]),
    1: ([0.0, 6.0, 6.5, 8.0, 13.0, 14.0, 15.0, 24.0],
        [10, 10, 170, 150, 120, 25, 10, 10]),
    2: ([0.0, 24.0], [5, 5]),
}

# Time-of-use tariff ($/kWh) by hour-of-day bands
_WEEKDAY_TARIFF = ([7, 9, 17, 21, 22], [0.085, 0.190, 0.145, 0.255, 0.150], 0.085)
_WEEKEND_TARIFF = ([8, 17, 21], [0.080, 0.110, 0.175], 0.080)


# --------------------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------------------
def waste_heat_kw(exhaust_flow_kg_h, exhaust_temp_c, ambient_temp_c, cp_kj_kgk: float = CP_EXHAUST_KJ_KGK):
    """Q_in = m_dot * Cp * (T_exhaust - T_ambient), with m_dot converted from kg/h to kg/s."""
    return np.asarray(exhaust_flow_kg_h) / 3600.0 * cp_kj_kgk * (
        np.asarray(exhaust_temp_c) - np.asarray(ambient_temp_c)
    )


def engine_response(load_pct, ambient_temp_c, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Map engine load and intake-air temperature to speed, exhaust mass flow and exhaust temperature.

    Speed follows the governor curve of the variable-speed drive, mass flow scales with speed,
    boost (load) and intake-air density, and the exhaust temperature rises with fuel per cycle and
    lags its steady-state value through the thermal inertia of the exhaust system.
    Pass ``rng`` to add sensor/process noise; without it the response is deterministic.
    """
    load = np.asarray(load_pct, dtype=float)
    ambient = np.asarray(ambient_temp_c, dtype=float)
    n = load.size
    x = np.clip((load - LOAD_RANGE[0]) / (LOAD_RANGE[1] - LOAD_RANGE[0]), 0.0, 1.0)

    rpm = RPM_RANGE[0] + (RPM_RANGE[1] - RPM_RANGE[0]) * x**0.85
    if rng is not None:
        rpm = rpm + rng.normal(0.0, 15.0, n)
    rpm = np.clip(rpm, *RPM_RANGE)
    speed = (rpm - RPM_RANGE[0]) / (RPM_RANGE[1] - RPM_RANGE[0])

    air_density_ratio = (273.15 + 25.0) / (273.15 + ambient)
    flow = (FLOW_RANGE[0] + (FLOW_RANGE[1] - FLOW_RANGE[0]) * (0.55 * speed + 0.45 * x)) * air_density_ratio
    if rng is not None:
        flow = flow * (1.0 + rng.normal(0.0, 0.015, n))
    flow = np.clip(flow, *FLOW_RANGE)

    steady_temp = EXHAUST_TEMP_RANGE[0] + (EXHAUST_TEMP_RANGE[1] - EXHAUST_TEMP_RANGE[0]) * x**0.9
    steady_temp = steady_temp + 0.4 * (ambient - 22.5)
    alpha = 1.0 - np.exp(-STEP_MINUTES / EXHAUST_TIME_CONSTANT_MIN)
    temp = np.empty(n)
    if n:
        temp[0] = steady_temp[0]
    for k in range(1, n):
        temp[k] = temp[k - 1] + alpha * (steady_temp[k] - temp[k - 1])
    if rng is not None:
        temp = temp + rng.normal(0.0, 2.5, n)
    temp = np.clip(temp, *EXHAUST_TEMP_RANGE)

    return {"engine_rpm": rpm, "exhaust_flow_kg_h": flow, "exhaust_temp_c": temp}


def apply_load_override(df: pd.DataFrame, load_factor: float) -> pd.DataFrame:
    """What-if scenario: scale the engine load and re-derive rpm, flow, exhaust temperature and Q_in.

    The recorded noise is preserved: each sensor keeps its residual against the deterministic
    engine map, so ``load_factor=1.0`` returns the original data unchanged.
    """
    out = df.copy()
    if abs(load_factor - 1.0) < 1e-9:
        return out
    new_load = np.clip(df["engine_load_pct"].to_numpy() * load_factor, *LOAD_RANGE)
    baseline = engine_response(df["engine_load_pct"], df["ambient_temp_c"])
    scenario = engine_response(new_load, df["ambient_temp_c"])
    limits = {"engine_rpm": RPM_RANGE, "exhaust_flow_kg_h": FLOW_RANGE, "exhaust_temp_c": EXHAUST_TEMP_RANGE}
    out["engine_load_pct"] = new_load
    for col in ENGINE_COLUMNS:
        out[col] = np.clip(df[col].to_numpy() + scenario[col] - baseline[col], *limits[col])
    out["q_in_kw"] = waste_heat_kw(out["exhaust_flow_kg_h"], out["exhaust_temp_c"], out["ambient_temp_c"])
    return out


# --------------------------------------------------------------------------------------
# Stochastic building blocks
# --------------------------------------------------------------------------------------
def _hour_of_day(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray(index.hour + index.minute / 60.0, dtype=float)


def day_type(index: pd.DatetimeIndex) -> np.ndarray:
    """0 = Monday-Friday, 1 = Saturday, 2 = Sunday."""
    dow = np.asarray(index.day_of_week)
    return np.where(dow < 5, 0, np.where(dow == 5, 1, 2))


def _day_number(index: pd.DatetimeIndex) -> np.ndarray:
    return np.asarray((index - index[0].normalize()) // pd.Timedelta(days=1), dtype=int)


def _daily_profile(index: pd.DatetimeIndex, knots: dict[int, tuple[list, list]]) -> np.ndarray:
    hour, kinds = _hour_of_day(index), day_type(index)
    out = np.empty(len(index))
    for kind, (hours, values) in knots.items():
        mask = kinds == kind
        out[mask] = np.interp(hour[mask], hours, values)
    return out


def _ar1(n: int, phi: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) process: x[k] = phi * x[k-1] + e[k], e ~ N(0, sigma)."""
    shocks = rng.normal(0.0, sigma, n)
    x = np.empty(n)
    x[0] = shocks[0] / np.sqrt(1.0 - phi**2)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + shocks[k]
    return x


def ambient_temperature(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Seasonal mean + multi-day weather anomalies + diurnal cycle (peak at 15:00) + noise."""
    hour = _hour_of_day(index)
    day_of_year = np.asarray(index.day_of_year, dtype=float) + hour / 24.0
    seasonal = 19.0 + 8.0 * np.cos(2.0 * np.pi * (day_of_year - 200.0) / 365.25)

    day_pos = np.asarray((index - index[0].normalize()) / pd.Timedelta(days=1), dtype=float)
    n_days = int(np.ceil(day_pos.max())) + 2
    centers = np.arange(n_days) + 0.5
    weather = np.interp(day_pos, centers, _ar1(n_days, 0.65, 2.0, rng))
    amplitude = np.interp(day_pos, centers, rng.uniform(4.5, 8.5, n_days))
    diurnal = amplitude * np.cos(2.0 * np.pi * (hour - 15.0) / 24.0)

    temp = seasonal + weather + diurnal + rng.normal(0.0, 0.3, len(index))
    return np.clip(temp, *AMBIENT_RANGE)


def production_load(index: pd.DatetimeIndex, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, int]]:
    """Scheduled load x daily intensity, plus rush orders, unplanned trips and process noise."""
    n = len(index)
    days = _day_number(index)
    schedule = _daily_profile(index, _LOAD_KNOTS)
    intensity = rng.normal(1.0, 0.03, days.max() + 1)[days]
    load = LOAD_RANGE[0] + (schedule - LOAD_RANGE[0]) * intensity

    hour, kinds = _hour_of_day(index), day_type(index)
    production = ((kinds == 0) & (hour >= 6) & (hour < 22)) | ((kinds == 1) & (hour >= 7.5) & (hour < 15))
    candidates = np.flatnonzero(production)
    n_days = n / STEPS_PER_DAY

    rush_starts = rng.choice(candidates, size=min(rng.poisson(0.2 * n_days), candidates.size), replace=False)
    for start in rush_starts:                                   # rush orders: +6-12 % for 2-5 h
        load[start:start + rng.integers(8, 21)] += rng.uniform(6.0, 12.0)

    trip_starts = rng.choice(candidates, size=min(rng.poisson(0.12 * n_days), candidates.size), replace=False)
    for start in trip_starts:                                   # trips / maintenance: minimum load for 1-4 h
        seg = slice(start, min(start + rng.integers(4, 17), n))
        load[seg] = rng.uniform(20.0, 24.0, seg.stop - seg.start)

    load = load + _ar1(n, 0.85, 2.0, rng)
    return np.clip(load, *LOAD_RANGE), {"rush_orders": len(rush_starts), "unplanned_trips": len(trip_starts)}


def grid_price(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Time-of-use tariff (weekday morning/evening peaks, cheaper weekends) x daily market index."""
    hour = np.asarray(index.hour)
    weekday = np.asarray(index.day_of_week) < 5

    def tariff(bands):
        limits, prices, default = bands
        return np.select([hour < h for h in limits], prices, default)

    base = np.where(weekday, tariff(_WEEKDAY_TARIFF), tariff(_WEEKEND_TARIFF))
    days = _day_number(index)
    market = rng.normal(1.0, 0.035, days.max() + 1)[days]
    price = base * market + rng.normal(0.0, 0.003, len(index))
    return np.clip(price, *PRICE_RANGE)


def _batch_heat_spikes(index: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Short process batches (oven start-ups, cleaning-in-place) adding 90-260 kW for 30 min - 2 h."""
    n = len(index)
    hour, kinds = _hour_of_day(index), day_type(index)
    window = ((kinds == 0) & (hour >= 6) & (hour < 20)) | ((kinds == 1) & (hour >= 7) & (hour < 13))
    candidates = np.flatnonzero(window)
    spikes = np.zeros(n)
    count = min(rng.poisson(1.4 * n / STEPS_PER_DAY * 6 / 7), candidates.size)
    for start in rng.choice(candidates, size=count, replace=False):
        duration = int(rng.integers(2, 9))
        shape = np.ones(duration)
        shape[0], shape[-1] = 0.5, 0.6
        seg = slice(start, min(start + duration, n))
        spikes[seg] += rng.uniform(90.0, 260.0) * shape[: seg.stop - seg.start]
    return spikes


def facility_heat_demand(index: pd.DatetimeIndex, ambient_c: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Base load + weather-driven losses + scheduled process heat + batch spikes + noise."""
    process = _daily_profile(index, _PROCESS_HEAT_KNOTS)
    weather = 8.5 * np.clip(22.0 - ambient_c, 0.0, None)
    demand = 112.0 + weather + process + _batch_heat_spikes(index, rng) + _ar1(len(index), 0.7, 9.0, rng)
    return np.clip(demand, *DEMAND_RANGE)


# --------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------
def build_dataset(days: int = 30, start: str = "2026-09-21", seed: int = 42) -> tuple[pd.DataFrame, dict[str, int]]:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=days * STEPS_PER_DAY, freq=f"{STEP_MINUTES}min", name="timestamp")

    ambient = ambient_temperature(index, rng)
    load, events = production_load(index, rng)
    engine = engine_response(load, ambient, rng)
    price = grid_price(index, rng)
    demand = facility_heat_demand(index, ambient, rng)

    df = pd.DataFrame(
        {
            "engine_rpm": engine["engine_rpm"],
            "engine_load_pct": load,
            "exhaust_temp_c": engine["exhaust_temp_c"],
            "exhaust_flow_kg_h": engine["exhaust_flow_kg_h"],
            "ambient_temp_c": ambient,
            "grid_electricity_price_kwh": price,
            "facility_heat_demand_kw": demand,
        },
        index=index,
    ).round({
        "engine_rpm": 1, "engine_load_pct": 2, "exhaust_temp_c": 2, "exhaust_flow_kg_h": 1,
        "ambient_temp_c": 2, "grid_electricity_price_kwh": 4, "facility_heat_demand_kw": 2,
    })
    # Ground truth computed from the recorded (rounded) columns so it can be reproduced from the CSV.
    df["q_in_kw"] = waste_heat_kw(df["exhaust_flow_kg_h"], df["exhaust_temp_c"], df["ambient_temp_c"]).round(3)
    return df.reset_index(), events


def print_summary(df: pd.DataFrame, path: Path, events: dict[str, int]) -> None:
    ts = pd.to_datetime(df["timestamp"])
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {path.resolve()}")
    print(f"Period : {ts.iloc[0]} -> {ts.iloc[-1]}  ({STEP_MINUTES}-minute steps)")
    print(f"Events : {events['rush_orders']} rush orders, {events['unplanned_trips']} unplanned engine trips\n")
    stats = df.drop(columns="timestamp").describe().T[["min", "mean", "max"]]
    print(stats.round(3).to_string())

    hours = STEP_MINUTES / 60.0
    q, demand = df["q_in_kw"], df["facility_heat_demand_kw"]
    surplus = (q - demand).clip(lower=0)
    deficit = (demand - q).clip(lower=0)
    print("\nHeat balance over the period")
    print(f"  waste heat available     {q.sum() * hours / 1000:9.1f} MWh_th")
    print(f"  facility heat demand     {demand.sum() * hours / 1000:9.1f} MWh_th")
    print(f"  surplus (Q_in > demand)  {surplus.sum() * hours / 1000:9.1f} MWh_th in {(surplus > 0).mean():.0%} of steps")
    print(f"  deficit (Q_in < demand)  {deficit.sum() * hours / 1000:9.1f} MWh_th")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--start", default="2026-09-21", help="first timestamp (a Monday keeps weeks aligned)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=OUTPUT_CSV)
    args = parser.parse_args()

    df, events = build_dataset(days=args.days, start=args.start, seed=args.seed)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print_summary(df, out, events)


if __name__ == "__main__":
    main()
