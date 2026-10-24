"""Smart heat dispatcher for the WHR plant: hourly routing of the recovered exhaust heat.

Every hour the dispatcher receives the predicted Q_in, the storage state of charge, the grid
electricity price and the facility heat demand, and routes the heat to
  1. facility heating        - displaces natural gas burnt in the backup boiler
  2. the ORC generator       - electricity valued at the grid price (most valuable at peak)
  3. the PCM thermal storage - excess heat that would otherwise be bypassed to the stack
  4. storage discharge       - back to the facility, or into the ORC during peak-price windows

Two strategies share one interface, ``decide(...) -> DispatchDecision``:
  RuleBasedDispatcher  transparent priority rules (the classic plant-operator heuristic)
  MPCDispatcher        receding-horizon linear programme solved with scipy.optimize.linprog
                       (HiGHS): 1 h / 4 h ML forecasts of Q_in, a day-type profile beyond 4 h,
                       the known tariff and the expected demand; only the first hour is executed.

Decisions are taken on *forecasts*; ``realize`` then applies them to the heat that actually
arrives during the hour, the diverter valves absorbing the forecast error.

Run ``python optimizer.py`` to simulate the hold-out week with both strategies.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from dataset_builder import day_type
from storage_simulator import ThermalStorageBuffer
from train_model import load_forecaster, load_sensor_data, predict_horizons

TOL = 1e-6          # kW below which a flow counts as zero
TIE_BREAK = 1e-6    # TND/kWh nudge that keeps the LP from cycling energy through the tank


@dataclass(frozen=True)
class PlantConfig:
    """Technical and economic parameters of the WHR plant."""

    orc_thermal_capacity_kw: float = 150.0     # max heat input to the ORC evaporator (about 22 kWe)
    orc_efficiency: float = 0.15               # net electrical efficiency, kWh_e per kWh_th
    gas_price_kwh: float = 0.135               # natural gas, TND/kWh of fuel ($0.045 at 3.0 TND/USD)
    boiler_efficiency: float = 0.90            # backup boiler, heat out / fuel in
    grid_co2_kg_kwh: float = 0.40              # grid emission factor, kg CO2 per kWh_e
    gas_co2_kg_kwh: float = 0.202              # natural gas emission factor, kg CO2 per kWh of fuel
    peak_price_threshold: float = 0.66         # rule-based: price (TND/kWh) that opens a peak window
    peak_discharge_reserve_soc: float = 0.40   # rule-based: SoC kept back for the facility at peak

    @property
    def heat_value_kwh(self) -> float:
        """TND saved per kWh of heat delivered to the facility (boiler gas not burnt)."""
        return self.gas_price_kwh / self.boiler_efficiency


@dataclass(frozen=True)
class DispatchDecision:
    """Hourly routing set-points in kW (hour averages), computed on predicted heat."""

    to_facility_kw: float = 0.0           # 1. recovered heat straight to facility heating
    to_orc_kw: float = 0.0                # 2. recovered heat to the ORC generator
    to_storage_kw: float = 0.0            # 3. excess heat diverted into the PCM storage
    storage_to_facility_kw: float = 0.0   # 4a. storage discharge to the facility
    storage_to_orc_kw: float = 0.0        # 4b. storage discharge to the ORC (peak-price window)
    bypass_kw: float = 0.0                # heat that cannot be used (stack bypass)
    boiler_kw: float = 0.0                # backup boiler heat for the remaining demand
    surplus_priority: tuple[str, ...] = ("storage", "orc")  # real-time order for heat beyond the facility need
    orc_before_facility: bool = False     # electricity worth more than the gas it would displace
    hold_storage: bool = False            # keep stored heat for later instead of covering facility deficits
    predicted_q_in_kw: float = 0.0


def _first(value) -> float:
    return float(np.atleast_1d(np.asarray(value, dtype=float))[0])


class RuleBasedDispatcher:
    """Priority rules: facility first (unless power is worth more than the gas it displaces);
    surplus to the ORC at peak price, otherwise into storage; storage covers facility deficits
    and, above a reserve SoC, feeds the ORC during peak windows."""

    name = "Rule-based heuristic"
    horizon_h = 1

    def __init__(self, storage: ThermalStorageBuffer, config: PlantConfig):
        self.storage, self.config = storage, config

    def decide(self, predicted_q_in_kw, current_soc: float, grid_price, facility_heat_demand_kw,
               ambient_temp_c: float = 20.0, duration_h: float = 1.0) -> DispatchDecision:
        q, price, demand = _first(predicted_q_in_kw), _first(grid_price), _first(facility_heat_demand_kw)
        cfg, tank, dt = self.config, self.storage, duration_h
        peak = price > cfg.peak_price_threshold
        orc_room = cfg.orc_thermal_capacity_kw

        # When power is worth more than the gas it displaces, the ORC is served before the facility
        orc_first = cfg.orc_efficiency * price > cfg.heat_value_kwh
        to_orc = min(q, orc_room) if orc_first else 0.0
        orc_room -= to_orc
        to_facility = min(q - to_orc, demand)                         # 1. facility heating
        surplus, deficit = q - to_orc - to_facility, demand - to_facility
        charge_room = tank.max_charge_power_kw(dt, soc=current_soc)
        if peak:                                                      # 2. sell power while the grid pays most
            orc_extra = min(surplus, orc_room)
            to_storage = min(surplus - orc_extra, charge_room)
        else:                                                         # 3. bank the excess for later
            to_storage = min(surplus, charge_room)
            orc_extra = min(surplus - to_storage, orc_room)
        to_orc += orc_extra
        orc_room -= orc_extra
        bypass = surplus - orc_extra - to_storage

        storage_to_facility = storage_to_orc = 0.0                    # 4. discharge
        if to_storage <= TOL:
            discharge_room = tank.max_discharge_power_kw(dt, soc=current_soc)
            storage_to_facility = min(deficit, discharge_room)
            if peak and current_soc > cfg.peak_discharge_reserve_soc:
                above_reserve_kwh = ((current_soc - cfg.peak_discharge_reserve_soc) * tank.capacity_kwh
                                     - storage_to_facility * dt / tank.discharge_efficiency)
                storage_to_orc = max(0.0, min(orc_room, discharge_room - storage_to_facility,
                                              above_reserve_kwh * tank.discharge_efficiency / dt))
        return DispatchDecision(
            to_facility_kw=to_facility, to_orc_kw=to_orc, to_storage_kw=to_storage,
            storage_to_facility_kw=storage_to_facility, storage_to_orc_kw=storage_to_orc,
            bypass_kw=bypass, boiler_kw=deficit - storage_to_facility,
            surplus_priority=("orc", "storage") if peak else ("storage", "orc"),
            orc_before_facility=orc_first,
            predicted_q_in_kw=q,
        )


class MPCDispatcher:
    """Receding-horizon LP over the next ``horizon_h`` hours.

    Decision variables per hour k (kW, plus the stored energy E_k in kWh):
        f  direct heat to facility     o  direct heat to ORC        c  heat into storage
        w  bypass                      sf storage -> facility       so storage -> ORC
        b  backup boiler               E  stored energy at the end of hour k
    Constraints:
        f + o + c + w = Q_hat_k                         every kW of predicted heat is routed
        f + sf + b    = D_hat_k                         the facility demand is always met
        o + so       <= ORC capacity;  sf + so <= discharge rating;  c <= charge rating
        E_k = E_{k-1} + eta_c c dt - (sf + so) dt / eta_d - U A (T_s - T_amb) dt
        SoC_min C <= E_k <= SoC_max C
    Objective (minimised): sum_k disc_k [gas cost of b - ORC electricity value] - value of E_H.
    """

    name = "MPC optimizer (SciPy LP)"
    VARIABLES = ("to_facility", "to_orc", "to_storage", "bypass", "storage_to_facility",
                 "storage_to_orc", "boiler", "energy")
    F, O, C, W, SF, SO, B, E = range(8)

    def __init__(self, storage: ThermalStorageBuffer, config: PlantConfig, horizon_h: int = 24,
                 terminal_value_factor: float = 0.9, discount_per_h: float = 1e-3):
        self.storage, self.config = storage, config
        self.horizon_h = int(horizon_h)
        self.terminal_value_factor = terminal_value_factor   # haircut on heat left in the tank at the horizon
        self.discount_per_h = discount_per_h                 # tiny time preference: use stored heat early
        self.solve_count = 0
        self.fallback_count = 0
        self._fallback = RuleBasedDispatcher(storage, config)
        self._structure: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}

    def _matrices(self, H: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """Constraint matrices depend only on the horizon length, so they are built once."""
        if (H, dt) not in self._structure:
            nv, k, tank = len(self.VARIABLES), np.arange(H), self.storage
            col = lambda j: k * nv + j
            A_eq = np.zeros((3 * H, H * nv))
            heat, demand, energy = 3 * k, 3 * k + 1, 3 * k + 2
            for j in (self.F, self.O, self.C, self.W):
                A_eq[heat, col(j)] = 1.0
            for j in (self.F, self.SF, self.B):
                A_eq[demand, col(j)] = 1.0
            A_eq[energy, col(self.E)] = 1.0
            A_eq[energy[1:], col(self.E)[:-1]] = -1.0
            A_eq[energy, col(self.C)] = -tank.charge_efficiency * dt
            A_eq[energy, col(self.SF)] = dt / tank.discharge_efficiency
            A_eq[energy, col(self.SO)] = dt / tank.discharge_efficiency

            A_ub = np.zeros((2 * H, H * nv))
            A_ub[2 * k, col(self.O)] = A_ub[2 * k, col(self.SO)] = 1.0          # ORC capacity
            A_ub[2 * k + 1, col(self.SF)] = A_ub[2 * k + 1, col(self.SO)] = 1.0  # discharge rating
            self._structure[(H, dt)] = (A_eq, A_ub)
        return self._structure[(H, dt)]

    def plan(self, predicted_q_in_kw, current_soc: float, grid_price, facility_heat_demand_kw,
             ambient_temp_c: float = 20.0, duration_h: float = 1.0) -> pd.DataFrame | None:
        """Solve the look-ahead LP; returns one row per hour, or None if HiGHS fails."""
        q = np.maximum(np.atleast_1d(np.asarray(predicted_q_in_kw, dtype=float)), 0.0)
        p = np.atleast_1d(np.asarray(grid_price, dtype=float))
        d = np.maximum(np.atleast_1d(np.asarray(facility_heat_demand_kw, dtype=float)), 0.0)
        H = min(q.size, p.size, d.size, self.horizon_h)
        q, p, d = q[:H], p[:H], d[:H]
        cfg, tank, dt = self.config, self.storage, duration_h
        nv, k = len(self.VARIABLES), np.arange(H)
        col = lambda j: k * nv + j
        A_eq, A_ub = self._matrices(H, dt)

        disc = (1.0 - self.discount_per_h) ** (k * dt)
        orc_value = cfg.orc_efficiency * p * dt * disc
        cost = np.zeros(H * nv)
        cost[col(self.B)] = cfg.heat_value_kwh * dt * disc
        cost[col(self.O)] = -orc_value
        cost[col(self.SO)] = -orc_value + TIE_BREAK
        cost[col(self.SF)] = TIE_BREAK
        cost[col(self.C)] = TIE_BREAK
        # Heat left in the tank at the horizon is valued at its dependable use, facility heating.
        # (Valuing it at the peak ORC price would make the LP hoard heat the ORC can never absorb.)
        cost[col(self.E)[-1]] = -self.terminal_value_factor * tank.discharge_efficiency * cfg.heat_value_kwh * disc[-1]

        # Standing loss at the current tank temperature; the "free" trajectory (no charge, no
        # discharge) bounds E from below when the tank already sits under SoC_min.
        e0 = current_soc * tank.capacity_kwh
        loss_kw = tank.standing_loss_kw(ambient_temp_c, soc=current_soc)
        e_free = np.maximum(0.0, e0 - loss_kw * dt * (k + 1))
        step_loss = np.concatenate(([e0], e_free[:-1])) - e_free

        b_eq = np.empty(3 * H)
        b_eq[3 * k], b_eq[3 * k + 1], b_eq[3 * k + 2] = q, d, -step_loss
        b_eq[2] += e0
        b_ub = np.empty(2 * H)
        b_ub[2 * k], b_ub[2 * k + 1] = cfg.orc_thermal_capacity_kw, tank.max_discharge_kw

        lower, upper = np.zeros(H * nv), np.full(H * nv, np.inf)
        upper[col(self.C)] = tank.max_charge_kw
        lower[col(self.E)] = np.minimum(tank.soc_min * tank.capacity_kwh, e_free)
        upper[col(self.E)] = np.maximum(tank.soc_max * tank.capacity_kwh, e_free)

        res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=np.column_stack([lower, upper]), method="highs")
        self.solve_count += 1
        if res.status != 0:
            return None
        plan = pd.DataFrame(res.x.reshape(H, nv), columns=self.VARIABLES)
        plan["soc"] = plan["energy"] / tank.capacity_kwh
        plan["q_forecast_kw"], plan["demand_forecast_kw"], plan["price"] = q, d, p
        return plan

    def decide(self, predicted_q_in_kw, current_soc: float, grid_price, facility_heat_demand_kw,
               ambient_temp_c: float = 20.0, duration_h: float = 1.0) -> DispatchDecision:
        plan = self.plan(predicted_q_in_kw, current_soc, grid_price, facility_heat_demand_kw,
                         ambient_temp_c, duration_h)
        if plan is None:
            self.fallback_count += 1
            return self._fallback.decide(predicted_q_in_kw, current_soc, grid_price, facility_heat_demand_kw,
                                         ambient_temp_c, duration_h)
        k0 = plan.iloc[0].to_dict()   # dict access: "to_orc" would clash with DataFrame.to_orc()
        q0, d0, p0 = k0["q_forecast_kw"], k0["demand_forecast_kw"], k0["price"]
        return DispatchDecision(
            to_facility_kw=k0["to_facility"], to_orc_kw=k0["to_orc"], to_storage_kw=k0["to_storage"],
            storage_to_facility_kw=k0["storage_to_facility"], storage_to_orc_kw=k0["storage_to_orc"],
            bypass_kw=k0["bypass"], boiler_kw=k0["boiler"],
            surplus_priority=self._surplus_priority(k0, p0),
            orc_before_facility=bool(k0["to_orc"] > TOL and k0["to_facility"] < min(q0, d0) - TOL),
            hold_storage=bool(d0 - q0 > TOL and k0["storage_to_facility"] <= TOL),
            predicted_q_in_kw=q0,
        )

    def _surplus_priority(self, k0: dict, price: float) -> tuple[str, ...]:
        """Which sink takes unexpected extra heat first: the one the plan favoured this hour."""
        orc, sto = k0["to_orc"] > TOL, k0["to_storage"] > TOL
        if orc and not sto:
            return ("orc", "storage")
        if sto and not orc:
            return ("storage", "orc")
        if orc and sto:   # one of them was saturated: the saturated one was preferred
            saturated = k0["to_orc"] >= self.config.orc_thermal_capacity_kw - 1e-3
            return ("orc", "storage") if saturated else ("storage", "orc")
        tank = self.storage
        round_trip_value = tank.charge_efficiency * tank.discharge_efficiency * self.config.heat_value_kwh
        return ("orc", "storage") if self.config.orc_efficiency * price > round_trip_value else ("storage", "orc")


def realize(decision: DispatchDecision, actual_q_in_kw: float, actual_demand_kw: float,
            storage: ThermalStorageBuffer, config: PlantConfig, ambient_temp_c: float = 20.0,
            duration_h: float = 1.0) -> dict[str, float]:
    """Apply an hourly decision to the heat and demand that actually occur, and update the tank."""
    dt = duration_h
    heat = max(0.0, float(actual_q_in_kw))
    demand = max(0.0, float(actual_demand_kw))
    orc_room = config.orc_thermal_capacity_kw

    # ORC ahead of the facility only when the plan found electricity worth more than gas
    to_orc = min(heat, orc_room) if decision.orc_before_facility else 0.0
    heat -= to_orc
    orc_room -= to_orc
    to_facility = min(heat, demand)
    heat -= to_facility
    deficit = demand - to_facility

    # Surplus: planned set-points first, then any leftover heat, in the plan's priority order
    orc_discharge_wanted = decision.storage_to_orc_kw > TOL and heat < orc_room - TOL
    can_charge = deficit <= TOL and not orc_discharge_wanted
    room = {"orc": orc_room, "storage": storage.max_charge_power_kw(dt) if can_charge else 0.0}
    planned = {"orc": max(0.0, decision.to_orc_kw - to_orc), "storage": decision.to_storage_kw}
    alloc = {"orc": 0.0, "storage": 0.0}
    for follow_plan in (True, False):
        for sink in decision.surplus_priority:
            limit = min(room[sink], planned[sink]) if follow_plan else room[sink]
            amount = max(0.0, min(heat, limit - alloc[sink]))
            alloc[sink] += amount
            heat -= amount
    to_orc += alloc["orc"]
    orc_room -= alloc["orc"]
    to_storage = storage.charge(alloc["storage"], dt) if alloc["storage"] > TOL else 0.0
    bypass = heat + alloc["storage"] - to_storage

    # Storage discharge: facility deficit (unless the plan holds the heat), ORC at peak per plan
    storage_to_facility = storage_to_orc = 0.0
    if deficit > TOL and to_storage <= TOL and (decision.storage_to_facility_kw > TOL or not decision.hold_storage):
        storage_to_facility = storage.discharge(deficit, dt)
    if decision.storage_to_orc_kw > TOL and to_storage <= TOL and orc_room > TOL:
        request = min(decision.storage_to_orc_kw, orc_room, storage.max_discharge_kw - storage_to_facility)
        storage_to_orc = storage.discharge(max(0.0, request), dt)
    standing_loss_kwh = storage.apply_standing_loss(ambient_temp_c, dt)

    return {
        "to_facility_kw": to_facility,
        "to_orc_kw": to_orc,
        "to_storage_kw": to_storage,
        "storage_to_facility_kw": storage_to_facility,
        "storage_to_orc_kw": storage_to_orc,
        "bypass_kw": bypass,
        "boiler_kw": deficit - storage_to_facility,
        "standing_loss_kw": standing_loss_kwh / dt,
        "soc": storage.get_soc(),
        "storage_temp_c": storage.temperature_c(),
    }


# --------------------------------------------------------------------------------------
# Inputs, simulation and KPIs
# --------------------------------------------------------------------------------------
def build_hourly_inputs(df: pd.DataFrame, forecasts: pd.DataFrame, profile_until) -> pd.DataFrame:
    """Hourly dispatcher inputs: actuals, target-aligned forecasts and day-type x hour profiles.

    The profiles (mean Q_in and demand per day type and hour) are computed from data before
    ``profile_until`` only, and serve as the forecast beyond the 4-hour ML horizon.
    """
    frame = df.join(forecasts).resample("1h").mean()
    hourly = pd.DataFrame({
        "q_in_kw": frame["q_in_kw"],
        "q_pred_1h_kw": frame["q_pred_1h_kw"],
        "q_pred_4h_kw": frame["q_pred_4h_kw"],
        "demand_kw": frame["facility_heat_demand_kw"],
        "price": frame["grid_electricity_price_kwh"],
        "ambient_c": frame["ambient_temp_c"],
    })
    kinds, hours = day_type(hourly.index), np.asarray(hourly.index.hour)
    history = np.asarray(hourly.index < pd.Timestamp(profile_until))
    keys = pd.MultiIndex.from_arrays([kinds, hours])
    for src, dst in (("q_in_kw", "q_profile_kw"), ("demand_kw", "demand_profile_kw")):
        past = hourly.loc[history, src]
        table = past.groupby([kinds[history], hours[history]]).mean()
        hourly[dst] = np.nan_to_num(table.reindex(keys).to_numpy(), nan=float(past.mean()))
    return hourly


class ForecastInputs:
    """Look-ahead paths available at the start of each hour - nothing after that instant is used."""

    def __init__(self, hourly: pd.DataFrame):
        self.q_1h = hourly["q_pred_1h_kw"].to_numpy(dtype=float)
        self.q_4h = hourly["q_pred_4h_kw"].to_numpy(dtype=float)
        self.q_profile = hourly["q_profile_kw"].to_numpy(dtype=float)
        self.demand = hourly["demand_kw"].to_numpy(dtype=float)
        self.demand_profile = hourly["demand_profile_kw"].to_numpy(dtype=float)
        self.price = hourly["price"].to_numpy(dtype=float)
        self.n = len(hourly)

    def at(self, i: int, horizon_h: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        H = max(1, min(horizon_h, self.n - i))
        q = self.q_profile[i:i + H].copy()                       # beyond 4 h: day-type profile
        ahead = self.q_4h[i + 1:i + min(4, H)]                   # hours 2-4: 4h-ahead model
        q[1:1 + ahead.size] = np.where(np.isnan(ahead), q[1:1 + ahead.size], ahead)
        if not np.isnan(self.q_1h[i]):                           # next hour: 1h-ahead model
            q[0] = self.q_1h[i]
        bias = self.demand[i - 1] - self.demand_profile[i - 1] if i > 0 else 0.0
        demand = np.maximum(0.0, self.demand_profile[i:i + H] + bias * 0.5 ** np.arange(H))
        return q, demand, self.price[i:i + H]                    # the tariff is published in advance


ROUTE_LABELS = {
    "to_storage_kw": "Charge storage",
    "storage_to_facility_kw": "Storage -> facility",
    "storage_to_orc_kw": "Storage -> ORC",
    "to_orc_kw": "ORC power",
    "bypass_kw": "Bypass",
}


def routing_label(row, threshold_kw: float = 1.0) -> str:
    parts = [label for col, label in ROUTE_LABELS.items() if row[col] > threshold_kw]
    if row["boiler_kw"] > threshold_kw:
        parts.append("boiler top-up" if parts else "Boiler top-up")
    return " + ".join(parts) or "Direct heating only"


def add_economics(flows: pd.DataFrame, config: PlantConfig, duration_h: float = 1.0) -> pd.DataFrame:
    """Gas savings, ORC revenue and avoided CO2 for every hour of a routing table."""
    out = flows.copy()
    out["orc_power_kw"] = config.orc_efficiency * (out["to_orc_kw"] + out["storage_to_orc_kw"])
    recovered = out["to_facility_kw"] + out["storage_to_facility_kw"]
    out["gas_saved_kwh"] = recovered * duration_h / config.boiler_efficiency
    out["gas_savings_tnd"] = out["gas_saved_kwh"] * config.gas_price_kwh
    out["orc_revenue_tnd"] = out["orc_power_kw"] * duration_h * out["price"]
    out["savings_tnd"] = out["gas_savings_tnd"] + out["orc_revenue_tnd"]
    out["co2_avoided_kg"] = (out["gas_saved_kwh"] * config.gas_co2_kg_kwh
                             + out["orc_power_kw"] * duration_h * config.grid_co2_kg_kwh)
    return out


def simulate(hourly: pd.DataFrame, dispatcher, config: PlantConfig, duration_h: float = 1.0) -> pd.DataFrame:
    """Closed loop, hour by hour: forecast -> decide -> realise on actual heat -> update the tank."""
    tank = dispatcher.storage
    inputs = ForecastInputs(hourly)
    q_act, d_act = hourly["q_in_kw"].to_numpy(float), hourly["demand_kw"].to_numpy(float)
    price, ambient = hourly["price"].to_numpy(float), hourly["ambient_c"].to_numpy(float)
    rows = []
    for i in range(len(hourly)):
        q_path, d_path, p_path = inputs.at(i, dispatcher.horizon_h)
        soc_start = tank.get_soc()
        decision = dispatcher.decide(q_path, soc_start, p_path, d_path,
                                     ambient_temp_c=ambient[max(i - 1, 0)], duration_h=duration_h)
        flows = realize(decision, q_act[i], d_act[i], tank, config, ambient[i], duration_h)
        flows.update(q_in_kw=q_act[i], q_forecast_kw=decision.predicted_q_in_kw, demand_kw=d_act[i],
                     price=price[i], ambient_c=ambient[i], soc_start=soc_start)
        rows.append(flows)
    results = pd.DataFrame(rows, index=hourly.index)
    results["action"] = results.apply(routing_label, axis=1)
    return add_economics(results, config, duration_h)


def no_storage_baseline(hourly: pd.DataFrame, config: PlantConfig, duration_h: float = 1.0) -> pd.DataFrame:
    """Same WHR plant without thermal storage: facility first, surplus to the ORC, rest bypassed."""
    q, d, p = (hourly[c].to_numpy(float) for c in ("q_in_kw", "demand_kw", "price"))
    cap = config.orc_thermal_capacity_kw
    orc_first = np.where(config.orc_efficiency * p > config.heat_value_kwh, np.minimum(q, cap), 0.0)
    to_facility = np.minimum(q - orc_first, d)
    to_orc = orc_first + np.minimum(q - orc_first - to_facility, cap - orc_first)
    zeros = np.zeros_like(q)
    flows = pd.DataFrame({
        "to_facility_kw": to_facility, "to_orc_kw": to_orc, "to_storage_kw": zeros,
        "storage_to_facility_kw": zeros, "storage_to_orc_kw": zeros,
        "bypass_kw": q - to_facility - to_orc, "boiler_kw": d - to_facility,
        "standing_loss_kw": zeros, "soc": np.nan, "q_in_kw": q, "demand_kw": d, "price": p,
    }, index=hourly.index)
    return add_economics(flows, config, duration_h)


def summarize(results: pd.DataFrame, duration_h: float = 1.0) -> dict[str, float]:
    def kwh(col: str) -> float:
        return float(results[col].sum() * duration_h)

    available, bypass = kwh("q_in_kw"), kwh("bypass_kw")
    return {
        "heat_available_kwh": available,
        "heat_recaptured_kwh": available - bypass,
        "recapture_rate": (available - bypass) / available if available else 0.0,
        "facility_heat_kwh": kwh("to_facility_kw") + kwh("storage_to_facility_kw"),
        "orc_heat_kwh": kwh("to_orc_kw") + kwh("storage_to_orc_kw"),
        "orc_electricity_kwh": kwh("orc_power_kw"),
        "storage_charged_kwh": kwh("to_storage_kw"),
        "storage_discharged_kwh": kwh("storage_to_facility_kw") + kwh("storage_to_orc_kw"),
        "standing_loss_kwh": kwh("standing_loss_kw"),
        "bypass_kwh": bypass,
        "boiler_heat_kwh": kwh("boiler_kw"),
        "gas_savings_tnd": float(results["gas_savings_tnd"].sum()),
        "orc_revenue_tnd": float(results["orc_revenue_tnd"].sum()),
        "total_savings_tnd": float(results["savings_tnd"].sum()),
        "co2_avoided_kg": float(results["co2_avoided_kg"].sum()),
        "final_soc": float(results["soc"].iloc[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the hold-out week with both dispatch strategies.")
    parser.add_argument("--capacity", type=float, default=2000.0, help="storage capacity (kWh)")
    parser.add_argument("--horizon", type=int, default=24, help="MPC look-ahead (hours)")
    parser.add_argument("--initial-soc", type=float, default=0.5)
    args = parser.parse_args()

    df, bundle = load_sensor_data(), load_forecaster()
    test_start = pd.Timestamp(bundle["test_start"])
    hourly = build_hourly_inputs(df, predict_horizons(bundle, df), profile_until=test_start)
    window = hourly.loc[test_start:]
    cfg = PlantConfig()
    print(f"Simulating {len(window)} h ({window.index[0]} -> {window.index[-1]}), "
          f"storage {args.capacity:.0f} kWh, start SoC {args.initial_soc:.0%}\n")

    summaries, logs = {}, {}
    for build in (lambda tank: MPCDispatcher(tank, cfg, horizon_h=args.horizon),
                  lambda tank: RuleBasedDispatcher(tank, cfg)):
        tank = ThermalStorageBuffer(capacity_kwh=args.capacity, initial_soc=args.initial_soc)
        dispatcher = build(tank)
        started = time.perf_counter()
        results = simulate(window, dispatcher, cfg)
        summaries[dispatcher.name] = summarize(results) | {
            "runtime_s": time.perf_counter() - started,
            "balance_error_kwh": tank.energy_balance_error_kwh(),
        }
        logs[dispatcher.name] = results
        if isinstance(dispatcher, MPCDispatcher):
            print(f"MPC: {dispatcher.solve_count} LP solves, {dispatcher.fallback_count} fallbacks")
    summaries["No storage (direct + ORC)"] = summarize(no_storage_baseline(window, cfg))

    header = (f"{'Strategy':<28}{'Recaptured':>11}{'Rate':>7}{'Gas TND':>9}{'ORC TND':>8}{'Total TND':>10}"
              f"{'CO2 t':>8}{'Boiler MWh':>11}{'Bypass MWh':>11}{'End SoC':>8}")
    print(header)
    print("-" * len(header))
    for name, s in summaries.items():
        end_soc = f"{s['final_soc']:.0%}" if not np.isnan(s["final_soc"]) else "-"
        print(f"{name:<28}{s['heat_recaptured_kwh'] / 1000:>9.1f}MWh{s['recapture_rate']:>7.1%}"
              f"{s['gas_savings_tnd']:>9.0f}{s['orc_revenue_tnd']:>8.0f}{s['total_savings_tnd']:>10.0f}"
              f"{s['co2_avoided_kg'] / 1000:>8.2f}{s['boiler_heat_kwh'] / 1000:>11.1f}"
              f"{s['bypass_kwh'] / 1000:>11.2f}{end_soc:>8}")
    for name in logs:
        s = summaries[name]
        print(f"{name}: {s['runtime_s']:.2f} s, storage energy-balance error {s['balance_error_kwh']:.1e} kWh")

    mpc = logs[MPCDispatcher.name]
    print("\nMPC hourly routing decisions - first 24 h of the hold-out week (kW)")
    cols = ["q_forecast_kw", "q_in_kw", "demand_kw", "price", "to_facility_kw", "to_orc_kw", "to_storage_kw",
            "storage_to_facility_kw", "storage_to_orc_kw", "boiler_kw", "soc", "action"]
    table = mpc[cols].head(24).rename(columns=lambda c: c.replace("_kw", "").replace("storage_to_", "sto->"))
    table.index = table.index.strftime("%a %H:%M")
    print(table.to_string(float_format=lambda v: f"{v:.2f}" if abs(v) < 1 else f"{v:.0f}"))


if __name__ == "__main__":
    main()
