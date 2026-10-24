"""Thermal storage buffer between the WHR heat exchanger and its heat consumers.

Default medium: a nitrate-salt phase-change material (KNO3-NaNO3 eutectic, melting point
about 222 C) in a shell-and-tube tank charged by the thermal-oil loop of the exhaust heat
exchanger. 70 % of the capacity is latent heat, so the tank sits on its melting plateau over
most of the usable SoC window. A sensible thermal-oil tank is available as an alternative.

Energy bookkeeping per control interval dt (powers in kW, energies in kWh):
    charge      E += eta_charge * P_in * dt              P_in  = heat accepted from the WHR loop
    discharge   E -= P_out / eta_discharge * dt          P_out = heat delivered to the consumer
    standing    E -= U * A * (T_storage - T_ambient) * dt
Control actions keep the SoC inside [SoC_min, SoC_max]. Standing losses are physics, not a
control action: they can drain the tank below SoC_min, after which discharge is blocked until
the tank is recharged.

Run `python storage_simulator.py` for a 24-hour demo with an energy-balance check.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StorageMedium:
    """Temperature of the storage medium as a function of state of charge."""

    name: str
    t_empty_c: float                 # temperature at SoC = 0
    t_full_c: float                  # temperature at SoC = 1
    t_melt_c: float | None = None    # melting point (None for a sensible-heat medium)
    latent_share: float = 0.0        # share of the capacity stored as latent heat

    def __post_init__(self):
        if not self.t_empty_c < self.t_full_c:
            raise ValueError("t_empty_c must be below t_full_c")
        if self.t_melt_c is not None and not (self.t_empty_c < self.t_melt_c < self.t_full_c):
            raise ValueError("t_melt_c must lie between t_empty_c and t_full_c")
        if not 0.0 <= self.latent_share < 1.0:
            raise ValueError("latent_share must be in [0, 1)")

    @property
    def is_pcm(self) -> bool:
        return self.t_melt_c is not None and self.latent_share > 0.0

    def _solid_share(self) -> float:
        # the sensible part is split between solid and liquid in proportion to their temperature spans
        return (1.0 - self.latent_share) * (self.t_melt_c - self.t_empty_c) / (self.t_full_c - self.t_empty_c)

    def temperature_c(self, soc: float) -> float:
        soc = min(max(soc, 0.0), 1.0)
        if not self.is_pcm:
            return self.t_empty_c + (self.t_full_c - self.t_empty_c) * soc
        solid = self._solid_share()
        liquid = 1.0 - self.latent_share - solid
        if soc < solid:                                   # sensible heating of the solid
            return self.t_empty_c + (self.t_melt_c - self.t_empty_c) * soc / solid
        if soc <= solid + self.latent_share:              # melting plateau
            return self.t_melt_c
        return self.t_melt_c + (self.t_full_c - self.t_melt_c) * (soc - solid - self.latent_share) / liquid

    def melt_fraction(self, soc: float) -> float:
        """Liquid share of the PCM (0 = fully solid, 1 = fully molten); 0 for sensible media."""
        if not self.is_pcm:
            return 0.0
        return min(max((soc - self._solid_share()) / self.latent_share, 0.0), 1.0)


NITRATE_SALT_PCM = StorageMedium("KNO3-NaNO3 eutectic PCM", t_empty_c=170.0, t_full_c=270.0,
                                 t_melt_c=222.0, latent_share=0.70)
THERMAL_OIL = StorageMedium("Synthetic thermal oil (sensible)", t_empty_c=180.0, t_full_c=300.0)


class ThermalStorageBuffer:
    """Thermal storage tank with conversion efficiencies, power limits, an SoC window and standing losses."""

    REFERENCE_CAPACITY_KWH = 2000.0
    REFERENCE_AREA_M2 = 45.0        # insulated cylinder of about 2.4 m x 4.8 m holding 2 MWh of PCM

    def __init__(
        self,
        capacity_kwh: float = 2000.0,
        charge_efficiency: float = 0.90,
        discharge_efficiency: float = 0.90,
        soc_min: float = 0.15,
        soc_max: float = 0.95,
        initial_soc: float = 0.50,
        max_charge_kw: float | None = None,
        max_discharge_kw: float | None = None,
        u_value_w_m2k: float = 0.40,
        surface_area_m2: float | None = None,
        medium: StorageMedium = NITRATE_SALT_PCM,
    ):
        if capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if not 0.0 <= soc_min < soc_max <= 1.0:
            raise ValueError("need 0 <= soc_min < soc_max <= 1")
        if not (0.0 < charge_efficiency <= 1.0 and 0.0 < discharge_efficiency <= 1.0):
            raise ValueError("efficiencies must be in (0, 1]")
        if not 0.0 <= initial_soc <= 1.0:
            raise ValueError("initial_soc must be in [0, 1]")

        self.capacity_kwh = float(capacity_kwh)
        self.charge_efficiency = float(charge_efficiency)
        self.discharge_efficiency = float(discharge_efficiency)
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        # Heat-exchanger rating: C/5 by default, i.e. the usable window is crossed in about 4 h
        self.max_charge_kw = float(max_charge_kw) if max_charge_kw is not None else 0.2 * self.capacity_kwh
        self.max_discharge_kw = float(max_discharge_kw) if max_discharge_kw is not None else 0.2 * self.capacity_kwh
        self.u_value_w_m2k = float(u_value_w_m2k)
        # Geometric similarity: surface area grows with volume^(2/3)
        self.surface_area_m2 = float(surface_area_m2) if surface_area_m2 is not None else (
            self.REFERENCE_AREA_M2 * (self.capacity_kwh / self.REFERENCE_CAPACITY_KWH) ** (2.0 / 3.0)
        )
        self.medium = medium

        self.energy_kwh = initial_soc * self.capacity_kwh
        self.initial_energy_kwh = self.energy_kwh
        self.heat_in_kwh = 0.0            # heat accepted at the inlet
        self.heat_out_kwh = 0.0           # heat delivered at the outlet
        self.conversion_loss_kwh = 0.0    # charge + discharge heat-exchanger losses
        self.standing_loss_kwh = 0.0      # losses through the insulation
        self.history: list[dict] = []

    # ---------------------------------------------------------------- state
    def get_soc(self) -> float:
        """State of charge as a fraction of the nominal capacity (0-1)."""
        return self.energy_kwh / self.capacity_kwh

    def _energy(self, soc: float | None) -> float:
        return self.energy_kwh if soc is None else soc * self.capacity_kwh

    def temperature_c(self, soc: float | None = None) -> float:
        return self.medium.temperature_c(self.get_soc() if soc is None else soc)

    @property
    def ua_kw_per_k(self) -> float:
        return self.u_value_w_m2k * self.surface_area_m2 / 1000.0

    def standing_loss_kw(self, ambient_temp_c: float, soc: float | None = None) -> float:
        """Q_loss = U * A * (T_storage - T_ambient), in kW."""
        return max(0.0, self.ua_kw_per_k * (self.temperature_c(soc) - ambient_temp_c))

    def usable_energy_kwh(self, soc: float | None = None) -> float:
        """Energy above SoC_min that may still be discharged."""
        return max(0.0, self._energy(soc) - self.soc_min * self.capacity_kwh)

    def headroom_kwh(self, soc: float | None = None) -> float:
        """Energy that may still be stored before SoC_max."""
        return max(0.0, self.soc_max * self.capacity_kwh - self._energy(soc))

    def max_charge_power_kw(self, duration_h: float = 1.0, soc: float | None = None) -> float:
        """Largest inlet heat flow accepted for `duration_h` hours without passing SoC_max."""
        return min(self.max_charge_kw, self.headroom_kwh(soc) / (self.charge_efficiency * duration_h))

    def max_discharge_power_kw(self, duration_h: float = 1.0, soc: float | None = None) -> float:
        """Largest outlet heat flow deliverable for `duration_h` hours without passing SoC_min."""
        return min(self.max_discharge_kw, self.usable_energy_kwh(soc) * self.discharge_efficiency / duration_h)

    # -------------------------------------------------------------- actions
    def charge(self, energy_kw: float, duration_h: float = 1.0) -> float:
        """Offer `energy_kw` of heat for `duration_h` hours; returns the heat flow accepted (kW)."""
        if energy_kw < 0:
            raise ValueError("charge power must be >= 0 (use discharge)")
        accepted = min(float(energy_kw), self.max_charge_power_kw(duration_h))
        heat = accepted * duration_h
        stored = heat * self.charge_efficiency
        self.energy_kwh += stored
        self.heat_in_kwh += heat
        self.conversion_loss_kwh += heat - stored
        return accepted

    def discharge(self, energy_kw: float, duration_h: float = 1.0) -> float:
        """Request `energy_kw` of heat at the outlet for `duration_h` hours; returns the heat flow delivered (kW)."""
        if energy_kw < 0:
            raise ValueError("discharge power must be >= 0 (use charge)")
        delivered = min(float(energy_kw), self.max_discharge_power_kw(duration_h))
        heat = delivered * duration_h
        drawn = heat / self.discharge_efficiency
        self.energy_kwh -= drawn
        self.heat_out_kwh += heat
        self.conversion_loss_kwh += drawn - heat
        return delivered

    def apply_standing_loss(self, ambient_temp_c: float, duration_h: float = 1.0) -> float:
        """Remove the standing loss U*A*(T_storage - T_ambient) over `duration_h`; returns the loss (kWh)."""
        loss = min(self.energy_kwh, self.standing_loss_kw(ambient_temp_c) * duration_h)
        self.energy_kwh -= loss
        self.standing_loss_kwh += loss
        return loss

    def step(self, charge_kw: float = 0.0, discharge_kw: float = 0.0, ambient_temp_c: float = 20.0,
             duration_h: float = 1.0, timestamp=None) -> dict:
        """One control interval: charge *or* discharge, then the standing loss. Logged to `history`."""
        if charge_kw > 0 and discharge_kw > 0:
            raise ValueError("charge and discharge in the same interval")
        accepted = self.charge(charge_kw, duration_h) if charge_kw > 0 else 0.0
        delivered = self.discharge(discharge_kw, duration_h) if discharge_kw > 0 else 0.0
        loss_kwh = self.apply_standing_loss(ambient_temp_c, duration_h)
        record = {
            "timestamp": timestamp,
            "charge_requested_kw": charge_kw,
            "charge_accepted_kw": accepted,
            "discharge_requested_kw": discharge_kw,
            "discharge_delivered_kw": delivered,
            "standing_loss_kw": loss_kwh / duration_h,
            "soc": self.get_soc(),
            "energy_kwh": self.energy_kwh,
            "temperature_c": self.temperature_c(),
            "melt_fraction": self.medium.melt_fraction(self.get_soc()),
        }
        self.history.append(record)
        return record

    # ------------------------------------------------------------- reporting
    def energy_balance_error_kwh(self) -> float:
        """First-law check: initial + heat in - heat out - all losses - final energy (should be ~0)."""
        return (self.initial_energy_kwh + self.heat_in_kwh - self.heat_out_kwh
                - self.conversion_loss_kwh - self.standing_loss_kwh - self.energy_kwh)

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def describe(self) -> dict:
        return {
            "medium": self.medium.name,
            "capacity_kwh": self.capacity_kwh,
            "usable_kwh": (self.soc_max - self.soc_min) * self.capacity_kwh,
            "soc_window": (self.soc_min, self.soc_max),
            "efficiency_charge_discharge": (self.charge_efficiency, self.discharge_efficiency),
            "max_charge_kw": self.max_charge_kw,
            "max_discharge_kw": self.max_discharge_kw,
            "ua_w_per_k": self.ua_kw_per_k * 1000.0,
            "surface_area_m2": self.surface_area_m2,
        }

    def __repr__(self) -> str:
        return (f"ThermalStorageBuffer({self.capacity_kwh:.0f} kWh {self.medium.name}, SoC {self.get_soc():.1%}, "
                f"{self.temperature_c():.0f} C, UA {self.ua_kw_per_k * 1000:.1f} W/K)")


def _demo() -> None:
    tank = ThermalStorageBuffer(capacity_kwh=2000.0, initial_soc=0.60)
    print(tank)
    print(f"Usable window {tank.soc_min:.0%}-{tank.soc_max:.0%} = {tank.describe()['usable_kwh']:.0f} kWh, "
          f"power limit {tank.max_charge_kw:.0f} kW, standing loss at 222 C / 20 C ambient = "
          f"{tank.standing_loss_kw(20.0):.2f} kW\n")

    # (hour, charge kW, discharge kW, ambient C): night deficit, morning peak demand that empties the
    # tank down to SoC_min, midday and afternoon surplus that fills it up to SoC_max, evening idle.
    schedule = (
        [(h, 0.0, 100.0, 16.0) for h in range(0, 5)]
        + [(h, 0.0, 300.0, 15.0) for h in range(5, 9)]
        + [(h, 150.0, 0.0, 22.0) for h in range(9, 14)]
        + [(h, 380.0, 0.0, 27.0) for h in range(14, 22)]
        + [(h, 0.0, 0.0, 21.0) for h in range(22, 24)]
    )
    print(f"{'hour':>4} {'charge kW':>10} {'accepted':>9} {'discharge kW':>13} {'delivered':>10} "
          f"{'loss kW':>8} {'SoC':>7} {'T C':>6} {'melted':>7}")
    for hour, chg, dis, amb in schedule:
        r = tank.step(charge_kw=chg, discharge_kw=dis, ambient_temp_c=amb, timestamp=hour)
        print(f"{hour:>4} {chg:>10.0f} {r['charge_accepted_kw']:>9.1f} {dis:>13.0f} {r['discharge_delivered_kw']:>10.1f} "
              f"{r['standing_loss_kw']:>8.2f} {r['soc']:>7.1%} {r['temperature_c']:>6.1f} {r['melt_fraction']:>7.0%}")

    print(f"\nHeat in {tank.heat_in_kwh:.0f} kWh | heat out {tank.heat_out_kwh:.0f} kWh | "
          f"conversion losses {tank.conversion_loss_kwh:.0f} kWh | standing losses {tank.standing_loss_kwh:.1f} kWh")
    print(f"Energy balance error: {tank.energy_balance_error_kwh():.2e} kWh")


if __name__ == "__main__":
    _demo()
