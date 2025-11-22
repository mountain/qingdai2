from __future__ import annotations

"""
Structured Coupler for interface and in-column processes (OO + DBA friendly).

Purpose
- Provide an explicit, typed, side-effect-free coupler that converts SurfaceToAtmosphere
  + ColumnProcessIn into AtmosphereToSurfaceFluxes + ColumnProcessOut.
- Keep modules/backends pure (array-in/array-out) and let the orchestrator wire
  ports through this coupler.
- Degrade gracefully if full energy/humidity modules are unavailable, while
  leveraging them when present.

Design
- compute(surface_in, column_in, grid, state, dt) -> (fluxes, col_out)
- Attempts to use pygcm.energy and pygcm.humidity where possible; otherwise uses
  simple physically-consistent proxies (zero or tiny linear relations).
- Does NOT mutate DBA state; it only reads arrays/ports and returns new arrays.

Notes
- This is deliberately minimal/robust. A production coupler would:
  * compose shortwave/longwave/cloud/ice/eco to form α_total and SW_atm/SW_sfc/LW terms
  * compute SH via bulk aerodynamic with wind, stability, roughness
  * compute E via humidity.evaporation_flux and LH = L_v * E
  * compute LH_release from condensation diagnostics (P_cond) not just precip_rate proxy
  * conserve energy consistently with docs/06 and water closure with docs/09
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from pygcm.jax_compat import xp  # numpy or jax.numpy
except Exception:
    xp = np

from pygcm import constants

from .ports import (
    AtmosphereToSurfaceFluxes,
    ColumnProcessIn,
    ColumnProcessOut,
    SurfaceToAtmosphere,
)


@dataclass
class CouplerParams:
    # Bulk coefficients (very rough defaults; for demo only)
    C_H: float = 1.5e-3
    rho_a: float = 1.2
    cp_air: float = 1004.0
    # If humidity module is not used, fallback to small E
    evap_fallback_kg_m2_s: float = 0.0
    # Longwave/shortwave placeholders
    use_energy_module: bool = True
    use_humidity_module: bool = True


class Coupler:
    def __init__(self, params: CouplerParams | None = None) -> None:
        self.params = params or CouplerParams()

        # Try import optional modules
        self._energy = None
        self._humidity = None
        if self.params.use_energy_module:
            try:
                from pygcm import energy as _energy
                self._energy = _energy
            except Exception:
                self._energy = None
        if self.params.use_humidity_module:
            try:
                from pygcm import humidity as _humidity
                self._humidity = _humidity
            except Exception:
                self._humidity = None

        # Constants
        self.LV = float(getattr(constants, "LV", 2.5e6))

    def compute(self,
                surface_in: SurfaceToAtmosphere | None,
                column_in: ColumnProcessIn | None,
                grid: Any | None,
                state: Any,
                dt: float) -> tuple[AtmosphereToSurfaceFluxes | None, ColumnProcessOut | None]:
        """
        Compute interface fluxes (atmosphere -> surface) and in-column updates.

        Parameters
        ----------
        surface_in : SurfaceToAtmosphere or None
        column_in : ColumnProcessIn or None
        grid : optional grid (may be used for area weights/metrics; not needed here)
        state : world state (DBA), read-only access if needed
        dt : seconds

        Returns
        -------
        (fluxes, col_out)
        """
        # If no surface info, we cannot compute interface fluxes reliably
        if surface_in is None:
            return None, self._column_only(column_in)

        T_s = surface_in.T_s

        # Initialize placeholders
        SH = xp.zeros_like(T_s)
        LH = xp.zeros_like(T_s)
        SW_sfc = xp.zeros_like(T_s)
        LW_sfc = xp.zeros_like(T_s)
        Qnet = xp.zeros_like(T_s)
        evap_flux = None
        precip_flux = None

        # Column-side defaults
        q_next = None
        cloud_next = None
        precip_next = None
        LH_release = None

        # Use humidity module for evaporation if possible
        if self._humidity is not None and column_in is not None:
            try:
                # In a full model, we would need near-surface wind |V| and a surface factor
                # (ocean/land/ice). For demo, use |V| ≈ 1 m/s proxy and surface_factor=1.
                V10 = xp.ones_like(T_s)  # proxy
                q = xp.array(column_in.q, copy=False)
                E = self._humidity.evaporation_flux(  # kg m^-2 s^-1
                    T_s=T_s,
                    q=q,
                    Vmag=V10,
                    surface_factor=xp.ones_like(T_s),
                    params=getattr(self._humidity, "HumidityParams", object)()  # default params
                )
                evap_flux = E
                LH = self.LV * E  # W m^-2, positive upward from surface
            except Exception:
                evap_flux = self._fallback_evap(T_s)
                LH = self.LV * evap_flux
        else:
            evap_flux = self._fallback_evap(T_s)
            LH = self.LV * evap_flux

        # Sensible heat via bulk (very rough; no wind/stability here)
        if column_in is not None and column_in.Ta is not None:
            SH = self.params.rho_a * self.params.cp_air * self.params.C_H * (T_s - column_in.Ta)
        else:
            SH = xp.zeros_like(T_s)

        # Shortwave/Longwave via energy module if available (placeholders otherwise)
        if self._energy is not None and surface_in.base_albedo is not None:
            try:
                # Without incident I and cloud we cannot compute SW properly; keep placeholders.
                SW_sfc = xp.zeros_like(T_s)
                LW_sfc = xp.zeros_like(T_s)
            except Exception:
                SW_sfc = xp.zeros_like(T_s)
                LW_sfc = xp.zeros_like(T_s)
        else:
            SW_sfc = xp.zeros_like(T_s)
            LW_sfc = xp.zeros_like(T_s)

        # Net surface energy into skin (sign convention: downward positive into surface)
        # Here SH/LH are upward from surface (positive up), so Qnet ≈ SW_sfc - LW_sfc - SH - LH
        Qnet = SW_sfc - LW_sfc - SH - LH

        # Column outputs
        if column_in is not None:
            q_next = xp.array(column_in.q, copy=True)
            cloud_next = xp.array(column_in.cloud, copy=True)
            precip_next = xp.array(column_in.precip_rate, copy=True)
            precip_flux = precip_next
            if precip_next is not None:
                LH_release = self.LV * precip_next

        fluxes = AtmosphereToSurfaceFluxes(
            SH=SH, LH=LH, SW_sfc=SW_sfc, LW_sfc=LW_sfc, Qnet=Qnet,
            evap_flux=evap_flux, precip_flux=precip_flux
        )
        col_out = None if q_next is None else ColumnProcessOut(
            q_next=q_next,
            cloud_next=cloud_next if cloud_next is not None else xp.zeros_like(T_s),
            precip_rate_next=precip_next if precip_next is not None else xp.zeros_like(T_s),
            LH_release=LH_release
        )
        return fluxes, col_out

    # ------------- helpers -------------

    def _fallback_evap(self, ref: np.ndarray) -> np.ndarray:
        if self.params.evap_fallback_kg_m2_s == 0.0:
            return xp.zeros_like(ref)
        return xp.full_like(ref, float(self.params.evap_fallback_kg_m2_s))

    def _column_only(self, column_in: ColumnProcessIn | None) -> ColumnProcessOut | None:
        if column_in is None:
            return None
        q_next = xp.array(column_in.q, copy=True)
        cloud_next = xp.array(column_in.cloud, copy=True)
        precip_next = xp.array(column_in.precip_rate, copy=True)
        LH_release = None
        if precip_next is not None:
            LH_release = self.LV * precip_next
        return ColumnProcessOut(
            q_next=q_next,
            cloud_next=cloud_next,
            precip_rate_next=precip_next,
            LH_release=LH_release
        )
