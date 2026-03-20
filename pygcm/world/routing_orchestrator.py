from __future__ import annotations

import os

import numpy as np

from pygcm.routing import RiverRouting
from pygcm.world.orchestrator_spec import WorldOrchestratorConfigureSpec


class RoutingOrchestrator:
    def __init__(self, backend: RiverRouting | None = None) -> None:
        self.backend = backend
        self._shape: tuple[int, int] | None = None
        self._steps = 0.0

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        rspec = spec.routing
        self._shape = tuple(spec.ocean.land_mask.shape)
        if not rspec.enabled:
            self.backend = None
            return
        if self.backend is not None:
            return
        if not os.path.exists(rspec.network_nc_path):
            self.backend = None
            return
        try:
            self.backend = RiverRouting(
                spec.ocean.grid,
                rspec.network_nc_path,
                dt_hydro_hours=rspec.dt_hydro_hours,
                treat_lake_as_water=rspec.treat_lake_as_water,
                alpha_lake=rspec.alpha_lake,
                diag=rspec.diag,
            )
        except Exception:
            self.backend = None

    def step(
        self,
        *,
        runoff_flux: np.ndarray,
        dt_seconds: float,
        precip_flux: np.ndarray | None = None,
        evap_flux: np.ndarray | None = None,
    ) -> None:
        if self.backend is None:
            return
        self._steps += 1.0
        self.backend.step(
            R_land_flux=runoff_flux,
            dt_seconds=dt_seconds,
            precip_flux=precip_flux,
            evap_flux=evap_flux,
        )

    def diagnostics(self) -> dict[str, object]:
        if self.backend is not None:
            d = self.backend.diagnostics()
            if isinstance(d, dict):
                d = dict(d)
                d["steps"] = float(self._steps)
            return d
        shape = self._shape if self._shape is not None else (1, 1)
        zeros: np.ndarray = np.zeros(shape, dtype=float)
        return {
            "flow_accum_kgps": zeros,
            "ocean_inflow_kgps": 0.0,
            "mass_closure_error_kg": 0.0,
            "lake_volume_kg": None,
            "steps": float(self._steps),
        }


class _LegacyRoutingAdapter:
    def __init__(self, backend) -> None:
        self.backend = backend
        self._steps = 0.0

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        return None

    def step(
        self,
        *,
        runoff_flux: np.ndarray,
        dt_seconds: float,
        precip_flux: np.ndarray | None = None,
        evap_flux: np.ndarray | None = None,
    ) -> None:
        self._steps += 1.0
        self.backend.step(
            R_land_flux=runoff_flux,
            dt_seconds=dt_seconds,
            precip_flux=precip_flux,
            evap_flux=evap_flux,
        )

    def diagnostics(self) -> dict[str, object]:
        d = self.backend.diagnostics()
        if isinstance(d, dict):
            d = dict(d)
            d["steps"] = float(self._steps)
        return d


def ensure_routing_orchestrator(routing_obj) -> RoutingOrchestrator | object:
    if routing_obj is None:
        return RoutingOrchestrator()
    if hasattr(routing_obj, "configure") and hasattr(routing_obj, "step"):
        return routing_obj
    if hasattr(routing_obj, "step") and hasattr(routing_obj, "diagnostics"):
        return _LegacyRoutingAdapter(routing_obj)
    raise TypeError("routing object must provide configure+step or legacy step+diagnostics")
