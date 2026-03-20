from __future__ import annotations

import numpy as np

from pygcm.ocean import WindDrivenSlabOcean
from pygcm.world.orchestrator_spec import WorldOrchestratorConfigureSpec


class OceanOrchestrator:
    def __init__(self, backend: WindDrivenSlabOcean | None = None) -> None:
        self.backend = backend

    def configure(
        self,
        *,
        spec: WorldOrchestratorConfigureSpec,
    ) -> None:
        ospec = spec.ocean
        if self.backend is None:
            self.backend = WindDrivenSlabOcean(
                ospec.grid,
                ospec.land_mask,
                H_m=ospec.H_m,
                init_Ts=ospec.init_Ts,
                rho_w=ospec.rho_w,
                cp_w=ospec.cp_w,
            )

    def step_and_write(
        self,
        *,
        state,
        dt: float,
        u_atm: np.ndarray,
        v_atm: np.ndarray,
        Q_net: np.ndarray,
        ice_mask: np.ndarray,
    ) -> None:
        if self.backend is None:
            raise RuntimeError("OceanOrchestrator is not configured")
        self.backend.step(dt, u_atm, v_atm, Q_net=Q_net, ice_mask=ice_mask)
        state.surface.Ts.write[:] = self.backend.Ts
        if state.ocean is not None:
            state.ocean.sst.write[:] = self.backend.Ts
            state.ocean.uo.write[:] = self.backend.uo
            state.ocean.vo.write[:] = self.backend.vo
            state.ocean.eta.write[:] = self.backend.eta


def ensure_ocean_orchestrator(ocean_obj) -> OceanOrchestrator | object:
    if ocean_obj is None:
        return OceanOrchestrator()
    if hasattr(ocean_obj, "configure") and hasattr(ocean_obj, "step_and_write"):
        return ocean_obj
    if hasattr(ocean_obj, "step") and hasattr(ocean_obj, "Ts"):
        return OceanOrchestrator(backend=ocean_obj)
    raise TypeError("ocean object must provide configure+step_and_write or legacy step+fields")
