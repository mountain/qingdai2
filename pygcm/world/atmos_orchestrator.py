from __future__ import annotations

from pygcm.world.atmos_api import make_engine
from pygcm.world.atmosphere import Atmosphere
from pygcm.world.orchestrator_spec import WorldOrchestratorConfigureSpec


class AtmosphereOrchestrator:
    def __init__(self, backend: Atmosphere | None = None) -> None:
        self.backend = backend

    def configure(
        self,
        *,
        spec: WorldOrchestratorConfigureSpec,
    ) -> None:
        aspec = spec.atmosphere
        if self.backend is None:
            engine = make_engine(
                "legacy_spectral",
                grid=aspec.grid,
                friction_map=aspec.friction_map,
                land_mask=aspec.land_mask,
                C_s_map=aspec.C_s_map,
                Cs_ocean=aspec.Cs_ocean,
                Cs_land=aspec.Cs_land,
                Cs_ice=aspec.Cs_ice,
            )
            self.backend = Atmosphere(engine=engine)
        try:
            if (
                hasattr(self.backend, "coupler")
                and getattr(self.backend, "coupler", None) is not None
                and hasattr(self.backend.coupler, "set_external_params")
            ):
                self.backend.coupler.set_external_params(
                    humidity_params=aspec.humidity_params,
                    energy_params=aspec.energy_params,
                )
        except Exception:
            pass

    def step_and_write(
        self,
        *,
        state,
        dt: float,
        h_eq,
        surface_in,
        column_in,
    ):
        if self.backend is None:
            raise RuntimeError("AtmosphereOrchestrator is not configured")
        return self.backend.time_step(
            state,
            dt,
            h_eq=h_eq,
            surface_in=surface_in,
            column_in=column_in,
        )


def ensure_atmos_orchestrator(atmos_obj) -> AtmosphereOrchestrator | object:
    if atmos_obj is None:
        return AtmosphereOrchestrator()
    if hasattr(atmos_obj, "configure") and hasattr(atmos_obj, "step_and_write"):
        return atmos_obj
    if hasattr(atmos_obj, "time_step"):
        return AtmosphereOrchestrator(backend=atmos_obj)
    raise TypeError("atmos object must provide configure+step_and_write or legacy time_step")
