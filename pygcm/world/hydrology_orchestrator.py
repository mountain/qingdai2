from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygcm import hydrology as hydro
from pygcm.world.orchestrator_spec import WorldOrchestratorConfigureSpec


@dataclass(frozen=True)
class HydrologyStepResult:
    runoff_flux: np.ndarray


class HydrologyOrchestrator:
    def __init__(self, backend=None) -> None:
        self.backend = backend
        self.land_mask: np.ndarray | None = None
        self.hparams = None
        self._diag: dict[str, float] = {
            "runoff_mean": 0.0,
            "evap_mean": 0.0,
            "precip_mean": 0.0,
            "steps": 0.0,
        }

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        self.land_mask = np.asarray(spec.hydrology.land_mask)
        self.hparams = spec.hydrology.hydrology_params

    def step_and_write(
        self,
        *,
        state,
        Ta: np.ndarray,
        precip_flux: np.ndarray,
        evap_flux: np.ndarray,
        dt: float,
        ref: np.ndarray,
    ) -> HydrologyStepResult:
        if self.land_mask is None or self.hparams is None:
            raise RuntimeError("HydrologyOrchestrator is not configured")
        if state.hydro is None:
            self._diag["steps"] += 1.0
            self._diag["runoff_mean"] = 0.0
            self._diag["evap_mean"] = float(np.nanmean(evap_flux))
            self._diag["precip_mean"] = float(np.nanmean(precip_flux))
            return HydrologyStepResult(runoff_flux=np.zeros_like(ref))
        P_rain, P_snow, _ = hydro.partition_precip_phase_smooth(
            precip_flux, Ta, self.hparams.snow_thresh_K, self.hparams.snow_t_band_K
        )
        land = (self.land_mask == 1).astype(float)
        SWE_next, melt_flux, _c, _a = hydro.snowpack_step(
            state.hydro.SWE.read,
            P_snow * land,
            Ta,
            self.hparams,
            dt,
        )
        W_next, runoff_flux = hydro.update_land_bucket(
            state.hydro.W_land.read,
            P_rain * land + melt_flux * land,
            evap_flux * land,
            self.hparams,
            dt,
        )
        state.hydro.SWE.write[:] = SWE_next
        state.hydro.W_land.write[:] = W_next
        self._diag["steps"] += 1.0
        self._diag["runoff_mean"] = float(np.nanmean(runoff_flux))
        self._diag["evap_mean"] = float(np.nanmean(evap_flux))
        self._diag["precip_mean"] = float(np.nanmean(precip_flux))
        return HydrologyStepResult(runoff_flux=runoff_flux)

    def diagnostics(self) -> dict[str, float]:
        return dict(self._diag)


class _LegacyHydrologyAdapter:
    def __init__(self, backend) -> None:
        self.backend = backend
        self._diag: dict[str, float] = {
            "runoff_mean": 0.0,
            "evap_mean": 0.0,
            "precip_mean": 0.0,
            "steps": 0.0,
        }

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        if hasattr(self.backend, "configure"):
            self.backend.configure(spec=spec)

    def step_and_write(
        self,
        *,
        state,
        Ta: np.ndarray,
        precip_flux: np.ndarray,
        evap_flux: np.ndarray,
        dt: float,
        ref: np.ndarray,
    ) -> HydrologyStepResult:
        out = self.backend.step(
            state=state,
            Ta=Ta,
            precip_flux=precip_flux,
            evap_flux=evap_flux,
            dt=dt,
            ref=ref,
        )
        if isinstance(out, dict) and "runoff_flux" in out:
            runoff = np.asarray(out["runoff_flux"])
            self._diag["steps"] += 1.0
            self._diag["runoff_mean"] = float(np.nanmean(runoff))
            self._diag["evap_mean"] = float(np.nanmean(evap_flux))
            self._diag["precip_mean"] = float(np.nanmean(precip_flux))
            return HydrologyStepResult(runoff_flux=runoff)
        if hasattr(out, "runoff_flux"):
            runoff = np.asarray(out.runoff_flux)
            self._diag["steps"] += 1.0
            self._diag["runoff_mean"] = float(np.nanmean(runoff))
            self._diag["evap_mean"] = float(np.nanmean(evap_flux))
            self._diag["precip_mean"] = float(np.nanmean(precip_flux))
            return HydrologyStepResult(runoff_flux=runoff)
        runoff = np.zeros_like(ref)
        self._diag["steps"] += 1.0
        self._diag["runoff_mean"] = 0.0
        self._diag["evap_mean"] = float(np.nanmean(evap_flux))
        self._diag["precip_mean"] = float(np.nanmean(precip_flux))
        return HydrologyStepResult(runoff_flux=runoff)

    def diagnostics(self) -> dict[str, float]:
        if hasattr(self.backend, "diagnostics"):
            try:
                d = self.backend.diagnostics()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        return dict(self._diag)


def ensure_hydrology_orchestrator(hydro_obj) -> HydrologyOrchestrator | object:
    if hydro_obj is None:
        return HydrologyOrchestrator()
    if hasattr(hydro_obj, "configure") and hasattr(hydro_obj, "step_and_write"):
        return hydro_obj
    if hasattr(hydro_obj, "step"):
        return _LegacyHydrologyAdapter(hydro_obj)
    raise TypeError("hydrology object must provide configure+step_and_write or legacy step")
