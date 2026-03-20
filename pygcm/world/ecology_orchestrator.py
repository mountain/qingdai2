from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pygcm.ecology import EcologyAdapter
from pygcm.world.orchestrator_spec import WorldOrchestratorConfigureSpec


@dataclass(frozen=True)
class EcologyStepResult:
    albedo: np.ndarray


class EcologyOrchestrator:
    def __init__(self, backend: EcologyAdapter | None = None) -> None:
        self.backend = backend
        self.enabled = False
        self.land_mask: np.ndarray | None = None
        self.day_in_seconds = 86400.0
        self.soil_water_cap = 150.0
        self.lai_albedo_weight = 1.0
        self.albedo_couple = False
        self._accum = 0.0
        self._diag: dict[str, float] = {
            "subdaily_calls": 0.0,
            "daily_calls": 0.0,
            "last_albedo_mean": 0.0,
        }

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        es = spec.ecology
        self.enabled = bool(es.enabled)
        self.land_mask = np.asarray(es.land_mask)
        self.day_in_seconds = float(es.day_in_seconds)
        self.soil_water_cap = float(es.soil_water_cap)
        self.lai_albedo_weight = float(es.lai_albedo_weight)
        self.albedo_couple = bool(es.albedo_couple)
        if self.backend is None and self.enabled:
            self.backend = EcologyAdapter(spec.ocean.grid, self.land_mask)

    def apply_albedo(
        self,
        *,
        base_albedo: np.ndarray,
        insolation: np.ndarray,
        cloud_eff: np.ndarray,
        dt: float,
    ) -> EcologyStepResult:
        albedo = base_albedo
        if not self.enabled or self.backend is None or (not self.albedo_couple):
            self._diag["last_albedo_mean"] = float(np.nanmean(albedo))
            return EcologyStepResult(albedo=albedo)
        self._diag["subdaily_calls"] += 1.0
        alpha_eco = self.backend.step_subdaily(insolation, cloud_eff, dt)
        if alpha_eco is None:
            self._diag["last_albedo_mean"] = float(np.nanmean(albedo))
            return EcologyStepResult(albedo=albedo)
        alpha_mix = np.where(np.isfinite(alpha_eco), alpha_eco, albedo)
        land_mask = self.land_mask if self.land_mask is not None else np.zeros_like(albedo)
        out = np.where(
            land_mask == 1,
            np.clip(
                (1.0 - self.lai_albedo_weight) * albedo + self.lai_albedo_weight * alpha_mix,
                0.0,
                1.0,
            ),
            albedo,
        )
        self._diag["last_albedo_mean"] = float(np.nanmean(out))
        return EcologyStepResult(albedo=out)

    def step_daily_if_needed(self, *, state, dt: float) -> None:
        self._accum += dt
        if self.backend is None or self._accum < self.day_in_seconds:
            return
        if state.hydro is not None:
            soil_idx = np.clip(
                state.hydro.W_land.write / max(1.0e-6, self.soil_water_cap),
                0.0,
                1.0,
            )
            self.backend.step_daily(soil_idx)
            self._diag["daily_calls"] += 1.0
        self._accum -= self.day_in_seconds

    def diagnostics(self) -> dict[str, float]:
        return dict(self._diag)


class _LegacyEcologyAdapter:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.day_in_seconds = 86400.0
        self.soil_water_cap = 150.0
        self.lai_albedo_weight = 1.0
        self.albedo_couple = False
        self.land_mask: np.ndarray | None = None
        self._accum = 0.0
        self.enabled = True
        self._diag: dict[str, float] = {
            "subdaily_calls": 0.0,
            "daily_calls": 0.0,
            "last_albedo_mean": 0.0,
        }

    def configure(self, *, spec: WorldOrchestratorConfigureSpec) -> None:
        es = spec.ecology
        self.enabled = bool(es.enabled)
        self.land_mask = np.asarray(es.land_mask)
        self.day_in_seconds = float(es.day_in_seconds)
        self.soil_water_cap = float(es.soil_water_cap)
        self.lai_albedo_weight = float(es.lai_albedo_weight)
        self.albedo_couple = bool(es.albedo_couple)

    def apply_albedo(
        self,
        *,
        base_albedo: np.ndarray,
        insolation: np.ndarray,
        cloud_eff: np.ndarray,
        dt: float,
    ) -> EcologyStepResult:
        if (not self.enabled) or (not self.albedo_couple):
            self._diag["last_albedo_mean"] = float(np.nanmean(base_albedo))
            return EcologyStepResult(albedo=base_albedo)
        self._diag["subdaily_calls"] += 1.0
        alpha_eco = self.backend.step_subdaily(insolation, cloud_eff, dt)
        if alpha_eco is None:
            self._diag["last_albedo_mean"] = float(np.nanmean(base_albedo))
            return EcologyStepResult(albedo=base_albedo)
        alpha_mix = np.where(np.isfinite(alpha_eco), alpha_eco, base_albedo)
        land_mask = self.land_mask if self.land_mask is not None else np.zeros_like(base_albedo)
        out = np.where(
            land_mask == 1,
            np.clip(
                (1.0 - self.lai_albedo_weight) * base_albedo + self.lai_albedo_weight * alpha_mix,
                0.0,
                1.0,
            ),
            base_albedo,
        )
        self._diag["last_albedo_mean"] = float(np.nanmean(out))
        return EcologyStepResult(albedo=out)

    def step_daily_if_needed(self, *, state, dt: float) -> None:
        self._accum += dt
        if self._accum < self.day_in_seconds:
            return
        if state.hydro is not None:
            soil_idx = np.clip(
                state.hydro.W_land.write / max(1.0e-6, self.soil_water_cap),
                0.0,
                1.0,
            )
            self.backend.step_daily(soil_idx)
            self._diag["daily_calls"] += 1.0
        self._accum -= self.day_in_seconds

    def diagnostics(self) -> dict[str, float]:
        if hasattr(self.backend, "diagnostics"):
            try:
                d = self.backend.diagnostics()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        return dict(self._diag)


def ensure_ecology_orchestrator(eco_obj) -> EcologyOrchestrator | object:
    if eco_obj is None:
        return EcologyOrchestrator()
    if (
        hasattr(eco_obj, "configure")
        and hasattr(eco_obj, "apply_albedo")
        and hasattr(eco_obj, "step_daily_if_needed")
    ):
        return eco_obj
    if hasattr(eco_obj, "step_subdaily") and hasattr(eco_obj, "step_daily"):
        return _LegacyEcologyAdapter(eco_obj)
    raise TypeError(
        "ecology object must provide orchestrator interface or legacy step_subdaily+step_daily"
    )
