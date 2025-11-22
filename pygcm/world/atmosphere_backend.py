from __future__ import annotations

"""
Atmosphere backends for DBA orchestrator.

This module lets us plug different numerical engines behind the DBA-friendly
world/atmosphere orchestrator without changing the DBA contract.

Backends should:
- Accept plain arrays (numpy/jax) as inputs (u,v,h, and optional auxiliaries).
- Return next-step arrays (u,v,h) as plain arrays.
- Remain pure from the caller's perspective: the orchestrator owns DBA read/write;
  the backend simply computes new fields (internally it may keep a persistent
  engine instance, like the legacy SpectralModel).

Provided implementations:
- LegacySpectralBackend: bridges to pygcm.dynamics.SpectralModel (mixed spectral/FD engine).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any, Dict

import numpy as np

try:
    from pygcm.jax_compat import xp  # type: ignore
except Exception:
    xp = np  # fallback


# -----------------------
# Backend base interface
# -----------------------

class AtmosphereBackend:
    """
    Minimal interface for an atmosphere backend engine.

    step(u, v, h, dt, **kwargs) -> (u_next, v_next, h_next)

    Inputs/outputs are plain arrays; DBA is handled by the orchestrator.
    """

    def step(self,
             u: np.ndarray,
             v: np.ndarray,
             h: np.ndarray,
             dt: float,
             **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


# ---------------------------------
# Legacy SpectralModel bridge
# ---------------------------------

@dataclass
class LegacySpectralParams:
    H_m: float = 8000.0
    tau_rad_s: float = 10 * 24 * 3600.0
    greenhouse_factor: float = 0.40
    # Optional capacities (for energy coupling consistency, use same defaults as run_simulation)
    Cs_ocean: float = 1000.0 * 4200.0 * 50.0  # rho_w * cp_w * H_mld
    Cs_land: float = 3.0e6
    Cs_ice: float = 5.0e6


class LegacySpectralBackend(AtmosphereBackend):
    """
    Bridge to pygcm.dynamics.SpectralModel.

    Notes:
    - We instantiate a SpectralModel once and keep it persistent (holds friction,
      base albedo coupling, greenhouse parameters, etc.).
    - Each call:
        1) copy u,v,h (and optionally T_s/cloud/q/h_ice if provided in kwargs) into the gcm state
        2) call gcm.time_step(Teq, dt)
        3) return gcm.u, gcm.v, gcm.h arrays (plain arrays)
    - The DBA is handled by the orchestrator (world/atmosphere).
    """

    def __init__(self,
                 grid,
                 friction_map: np.ndarray,
                 land_mask: np.ndarray,
                 *,
                 params: Optional[LegacySpectralParams] = None,
                 C_s_map: Optional[np.ndarray] = None,
                 Cs_ocean: Optional[float] = None,
                 Cs_land: Optional[float] = None,
                 Cs_ice: Optional[float] = None,
                 greenhouse_factor: Optional[float] = None) -> None:
        from pygcm.dynamics import SpectralModel  # import here to avoid world-level hard dependency

        p = params or LegacySpectralParams()
        self.grid = grid
        self.land_mask = land_mask

        # Allow overrides while keeping sane defaults
        if greenhouse_factor is not None:
            p.greenhouse_factor = greenhouse_factor
        if Cs_ocean is not None:
            p.Cs_ocean = Cs_ocean
        if Cs_land is not None:
            p.Cs_land = Cs_land
        if Cs_ice is not None:
            p.Cs_ice = Cs_ice

        self.gcm = SpectralModel(
            grid,
            friction_map,
            H=p.H_m,
            tau_rad=p.tau_rad_s,
            greenhouse_factor=p.greenhouse_factor,
            C_s_map=C_s_map,
            land_mask=land_mask,
            Cs_ocean=p.Cs_ocean,
            Cs_land=p.Cs_land,
            Cs_ice=p.Cs_ice,
        )

    def step(self,
             u: np.ndarray,
             v: np.ndarray,
             h: np.ndarray,
             dt: float,
             **kwargs: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        kwargs may include (all optional, pass-through if present):
          - Teq: ndarray, equilibrium temperature forcing for the legacy engine
          - T_s: ndarray, surface temperature (for coupled paths)
          - cloud_cover: ndarray
          - q: ndarray
          - h_ice: ndarray
        """
        # 1) Copy read fields into legacy engine
        self.gcm.u = xp.asarray(u)
        self.gcm.v = xp.asarray(v)
        self.gcm.h = xp.asarray(h)

        # Optional coupled fields (best-effort; ignored if not present in legacy model)
        if "T_s" in kwargs and hasattr(self.gcm, "T_s"):
            self.gcm.T_s = xp.asarray(kwargs["T_s"])
        if "cloud_cover" in kwargs and hasattr(self.gcm, "cloud_cover"):
            self.gcm.cloud_cover = xp.clip(xp.asarray(kwargs["cloud_cover"]), 0.0, 1.0)
        if "q" in kwargs and hasattr(self.gcm, "q"):
            self.gcm.q = xp.asarray(kwargs["q"])
        if "h_ice" in kwargs and hasattr(self.gcm, "h_ice"):
            self.gcm.h_ice = xp.maximum(xp.asarray(kwargs["h_ice"]), 0.0)

        # 2) Advance legacy engine
        Teq = kwargs.get("Teq", None)
        if Teq is None:
            # If no forcing is provided, do a neutral step (no-op forcing)
            # by passing a zero field of same shape as h (legacy API requires a Teq).
            Teq = xp.zeros_like(h, dtype=h.dtype)
        self.gcm.time_step(Teq, float(dt))

        # 3) Return next-step arrays (plain arrays)
        return (np.asarray(self.gcm.u),
                np.asarray(self.gcm.v),
                np.asarray(self.gcm.h))
