from __future__ import annotations
"""
Atmosphere API: unified interfaces for engines and couplers (OO + DBA friendly).

Intent
- Keep every file small and focused (<300 LOC target) and DRY.
- Provide a single, normative interface so different numerical schemes
  (spectral, FD, barotropic, baroclinic) can plug into the same orchestrator.
- Keep DBA ownership in orchestrators; engines operate on plain arrays.

Key concepts
- AtmosEngine: array-in/array-out engine for 1-layer (u,v,h).
- AtmosCoupler: computes interface fluxes and in-column updates from typed ports.
- Ports live in pygcm/world/ports.py (SurfaceToAtmosphere, ColumnProcessIn, ...).

Notes
- Engines should be pure from the orchestrator's perspective: no hidden swaps,
  no mutation of orchestrator-owned buffers.
- Backends may keep internal persistent state (e.g., spectral transforms) but
  must expose array → array step().
"""

from typing import Optional, Tuple, Protocol, Any
import numpy as np
from .ports import SurfaceToAtmosphere, AtmosphereToSurfaceFluxes, ColumnProcessIn, ColumnProcessOut


# --------------------------
# Engines (1-layer)
# --------------------------

class AtmosEngine(Protocol):
    """
    Minimal 1-layer engine interface.
    step(u,v,h, dt, **kwargs) -> (u_next, v_next, h_next)
    """
    def step(
        self,
        u: np.ndarray,
        v: np.ndarray,
        h: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


# --------------------------
# Coupler
# --------------------------

class AtmosCoupler(Protocol):
    """
    Coupler interface. Converts surface/column inputs to interface fluxes
    and in-column outputs (without mutating the DBA state).
    """
    def compute(
        self,
        surface_in: Optional[SurfaceToAtmosphere],
        column_in: Optional[ColumnProcessIn],
        grid: Optional[Any],
        state: Any,
        dt: float,
    ) -> Tuple[Optional[AtmosphereToSurfaceFluxes], Optional[ColumnProcessOut]]:
        ...


# --------------------------
# Factories (optional)
# --------------------------

def make_engine(kind: str, **kwargs: Any) -> AtmosEngine:
    """
    Simple factory to build engines by name.
    kind:
      - "legacy_spectral": uses LegacySpectralBackend bridge (spectral/FD mixed)
      - "demo_relax":     demo linear-relaxation kernel engine (imported lazily)
    """
    if kind == "legacy_spectral":
        from .atmosphere_backend import LegacySpectralBackend
        return LegacySpectralBackend(**kwargs)  # type: ignore[return-value]
    if kind == "demo_relax":
        from .atmos_kernels import DemoRelaxEngine
        return DemoRelaxEngine(**kwargs)
    raise ValueError(f"Unknown AtmosEngine kind: {kind!r}")


def make_coupler(kind: str = "default", **kwargs: Any) -> AtmosCoupler:
    """
    Small factory to construct a coupler.
    kind:
      - "default": world.coupler.Coupler (uses energy/humidity if available)
    """
    if kind == "default":
        from .coupler import Coupler, CouplerParams
        params = kwargs.get("params") or CouplerParams()
        return Coupler(params)  # type: ignore[return-value]
    raise ValueError(f"Unknown AtmosCoupler kind: {kind!r}")
