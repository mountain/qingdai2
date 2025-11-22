from __future__ import annotations
"""
World API (OO + DBA friendly): unified interfaces for layers and couplers.

Intent
- Provide a normative, DRY and small interface layer (<=300 LOC) under world.api
  so that different numerical schemes (spectral, FD, barotropic/baroclinic) and
  different grid choices can plug into the same orchestration topology.
- Keep DoubleBufferingArray (DBA) ownership in orchestrators; engines operate on plain arrays.

Scope
- Layer Engines (array-in/array-out Protocols):
  * AtmosEngine: 1-layer (u,v,h) dynamics
  * OceanEngine: surface wind-driven slab or barotropic layer (uo, vo, eta, SST)
  * LandEngine: hydrology/thermodynamics state updates (shape-agnostic dict->dict)
  * EcologyEngine: vegetation/population state updates (shape-agnostic)
- Couplers (typed Ports → fluxes/updates Protocols):
  * RadiationCoupler, HumidityCoupler, CloudCoupler, WindStressCoupler
  * OceanSurfaceCoupler (Qnet/ice factors to SST update helpers)
  * RoutingCoupler (runoff routing & lakes integration)
- Factories: minimal name→constructor hooks to keep orchestrator code small.

Notes
- Use with typed ports from pygcm.world.ports (SurfaceToAtmosphere, ColumnProcessIn, etc.).
- Engines/couplers should be side-effect free w.r.t. DBA: no hidden swaps; pure arrays in/out.
- Backends may own internal persistent state (e.g., spectral transforms) but expose a pure step interface.

This API is an evolution/abstraction of P020 OO refactor, intended as the orchestrator contract.
"""

from typing import Protocol, Optional, Tuple, Dict, Any
import numpy as np

# Reuse existing ports (extend as needed)
from ..ports import (
    SurfaceToAtmosphere,
    AtmosphereToSurfaceFluxes,
    ColumnProcessIn,
    ColumnProcessOut,
)

# --------------------------------------------------------------------
# Layer Engine Protocols (array-in/array-out; orchestrator owns DBA)
# --------------------------------------------------------------------

class AtmosEngine(Protocol):
    """
    Minimal 1-layer atmosphere engine.
    step(u, v, h, dt, **kwargs) -> (u_next, v_next, h_next)
    Optional kwargs may include h_eq, Teq, base_albedo, cloud_cover, q, h_ice, etc.
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


class OceanEngine(Protocol):
    """
    Minimal ocean surface/barotropic engine.
    step(uo, vo, eta, sst, dt, **kwargs) -> (uo_next, vo_next, eta_next, sst_next)
    Optional kwargs may include wind_stress (tau_x, tau_y), Qnet, ice_mask, K parameters, etc.
    """
    def step(
        self,
        uo: np.ndarray,
        vo: np.ndarray,
        eta: np.ndarray,
        sst: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...


class LandEngine(Protocol):
    """
    Land surface/hydrology engine (shape-agnostic).
    Accepts/returns plain dictionaries of arrays to avoid premature commitment.
    step(inputs, dt, **kwargs) -> outputs
    Typical inputs: {'W_land':..., 'SWE':..., 'T_soil':..., 'mask':...}
    """
    def step(
        self,
        inputs: Dict[str, np.ndarray],
        dt: float,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        ...


class EcologyEngine(Protocol):
    """
    Ecology/biogeography engine (shape-agnostic).
    step(state, dt, **kwargs) -> state_next (dict of arrays or typed object)
    """
    def step(
        self,
        state: Dict[str, np.ndarray],
        dt: float,
        **kwargs: Any,
    ) -> Dict[str, np.ndarray]:
        ...


# --------------------------------------------------------------------
# Coupler Protocols (Ports → fluxes/updates)
# --------------------------------------------------------------------

class RadiationCoupler(Protocol):
    """
    Compute radiative terms for atmosphere/surface from ports and state.
    Returns (SW_sfc, LW_sfc) and optionally SW_atm/LW_atm if needed.
    """
    def compute(
        self,
        surface: SurfaceToAtmosphere,
        column: Optional[ColumnProcessIn],
        grid: Optional[Any],
        state: Any,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        ...


class HumidityCoupler(Protocol):
    """
    Compute evaporation/condensation and associated latent heat terms.
    Returns dict with E, LH (surface), P_cond, LH_release (atmos), optionally q_next.
    """
    def compute(
        self,
        surface: SurfaceToAtmosphere,
        column: ColumnProcessIn,
        grid: Optional[Any],
        state: Any,
        dt: float,
    ) -> Dict[str, np.ndarray]:
        ...


class CloudCoupler(Protocol):
    """
    Diagnose/advance cloud cover (and related optical properties) from dynamics and column state.
    Returns dict with cloud_next and optional optical depth proxies.
    """
    def compute(
        self,
        column: ColumnProcessIn,
        dynamics_fields: Dict[str, np.ndarray],
        grid: Optional[Any],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        ...


class WindStressCoupler(Protocol):
    """
    Compute surface wind stress for ocean (and optionally drag for atmosphere) from 10m winds and surface currents.
    Returns dict with tau_x, tau_y (and optional caps/efficiencies).
    """
    def compute(
        self,
        u10: np.ndarray,
        v10: np.ndarray,
        uo: Optional[np.ndarray],
        vo: Optional[np.ndarray],
        grid: Optional[Any],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        ...


class OceanSurfaceCoupler(Protocol):
    """
    Map surface energy terms to SST increment (Qnet / (rho cp H)) with ice modifiers.
    Returns dict with sst_next or dT_sst.
    """
    def compute(
        this,
        Qnet: np.ndarray,
        ice_mask: Optional[np.ndarray],
        params: Dict[str, Any],
        grid: Optional[Any],
        dt: float,
    ) -> Dict[str, np.ndarray]:
        ...


class RoutingCoupler(Protocol):
    """
    Route land runoff to ocean/lakes given a precomputed network.
    Returns dict with flow_accum, lake_updates, ocean_inflow diagnostics.
    """
    def compute(
        self,
        runoff_flux: np.ndarray,
        grid: Optional[Any],
        dt: float,
        **network: Any,
    ) -> Dict[str, np.ndarray]:
        ...


# --------------------------------------------------------------------
# Minimal Factories (optional; can be replaced by registry)
# --------------------------------------------------------------------

def make_atmos_engine(kind: str, **kwargs: Any) -> AtmosEngine:
    if kind == "legacy_spectral":
        from ..atmosphere_backend import LegacySpectralBackend
        return LegacySpectralBackend(**kwargs)  # type: ignore[return-value]
    if kind == "demo_relax":
        from ..atmos_kernels import DemoRelaxEngine
        return DemoRelaxEngine(**kwargs)
    raise ValueError(f"Unknown AtmosEngine kind: {kind!r}")


def make_coupler(kind: str, **kwargs: Any) -> Any:
    if kind == "radiation":
        # placeholder example; user can provide concrete module
        raise NotImplementedError("radiation coupler not wired; supply implementation")
    if kind == "humidity":
        raise NotImplementedError("humidity coupler not wired; supply implementation")
    if kind == "cloud":
        raise NotImplementedError("cloud coupler not wired; supply implementation")
    if kind == "wind_stress":
        raise NotImplementedError("wind_stress coupler not wired; supply implementation")
    if kind == "ocean_surface":
        raise NotImplementedError("ocean_surface coupler not wired; supply implementation")
    if kind == "routing":
        from ..routing import RiverRouting  # example: not a perfect coupler fit, but close
        return RiverRouting(**kwargs)  # type: ignore[return-value]
    if kind == "default":
        from ..coupler import Coupler, CouplerParams
        return Coupler(kwargs.get("params") or CouplerParams())  # type: ignore[return-value]
    raise ValueError(f"Unknown coupler kind: {kind!r}")
