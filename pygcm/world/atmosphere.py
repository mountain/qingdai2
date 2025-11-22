from __future__ import annotations
"""
Atmosphere orchestrator (DBA-friendly) — DRY, <300 LOC, engine/coupler pluggable.

Design goals
- Keep this file small and focused on orchestration only (no physics kernels here).
- Unify numerical schemes (spectral, FD, barotropic) behind a single engine interface.
- Make interface/column couplings explicit via typed ports and a separate coupler.
- Preserve DBA contract: read .read, write .write, never swap here.

Key collaborators
- Engines (array-in/array-out): pygcm/world/atmos_api.py::AtmosEngine
  * default: DemoRelaxEngine (pygcm/world/atmos_kernels.py)
  * legacy spectral bridge: LegacySpectralBackend (pygcm/world/atmosphere_backend.py)
- Coupler (ports-in → fluxes/out): pygcm/world/coupler.py::Coupler
- Typed ports: pygcm/world/ports.py
- Step diagnostics (read/write invariants): pygcm/world/diagnostics.py

Usage
-----
from pygcm.world.atmosphere import Atmosphere
from pygcm.world.atmos_api import make_engine, make_coupler
from pygcm.world.ports import SurfaceToAtmosphere, ColumnProcessIn

engine = make_engine("legacy_spectral", grid=..., friction_map=..., land_mask=..., C_s_map=...)
coupler = make_coupler("default")
atm = Atmosphere(engine=engine, coupler=coupler)

fluxes, col_out = atm.time_step(state, dt,
                                h_eq=None,
                                surface_in=SurfaceToAtmosphere(...),
                                column_in=ColumnProcessIn(...))
# world swaps at step end:
state.swap_all()
"""

from dataclasses import dataclass
from typing import Optional, Tuple as _Tuple, Any

import numpy as np

try:
    from pygcm.jax_compat import xp  # numpy or jax.numpy (engines operate on arrays)
except Exception:
    xp = np  # fallback

from .ports import (
    SurfaceToAtmosphere,
    AtmosphereToSurfaceFluxes,
    ColumnProcessIn,
    ColumnProcessOut,
)
from .atmos_api import AtmosEngine, AtmosCoupler, make_engine, make_coupler


@dataclass
class AtmosParams:
    """Optional wrapper to carry defaults that engines may use (kept minimal here)."""
    # These are not used directly in this file; engines own their physical params.
    pass


class Atmosphere:
    """
    Thin DBA orchestrator for 1-layer atmosphere.

    Responsibilities
    - Pull READ arrays (u, v, h) from state
    - Delegate dynamics to engine.step(u, v, h, dt, **kwargs) and WRITE results
    - Delegate interface/column couplings to coupler.compute(...) and return outputs
    """

    def __init__(self,
                 params: Optional[AtmosParams] = None,
                 *,
                 engine: Optional[AtmosEngine] = None,
                 coupler: Optional[AtmosCoupler] = None,
                 engine_kind: str = "demo_relax",
                 engine_kwargs: Optional[dict] = None,
                 coupler_kind: str = "default",
                 coupler_kwargs: Optional[dict] = None) -> None:
        self.params = params or AtmosParams()
        # Engines/coupler can be injected or created via factories
        self.engine: AtmosEngine = engine or make_engine(engine_kind, **(engine_kwargs or {}))
        self.coupler: AtmosCoupler = coupler or make_coupler(coupler_kind, **(coupler_kwargs or {}))

    def time_step(self,
                  state,
                  dt: float,
                  *,
                  h_eq: Optional[np.ndarray] = None,
                  surface_in: Optional[SurfaceToAtmosphere] = None,
                  column_in: Optional[ColumnProcessIn] = None) -> _Tuple[Optional[AtmosphereToSurfaceFluxes], Optional[ColumnProcessOut]]:
        """
        Advance (u, v, h) one step (WRITE only; no swap). Return (fluxes, col_out).

        Parameters
        ----------
        state : world state (with DBA fields state.atmos.u/v/h)
        dt : seconds
        h_eq : optional height equilibrium field (engine-specific optional)
        surface_in : SurfaceToAtmosphere | None
        column_in : ColumnProcessIn | None
        """
        # READ buffers
        u = state.atmos.u.read
        v = state.atmos.v.read
        h = state.atmos.h.read

        # Delegate to engine (arrays only)
        u_next, v_next, h_next = self.engine.step(u, v, h, float(dt), h_eq=h_eq)

        # WRITE buffers
        state.atmos.u.write[:] = u_next
        state.atmos.v.write[:] = v_next
        state.atmos.h.write[:] = h_next

        # Coupling (pure computation, array-in/array-out)
        fluxes, col_out = self.coupler.compute(surface_in, column_in, getattr(state, "grid", None), state, float(dt))
        return fluxes, col_out


# --- Optional: diagnostics convenience wrappers (proxy to world.diagnostics) ---

def invariants_from_read(state, grid, mask=None):
    from .diagnostics import invariants_from_read as _ifr
    return _ifr(state, grid, mask)


def invariants_from_write(state, grid, mask=None):
    from .diagnostics import invariants_from_write as _ifw
    return _ifw(state, grid, mask)


def diagnostics_report_read_write(state, grid, mask=None):
    from .diagnostics import diagnostics_report as _rep, invariants_from_read as _ifr, invariants_from_write as _ifw
    prev = _ifr(state, grid, mask)
    nxt = _ifw(state, grid, mask)
    return _rep(prev, nxt)
