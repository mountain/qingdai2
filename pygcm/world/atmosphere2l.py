from __future__ import annotations

"""
DBA-friendly minimal two-layer (baroclinic) atmosphere scaffold.

Purpose
- Demonstrate how the existing DBA orchestrator pattern scales from 1-layer SW to a
  minimal 2-layer baroclinic model (lowest computational cost variant).
- Keep pure-kernel + thin-orchestrator design so JAX/NumPy duality and backends remain possible.
- Avoid invasive changes to global WorldState by providing a self-contained TwoLayerState
  builder; later, this can be integrated into world/state.py when desired.

Design
- State (TwoLayerState) holds six DBAs: (u1, v1, h1) for upper-layer, (u2, v2, h2) for lower-layer.
- Orchestrator (Atmosphere2L) reads from .read buffers and writes to .write buffers only.
- Pure kernels implement a toy barotropic/baroclinic coupling (linear relaxation + weak internal mode).
  Replace with proper two-layer shallow-water numerics as needed.
- Optional backends can be added later (e.g., LegacyTwoLayerBackend) similar to AtmosphereBackend.

Diagnostics
- For quick checks, helper functions provide layer-combined barotropic proxies (mass, momentum, AAM, KE).
  For full invariants, prefer world/diagnostics with proper extensions.

Usage
-----
from pygcm.world.atmosphere2l import new_two_layer_state, TwoLayerParams, Atmosphere2L

shape = (n_lat, n_lon)
s2 = new_two_layer_state(shape)
atm2 = Atmosphere2L(TwoLayerParams())

# one step
atm2.time_step(s2, dt)

# driver handles swap:
s2.swap_all()
"""

from dataclasses import dataclass

import numpy as np

try:
    from pygcm.jax_compat import xp
except Exception:
    xp = np

from pygcm.numerics.double_buffer import DoubleBufferingArray as DBA

# ---------------------------
# Two-layer DBA state
# ---------------------------


@dataclass
class TwoLayerState:
    u1: DBA
    v1: DBA
    h1: DBA
    u2: DBA
    v2: DBA
    h2: DBA

    def swap_all(self) -> None:
        for name in ("u1", "v1", "h1", "u2", "v2", "h2"):
            getattr(self, name).swap()


def _new_dba(shape, dtype=np.float64, fill=0.0) -> DBA:
    dba = DBA(shape, dtype=dtype, initial_value=fill)
    return dba


def new_two_layer_state(
    shape: tuple[int, int], *, dtype=np.float64, u1=0.0, v1=0.0, h1=0.0, u2=0.0, v2=0.0, h2=0.0
) -> TwoLayerState:
    return TwoLayerState(
        u1=_new_dba(shape, dtype=dtype, fill=u1),
        v1=_new_dba(shape, dtype=dtype, fill=v1),
        h1=_new_dba(shape, dtype=dtype, fill=h1),
        u2=_new_dba(shape, dtype=dtype, fill=u2),
        v2=_new_dba(shape, dtype=dtype, fill=v2),
        h2=_new_dba(shape, dtype=dtype, fill=h2),
    )


# ---------------------------
# Pure-kernel demo numerics
# ---------------------------


@dataclass
class TwoLayerParams:
    # layer reference depths (m); only enter demo coupling scales
    H1_m: float = 5000.0
    H2_m: float = 5000.0
    # gravities
    g: float = 9.81  # external mode surface gravity
    g_red: float = 0.2  # reduced gravity for internal mode (demo scale)
    # simple linear relaxations (s)
    tau_u_s: float = 2.0 * 24 * 3600.0
    tau_h_s: float = 2.0 * 24 * 3600.0
    # diffusive smoothing (demo ∇^4)
    k4_u: float = 0.0
    k4_h: float = 0.0
    # demo metrics
    dx_m: float = 1.0e5
    dy_m: float = 1.0e5


def _relax_tendency(field: np.ndarray, tau_s: float) -> np.ndarray:
    if tau_s <= 0.0:
        return xp.zeros_like(field)
    return -field / tau_s


def _laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    f_ip1 = xp.roll(f, -1, axis=1)
    f_im1 = xp.roll(f, +1, axis=1)
    f_jp1 = xp.concatenate([f[1:, :], f[-1:, :]], axis=0)
    f_jm1 = xp.concatenate([f[:1, :], f[:-1, :]], axis=0)
    return (f_ip1 + f_im1 - 2.0 * f) / (dx * dx) + (f_jp1 + f_jm1 - 2.0 * f) / (dy * dy)


def _hyperdiff(f: np.ndarray, dx: float, dy: float, k4: float) -> np.ndarray:
    if k4 == 0.0:
        return xp.zeros_like(f)
    lap = _laplacian(f, dx, dy)
    lap2 = _laplacian(lap, dx, dy)
    return -k4 * lap2


def two_layer_tendencies(
    u1: np.ndarray,
    v1: np.ndarray,
    h1: np.ndarray,
    u2: np.ndarray,
    v2: np.ndarray,
    h2: np.ndarray,
    p: TwoLayerParams,
    h1_eq: np.ndarray | None,
    h2_eq: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    """
    Demo tendencies:
      - Relaxation on u1,u2,h1,h2
      - Small ∇^4 damping on u and h
      - Weak baroclinic coupling: internal mode exchange proportional to (h1 - h2)
    Replace with proper two-layer SW numerics (pressure gradient, Coriolis, advection) as needed.
    """
    du1 = _relax_tendency(u1, p.tau_u_s) + _hyperdiff(u1, p.dx_m, p.dy_m, p.k4_u)
    dv1 = _relax_tendency(v1, p.tau_u_s) + _hyperdiff(v1, p.dx_m, p.dy_m, p.k4_u)
    du2 = _relax_tendency(u2, p.tau_u_s) + _hyperdiff(u2, p.dx_m, p.dy_m, p.k4_u)
    dv2 = _relax_tendency(v2, p.tau_u_s) + _hyperdiff(v2, p.dx_m, p.dy_m, p.k4_u)

    # height relaxations (toward equilibria if provided)
    target1 = 0.0 if h1_eq is None else h1_eq
    target2 = 0.0 if h2_eq is None else h2_eq
    dh1 = -(h1 - target1) / p.tau_h_s + _hyperdiff(h1, p.dx_m, p.dy_m, p.k4_h)
    dh2 = -(h2 - target2) / p.tau_h_s + _hyperdiff(h2, p.dx_m, p.dy_m, p.k4_h)

    # weak internal mode coupling (demo): push h1 and h2 toward each other with strength ~ g_red
    # This mimics a crude baroclinic restoring without computing gradients.
    alpha = 0.5 * p.g_red / max(p.H1_m + p.H2_m, 1.0)
    d = h1 - h2
    dh1 += -alpha * d
    dh2 += +alpha * d

    return du1, dv1, dh1, du2, dv2, dh2


# ---------------------------
# Orchestrator
# ---------------------------


class Atmosphere2L:
    """
    DBA-friendly two-layer atmosphere orchestrator (demo).
    """

    def __init__(self, params: TwoLayerParams | None = None) -> None:
        self.params = params or TwoLayerParams()
        self.backend = None  # placeholder for future two-layer backend

    def time_step(
        self,
        state2: TwoLayerState,
        dt: float,
        *,
        h1_eq: np.ndarray | None = None,
        h2_eq: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> None:
        """
        Advance two-layer state by one step. Writes to WRITE buffers only; no swap.
        """
        u1 = state2.u1.read
        v1 = state2.v1.read
        h1 = state2.h1.read
        u2 = state2.u2.read
        v2 = state2.v2.read
        h2 = state2.h2.read

        if self.backend is not None:
            # Future: delegate to a two-layer backend with an interface like:
            # u1n,v1n,h1n,u2n,v2n,h2n = self.backend.step2l(u1,v1,h1, u2,v2,h2, dt, **(extra or {}))
            # state2.u1.write[:] = u1n; ... etc.
            raise NotImplementedError("Two-layer backend is not yet plugged. Use demo kernels.")
        else:
            du1, dv1, dh1, du2, dv2, dh2 = two_layer_tendencies(
                u1, v1, h1, u2, v2, h2, self.params, h1_eq, h2_eq
            )
            state2.u1.write[:] = u1 + du1 * dt
            state2.v1.write[:] = v1 + dv1 * dt
            state2.h1.write[:] = h1 + dh1 * dt
            state2.u2.write[:] = u2 + du2 * dt
            state2.v2.write[:] = v2 + dv2 * dt
            state2.h2.write[:] = h2 + dh2 * dt


# ---------------------------
# Quick diagnostics helpers
# ---------------------------


def barotropic_mass(h1: np.ndarray, h2: np.ndarray) -> float:
    return float(xp.sum(xp.nan_to_num(h1 + h2)))


def barotropic_ke(
    u1: np.ndarray, v1: np.ndarray, h1: np.ndarray, u2: np.ndarray, v2: np.ndarray, h2: np.ndarray
) -> float:
    return float(xp.sum(0.5 * (h1 * (u1 * u1 + v1 * v1) + h2 * (u2 * u2 + v2 * v2))))


def baroclinic_contrast(h1: np.ndarray, h2: np.ndarray) -> float:
    # proxy for internal mode activity
    return float(xp.sum(xp.abs(h1 - h2)))
