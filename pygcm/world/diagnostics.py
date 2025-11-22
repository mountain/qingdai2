from __future__ import annotations

"""
DBA-friendly diagnostics utilities.

Purpose
- Provide clear, side-effect-free invariants/diagnostics that work with DoubleBufferingArray (DBA)
  by accepting either DBA fields (we read .read) or plain arrays.
- Support step-level "pre/post" checks by evaluating on read-buffers (old state) and write-buffers
  (next state before swap_all), so conservation deltas can be computed without interfering with
  the driver’s buffer swap.

Notes
- All functions are pure (no global mutation). Callers decide where to log/print.
- Weighting uses cos(phi) area weights. Absolute multiplicative constants (a^2, ΔφΔλ) are omitted
  so that step-wise deltas remain meaningful and comparable on a fixed grid. If absolute units are
  desired, those constants can be reinstated uniformly.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    # Optional: if JAX backend is enabled, xp becomes jax.numpy; otherwise numpy
    from pygcm.jax_compat import xp
except Exception:
    xp = np  # Fallback to numpy

from pygcm import constants


def _as_array(x: Any):
    """
    Accept either a DBA-like object with .read or a plain ndarray, and return a plain array.
    """
    if hasattr(x, "read"):
        return x.read  # DBA → ndarray view
    return x


def area_weights(grid, normalize: bool = False, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Return cos(phi) area weights (shape [lat,lon]).
    If normalize=True, divide by the global sum (masked if provided).

    Parameters
    ----------
    grid : object with lat_mesh (deg)
    normalize : bool
    mask : optional boolean or {0,1} array, where 1/True keeps, 0/False excludes

    Returns
    -------
    w : ndarray
    """
    phi = xp.deg2rad(grid.lat_mesh)
    w = xp.maximum(xp.cos(phi), 0.0)
    if mask is not None:
        w = w * mask
    if normalize:
        s = float(xp.sum(w)) + 1e-15
        w = w / s
    return w


def integrate(field: np.ndarray, w: np.ndarray) -> float:
    """
    Weighted integral (up to a global constant factor).
    """
    return float(xp.sum(xp.nan_to_num(field) * w))


def mass_integral(h: np.ndarray, w: np.ndarray) -> float:
    """
    "Mass" integral for shallow-water layer (proportional to ∫ h dA).
    """
    return integrate(h, w)


def zonal_momentum(u: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
    """
    Pseudo-zonal momentum (proportional to ∫ h u dA).
    Useful for step-wise deltas on a fixed grid.
    """
    return integrate(h * u, w)


def axial_angular_momentum(u: np.ndarray, h: np.ndarray, grid, w: np.ndarray) -> float:
    """
    Axial Absolute Angular Momentum (AAM) proxy for shallow water.

    For a thin layer of depth h and zonal wind u on a sphere:
        m = a cos(phi) * (u + Ω a cos(phi))    [per-mass about spin axis]
    Total AAM ~ ∫ h * m dA.

    We compute the integral up to a constant factor (a^2, ΔφΔλ omitted for step-wise comparisons).
    """
    a = float(constants.PLANET_RADIUS)
    Omega = float(constants.PLANET_OMEGA)
    phi = xp.deg2rad(grid.lat_mesh)
    cosphi = xp.cos(phi)
    m = a * cosphi * (u + Omega * a * cosphi)
    return integrate(h * m, w)


def kinetic_energy(u: np.ndarray, v: np.ndarray, h: np.ndarray, w: np.ndarray) -> float:
    """
    Kinetic energy proxy: ∫ 0.5 h (u^2 + v^2) dA (up to constant factor).
    """
    return integrate(0.5 * h * (u * u + v * v), w)


def potential_energy(h: np.ndarray, w: np.ndarray, g: float = constants.G) -> float:
    """
    Potential energy proxy: ∫ 0.5 g h^2 dA (up to constant factor).
    """
    return integrate(0.5 * float(g) * h * h, w)


@dataclass
class Invariants:
    mass: float
    px: float
    aam: float
    ke: float
    pe: float


def invariants_from_read(state, grid, mask: np.ndarray | None = None) -> Invariants:
    """
    Compute invariants on READ buffers (current/old state).
    """
    u = _as_array(state.atmos.u)
    v = _as_array(state.atmos.v)
    h = _as_array(state.atmos.h)

    w = area_weights(grid, normalize=False, mask=mask)

    m = mass_integral(h, w)
    px = zonal_momentum(u, h, w)
    aam = axial_angular_momentum(u, h, grid, w)
    ke = kinetic_energy(u, v, h, w)
    pe = potential_energy(h, w)

    return Invariants(mass=m, px=px, aam=aam, ke=ke, pe=pe)


def invariants_from_write(state, grid, mask: np.ndarray | None = None) -> Invariants:
    """
    Compute invariants on WRITE buffers (next state before swap).
    """
    # Access WRITE explicitly (safe pattern across modules)
    u = state.atmos.u.write
    v = state.atmos.v.write
    h = state.atmos.h.write

    w = area_weights(grid, normalize=False, mask=mask)

    m = mass_integral(h, w)
    px = zonal_momentum(u, h, w)
    aam = axial_angular_momentum(u, h, grid, w)
    ke = kinetic_energy(u, v, h, w)
    pe = potential_energy(h, w)

    return Invariants(mass=m, px=px, aam=aam, ke=ke, pe=pe)


def step_deltas(prev: Invariants, nxt: Invariants) -> dict[str, float]:
    """
    Return simple deltas (next - prev) for quick checks.
    """
    return {
        "d_mass": nxt.mass - prev.mass,
        "d_px": nxt.px - prev.px,
        "d_aam": nxt.aam - prev.aam,
        "d_ke": nxt.ke - prev.ke,
        "d_pe": nxt.pe - prev.pe,
    }


def diagnostics_report(prev: Invariants, nxt: Invariants) -> dict[str, float]:
    """
    Convenience helper returning both absolute values and deltas in one dict.
    """
    d = step_deltas(prev, nxt)
    return {
        "mass_old": prev.mass,
        "px_old": prev.px,
        "aam_old": prev.aam,
        "ke_old": prev.ke,
        "pe_old": prev.pe,
        "mass_new": nxt.mass,
        "px_new": nxt.px,
        "aam_new": nxt.aam,
        "ke_new": nxt.ke,
        "pe_new": nxt.pe,
        **d,
    }


# Example usage pattern (to be used by orchestrators or the world driver):
#
#   prev = invariants_from_read(state, grid, mask=land_or_global_mask)
#   ... module.time_step(state, dt)  # writes next to .write
#   nxt = invariants_from_write(state, grid, mask=land_or_global_mask)
#   rep = diagnostics_report(prev, nxt)
#   # log rep, assert small deltas, etc.
#
# The pattern works cleanly with DBA: no swap needed to obtain per-step "before/after".
