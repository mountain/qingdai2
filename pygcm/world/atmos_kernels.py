from __future__ import annotations
"""
Demo relaxation engine (array-in/array-out) for 1-layer atmosphere.

Purpose
- Provide a tiny, pure, and reusable engine implementation that matches AtmosEngine
  (see pygcm/world/atmos_api.py) to keep orchestrator DRY and files small.
- Encapsulate demo kernels (linear relaxation + optional weak ∇⁴) here so that
  pygcm/world/atmosphere.py can focus on orchestration + coupling only.

Notes
- This engine operates on plain arrays and returns new arrays; it does not know DBA.
- Replace with production engines as needed (FD, spectral, hybrid). They only need to
  implement the same step signature.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
import numpy as np

try:
    from pygcm.jax_compat import xp  # numpy or jax.numpy
except Exception:
    xp = np


@dataclass
class DemoRelaxParams:
    tau_relax_u_s: float = 2.0 * 24 * 3600.0  # 2 days
    tau_relax_v_s: float = 2.0 * 24 * 3600.0
    tau_relax_h_s: float = 2.0 * 24 * 3600.0
    k4_u: float = 0.0  # m^4/s, hyperdiffusion demo
    k4_v: float = 0.0
    k4_h: float = 0.0
    dx_m: float = 1.0e5  # ~100 km
    dy_m: float = 1.0e5


class DemoRelaxEngine:
    """
    Minimal 1-layer engine:
        du/dt = -u / tau_u + (-k4 ∇⁴ u)
        dv/dt = -v / tau_v + (-k4 ∇⁴ v)
        dh/dt = -(h - h_eq) / tau_h + (-k4 ∇⁴ h)

    It accepts optional "h_eq" via kwargs to relax height toward a target.
    Other kwargs are ignored (keeps signature compatible with orchestrator).
    """

    def __init__(self, params: Optional[DemoRelaxParams] = None) -> None:
        self.p = params or DemoRelaxParams()

    # -------------- kernels --------------

    @staticmethod
    def _relax(field: np.ndarray, tau_s: float) -> np.ndarray:
        if tau_s <= 0.0:
            return xp.zeros_like(field)
        return -field / tau_s

    @staticmethod
    def _laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
        f_ip1 = xp.roll(f, -1, axis=1)
        f_im1 = xp.roll(f, +1, axis=1)
        f_jp1 = xp.concatenate([f[1:, :], f[-1:, :]], axis=0)
        f_jm1 = xp.concatenate([f[:1, :], f[:-1, :]], axis=0)
        return (f_ip1 + f_im1 - 2.0 * f) / (dx * dx) + (f_jp1 + f_jm1 - 2.0 * f) / (dy * dy)

    def _hyperdiff(self, f: np.ndarray, k4: float) -> np.ndarray:
        if k4 == 0.0:
            return xp.zeros_like(f)
        lap = self._laplacian(f, self.p.dx_m, self.p.dy_m)
        lap2 = self._laplacian(lap, self.p.dx_m, self.p.dy_m)
        return -k4 * lap2

    # -------------- engine API --------------

    def step(
        self,
        u: np.ndarray,
        v: np.ndarray,
        h: np.ndarray,
        dt: float,
        **kwargs: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Accepts optional:
          - h_eq: ndarray, equilibrium target for height relaxation
        """
        h_eq = kwargs.get("h_eq", None)

        du_dt = self._relax(u, self.p.tau_relax_u_s) + self._hyperdiff(u, self.p.k4_u)
        dv_dt = self._relax(v, self.p.tau_relax_v_s) + self._hyperdiff(v, self.p.k4_v)

        if self.p.tau_relax_h_s <= 0.0:
            dh_dt = self._hyperdiff(h, self.p.k4_h)
        else:
            target = 0.0 if h_eq is None else h_eq
            dh_dt = -(h - target) / self.p.tau_relax_h_s + self._hyperdiff(h, self.p.k4_h)

        u_next = u + du_dt * dt
        v_next = v + dv_dt * dt
        h_next = h + dh_dt * dt
        return u_next, v_next, h_next
