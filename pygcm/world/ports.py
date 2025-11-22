from __future__ import annotations

"""
Typed coupling ports for OO orchestrator (DBA-friendly).

Goal
- Provide explicit, typed "ports" for interface couplings (air–sea, land–atmosphere)
  and in-column microphysics/process couplings (humidity, clouds, precipitation).
- Remove ad-hoc dicts from critical APIs; keep backends pure (array-in/array-out)
  while giving the orchestrator well-defined structures to pass around.

Notes
- All fields are arrays on the model's lat×lon grid unless noted.
- These are light dataclasses; no logic. Orchestrators assemble them and pass to
  modules/backends which may choose to consume some/all fields.
- Backends that cannot consume structured ports (e.g., legacy engines) can use
  a thin adapter to map selected fields to their own inputs.

Examples
--------
# Build interface coupling (surface -> atmosphere)
c_in = SurfaceToAtmosphere(
    T_s=T_s,                   # surface temperature (K)
    land_mask=land_mask,       # 0/1
    ice_mask=ice_mask,         # 0/1 (optional)
    base_albedo=alpha_base,    # base albedo (pre-cloud/eco/ice)
    friction_map=friction_map, # surface friction (s^-1 or roughness proxy)
)

# Build in-column inputs (to atmosphere column physics)
col_in = ColumnProcessIn(
    q=q,                         # specific humidity (kg/kg)
    cloud=cloud,                 # cloud fraction [0,1]
    precip_rate=precip_rate,     # kg m^-2 s^-1
    Ta=Ta,                       # column air temp proxy (K) (optional)
    RH=RH,                       # relative humidity (optional)
)

# Atmosphere returns interface fluxes & column updates
# (A backend may not supply these; orchestrator can compute them via energy/humidity modules.)
fluxes = AtmosphereToSurfaceFluxes(
    SH=SH, LH=LH, SW_sfc=SW_sfc, LW_sfc=LW_sfc, Qnet=Qnet,
    evap_flux=evap_flux, precip_flux=precip_flux
)
col_out = ColumnProcessOut(
    q_next=q_next, cloud_next=cloud_next,
    precip_rate_next=precip_rate_next,
    LH_release=LH_release
)
"""

from dataclasses import dataclass

import numpy as np

# ------------------------------
# Interface coupling (surface -> atmosphere)
# ------------------------------


@dataclass
class SurfaceToAtmosphere:
    T_s: np.ndarray  # surface temperature (K)
    land_mask: np.ndarray  # {0,1}
    ice_mask: np.ndarray | None = None  # {0,1}
    base_albedo: np.ndarray | None = None  # scalar or map before cloud/eco/ice
    friction_map: np.ndarray | None = None  # s^-1 or a roughness proxy


# ------------------------------
# Interface fluxes (atmosphere -> surface)
# ------------------------------


@dataclass
class AtmosphereToSurfaceFluxes:
    SH: np.ndarray  # sensible heat flux (W m^-2), + upward from surface
    LH: np.ndarray  # latent heat flux  (W m^-2), + upward from surface
    SW_sfc: np.ndarray  # shortwave absorbed at surface (W m^-2)
    LW_sfc: np.ndarray  # net longwave at surface (W m^-2), positive downward
    Qnet: np.ndarray  # net surface energy to ocean/land skin (W m^-2)
    evap_flux: np.ndarray | None = None  # kg m^-2 s^-1 (E)
    precip_flux: np.ndarray | None = None  # kg m^-2 s^-1 (P)


# ------------------------------
# In-column coupling (inputs/outputs)
# ------------------------------


@dataclass
class ColumnProcessIn:
    q: np.ndarray  # specific humidity (kg/kg)
    cloud: np.ndarray  # cloud fraction [0,1]
    precip_rate: np.ndarray  # kg m^-2 s^-1
    Ta: np.ndarray | None = None  # air temp proxy (K)
    RH: np.ndarray | None = None  # relative humidity [0,1]


@dataclass
class ColumnProcessOut:
    q_next: np.ndarray
    cloud_next: np.ndarray
    precip_rate_next: np.ndarray
    LH_release: np.ndarray | None = None  # latent heat release to atmosphere (W m^-2)
