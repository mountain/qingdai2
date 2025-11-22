from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import DTypeLike

try:
    # Preferred import path within package
    from pygcm.jax_compat import xp  # numpy or jax.numpy
    from pygcm.numerics.double_buffer import DoubleBufferingArray as DBA
except Exception:  # pragma: no cover - fallback for direct executions
    from ..jax_compat import xp
    from ..numerics.double_buffer import DoubleBufferingArray as DBA


@dataclass
class AtmosState:
    """Atmosphere prognostic/diagnostic fields wrapped by DoubleBufferingArray."""

    u: DBA  # zonal wind (m/s)
    v: DBA  # meridional wind (m/s)
    h: DBA  # geopotential height proxy or layer thickness (SI)
    Ta: DBA  # air temperature (K)
    q: DBA  # specific humidity (kg/kg)
    cloud: DBA  # cloud fraction (0..1)


@dataclass
class OceanState:
    """Ocean single-layer shallow-water + SST wrapped by DoubleBufferingArray."""

    uo: DBA  # ocean zonal velocity (m/s)
    vo: DBA  # ocean meridional velocity (m/s)
    eta: DBA  # sea surface height anomaly (m)
    sst: DBA  # sea surface temperature (K)


@dataclass
class SurfaceState:
    """Surface fields (land/sea/ice) wrapped by DoubleBufferingArray."""

    Ts: DBA  # surface temperature (K)
    h_ice: DBA  # sea-ice thickness (m), can be 0 over land/open ocean


@dataclass
class HydroState:
    """Hydrology reservoirs wrapped by DoubleBufferingArray."""

    W_land: DBA  # land water storage/bucket (mm or kg/m^2, per implementation)
    SWE: DBA  # snow water equivalent (mm or kg/m^2)


@dataclass
class WorldState:
    """Aggregate state for DB-aware world. All grid-shaped fields are DBAs."""

    atmos: AtmosState
    surface: SurfaceState
    ocean: OceanState | None = None
    hydro: HydroState | None = None
    t_seconds: float = 0.0

    def swap_all(self) -> None:
        """Atomically flip all DBA buffers (read<=>write) for the next step."""

        def _swap_namespace(ns: object) -> None:
            for _, value in vars(ns).items():
                if isinstance(value, DBA):
                    value.swap()

        _swap_namespace(self.atmos)
        _swap_namespace(self.surface)
        if self.ocean is not None:
            _swap_namespace(self.ocean)
        if self.hydro is not None:
            _swap_namespace(self.hydro)


# ---------- Helpers to construct state with DBAs ----------


def _alloc_dba(
    shape: tuple[int, int], dtype: DTypeLike = np.float64, initial_value: float = 0.0
) -> DBA:
    """Allocate a DoubleBufferingArray with given shape, dtype and initial fill."""
    dba = DBA(shape, dtype=dtype, initial_value=initial_value)
    # Ensure both buffers have the same initial content (in case backend does lazy)
    dba.write[:] = xp.asarray(initial_value, dtype=dtype)
    dba.swap()
    dba.write[:] = xp.asarray(initial_value, dtype=dtype)
    dba.swap()
    return dba


def dba_from_array(arr: np.ndarray) -> DBA:
    """Create a DBA initialized from an existing ndarray (or jax array)."""
    arr = xp.asarray(arr)
    dba = DBA(arr.shape, dtype=arr.dtype, initial_value=0.0)
    # Fill both buffers with the provided array
    dba.write[:] = arr
    dba.swap()
    dba.write[:] = arr
    dba.swap()
    return dba


def zeros_world_state(
    shape: tuple[int, int],
    *,
    include_ocean: bool = True,
    include_hydro: bool = True,
    dtype: DTypeLike = np.float64,
) -> WorldState:
    """Construct a zero-initialized WorldState with DBAs of the given shape.

    Parameters
    ----------
    shape : (n_lat, n_lon)
        Grid shape for all 2D fields.
    include_ocean : bool
        Whether to allocate ocean fields.
    include_hydro : bool
        Whether to allocate hydrology fields.
    dtype : numpy dtype
        Data type for all buffers.

    Returns
    -------
    WorldState
        A world state with DoubleBufferingArray-backed fields.
    """
    # Atmosphere
    atmos = AtmosState(
        u=_alloc_dba(shape, dtype),
        v=_alloc_dba(shape, dtype),
        h=_alloc_dba(shape, dtype),
        Ta=_alloc_dba(shape, dtype),
        q=_alloc_dba(shape, dtype),
        cloud=_alloc_dba(shape, dtype),
    )

    # Surface
    surface = SurfaceState(
        Ts=_alloc_dba(shape, dtype),
        h_ice=_alloc_dba(shape, dtype),
    )

    ocean: OceanState | None = None
    hydro: HydroState | None = None

    if include_ocean:
        ocean = OceanState(
            uo=_alloc_dba(shape, dtype),
            vo=_alloc_dba(shape, dtype),
            eta=_alloc_dba(shape, dtype),
            sst=_alloc_dba(shape, dtype),
        )

    if include_hydro:
        hydro = HydroState(
            W_land=_alloc_dba(shape, dtype),
            SWE=_alloc_dba(shape, dtype),
        )

    return WorldState(atmos=atmos, surface=surface, ocean=ocean, hydro=hydro, t_seconds=0.0)


# ---------- Optional convenience for grid-aware construction ----------


def zeros_world_state_from_grid(grid) -> WorldState:
    """Create a zero-initialized WorldState from a SphericalGrid-like object.

    The grid is expected to provide lat/lon 1D arrays (or shape attributes) that
    define the target 2D field shape (n_lat, n_lon).
    """
    # Compatible with pygcm.grid.SphericalGrid
    try:
        n_lat = int(len(grid.lat))
        n_lon = int(len(grid.lon))
    except Exception as err:
        # Fallback to attributes if available
        n_lat = int(getattr(grid, "n_lat", 0))
        n_lon = int(getattr(grid, "n_lon", 0))
        if n_lat == 0 or n_lon == 0:
            raise ValueError("Unable to infer grid shape for world state allocation.") from err
    return zeros_world_state((n_lat, n_lon))
