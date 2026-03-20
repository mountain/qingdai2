from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AtmosphereConfigureSpec:
    grid: object
    friction_map: np.ndarray
    land_mask: np.ndarray
    C_s_map: np.ndarray
    Cs_ocean: float
    Cs_land: float
    Cs_ice: float
    humidity_params: object | None = None
    energy_params: object | None = None


@dataclass(frozen=True)
class OceanConfigureSpec:
    grid: object
    land_mask: np.ndarray
    H_m: float
    init_Ts: np.ndarray
    rho_w: float
    cp_w: float


@dataclass(frozen=True)
class HydrologyConfigureSpec:
    land_mask: np.ndarray
    hydrology_params: object


@dataclass(frozen=True)
class EcologyConfigureSpec:
    enabled: bool
    land_mask: np.ndarray
    day_in_seconds: float
    soil_water_cap: float
    lai_albedo_weight: float
    albedo_couple: bool


@dataclass(frozen=True)
class RoutingConfigureSpec:
    enabled: bool
    network_nc_path: str
    dt_hydro_hours: float
    treat_lake_as_water: bool
    alpha_lake: float | None
    diag: bool


@dataclass(frozen=True)
class WorldOrchestratorConfigureSpec:
    atmosphere: AtmosphereConfigureSpec
    ocean: OceanConfigureSpec
    hydrology: HydrologyConfigureSpec
    ecology: EcologyConfigureSpec
    routing: RoutingConfigureSpec


def build_world_orchestrator_spec(
    *,
    grid: object,
    friction_map: np.ndarray,
    land_mask: np.ndarray,
    C_s_map: np.ndarray,
    Cs_ocean: float,
    Cs_land: float,
    Cs_ice: float,
    H_m: float,
    rho_w: float,
    cp_w: float,
    hydrology_params: object,
    ecology_enabled: bool,
    ecology_day_in_seconds: float,
    ecology_soil_water_cap: float,
    ecology_lai_albedo_weight: float,
    ecology_albedo_couple: bool,
    routing_enabled: bool = False,
    routing_network_nc_path: str = "data/hydrology.nc",
    routing_dt_hydro_hours: float = 6.0,
    routing_treat_lake_as_water: bool = True,
    routing_alpha_lake: float | None = None,
    routing_diag: bool = True,
    humidity_params: object | None = None,
    energy_params: object | None = None,
) -> WorldOrchestratorConfigureSpec:
    return WorldOrchestratorConfigureSpec(
        atmosphere=AtmosphereConfigureSpec(
            grid=grid,
            friction_map=friction_map,
            land_mask=land_mask,
            C_s_map=C_s_map,
            Cs_ocean=float(Cs_ocean),
            Cs_land=float(Cs_land),
            Cs_ice=float(Cs_ice),
            humidity_params=humidity_params,
            energy_params=energy_params,
        ),
        ocean=OceanConfigureSpec(
            grid=grid,
            land_mask=land_mask,
            H_m=float(H_m),
            init_Ts=np.full_like(land_mask, 288.0, dtype=float),
            rho_w=float(rho_w),
            cp_w=float(cp_w),
        ),
        hydrology=HydrologyConfigureSpec(
            land_mask=land_mask,
            hydrology_params=hydrology_params,
        ),
        ecology=EcologyConfigureSpec(
            enabled=bool(ecology_enabled),
            land_mask=land_mask,
            day_in_seconds=float(ecology_day_in_seconds),
            soil_water_cap=float(ecology_soil_water_cap),
            lai_albedo_weight=float(ecology_lai_albedo_weight),
            albedo_couple=bool(ecology_albedo_couple),
        ),
        routing=RoutingConfigureSpec(
            enabled=bool(routing_enabled),
            network_nc_path=str(routing_network_nc_path),
            dt_hydro_hours=float(routing_dt_hydro_hours),
            treat_lake_as_water=bool(routing_treat_lake_as_water),
            alpha_lake=(
                None if routing_alpha_lake is None else float(routing_alpha_lake)
            ),
            diag=bool(routing_diag),
        ),
    )
