import numpy as np

from pygcm.world.orchestrator_spec import build_world_orchestrator_spec


def test_build_world_orchestrator_spec_shapes_and_groups():
    land_mask = np.zeros((4, 6), dtype=int)
    friction_map = np.ones((4, 6), dtype=float) * 0.2
    c_s_map = np.ones((4, 6), dtype=float) * 2.0e7
    hum_params = object()
    energy_params = object()
    hydro_params = object()
    spec = build_world_orchestrator_spec(
        grid=object(),
        friction_map=friction_map,
        land_mask=land_mask,
        C_s_map=c_s_map,
        Cs_ocean=2.1e8,
        Cs_land=3.0e6,
        Cs_ice=5.0e6,
        H_m=50.0,
        rho_w=1000.0,
        cp_w=4200.0,
        hydrology_params=hydro_params,
        ecology_enabled=True,
        ecology_day_in_seconds=86400.0,
        ecology_soil_water_cap=150.0,
        ecology_lai_albedo_weight=0.8,
        ecology_albedo_couple=True,
        routing_enabled=True,
        routing_network_nc_path="data/hydrology.nc",
        routing_dt_hydro_hours=3.0,
        routing_treat_lake_as_water=False,
        routing_alpha_lake=0.3,
        routing_diag=False,
        humidity_params=hum_params,
        energy_params=energy_params,
    )
    assert spec.atmosphere.C_s_map.shape == land_mask.shape
    assert spec.ocean.init_Ts.shape == land_mask.shape
    assert spec.atmosphere.humidity_params is hum_params
    assert spec.atmosphere.energy_params is energy_params
    assert spec.hydrology.hydrology_params is hydro_params
    assert spec.ecology.enabled is True
    assert spec.ecology.day_in_seconds == 86400.0
    assert spec.ecology.soil_water_cap == 150.0
    assert spec.ecology.lai_albedo_weight == 0.8
    assert spec.ecology.albedo_couple is True
    assert spec.routing.enabled is True
    assert spec.routing.network_nc_path == "data/hydrology.nc"
    assert spec.routing.dt_hydro_hours == 3.0
    assert spec.routing.treat_lake_as_water is False
    assert spec.routing.alpha_lake == 0.3
    assert spec.routing.diag is False
