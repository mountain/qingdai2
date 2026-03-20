import pytest

from pygcm.world import SimConfig


def test_sim_config_validation_positive():
    cfg = SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0)
    assert cfg.n_lat == 16
    assert cfg.n_lon == 32
    assert cfg.dt_seconds == 300.0


def test_sim_config_validation_rejects_invalid():
    with pytest.raises(ValueError):
        SimConfig(n_lat=0, n_lon=32, dt_seconds=300.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=-1, dt_seconds=300.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=0.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, sim_days=0.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, total_years=-1.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, default_orbit_fraction=0.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, routing_dt_hydro_hours=0.0)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, routing_alpha_lake=-0.1)
    with pytest.raises(ValueError):
        SimConfig(n_lat=16, n_lon=32, dt_seconds=300.0, world_diagnostics_schema_version=0)


def test_sim_config_from_env_runtime_fields(monkeypatch):
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_METADATA_ENABLE", "0")
    monkeypatch.setenv("QD_OO_METADATA_JSON", "/tmp/qd-meta.json")
    monkeypatch.setenv("QD_TOTAL_YEARS", "0.2")
    monkeypatch.setenv("QD_SIM_DAYS", "")
    monkeypatch.setenv("QD_OO_DEFAULT_ORBIT_FRACTION", "0.015")
    monkeypatch.setenv("QD_HYDRO_NETCDF", "data/hydrology.nc")
    monkeypatch.setenv("QD_HYDRO_DT_HOURS", "3")
    monkeypatch.setenv("QD_TREAT_LAKE_AS_WATER", "0")
    monkeypatch.setenv("QD_ALPHA_LAKE", "0.2")
    monkeypatch.setenv("QD_HYDRO_DIAG", "0")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_ENABLE", "1")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_JSON", "/tmp/qd-world-diag.json")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_SCHEMA_VERSION", "1")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_STRICT_VALIDATE", "1")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_ALLOW_BACKCOMPAT", "0")
    cfg = SimConfig.from_env()
    s = cfg.snapshot()
    assert s["runtime_control"]["oo_config_diag"] is False
    assert s["runtime_control"]["oo_metadata_enable"] is False
    assert s["runtime_control"]["oo_metadata_json"] == "/tmp/qd-meta.json"
    assert s["runtime_control"]["total_years"] == 0.2
    assert s["runtime_control"]["sim_days"] is None
    assert s["runtime_control"]["default_orbit_fraction"] == 0.015
    assert s["runtime_control"]["routing_netcdf"] == "data/hydrology.nc"
    assert s["runtime_control"]["routing_dt_hydro_hours"] == 3.0
    assert s["runtime_control"]["routing_treat_lake_as_water"] is False
    assert s["runtime_control"]["routing_alpha_lake"] == 0.2
    assert s["runtime_control"]["routing_diag"] is False
    assert s["runtime_control"]["world_diagnostics_enable"] is True
    assert s["runtime_control"]["world_diagnostics_json"] == "/tmp/qd-world-diag.json"
    assert s["runtime_control"]["world_diagnostics_schema_version"] == 1
    assert s["runtime_control"]["world_diagnostics_strict_validation"] is True
    assert s["runtime_control"]["world_diagnostics_allow_backward_compat"] is False


def test_sim_config_snapshot_roundtrip():
    cfg = SimConfig(
        n_lat=24,
        n_lon=48,
        dt_seconds=600.0,
        use_ecology=False,
        oo_metadata_enable=False,
        oo_metadata_json="/tmp/meta.json",
        sim_days=3.0,
        default_orbit_fraction=0.01,
    )
    cfg2 = SimConfig.from_snapshot(cfg.snapshot())
    assert cfg2 == cfg
