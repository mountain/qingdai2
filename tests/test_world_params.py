import pytest
from pygcm.world import EcologyParams, ParamsRegistry, PhysicsParams, SpectralBands


def test_params_registry_from_env(monkeypatch):
    monkeypatch.setenv("QD_GH_FACTOR", "0.6")
    monkeypatch.setenv("QD_CE", "0.0015")
    monkeypatch.setenv("QD_LV", "2500000")
    monkeypatch.setenv("QD_ECO_SPECTRAL_BANDS", "12")
    monkeypatch.setenv("QD_ECO_LAI_ALBEDO_WEIGHT", "0.8")
    monkeypatch.setenv("QD_ECO_FEEDBACK_MODE", "daily")
    monkeypatch.setenv("QD_MLD_M", "55")
    monkeypatch.setenv("QD_Q_INIT_RH", "0.6")
    monkeypatch.setenv("QD_OO_DIAG_EVERY", "77")
    monkeypatch.setenv("QD_ECO_SOIL_WATER_CAP", "40")
    monkeypatch.setenv("QD_ECO_ALBEDO_COUPLE", "0")
    monkeypatch.setenv("QD_RUNOFF_TAU_DAYS", "8.5")
    monkeypatch.setenv("QD_WATER_DIAG", "0")
    monkeypatch.setenv("QD_SW_A0", "0.08")
    monkeypatch.setenv("QD_ENERGY_DIAG", "0")
    pr = ParamsRegistry.from_env()
    s = pr.snapshot()
    g = s["parameter_groups"]
    assert g["physics"]["gh_factor"] == 0.6
    assert g["physics"]["mld_m"] == 55.0
    assert g["physics"]["q_init_rh"] == 0.6
    assert g["physics"]["oo_diag_every"] == 77
    assert g["physics"]["runoff_tau_days"] == 8.5
    assert g["physics"]["water_diag"] is False
    assert g["physics"]["sw_a0"] == 0.08
    assert g["physics"]["energy_diag"] is False
    assert g["bands"]["nbands"] == 12
    assert g["ecology"]["feedback_mode"] == "daily"
    assert g["ecology"]["soil_water_cap"] == 40.0
    assert g["ecology"]["albedo_couple"] is False


def test_params_validation_rejects_invalid():
    with pytest.raises(ValueError):
        PhysicsParams(gh_factor=1.2)
    with pytest.raises(ValueError):
        PhysicsParams(q_init_rh=1.5)
    with pytest.raises(ValueError):
        PhysicsParams(oo_diag_every=0)
    with pytest.raises(ValueError):
        PhysicsParams(t_floor=0.0)
    with pytest.raises(ValueError):
        SpectralBands(nbands=0)
    with pytest.raises(ValueError):
        EcologyParams(lai_albedo_weight=-0.1)
    with pytest.raises(ValueError):
        EcologyParams(feedback_mode="bad")
    with pytest.raises(ValueError):
        EcologyParams(soil_water_cap=0.0)


def test_params_registry_snapshot_roundtrip():
    pr = ParamsRegistry(
        physics=PhysicsParams(gh_factor=0.55, mld_m=45.0, oo_diag_every=33),
        bands=SpectralBands(nbands=8),
        ecology=EcologyParams(lai_albedo_weight=0.9, feedback_mode="daily"),
    )
    pr2 = ParamsRegistry.from_snapshot(pr.snapshot())
    assert pr2 == pr
