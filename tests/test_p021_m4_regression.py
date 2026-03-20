import os

from pygcm.world import QingdaiWorld


def test_p021_m4_regression_thresholds(monkeypatch):
    monkeypatch.setenv("QD_N_LAT", "24")
    monkeypatch.setenv("QD_N_LON", "48")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_ECO_ALBEDO_COUPLE", "1")
    monkeypatch.setenv("QD_OO_DIAG_EVERY", "1000000")
    world = QingdaiWorld.create_default()
    world.run(duration_days=2.0)
    metrics = world.m4_metrics
    assert metrics["energy_mean_abs_toa"] <= float(os.getenv("P021_M4_TOA_MAX", "90.0"))
    assert metrics["energy_mean_abs_sfc"] <= float(os.getenv("P021_M4_SFC_MAX", "450.0"))
    assert metrics["energy_mean_abs_atm"] <= float(os.getenv("P021_M4_ATM_MAX", "350.0"))
    assert metrics["water_mean_abs_residual"] <= float(os.getenv("P021_M4_WATER_MAX", "1e-8"))
