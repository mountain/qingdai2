import numpy as np

from pygcm.world.world_step_ops import blend_ecology_albedo, resolve_total_seconds


def test_resolve_total_seconds_priority():
    day = 100.0
    orb = 1000.0
    assert resolve_total_seconds(
        dt=10.0,
        day_in_seconds=day,
        orbital_period=orb,
        duration_days=2.0,
        n_steps=3,
        total_years=4.0,
        sim_days=5.0,
        default_orbit_fraction=0.01,
    ) == 200.0
    assert resolve_total_seconds(
        dt=10.0,
        day_in_seconds=day,
        orbital_period=orb,
        duration_days=None,
        n_steps=3,
        total_years=4.0,
        sim_days=5.0,
        default_orbit_fraction=0.01,
    ) == 30.0


def test_blend_ecology_albedo_no_eco_returns_base():
    base = np.full((2, 3), 0.2, dtype=float)
    out = blend_ecology_albedo(
        base_albedo=base,
        land_mask=np.ones((2, 3), dtype=int),
        eco=None,
        eco_weight=1.0,
        eco_albedo_couple=True,
        insolation=np.full((2, 3), 500.0),
        cloud_eff=np.full((2, 3), 0.2),
        dt=300.0,
    )
    assert np.allclose(out, base)
