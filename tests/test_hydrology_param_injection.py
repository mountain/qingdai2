import numpy as np

from pygcm.hydrology import HydrologyParams, snowpack_step


def test_snowpack_step_uses_params_albedo():
    s = np.zeros((2, 3), dtype=float)
    p = np.zeros((2, 3), dtype=float)
    t = np.full((2, 3), 270.0, dtype=float)
    hp = HydrologyParams(snow_albedo_fresh=0.85)
    _, _, _, alpha = snowpack_step(s, p, t, hp, dt=300.0)
    assert np.allclose(alpha, 0.85)
