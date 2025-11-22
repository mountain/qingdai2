import numpy as np
from pygcm.world.atmosphere import AtmosParams, Atmosphere
from pygcm.world.state import zeros_world_state


def test_atmosphere_time_step_writes_only_write_buffers_and_requires_swap():
    shape = (4, 8)
    ws = zeros_world_state(shape, include_ocean=False, include_hydro=False)

    # Seed read buffers
    ws.atmos.u.write[:] = 2.0
    ws.atmos.v.write[:] = -3.0
    ws.atmos.h.write[:] = 5.0
    ws.swap_all()  # make seeds visible in .read

    # Configure atmosphere with finite relaxation times
    params = AtmosParams(
        tau_relax_u_s=100.0,
        tau_relax_v_s=100.0,
        tau_relax_h_s=100.0,
        k4_u=0.0,
        k4_v=0.0,
        k4_h=0.0,
    )
    atmos = Atmosphere(params)

    # Advance one step
    dt = 10.0
    atmos.time_step(ws, dt, h_eq=None)

    # Before swap: read should still reflect old values (seeds)
    assert np.allclose(np.asarray(ws.atmos.u.read), 2.0)
    assert np.allclose(np.asarray(ws.atmos.v.read), -3.0)
    assert np.allclose(np.asarray(ws.atmos.h.read), 5.0)

    # After swap: read should reflect next step written values
    ws.swap_all()
    u1 = np.asarray(ws.atmos.u.read)
    v1 = np.asarray(ws.atmos.v.read)
    h1 = np.asarray(ws.atmos.h.read)

    # Expected linear relaxation: x_next = x_old + (-x_old/tau)*dt = x_old*(1 - dt/tau)
    factor = 1.0 - dt / params.tau_relax_u_s
    assert np.allclose(u1, 2.0 * factor)
    factor = 1.0 - dt / params.tau_relax_v_s
    assert np.allclose(v1, -3.0 * factor)
    factor = 1.0 - dt / params.tau_relax_h_s
    assert np.allclose(h1, 5.0 * factor)


def test_atmosphere_hyperdiffusion_demo_changes_field_structure():
    shape = (6, 12)
    ws = zeros_world_state(shape, include_ocean=False, include_hydro=False)

    # Seed a delta in the center for u, v, h
    ws.atmos.u.write[:] = 0.0
    ws.atmos.v.write[:] = 0.0
    ws.atmos.h.write[:] = 0.0
    ws.swap_all()
    ws.atmos.u.write[shape[0] // 2, shape[1] // 2] = 1.0
    ws.atmos.v.write[shape[0] // 2, shape[1] // 2] = -1.0
    ws.atmos.h.write[shape[0] // 2, shape[1] // 2] = 2.0
    ws.swap_all()

    params = AtmosParams(
        tau_relax_u_s=1.0e9,  # nearly no relaxation
        tau_relax_v_s=1.0e9,
        tau_relax_h_s=1.0e9,
        k4_u=1.0e10,  # strong artificial diffusion for test visibility
        k4_v=1.0e10,
        k4_h=1.0e10,
        dx_m=1.0e5,
        dy_m=1.0e5,
    )
    atmos = Atmosphere(params)
    dt = 10.0
    atmos.time_step(ws, dt)

    # Swap then check that peaks have diffused into neighbors
    ws.swap_all()
    u1 = np.asarray(ws.atmos.u.read)
    v1 = np.asarray(ws.atmos.v.read)
    h1 = np.asarray(ws.atmos.h.read)

    # The exact values are not important; we only assert "smoothing" happened:
    # center value reduced in magnitude, and some neighbors are non-zero.
    ci, cj = shape[0] // 2, shape[1] // 2
    assert abs(u1[ci, cj]) < 1.0
    assert abs(v1[ci, cj]) < 1.0
    assert abs(h1[ci, cj]) < 2.0
    # Some neighbor cells should have non-zero signal now
    neighbors = [(ci + 1, cj), (ci - 1, cj), (ci, cj + 1), (ci, cj - 1)]
    assert any(abs(u1[i, j]) > 0 for i, j in neighbors)
    assert any(abs(v1[i, j]) > 0 for i, j in neighbors)
    assert any(abs(h1[i, j]) > 0 for i, j in neighbors)
