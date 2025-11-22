import numpy as np
from pygcm.numerics.double_buffer import DoubleBufferingArray as DBA
from pygcm.world.state import (
    WorldState,
    dba_from_array,
    zeros_world_state,
    zeros_world_state_from_grid,
)


def test_zeros_world_state_allocates_dbas():
    shape = (6, 12)
    ws = zeros_world_state(shape, include_ocean=True, include_hydro=True)
    assert isinstance(ws, WorldState)
    # Atmosphere DBAs
    assert isinstance(ws.atmos.u, DBA)
    assert isinstance(ws.atmos.v, DBA)
    assert isinstance(ws.atmos.h, DBA)
    assert isinstance(ws.atmos.Ta, DBA)
    assert isinstance(ws.atmos.q, DBA)
    assert isinstance(ws.atmos.cloud, DBA)
    # Surface DBAs
    assert isinstance(ws.surface.Ts, DBA)
    assert isinstance(ws.surface.h_ice, DBA)
    # Ocean/ Hydro present
    assert ws.ocean is not None
    assert ws.hydro is not None
    assert isinstance(ws.ocean.sst, DBA)
    assert isinstance(ws.hydro.W_land, DBA)


def test_swap_all_semantics():
    shape = (4, 8)
    ws = zeros_world_state(shape, include_ocean=False, include_hydro=False)

    # write does not affect read before swap
    ws.atmos.u.write[:] = 1.0
    assert np.allclose(np.asarray(ws.atmos.u.read), 0.0)

    # after swap_all(), read reflects previously written value
    ws.swap_all()
    assert np.allclose(np.asarray(ws.atmos.u.read), 1.0)

    # write a new value, ensure isolation again until swap
    ws.atmos.u.write[:] = 2.0
    assert np.allclose(np.asarray(ws.atmos.u.read), 1.0)
    ws.swap_all()
    assert np.allclose(np.asarray(ws.atmos.u.read), 2.0)


def test_dba_from_array_initializes_both_buffers():
    arr = np.ones((3, 5), dtype=np.float64) * 7.0
    dba = dba_from_array(arr)
    # read reflects provided array
    assert np.allclose(np.asarray(dba.read), 7.0)
    # when writing a different value then swapping, read should change accordingly
    dba.write[:] = 9.0
    # not yet visible
    assert np.allclose(np.asarray(dba.read), 7.0)
    dba.swap()
    assert np.allclose(np.asarray(dba.read), 9.0)


class _DummyGrid:
    def __init__(self, n_lat, n_lon):
        # mimic SphericalGrid interface
        self.lat = list(range(n_lat))
        self.lon = list(range(n_lon))


def test_zeros_world_state_from_grid_with_latlon_lists():
    grid = _DummyGrid(3, 7)
    ws = zeros_world_state_from_grid(grid)
    # Simply ensure fields have expected shape
    assert ws.atmos.u.read.shape == (3, 7)
    assert ws.surface.Ts.read.shape == (3, 7)


def test_zeros_world_state_from_grid_with_attrs():
    class _GridAttrs:
        def __init__(self, n_lat, n_lon):
            self.n_lat = n_lat
            self.n_lon = n_lon

    grid = _GridAttrs(5, 9)
    ws = zeros_world_state_from_grid(grid)
    assert ws.atmos.v.read.shape == (5, 9)
