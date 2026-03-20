import numpy as np
from pygcm.energy import EnergyParams
from pygcm.humidity import HumidityParams
from pygcm.world.coupler import Coupler
from pygcm.world.ports import ColumnProcessIn, SurfaceToAtmosphere


def test_coupler_uses_insolation_and_ports():
    shape = (4, 6)
    Ts = np.full(shape, 288.0)
    land_mask = np.zeros(shape, dtype=int)
    base_albedo = np.full(shape, 0.2)
    insolation = np.full(shape, 600.0)
    surface_in = SurfaceToAtmosphere(
        T_s=Ts,
        land_mask=land_mask,
        ice_mask=np.zeros(shape, dtype=bool),
        base_albedo=base_albedo,
        insolation=insolation,
    )
    column_in = ColumnProcessIn(
        q=np.full(shape, 0.01),
        cloud=np.full(shape, 0.3),
        precip_rate=np.full(shape, 1.0e-5),
        Ta=np.full(shape, 285.0),
        RH=np.full(shape, 0.7),
        u10=np.full(shape, 3.0),
        v10=np.full(shape, 2.0),
    )
    coupler = Coupler()
    fluxes, col_out = coupler.compute(surface_in, column_in, grid=None, state=None, dt=300.0)
    assert fluxes is not None
    assert col_out is not None
    assert np.nanmean(fluxes.SW_sfc) > 0.0
    assert np.isfinite(np.nanmean(fluxes.LW_sfc))
    assert np.nanmean(fluxes.Qnet) != 0.0
    assert np.nanmean(col_out.precip_rate_next) > 0.0


def test_coupler_column_only_fallback():
    shape = (3, 5)
    coupler = Coupler()
    column_in = ColumnProcessIn(
        q=np.full(shape, 0.01),
        cloud=np.full(shape, 0.2),
        precip_rate=np.full(shape, 5.0e-6),
    )
    fluxes, col_out = coupler.compute(None, column_in, grid=None, state=None, dt=300.0)
    assert fluxes is None
    assert col_out is not None
    assert np.allclose(col_out.q_next, column_in.q)


def test_coupler_explicit_param_injection():
    shape = (2, 4)
    coupler = Coupler(
        energy_params=EnergyParams(sw_a0=0.1, sw_kc=0.1),
        humidity_params=HumidityParams(C_E=2.0e-3, L_v=2.4e6),
    )
    surface_in = SurfaceToAtmosphere(
        T_s=np.full(shape, 289.0),
        land_mask=np.zeros(shape, dtype=int),
        ice_mask=np.zeros(shape, dtype=bool),
        base_albedo=np.full(shape, 0.2),
        insolation=np.full(shape, 500.0),
    )
    column_in = ColumnProcessIn(
        q=np.full(shape, 0.01),
        cloud=np.full(shape, 0.2),
        precip_rate=np.full(shape, 1.0e-6),
        Ta=np.full(shape, 286.0),
        u10=np.full(shape, 2.0),
        v10=np.zeros(shape),
    )
    fluxes, _ = coupler.compute(surface_in, column_in, grid=None, state=None, dt=300.0)
    assert fluxes is not None
    assert np.nanmean(fluxes.SW_sfc) > 0.0
