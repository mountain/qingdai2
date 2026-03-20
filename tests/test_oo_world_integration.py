import numpy as np

from pygcm.world import QingdaiWorld


def test_world_run_updates_hydro_and_surface(monkeypatch):
    monkeypatch.setenv("QD_N_LAT", "16")
    monkeypatch.setenv("QD_N_LON", "32")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_ECO_ALBEDO_COUPLE", "1")
    monkeypatch.setenv("QD_OO_DIAG_EVERY", "99999")
    world = QingdaiWorld.create_default()
    world.run(n_steps=4)


def test_world_run_uses_default_orbit_fraction(monkeypatch):
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_DEFAULT_ORBIT_FRACTION", "1e-8")
    monkeypatch.setenv("QD_TOTAL_YEARS", "")
    monkeypatch.setenv("QD_SIM_DAYS", "")
    world = QingdaiWorld.create_default()
    world.run()
    assert world.m4_metrics["steps"] == 1


def test_world_run_uses_injected_ocean_orchestrator(monkeypatch):
    class FakeOcean:
        def __init__(self):
            self.configure_called = 0
            self.step_called = 0

        def configure(self, *, spec):
            self.configure_called += 1
            self._shape = spec.ocean.init_Ts.shape

        def step_and_write(self, *, state, dt, u_atm, v_atm, Q_net, ice_mask):
            self.step_called += 1
            state.surface.Ts.write[:] = np.full(self._shape, 287.0)
            if state.ocean is not None:
                state.ocean.sst.write[:] = np.full(self._shape, 287.0)
                state.ocean.uo.write[:] = 0.0
                state.ocean.vo.write[:] = 0.0
                state.ocean.eta.write[:] = 0.0

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    fake = FakeOcean()
    world = QingdaiWorld.create_default()
    world.ocean = fake
    world.run(n_steps=2)
    assert fake.configure_called == 1
    assert fake.step_called == 2


def test_world_run_wraps_legacy_ocean_backend(monkeypatch):
    class LegacyOcean:
        def __init__(self, shape):
            self.step_called = 0
            self.Ts = np.full(shape, 288.0)
            self.uo = np.zeros(shape)
            self.vo = np.zeros(shape)
            self.eta = np.zeros(shape)

        def step(self, dt, u_atm, v_atm, Q_net=None, ice_mask=None):
            self.step_called += 1
            self.Ts = np.full_like(self.Ts, 286.0)

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    legacy = LegacyOcean((12, 24))
    world.ocean = legacy
    world.run(n_steps=2)
    assert hasattr(world.ocean, "step_and_write")
    assert legacy.step_called == 2


def test_world_run_uses_injected_atmos_orchestrator(monkeypatch):
    class FakeAtmos:
        def __init__(self):
            self.configure_called = 0
            self.step_called = 0

        def configure(self, *, spec):
            self.configure_called += 1

        def step_and_write(self, *, state, dt, h_eq, surface_in, column_in):
            self.step_called += 1
            state.atmos.u.write[:] = state.atmos.u.read
            state.atmos.v.write[:] = state.atmos.v.read
            state.atmos.h.write[:] = state.atmos.h.read
            return None, None

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    fake = FakeAtmos()
    world = QingdaiWorld.create_default()
    world.atmos = fake
    world.run(n_steps=2)
    assert fake.configure_called == 1
    assert fake.step_called == 2


def test_world_run_wraps_legacy_atmos_backend(monkeypatch):
    class LegacyAtmos:
        def __init__(self):
            self.step_called = 0

        def time_step(self, state, dt, *, h_eq=None, surface_in=None, column_in=None):
            self.step_called += 1
            state.atmos.u.write[:] = state.atmos.u.read
            state.atmos.v.write[:] = state.atmos.v.read
            state.atmos.h.write[:] = state.atmos.h.read
            return None, None

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    legacy = LegacyAtmos()
    world.atmos = legacy
    world.run(n_steps=2)
    assert hasattr(world.atmos, "step_and_write")
    assert legacy.step_called == 2


def test_world_run_uses_injected_routing_orchestrator(monkeypatch):
    class FakeRouting:
        def __init__(self):
            self.configure_called = 0
            self.step_called = 0

        def configure(self, *, spec):
            self.configure_called += 1

        def step(self, *, runoff_flux, dt_seconds, precip_flux=None, evap_flux=None):
            self.step_called += 1

        def diagnostics(self):
            return {
                "flow_accum_kgps": np.zeros((12, 24)),
                "ocean_inflow_kgps": 0.0,
                "mass_closure_error_kg": 0.0,
                "lake_volume_kg": None,
            }

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_HYDRO_ENABLE", "1")
    fake = FakeRouting()
    world = QingdaiWorld.create_default()
    world.routing = fake
    world.run(n_steps=2)
    assert fake.configure_called == 1
    assert fake.step_called == 2


def test_world_run_wraps_legacy_routing_backend(monkeypatch):
    class LegacyRouting:
        def __init__(self):
            self.step_called = 0

        def step(self, *, R_land_flux, dt_seconds, precip_flux=None, evap_flux=None):
            self.step_called += 1

        def diagnostics(self):
            return {
                "flow_accum_kgps": np.zeros((12, 24)),
                "ocean_inflow_kgps": 0.0,
                "mass_closure_error_kg": 0.0,
                "lake_volume_kg": None,
            }

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_HYDRO_ENABLE", "1")
    world = QingdaiWorld.create_default()
    legacy = LegacyRouting()
    world.routing = legacy
    world.run(n_steps=2)
    assert hasattr(world.routing, "configure")
    assert legacy.step_called == 2


def test_world_run_uses_injected_hydrology_orchestrator(monkeypatch):
    class FakeHydrology:
        def __init__(self):
            self.configure_called = 0
            self.step_called = 0

        def configure(self, *, spec):
            self.configure_called += 1

        def step_and_write(self, *, state, Ta, precip_flux, evap_flux, dt, ref):
            self.step_called += 1
            return type("H", (), {"runoff_flux": np.zeros_like(ref)})()

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    fake = FakeHydrology()
    world.hydrology = fake
    world.run(n_steps=2)
    assert fake.configure_called == 1
    assert fake.step_called == 2


def test_world_run_wraps_legacy_hydrology_backend(monkeypatch):
    class LegacyHydrology:
        def __init__(self):
            self.step_called = 0

        def step(self, *, state, Ta, precip_flux, evap_flux, dt, ref):
            self.step_called += 1
            return {"runoff_flux": np.zeros_like(ref)}

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    legacy = LegacyHydrology()
    world.hydrology = legacy
    world.run(n_steps=2)
    assert hasattr(world.hydrology, "step_and_write")
    assert legacy.step_called == 2


def test_world_run_keeps_hydro_routing_order(monkeypatch):
    call_order = []

    class FakeHydrology:
        def configure(self, *, spec):
            return None

        def step_and_write(self, *, state, Ta, precip_flux, evap_flux, dt, ref):
            call_order.append("hydrology")
            return type("H", (), {"runoff_flux": np.zeros_like(ref)})()

    class FakeRouting:
        def configure(self, *, spec):
            return None

        def step(self, *, runoff_flux, dt_seconds, precip_flux=None, evap_flux=None):
            call_order.append("routing")

        def diagnostics(self):
            return {
                "flow_accum_kgps": np.zeros((12, 24)),
                "ocean_inflow_kgps": 0.0,
                "mass_closure_error_kg": 0.0,
                "lake_volume_kg": None,
            }

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    world.hydrology = FakeHydrology()
    world.routing = FakeRouting()
    world.run(n_steps=1)
    assert call_order[:2] == ["hydrology", "routing"]


def test_world_run_uses_injected_ecology_orchestrator(monkeypatch):
    class FakeEcology:
        def __init__(self):
            self.configure_called = 0
            self.apply_called = 0
            self.daily_called = 0

        def configure(self, *, spec):
            self.configure_called += 1

        def apply_albedo(self, *, base_albedo, insolation, cloud_eff, dt):
            self.apply_called += 1
            return type("E", (), {"albedo": base_albedo})()

        def step_daily_if_needed(self, *, state, dt):
            self.daily_called += 1

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    fake = FakeEcology()
    world.ecology = fake
    world.run(n_steps=2)
    assert fake.configure_called == 1
    assert fake.apply_called == 2
    assert fake.daily_called == 2


def test_world_run_wraps_legacy_ecology_backend(monkeypatch):
    class LegacyEcology:
        def __init__(self):
            self.subdaily_called = 0
            self.daily_called = 0

        def step_subdaily(self, I_total, cloud_eff, dt_seconds):
            self.subdaily_called += 1
            return np.full_like(I_total, 0.2)

        def step_daily(self, soil_water_index):
            self.daily_called += 1

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    legacy = LegacyEcology()
    world.ecology = legacy
    world.run(n_steps=2)
    assert hasattr(world.ecology, "apply_albedo")
    assert legacy.subdaily_called == 2


def test_world_run_keeps_hydro_routing_ecology_order(monkeypatch):
    call_order = []

    class FakeHydrology:
        def configure(self, *, spec):
            return None

        def step_and_write(self, *, state, Ta, precip_flux, evap_flux, dt, ref):
            call_order.append("hydrology")
            return type("H", (), {"runoff_flux": np.zeros_like(ref)})()

    class FakeRouting:
        def configure(self, *, spec):
            return None

        def step(self, *, runoff_flux, dt_seconds, precip_flux=None, evap_flux=None):
            call_order.append("routing")

        def diagnostics(self):
            return {
                "flow_accum_kgps": np.zeros((12, 24)),
                "ocean_inflow_kgps": 0.0,
                "mass_closure_error_kg": 0.0,
                "lake_volume_kg": None,
            }

    class FakeEcology:
        def configure(self, *, spec):
            return None

        def apply_albedo(self, *, base_albedo, insolation, cloud_eff, dt):
            return type("E", (), {"albedo": base_albedo})()

        def step_daily_if_needed(self, *, state, dt):
            call_order.append("ecology")
            return None

    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    world = QingdaiWorld.create_default()
    world.hydrology = FakeHydrology()
    world.routing = FakeRouting()
    world.ecology = FakeEcology()
    world.run(n_steps=1)
    assert call_order[:3] == ["hydrology", "routing", "ecology"]


def test_world_run_emits_unified_world_diagnostics(monkeypatch):
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "1")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_DIAG_EVERY", "1")
    world = QingdaiWorld.create_default()
    world.run(n_steps=2)
    diag = world.world_diagnostics
    assert diag["schema_version"] == 1
    assert diag["steps"] == 2
    assert "summary" in diag
    assert "last_step" in diag
    assert "samples" in diag
    assert len(diag["samples"]) == 2
    assert "hydrology" in diag["last_step"]
    assert "routing" in diag["last_step"]
    assert "ecology" in diag["last_step"]
