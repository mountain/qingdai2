"""
P020 Phase 0: world/ skeleton (façade) with DI-ready stubs.

This module introduces minimal classes to support the QD_USE_OO switch and to
prepare for Phase 1–5 without changing runtime behavior yet. In Phase 0, the
legacy engine in scripts/run_simulation.py remains the execution path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

# Typing-only import to keep mypy happy without requiring runtime import
if TYPE_CHECKING:  # pragma: no cover
    from pygcm.grid import SphericalGrid as _SphericalGrid
else:  # pragma: no cover
    _SphericalGrid = object  # sentinel for type hints

# Runtime import (optional; keeps Phase 0 lightweight)
try:
    from pygcm.grid import SphericalGrid as _RuntimeSphericalGrid
except Exception:  # pragma: no cover
    _RuntimeSphericalGrid = None


# ---------------------------
# Configuration & Parameters
# ---------------------------


@dataclass(frozen=True)
class SimConfig:
    """Minimal simulation configuration for OO world (env-driven)."""

    n_lat: int = 121
    n_lon: int = 240
    dt_seconds: float = 300.0
    use_ocean: bool = True
    use_ecology: bool = True
    use_routing: bool = True
    oo_config_diag: bool = True
    oo_metadata_enable: bool = True
    oo_metadata_json: str = ""
    total_years: float | None = None
    sim_days: float | None = None
    default_orbit_fraction: float = 0.02
    routing_netcdf: str = "data/hydrology.nc"
    routing_dt_hydro_hours: float = 6.0
    routing_treat_lake_as_water: bool = True
    routing_alpha_lake: float | None = None
    routing_diag: bool = True
    world_diagnostics_enable: bool = True
    world_diagnostics_json: str = ""
    world_diagnostics_schema_version: int = 1
    world_diagnostics_strict_validation: bool = True
    world_diagnostics_allow_backward_compat: bool = False

    def __post_init__(self) -> None:
        if int(self.n_lat) <= 0 or int(self.n_lon) <= 0:
            raise ValueError("n_lat/n_lon must be positive")
        if float(self.dt_seconds) <= 0.0:
            raise ValueError("dt_seconds must be positive")
        if self.total_years is not None and float(self.total_years) <= 0.0:
            raise ValueError("total_years must be positive when set")
        if self.sim_days is not None and float(self.sim_days) <= 0.0:
            raise ValueError("sim_days must be positive when set")
        if float(self.default_orbit_fraction) <= 0.0:
            raise ValueError("default_orbit_fraction must be positive")
        if float(self.routing_dt_hydro_hours) <= 0.0:
            raise ValueError("routing_dt_hydro_hours must be positive")
        if self.routing_alpha_lake is not None and float(self.routing_alpha_lake) < 0.0:
            raise ValueError("routing_alpha_lake must be >= 0 when set")
        if int(self.world_diagnostics_schema_version) <= 0:
            raise ValueError("world_diagnostics_schema_version must be positive")

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "domain": {
                "grid": {
                    "n_lat": int(self.n_lat),
                    "n_lon": int(self.n_lon),
                },
                "subsystems": {
                    "use_ocean": bool(self.use_ocean),
                    "use_ecology": bool(self.use_ecology),
                    "use_routing": bool(self.use_routing),
                },
                "time_step": {"dt_seconds": float(self.dt_seconds)},
            },
            "runtime_control": {
                "oo_config_diag": bool(self.oo_config_diag),
                "oo_metadata_enable": bool(self.oo_metadata_enable),
                "oo_metadata_json": str(self.oo_metadata_json),
                "total_years": None if self.total_years is None else float(self.total_years),
                "sim_days": None if self.sim_days is None else float(self.sim_days),
                "default_orbit_fraction": float(self.default_orbit_fraction),
                "routing_netcdf": str(self.routing_netcdf),
                "routing_dt_hydro_hours": float(self.routing_dt_hydro_hours),
                "routing_treat_lake_as_water": bool(self.routing_treat_lake_as_water),
                "routing_alpha_lake": (
                    None if self.routing_alpha_lake is None else float(self.routing_alpha_lake)
                ),
                "routing_diag": bool(self.routing_diag),
                "world_diagnostics_enable": bool(self.world_diagnostics_enable),
                "world_diagnostics_json": str(self.world_diagnostics_json),
                "world_diagnostics_schema_version": int(self.world_diagnostics_schema_version),
                "world_diagnostics_strict_validation": bool(
                    self.world_diagnostics_strict_validation
                ),
                "world_diagnostics_allow_backward_compat": bool(
                    self.world_diagnostics_allow_backward_compat
                ),
            },
        }

    @classmethod
    def from_snapshot(cls, doc: dict) -> SimConfig:
        def _fopt(v) -> float | None:
            if v is None:
                return None
            return float(str(v))

        if "domain" in doc and "runtime_control" in doc:
            d = doc["domain"]
            r = doc["runtime_control"]
            return cls(
                n_lat=int(d["grid"]["n_lat"]),
                n_lon=int(d["grid"]["n_lon"]),
                dt_seconds=float(d["time_step"]["dt_seconds"]),
                use_ocean=bool(d["subsystems"]["use_ocean"]),
                use_ecology=bool(d["subsystems"]["use_ecology"]),
                use_routing=bool(d["subsystems"]["use_routing"]),
                oo_config_diag=bool(r["oo_config_diag"]),
                oo_metadata_enable=bool(r["oo_metadata_enable"]),
                oo_metadata_json=str(r["oo_metadata_json"]),
                total_years=_fopt(r["total_years"]),
                sim_days=_fopt(r["sim_days"]),
                default_orbit_fraction=float(r["default_orbit_fraction"]),
                routing_netcdf=str(r.get("routing_netcdf", "data/hydrology.nc")),
                routing_dt_hydro_hours=float(r.get("routing_dt_hydro_hours", 6.0)),
                routing_treat_lake_as_water=bool(r.get("routing_treat_lake_as_water", True)),
                routing_alpha_lake=_fopt(r.get("routing_alpha_lake")),
                routing_diag=bool(r.get("routing_diag", True)),
                world_diagnostics_enable=bool(r.get("world_diagnostics_enable", True)),
                world_diagnostics_json=str(r.get("world_diagnostics_json", "")),
                world_diagnostics_schema_version=int(r.get("world_diagnostics_schema_version", 1)),
                world_diagnostics_strict_validation=bool(
                    r.get("world_diagnostics_strict_validation", True)
                ),
                world_diagnostics_allow_backward_compat=bool(
                    r.get("world_diagnostics_allow_backward_compat", False)
                ),
            )
        total_years_val = doc.get("total_years")
        sim_days_val = doc.get("sim_days")
        return cls(
            n_lat=int(doc["n_lat"]),
            n_lon=int(doc["n_lon"]),
            dt_seconds=float(doc["dt_seconds"]),
            use_ocean=bool(doc["use_ocean"]),
            use_ecology=bool(doc["use_ecology"]),
            use_routing=bool(doc["use_routing"]),
            oo_config_diag=bool(doc.get("oo_config_diag", True)),
            oo_metadata_enable=bool(doc.get("oo_metadata_enable", True)),
            oo_metadata_json=str(doc.get("oo_metadata_json", "")),
            total_years=_fopt(total_years_val),
            sim_days=_fopt(sim_days_val),
            default_orbit_fraction=float(doc.get("default_orbit_fraction", 0.02)),
            routing_netcdf=str(doc.get("routing_netcdf", "data/hydrology.nc")),
            routing_dt_hydro_hours=float(doc.get("routing_dt_hydro_hours", 6.0)),
            routing_treat_lake_as_water=bool(doc.get("routing_treat_lake_as_water", True)),
            routing_alpha_lake=_fopt(doc.get("routing_alpha_lake")),
            routing_diag=bool(doc.get("routing_diag", True)),
            world_diagnostics_enable=bool(doc.get("world_diagnostics_enable", True)),
            world_diagnostics_json=str(doc.get("world_diagnostics_json", "")),
            world_diagnostics_schema_version=int(doc.get("world_diagnostics_schema_version", 1)),
            world_diagnostics_strict_validation=bool(
                doc.get("world_diagnostics_strict_validation", True)
            ),
            world_diagnostics_allow_backward_compat=bool(
                doc.get("world_diagnostics_allow_backward_compat", False)
            ),
        )

    @classmethod
    def from_env(cls) -> SimConfig:
        def _ibool(name: str, default: str = "1") -> bool:
            try:
                return int(os.getenv(name, default)) == 1
            except Exception:
                return default == "1"

        def _int(name: str, default: str) -> int:
            try:
                return int(os.getenv(name, default))
            except Exception:
                return int(default)

        def _float(name: str, default: str) -> float:
            try:
                return float(os.getenv(name, default))
            except Exception:
                return float(default)

        def _fopt(name: str) -> float | None:
            raw = os.getenv(name, "").strip()
            if raw in ("", "None", "none", "null"):
                return None
            try:
                return float(raw)
            except Exception:
                return None

        def _s(name: str, default: str = "") -> str:
            try:
                return str(os.getenv(name, default))
            except Exception:
                return str(default)

        return cls(
            n_lat=_int("QD_N_LAT", "121"),
            n_lon=_int("QD_N_LON", "240"),
            dt_seconds=_float("QD_DT_SECONDS", "300"),
            use_ocean=_ibool("QD_USE_OCEAN", "1"),
            use_ecology=_ibool("QD_ECO_ENABLE", "1"),
            use_routing=_ibool("QD_HYDRO_ENABLE", "1"),
            oo_config_diag=_ibool("QD_OO_CONFIG_DIAG", "1"),
            oo_metadata_enable=_ibool("QD_OO_METADATA_ENABLE", "1"),
            oo_metadata_json=_s("QD_OO_METADATA_JSON", "").strip(),
            total_years=_fopt("QD_TOTAL_YEARS"),
            sim_days=_fopt("QD_SIM_DAYS"),
            default_orbit_fraction=_float("QD_OO_DEFAULT_ORBIT_FRACTION", "0.02"),
            routing_netcdf=_s("QD_HYDRO_NETCDF", "data/hydrology.nc").strip(),
            routing_dt_hydro_hours=_float("QD_HYDRO_DT_HOURS", "6"),
            routing_treat_lake_as_water=_ibool("QD_TREAT_LAKE_AS_WATER", "1"),
            routing_alpha_lake=_fopt("QD_ALPHA_LAKE"),
            routing_diag=_ibool("QD_HYDRO_DIAG", "1"),
            world_diagnostics_enable=_ibool("QD_OO_WORLD_DIAG_ENABLE", "1"),
            world_diagnostics_json=_s("QD_OO_WORLD_DIAG_JSON", "").strip(),
            world_diagnostics_schema_version=_int("QD_OO_WORLD_DIAG_SCHEMA_VERSION", "1"),
            world_diagnostics_strict_validation=_ibool("QD_OO_WORLD_DIAG_STRICT_VALIDATE", "1"),
            world_diagnostics_allow_backward_compat=_ibool(
                "QD_OO_WORLD_DIAG_ALLOW_BACKCOMPAT", "0"
            ),
        )


@dataclass(frozen=True)
class PhysicsParams:
    gh_factor: float = 0.582
    ce: float = 1.3e-3
    lv: float = 2.5e6
    rho_w: float = 1000.0
    cp_w: float = 4200.0
    mld_m: float = 50.0
    cs_land: float = 3.0e6
    cs_ice: float = 5.0e6
    q_init_rh: float = 0.5
    ch: float = 1.5e-3
    cp_a: float = 1004.0
    bowen_land: float = 0.7
    bowen_ocean: float = 0.3
    oo_diag_every: int = 200
    rho_a: float = 1.2
    h_mbl: float = 800.0
    p0: float = 1.0e5
    tau_cond: float = 1800.0
    ocean_evap_scale: float = 1.0
    land_evap_scale: float = 0.5
    ice_evap_scale: float = 0.05
    runoff_tau_days: float = 10.0
    wland_cap_mm: float | None = None
    snow_thresh_k: float = 273.15
    snow_melt_rate_mm_day: float = 5.0
    snow_t_band_k: float = 1.5
    snow_melt_mode: str = "degree_day"
    snow_ddf_mm_per_k_day: float = 3.0
    snow_melt_tref_k: float = 273.15
    swe_ref_mm: float = 15.0
    swe_max_mm: float | None = None
    snow_albedo_fresh: float = 0.70
    water_diag: bool = True
    sw_a0: float = 0.06
    sw_kc: float = 0.20
    lw_eps0: float = 0.70
    lw_kc: float = 0.20
    t_floor: float = 150.0
    c_sfc: float = 2.0e7
    energy_diag: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= float(self.gh_factor) <= 1.0):
            raise ValueError("gh_factor must be in [0,1]")
        if float(self.ce) <= 0.0:
            raise ValueError("ce must be positive")
        if float(self.lv) <= 0.0:
            raise ValueError("lv must be positive")
        if float(self.rho_w) <= 0.0 or float(self.cp_w) <= 0.0 or float(self.mld_m) <= 0.0:
            raise ValueError("rho_w/cp_w/mld_m must be positive")
        if float(self.cs_land) <= 0.0 or float(self.cs_ice) <= 0.0:
            raise ValueError("cs_land/cs_ice must be positive")
        if not (0.0 <= float(self.q_init_rh) <= 1.0):
            raise ValueError("q_init_rh must be in [0,1]")
        if float(self.ch) <= 0.0 or float(self.cp_a) <= 0.0:
            raise ValueError("ch/cp_a must be positive")
        if float(self.oo_diag_every) <= 0:
            raise ValueError("oo_diag_every must be positive")
        if float(self.rho_a) <= 0.0 or float(self.h_mbl) <= 0.0 or float(self.p0) <= 0.0:
            raise ValueError("rho_a/h_mbl/p0 must be positive")
        if float(self.tau_cond) <= 0.0:
            raise ValueError("tau_cond must be positive")
        if float(self.runoff_tau_days) <= 0.0:
            raise ValueError("runoff_tau_days must be positive")
        if float(self.snow_t_band_k) <= 0.0:
            raise ValueError("snow_t_band_k must be positive")
        if str(self.snow_melt_mode) not in ("degree_day", "constant"):
            raise ValueError("snow_melt_mode must be degree_day or constant")
        if float(self.swe_ref_mm) <= 0.0:
            raise ValueError("swe_ref_mm must be positive")
        if float(self.t_floor) <= 0.0 or float(self.c_sfc) <= 0.0:
            raise ValueError("t_floor/c_sfc must be positive")

    @classmethod
    def from_env(cls) -> PhysicsParams:
        def _f(name: str, default: str) -> float:
            try:
                return float(os.getenv(name, default))
            except Exception:
                return float(default)

        def _i(name: str, default: str) -> int:
            try:
                return int(os.getenv(name, default))
            except Exception:
                return int(default)

        def _b(name: str, default: str) -> bool:
            try:
                return int(os.getenv(name, default)) == 1
            except Exception:
                return default == "1"

        def _fopt(name: str) -> float | None:
            raw = os.getenv(name, "").strip()
            if raw in ("", "None", "none", "null"):
                return None
            try:
                return float(raw)
            except Exception:
                return None

        return cls(
            gh_factor=_f("QD_GH_FACTOR", "0.582"),
            ce=_f("QD_CE", "1.3e-3"),
            lv=_f("QD_LV", "2.5e6"),
            rho_w=_f("QD_RHO_W", "1000"),
            cp_w=_f("QD_CP_W", "4200"),
            mld_m=_f("QD_MLD_M", "50"),
            cs_land=_f("QD_CS_LAND", "3e6"),
            cs_ice=_f("QD_CS_ICE", "5e6"),
            q_init_rh=_f("QD_Q_INIT_RH", "0.5"),
            ch=_f("QD_CH", "1.5e-3"),
            cp_a=_f("QD_CP_A", "1004.0"),
            bowen_land=_f("QD_BOWEN_LAND", "0.7"),
            bowen_ocean=_f("QD_BOWEN_OCEAN", "0.3"),
            oo_diag_every=_i("QD_OO_DIAG_EVERY", "200"),
            rho_a=_f("QD_RHO_A", "1.2"),
            h_mbl=_f("QD_MBL_H", "800.0"),
            p0=_f("QD_P0", "1.0e5"),
            tau_cond=_f("QD_TAU_COND", "1800.0"),
            ocean_evap_scale=_f("QD_OCEAN_EVAP_SCALE", "1.0"),
            land_evap_scale=_f("QD_LAND_EVAP_SCALE", "0.5"),
            ice_evap_scale=_f("QD_ICE_EVAP_SCALE", "0.05"),
            runoff_tau_days=_f("QD_RUNOFF_TAU_DAYS", "10.0"),
            wland_cap_mm=_fopt("QD_WLAND_CAP"),
            snow_thresh_k=_f("QD_SNOW_THRESH", "273.15"),
            snow_melt_rate_mm_day=_f("QD_SNOW_MELT_RATE", "5.0"),
            snow_t_band_k=_f("QD_SNOW_T_BAND", "1.5"),
            snow_melt_mode=os.getenv("QD_SNOW_MELT_MODE", "degree_day").strip().lower(),
            snow_ddf_mm_per_k_day=_f("QD_SNOW_DDF_MM_PER_K_DAY", "3.0"),
            snow_melt_tref_k=_f("QD_SNOW_MELT_TREF", "273.15"),
            swe_ref_mm=_f("QD_SWE_REF_MM", "15.0"),
            swe_max_mm=_fopt("QD_SWE_MAX_MM"),
            snow_albedo_fresh=_f("QD_SNOW_ALBEDO_FRESH", "0.70"),
            water_diag=_b("QD_WATER_DIAG", "1"),
            sw_a0=_f("QD_SW_A0", "0.06"),
            sw_kc=_f("QD_SW_KC", "0.20"),
            lw_eps0=_f("QD_LW_EPS0", "0.70"),
            lw_kc=_f("QD_LW_KC", "0.20"),
            t_floor=_f("QD_T_FLOOR", "150.0"),
            c_sfc=_f("QD_CS", "2.0e7"),
            energy_diag=_b("QD_ENERGY_DIAG", "1"),
        )

    def snapshot(self) -> dict:
        return {
            "gh_factor": float(self.gh_factor),
            "ce": float(self.ce),
            "lv": float(self.lv),
            "rho_w": float(self.rho_w),
            "cp_w": float(self.cp_w),
            "mld_m": float(self.mld_m),
            "cs_land": float(self.cs_land),
            "cs_ice": float(self.cs_ice),
            "q_init_rh": float(self.q_init_rh),
            "ch": float(self.ch),
            "cp_a": float(self.cp_a),
            "bowen_land": float(self.bowen_land),
            "bowen_ocean": float(self.bowen_ocean),
            "oo_diag_every": int(self.oo_diag_every),
            "rho_a": float(self.rho_a),
            "h_mbl": float(self.h_mbl),
            "p0": float(self.p0),
            "tau_cond": float(self.tau_cond),
            "ocean_evap_scale": float(self.ocean_evap_scale),
            "land_evap_scale": float(self.land_evap_scale),
            "ice_evap_scale": float(self.ice_evap_scale),
            "runoff_tau_days": float(self.runoff_tau_days),
            "wland_cap_mm": None if self.wland_cap_mm is None else float(self.wland_cap_mm),
            "snow_thresh_k": float(self.snow_thresh_k),
            "snow_melt_rate_mm_day": float(self.snow_melt_rate_mm_day),
            "snow_t_band_k": float(self.snow_t_band_k),
            "snow_melt_mode": str(self.snow_melt_mode),
            "snow_ddf_mm_per_k_day": float(self.snow_ddf_mm_per_k_day),
            "snow_melt_tref_k": float(self.snow_melt_tref_k),
            "swe_ref_mm": float(self.swe_ref_mm),
            "swe_max_mm": None if self.swe_max_mm is None else float(self.swe_max_mm),
            "snow_albedo_fresh": float(self.snow_albedo_fresh),
            "water_diag": bool(self.water_diag),
            "sw_a0": float(self.sw_a0),
            "sw_kc": float(self.sw_kc),
            "lw_eps0": float(self.lw_eps0),
            "lw_kc": float(self.lw_kc),
            "t_floor": float(self.t_floor),
            "c_sfc": float(self.c_sfc),
            "energy_diag": bool(self.energy_diag),
        }


@dataclass(frozen=True)
class SpectralBands:
    nbands: int = 16

    def __post_init__(self) -> None:
        if int(self.nbands) <= 0:
            raise ValueError("nbands must be positive")

    @classmethod
    def from_env(cls) -> SpectralBands:
        try:
            nb = int(os.getenv("QD_ECO_SPECTRAL_BANDS", "16"))
        except Exception:
            nb = 16
        return cls(nbands=nb)

    def snapshot(self) -> dict:
        return {"nbands": int(self.nbands)}


@dataclass(frozen=True)
class EcologyParams:
    lai_albedo_weight: float = 1.0
    feedback_mode: str = "instant"
    soil_water_cap: float = 50.0
    albedo_couple: bool = True

    def __post_init__(self) -> None:
        if float(self.lai_albedo_weight) < 0.0:
            raise ValueError("lai_albedo_weight must be >= 0")
        if str(self.feedback_mode) not in ("instant", "daily"):
            raise ValueError("feedback_mode must be instant or daily")
        if float(self.soil_water_cap) <= 0.0:
            raise ValueError("soil_water_cap must be positive")

    @classmethod
    def from_env(cls) -> EcologyParams:
        try:
            w = float(os.getenv("QD_ECO_LAI_ALBEDO_WEIGHT", "1.0"))
        except Exception:
            w = 1.0
        try:
            swc = float(os.getenv("QD_ECO_SOIL_WATER_CAP", "50.0"))
        except Exception:
            swc = 50.0
        try:
            couple = int(os.getenv("QD_ECO_ALBEDO_COUPLE", "1")) == 1
        except Exception:
            couple = True
        mode = os.getenv("QD_ECO_FEEDBACK_MODE", "instant").strip().lower()
        return cls(
            lai_albedo_weight=w,
            feedback_mode=mode,
            soil_water_cap=swc,
            albedo_couple=couple,
        )

    def snapshot(self) -> dict:
        return {
            "lai_albedo_weight": float(self.lai_albedo_weight),
            "feedback_mode": str(self.feedback_mode),
            "soil_water_cap": float(self.soil_water_cap),
            "albedo_couple": bool(self.albedo_couple),
        }


@dataclass(frozen=True)
class ParamsRegistry:
    """Aggregate of parameter models."""

    physics: PhysicsParams = PhysicsParams()
    bands: SpectralBands = SpectralBands()
    ecology: EcologyParams = EcologyParams()

    @classmethod
    def from_env(cls) -> ParamsRegistry:
        return cls(
            physics=PhysicsParams.from_env(),
            bands=SpectralBands.from_env(),
            ecology=EcologyParams.from_env(),
        )

    def snapshot(self) -> dict:
        return {
            "schema_version": 1,
            "parameter_groups": {
                "physics": self.physics.snapshot(),
                "bands": self.bands.snapshot(),
                "ecology": self.ecology.snapshot(),
            },
        }

    @classmethod
    def from_snapshot(cls, doc: dict) -> ParamsRegistry:
        groups = doc.get("parameter_groups", doc)
        return cls(
            physics=PhysicsParams(**groups["physics"]),
            bands=SpectralBands(**groups["bands"]),
            ecology=EcologyParams(**groups["ecology"]),
        )


# -------------
# World State
# -------------


@dataclass
class WorldState:
    """Minimal world state for Phase 0. Extended in later phases."""

    t_seconds: float = 0.0


# ----------------
# Qingdai World
# ----------------


class QingdaiWorld:
    """
    Phase 0 façade:
    - Supports DI (dependency injection) via constructor keyword args.
    - Provides create_default() to assemble from env + grid.
    - Implements nominal double-buffer layout without mutating legacy engine.
    """

    def __init__(
        self,
        config: SimConfig,
        params: ParamsRegistry,
        grid: _SphericalGrid | None,
        *,
        state: WorldState | None = None,
        atmos=None,
        ocean=None,
        surface=None,
        hydrology=None,
        routing=None,
        ecology=None,
        forcing=None,
    ) -> None:
        self.config = config
        self.params = params
        self.grid = grid
        # Double buffer placeholders (Phase 0: no external use)
        self.current_state = state or WorldState(t_seconds=0.0)
        self.next_state = WorldState(t_seconds=0.0)
        self.world_diagnostics: dict[str, object] = {}

        # DI slots (kept for future phases; unused in Phase 0)
        self.atmos = atmos
        self.ocean = ocean
        self.surface = surface
        self.hydrology = hydrology
        self.routing = routing
        self.ecology = ecology
        self.forcing = forcing

    @classmethod
    def create_default(cls) -> QingdaiWorld:
        cfg = SimConfig.from_env()
        grd = (
            _RuntimeSphericalGrid(cfg.n_lat, cfg.n_lon)
            if _RuntimeSphericalGrid is not None
            else None
        )
        pr = ParamsRegistry.from_env()
        if bool(cfg.oo_config_diag):
            print(f"[P020] SimConfig snapshot: {cfg.snapshot()}")
            print(f"[P020] Params snapshot: {pr.snapshot()}")
        return cls(cfg, pr, grd)

    # Phase 0: value-semantics step() without touching legacy runtime
    def step(self) -> WorldState:
        """Advance internal clock by one dt and swap buffers (placeholder)."""
        self.next_state.t_seconds = self.current_state.t_seconds + float(self.config.dt_seconds)
        # Buffer swap (no copies)
        self.current_state, self.next_state = self.next_state, self.current_state
        return self.current_state

    def run(self, n_steps: int | None = None, duration_days: float | None = None) -> None:
        import json

        import numpy as np

        from pygcm import constants as const
        from pygcm import energy as _energy
        from pygcm import humidity as hum
        from pygcm import hydrology as hydro
        from pygcm.forcing import ThermalForcing
        from pygcm.orbital import OrbitalSystem
        from pygcm.topography import create_land_sea_mask, generate_base_properties
        from pygcm.world.atmos_orchestrator import ensure_atmos_orchestrator
        from pygcm.world.diagnostics import (
            make_world_diagnostics_document,
            validate_world_diagnostics,
            world_diagnostics_to_jsonable,
        )
        from pygcm.world.ecology_orchestrator import ensure_ecology_orchestrator
        from pygcm.world.hydrology_orchestrator import ensure_hydrology_orchestrator
        from pygcm.world.ocean_orchestrator import ensure_ocean_orchestrator
        from pygcm.world.orchestrator_spec import build_world_orchestrator_spec
        from pygcm.world.ports import ColumnProcessIn, SurfaceToAtmosphere
        from pygcm.world.routing_orchestrator import ensure_routing_orchestrator
        from pygcm.world.state import zeros_world_state_from_grid
        from pygcm.world.world_step_ops import (
            compute_column_step,
            compute_energy_diag,
            normalize_fluxes,
            resolve_total_seconds,
        )

        grid = self.grid
        if grid is None:
            return
        state = zeros_world_state_from_grid(grid)
        land_mask = create_land_sea_mask(grid)
        base_albedo_map, friction_map = generate_base_properties(land_mask)
        land_frac = float(np.mean(land_mask == 1))
        self.run_metadata = {
            "schema_version": 1,
            "grid": {"n_lat": int(self.config.n_lat), "n_lon": int(self.config.n_lon)},
            "config": self.config.snapshot(),
            "params": self.params.snapshot(),
            "topography": {"land_fraction": land_frac},
        }
        meta_out = str(self.config.oo_metadata_json).strip()
        if bool(self.config.oo_metadata_enable) and meta_out:
            try:
                with open(meta_out, "w", encoding="utf-8") as fp:
                    json.dump(self.run_metadata, fp, ensure_ascii=False, indent=2)
            except Exception as _me:
                print(f"[P020] metadata write failed: {_me}")
        state.surface.Ts.write[:] = 288.0
        state.surface.Ts.swap()
        state.surface.Ts.write[:] = 288.0
        state.surface.Ts.swap()
        pp = self.params.physics
        ep = self.params.ecology
        rho_w = float(pp.rho_w)
        cp_w = float(pp.cp_w)
        H_mld = float(pp.mld_m)
        Cs_ocean = rho_w * cp_w * H_mld
        Cs_land = float(pp.cs_land)
        Cs_ice = float(pp.cs_ice)
        C_s_map = np.where(land_mask == 1, Cs_land, Cs_ocean).astype(float)
        atm = cast(Any, ensure_atmos_orchestrator(self.atmos))
        self.atmos = atm
        orbital = OrbitalSystem()
        forcing = ThermalForcing(grid, orbital)
        hparams = hydro.HydrologyParams(
            runoff_tau_days=float(pp.runoff_tau_days),
            wland_cap_mm=pp.wland_cap_mm,
            snow_thresh_K=float(pp.snow_thresh_k),
            snow_melt_rate_mm_day=float(pp.snow_melt_rate_mm_day),
            rho_w=float(pp.rho_w),
            snow_t_band_K=float(pp.snow_t_band_k),
            snow_melt_mode=str(pp.snow_melt_mode),
            snow_ddf_mm_per_k_day=float(pp.snow_ddf_mm_per_k_day),
            snow_melt_tref_K=float(pp.snow_melt_tref_k),
            swe_enable=True,
            swe_ref_mm=float(pp.swe_ref_mm),
            swe_max_mm=pp.swe_max_mm,
            snow_albedo_fresh=float(pp.snow_albedo_fresh),
            diag=bool(pp.water_diag),
        )
        hum_params = hum.HumidityParams(
            C_E=float(pp.ce),
            rho_a=float(pp.rho_a),
            h_mbl=float(pp.h_mbl),
            L_v=float(pp.lv),
            p0=float(pp.p0),
            ocean_evap_scale=float(pp.ocean_evap_scale),
            land_evap_scale=float(pp.land_evap_scale),
            ice_evap_scale=float(pp.ice_evap_scale),
            tau_cond=float(pp.tau_cond),
            diag=bool(pp.water_diag),
        )
        eparams = _energy.EnergyParams(
            sw_a0=float(pp.sw_a0),
            sw_kc=float(pp.sw_kc),
            lw_eps0=float(pp.lw_eps0),
            lw_kc=float(pp.lw_kc),
            t_floor=float(pp.t_floor),
            c_sfc=float(pp.c_sfc),
            diag=bool(pp.energy_diag),
        )
        dt = float(self.config.dt_seconds)
        day_in_seconds = 2 * np.pi / float(const.PLANET_OMEGA)
        orchestrator_spec = build_world_orchestrator_spec(
            grid=grid,
            friction_map=friction_map,
            land_mask=land_mask,
            C_s_map=C_s_map,
            Cs_ocean=Cs_ocean,
            Cs_land=Cs_land,
            Cs_ice=Cs_ice,
            H_m=H_mld,
            rho_w=float(pp.rho_w),
            cp_w=float(pp.cp_w),
            routing_enabled=bool(self.config.use_routing),
            routing_network_nc_path=str(self.config.routing_netcdf),
            routing_dt_hydro_hours=float(self.config.routing_dt_hydro_hours),
            routing_treat_lake_as_water=bool(self.config.routing_treat_lake_as_water),
            routing_alpha_lake=self.config.routing_alpha_lake,
            routing_diag=bool(self.config.routing_diag),
            hydrology_params=hparams,
            ecology_enabled=bool(self.config.use_ecology),
            ecology_day_in_seconds=day_in_seconds,
            ecology_soil_water_cap=float(ep.soil_water_cap),
            ecology_lai_albedo_weight=float(ep.lai_albedo_weight),
            ecology_albedo_couple=bool(ep.albedo_couple),
            humidity_params=hum_params,
            energy_params=eparams,
        )
        atm.configure(spec=orchestrator_spec)
        rh0 = float(pp.q_init_rh)
        q0 = hum.q_init(state.surface.Ts.read, RH0=rh0, p0=hum_params.p0)
        state.atmos.q.write[:] = q0
        state.atmos.cloud.write[:] = 0.0
        state.atmos.Ta.write[:] = 288.0
        if state.hydro is not None:
            state.hydro.W_land.write[:] = 10.0
            state.hydro.SWE.write[:] = 0.0
        state.swap_all()
        total_seconds = resolve_total_seconds(
            dt=dt,
            day_in_seconds=day_in_seconds,
            orbital_period=float(orbital.T_planet),
            duration_days=duration_days,
            n_steps=n_steps,
            total_years=self.config.total_years,
            sim_days=self.config.sim_days,
            default_orbit_fraction=self.config.default_orbit_fraction,
        )
        t0 = 0.0
        steps = int(max(1, total_seconds // dt))
        ocean = cast(Any, ensure_ocean_orchestrator(self.ocean))
        self.ocean = ocean
        ocean.configure(spec=orchestrator_spec)
        hydrology = cast(Any, ensure_hydrology_orchestrator(self.hydrology))
        self.hydrology = hydrology
        hydrology.configure(spec=orchestrator_spec)
        routing = cast(Any, ensure_routing_orchestrator(self.routing))
        self.routing = routing
        routing.configure(spec=orchestrator_spec)
        ecology = cast(Any, ensure_ecology_orchestrator(self.ecology))
        self.ecology = ecology
        ecology.configure(spec=orchestrator_spec)
        energy_abs_toa = 0.0
        energy_abs_sfc = 0.0
        energy_abs_atm = 0.0
        energy_count = 0
        water_abs_residual = 0.0
        water_count = 0
        water_prev_total = None
        world_diag_samples: list[dict[str, object]] = []
        last_world_diag: dict[str, object] = {}
        for i in range(steps):
            t = t0 + i * dt
            insolation = forcing.calculate_insolation(t)
            col_step = compute_column_step(
                h_read=state.atmos.h.read,
                q_read=state.atmos.q.read,
                cloud_read=state.atmos.cloud.read,
                dt=dt,
                hum_mod=hum,
                hum_params=hum_params,
            )
            eco_out = ecology.apply_albedo(
                base_albedo=base_albedo_map,
                insolation=insolation,
                cloud_eff=col_step.cloud_eff,
                dt=dt,
            )
            albedo = eco_out.albedo
            Teq = forcing.calculate_equilibrium_temp(t, albedo)
            surface_in = SurfaceToAtmosphere(
                T_s=state.surface.Ts.read,
                land_mask=land_mask,
                ice_mask=(state.surface.h_ice.read > 1.0e-6),
                base_albedo=albedo,
                friction_map=friction_map,
                insolation=insolation,
            )
            column_in = ColumnProcessIn(
                q=col_step.q_next,
                cloud=col_step.cloud_eff,
                precip_rate=col_step.P_cond,
                Ta=col_step.Ta,
                RH=col_step.RH,
                u10=state.atmos.u.read,
                v10=state.atmos.v.read,
            )
            fluxes, col_out = atm.step_and_write(
                state=state,
                dt=dt,
                h_eq=Teq,
                surface_in=surface_in,
                column_in=column_in,
            )
            if col_out is not None:
                state.atmos.q.write[:] = np.clip(col_out.q_next, 0.0, 0.2)
                state.atmos.cloud.write[:] = np.clip(col_out.cloud_next, 0.0, 1.0)
                precip_flux = np.maximum(0.0, col_out.precip_rate_next)
            else:
                state.atmos.q.write[:] = np.clip(col_step.q_next, 0.0, 0.2)
                precip_flux = np.maximum(0.0, col_step.P_cond)
            state.atmos.Ta.write[:] = 288.0 + (9.81 / 1004.0) * state.atmos.h.write
            evap_flux, Q_net, _precip_from_flux = normalize_fluxes(fluxes, albedo)
            hydro_out = hydrology.step_and_write(
                state=state,
                precip_flux=precip_flux,
                Ta=col_step.Ta,
                evap_flux=evap_flux,
                dt=dt,
                ref=albedo,
            )
            runoff_flux = hydro_out.runoff_flux
            routing.step(
                runoff_flux=runoff_flux,
                dt_seconds=dt,
                precip_flux=precip_flux,
                evap_flux=evap_flux,
            )
            routing_diag = routing.diagnostics() if hasattr(routing, "diagnostics") else {}
            hydrology_diag = hydrology.diagnostics() if hasattr(hydrology, "diagnostics") else {}
            diagE = compute_energy_diag(
                energy_mod=_energy,
                eparams=eparams,
                insolation=insolation,
                albedo=albedo,
                cloud_eff=col_step.cloud_eff,
                Ts_read=state.surface.Ts.read,
                Ta=col_step.Ta,
                land_mask=land_mask,
                ice_mask=(state.surface.h_ice.read > 1.0e-6),
                u_read=state.atmos.u.read,
                v_read=state.atmos.v.read,
                evap_flux=evap_flux,
                lat_mesh=grid.lat_mesh,
                ch=float(pp.ch),
                cp_air=float(pp.cp_a),
                bowen_land=float(pp.bowen_land),
                bowen_ocean=float(pp.bowen_ocean),
                latent_heat=float(hum_params.L_v),
            )
            energy_abs_toa += abs(float(diagE["TOA_net"]))
            energy_abs_sfc += abs(float(diagE["SFC_net"]))
            energy_abs_atm += abs(float(diagE["ATM_net"]))
            energy_count += 1
            if state.hydro is not None:
                diagW = hydro.diagnose_water_closure(
                    lat_mesh=grid.lat_mesh,
                    q=state.atmos.q.write,
                    rho_a=hum_params.rho_a,
                    h_mbl=hum_params.h_mbl,
                    h_ice=state.surface.h_ice.read,
                    rho_i=917.0,
                    W_land=state.hydro.W_land.write,
                    S_snow=state.hydro.SWE.write,
                    E_flux=evap_flux,
                    P_flux=precip_flux,
                    R_flux=runoff_flux,
                    dt_since_prev=(dt if water_prev_total is not None else None),
                    prev_total=water_prev_total,
                )
                if "closure_residual" in diagW:
                    water_abs_residual += abs(float(diagW["closure_residual"]))
                    water_count += 1
                water_prev_total = float(diagW["total_reservoir_mean"])
            ice_mask = np.zeros_like(albedo, dtype=bool)
            ocean.step_and_write(
                state=state,
                dt=dt,
                u_atm=state.atmos.u.read,
                v_atm=state.atmos.v.read,
                Q_net=Q_net,
                ice_mask=ice_mask,
            )
            ecology.step_daily_if_needed(state=state, dt=dt)
            ecology_diag = ecology.diagnostics() if hasattr(ecology, "diagnostics") else {}
            routing_flow_accum = routing_diag.get("flow_accum_kgps", np.zeros_like(runoff_flux))
            last_world_diag = {
                "step": int(i + 1),
                "energy": {
                    "toa_net": float(diagE["TOA_net"]),
                    "sfc_net": float(diagE["SFC_net"]),
                    "atm_net": float(diagE["ATM_net"]),
                },
                "water": {
                    "evap_mean": float(np.nanmean(evap_flux)),
                    "precip_mean": float(np.nanmean(precip_flux)),
                    "runoff_mean": float(np.nanmean(runoff_flux)),
                },
                "hydrology": hydrology_diag,
                "routing": {
                    "steps": float(routing_diag.get("steps", 0.0)),
                    "ocean_inflow_kgps": float(routing_diag.get("ocean_inflow_kgps", 0.0)),
                    "mass_closure_error_kg": float(routing_diag.get("mass_closure_error_kg", 0.0)),
                    "flow_accum_mean_kgps": float(np.nanmean(routing_flow_accum)),
                },
                "ecology": ecology_diag,
            }
            if (i + 1) % max(1, int(pp.oo_diag_every)) == 0:
                world_diag_samples.append(last_world_diag)
            state.swap_all()
            self.current_state.t_seconds += dt
        energy_mean_abs_toa = energy_abs_toa / max(1, energy_count)
        energy_mean_abs_sfc = energy_abs_sfc / max(1, energy_count)
        energy_mean_abs_atm = energy_abs_atm / max(1, energy_count)
        water_mean_abs_residual = water_abs_residual / max(1, water_count)
        self.m4_metrics = {
            "energy_mean_abs_toa": energy_mean_abs_toa,
            "energy_mean_abs_sfc": energy_mean_abs_sfc,
            "energy_mean_abs_atm": energy_mean_abs_atm,
            "water_mean_abs_residual": water_mean_abs_residual,
            "steps": steps,
        }
        self.world_diagnostics = make_world_diagnostics_document(
            schema_version=int(self.config.world_diagnostics_schema_version),
            steps=int(steps),
            summary={
                "energy_mean_abs_toa": float(energy_mean_abs_toa),
                "energy_mean_abs_sfc": float(energy_mean_abs_sfc),
                "energy_mean_abs_atm": float(energy_mean_abs_atm),
                "water_mean_abs_residual": float(water_mean_abs_residual),
            },
            last_step=last_world_diag,
            samples=world_diag_samples,
            strict=bool(self.config.world_diagnostics_strict_validation),
            allow_backward_compat=bool(self.config.world_diagnostics_allow_backward_compat),
        ).to_dict()
        validate_world_diagnostics(
            self.world_diagnostics,
            expected_schema_version=int(self.config.world_diagnostics_schema_version),
        )
        diag_out = str(self.config.world_diagnostics_json).strip()
        if bool(self.config.world_diagnostics_enable) and diag_out:
            with open(diag_out, "w", encoding="utf-8") as fp:
                json.dump(
                    world_diagnostics_to_jsonable(self.world_diagnostics),
                    fp,
                    ensure_ascii=False,
                    indent=2,
                )
        print(
            f"[P021-DIAG] steps={steps} | energy_mean_abs TOA={energy_mean_abs_toa:.3f} "
            f"SFC={energy_mean_abs_sfc:.3f} ATM={energy_mean_abs_atm:.3f} | "
            f"water_residual={water_mean_abs_residual:.3e}"
        )
        try:
            import numpy as _np

            u = _np.asarray(state.atmos.u.read)
            v = _np.asarray(state.atmos.v.read)
            h = _np.asarray(state.atmos.h.read)
            print(
                f"[P020] OO world completed {steps} step(s) | max|u|={_np.max(_np.abs(u)):.2f} m/s | max|v|={_np.max(_np.abs(v)):.2f} m/s | max|h|={_np.max(_np.abs(h)):.2f}"
            )
        except Exception:
            pass


__all__ = [
    "SimConfig",
    "PhysicsParams",
    "SpectralBands",
    "EcologyParams",
    "ParamsRegistry",
    "WorldState",
    "QingdaiWorld",
]
