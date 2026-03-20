#!/usr/bin/env python3
"""
Benchmark JAX acceleration (P016 M1)

Usage examples:
  # NumPy/Scipy backend
  QD_USE_JAX=0 python3 -m scripts.benchmark_jax --steps 50 --nlat 121 --nlon 240

  # JAX backend on CPU
  QD_USE_JAX=1 QD_JAX_PLATFORM=cpu python3 -m scripts.benchmark_jax --steps 50 --nlat 121 --nlon 240

  # JAX backend on GPU (if available)
  QD_USE_JAX=1 QD_JAX_PLATFORM=gpu python3 -m scripts.benchmark_jax --steps 50 --nlat 121 --nlon 240

Notes:
- The script reads QD_USE_JAX (0/1) before importing pygcm modules so it respects the chosen backend.
- Times a short loop of gcm.time_step (and optional ocean.step) to produce per-step timing.
"""
import os
import sys
import time
import argparse
import json
import tracemalloc
from typing import Any
import numpy as np
resource = None
try:
    import resource as _resource
    resource = _resource
except Exception:
    pass

# Respect env backend selection BEFORE importing pygcm modules
USE_JAX = int(os.getenv("QD_USE_JAX", "0")) == 1
PLAT = os.getenv("QD_JAX_PLATFORM", "(default)")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygcm.constants as constants
from pygcm.grid import SphericalGrid
from pygcm.orbital import OrbitalSystem
from pygcm.forcing import ThermalForcing
from pygcm.dynamics import SpectralModel
from pygcm.topography import create_land_sea_mask, generate_base_properties
from pygcm import energy as energy
from pygcm.ocean import WindDrivenSlabOcean
from pygcm.jax_compat import is_enabled as JAX_IS_ENABLED, backend as JAX_BACKEND


def _memory_peak_mb() -> float:
    py_peak = 0.0
    try:
        _, peak = tracemalloc.get_traced_memory()
        py_peak = float(peak) / (1024.0 * 1024.0)
    except Exception:
        py_peak = 0.0
    rss_peak = 0.0
    if resource is not None:
        try:
            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform == "darwin":
                rss_peak = rss / (1024.0 * 1024.0)
            else:
                rss_peak = rss / 1024.0
        except Exception:
            rss_peak = 0.0
    return max(py_peak, rss_peak)


def run_benchmark(nlat: int, nlon: int, steps: int, dt: float, with_ocean: bool) -> dict[str, Any]:
    print(f"[Benchmark] Backend: JAX={JAX_IS_ENABLED()} (QD_USE_JAX={int(USE_JAX)}) platform={PLAT} backend={JAX_BACKEND()}")
    if JAX_IS_ENABLED() and JAX_BACKEND() in ("cpu", "metal"):
        print("[Benchmark][Note] JAX backend is cpu/metal; this path is usually slower than pure NumPy for this problem size.")
        print("  - Recommended: set QD_USE_JAX=0 (pure NumPy), or use Linux+CUDA with jax[cudaXX] for actual GPU speedup.")
    print(f"[Benchmark] Grid: {nlat}x{nlon}, steps={steps}, dt={dt}s, with_ocean={with_ocean}")
    grid = SphericalGrid(n_lat=nlat, n_lon=nlon)

    # Simple procedural land/sea + base maps
    land_mask = create_land_sea_mask(grid)
    base_albedo_map, friction_map = generate_base_properties(land_mask)

    # Slab ocean surface heat capacity map
    rho_w = float(os.getenv("QD_RHO_W", "1000"))
    cp_w = float(os.getenv("QD_CP_W", "4200"))
    H_mld = float(os.getenv("QD_MLD_M", "50"))
    Cs_ocean = rho_w * cp_w * H_mld
    Cs_land = float(os.getenv("QD_CS_LAND", "3e6"))
    Cs_ice = float(os.getenv("QD_CS_ICE", "5e6"))
    C_s_map = np.where(land_mask == 1, Cs_land, Cs_ocean).astype(float)

    orbital_sys = OrbitalSystem()
    forcing = ThermalForcing(grid, orbital_sys)
    eparams = energy.get_energy_params_from_env()

    # Atmos dynamics
    gcm = SpectralModel(
        grid, friction_map, H=8000, tau_rad=10 * 24 * 3600,
        greenhouse_factor=float(os.getenv("QD_GH_FACTOR", "0.40")),
        C_s_map=C_s_map, land_mask=land_mask, Cs_ocean=Cs_ocean, Cs_land=Cs_land, Cs_ice=Cs_ice
    )

    # Optional ocean
    ocean = None
    if with_ocean:
        H_ocean = float(os.getenv("QD_OCEAN_H_M", str(H_mld)))
        init_Ts = np.where(land_mask == 0, gcm.T_s, 288.0)
        ocean = WindDrivenSlabOcean(grid, land_mask, H_ocean, init_Ts=init_Ts)

    # Cloud/albedo constants
    alpha_ice = 0.6
    alpha_cloud = 0.5

    tracemalloc.start()
    warmup = min(2, max(0, steps // 10))
    if warmup > 0:
        for _ in range(warmup):
            insA, insB = forcing.calculate_insolation_components(0.0)
            gcm.isr_A, gcm.isr_B = insA, insB
            gcm.isr = insA + insB
            ice_frac = 1.0 - np.exp(-np.maximum(gcm.h_ice, 0.0) / max(1e-6, float(os.getenv("QD_HICE_REF", "0.5"))))
            albedo = np.where(land_mask == 0, 0.08, base_albedo_map)  # simple mix
            Teq = forcing.calculate_equilibrium_temp(0.0, albedo)
            gcm.time_step(Teq, dt, albedo=albedo)
            if ocean is not None:
                cloud_eff = getattr(gcm, "cloud_eff_last", gcm.cloud_cover)
                SW_atm, SW_sfc, _ = energy.shortwave_radiation(gcm.isr, albedo, cloud_eff, eparams)
                T_a = 288.0 + (9.81 / 1004.0) * gcm.h
                use_lw_v2 = int(os.getenv("QD_LW_V2", "1")) == 1
                if use_lw_v2:
                    eps_sfc_map = energy.surface_emissivity_map(land_mask, ice_frac)
                    _LW_atm, LW_sfc, _OLR, _DLR, _eps = energy.longwave_radiation_v2(gcm.T_s, T_a, cloud_eff, eps_sfc_map, eparams)
                else:
                    _LW_atm, LW_sfc, _OLR, _DLR, _eps = energy.longwave_radiation(gcm.T_s, T_a, cloud_eff, eparams)
                C_H = float(os.getenv("QD_CH", "1.5e-3"))
                cp_air = float(os.getenv("QD_CP_A", "1004.0"))
                rho_air = float(getattr(getattr(gcm, "hum_params", None), "rho_a", 1.2))
                B_land = float(os.getenv("QD_BOWEN_LAND", "0.7"))
                B_ocean = float(os.getenv("QD_BOWEN_OCEAN", "0.3"))
                SH_arr, _ = energy.boundary_layer_fluxes(gcm.T_s, T_a, gcm.u, gcm.v, land_mask, C_H=C_H, rho=rho_air, c_p=cp_air, B_land=B_land, B_ocean=B_ocean)
                LH_arr = getattr(gcm, "LH_last", 0.0)
                if np.isscalar(LH_arr):
                    scalar_lh = np.asarray(LH_arr).item()
                    LH_arr = np.full_like(gcm.T_s, float(scalar_lh))
                Q_net = SW_sfc - LW_sfc - SH_arr - LH_arr
                ocean.step(dt, gcm.u, gcm.v, Q_net=Q_net, ice_mask=(gcm.h_ice > 0.0))
                ocean_open = (land_mask == 0) & (gcm.h_ice <= 0.0)
                gcm.T_s = np.where(ocean_open, ocean.Ts, gcm.T_s)

    # Timed loop
    t0 = time.perf_counter()
    sim_time = 0.0
    for i in range(steps):
        t = i * dt
        insA, insB = forcing.calculate_insolation_components(t)
        gcm.isr_A, gcm.isr_B = insA, insB
        gcm.isr = insA + insB
        ice_frac = 1.0 - np.exp(-np.maximum(gcm.h_ice, 0.0) / max(1e-6, float(os.getenv("QD_HICE_REF", "0.5"))))
        albedo = np.where(land_mask == 0, 0.08, base_albedo_map)  # simple mix
        Teq = forcing.calculate_equilibrium_temp(t, albedo)
        gcm.time_step(Teq, dt, albedo=albedo)

        if ocean is not None:
            cloud_eff = getattr(gcm, "cloud_eff_last", gcm.cloud_cover)
            SW_atm, SW_sfc, _ = energy.shortwave_radiation(gcm.isr, albedo, cloud_eff, eparams)
            T_a = 288.0 + (9.81 / 1004.0) * gcm.h
            use_lw_v2 = int(os.getenv("QD_LW_V2", "1")) == 1
            if use_lw_v2:
                eps_sfc_map = energy.surface_emissivity_map(land_mask, ice_frac)
                _LW_atm, LW_sfc, _OLR, _DLR, _eps = energy.longwave_radiation_v2(gcm.T_s, T_a, cloud_eff, eps_sfc_map, eparams)
            else:
                _LW_atm, LW_sfc, _OLR, _DLR, _eps = energy.longwave_radiation(gcm.T_s, T_a, cloud_eff, eparams)
            C_H = float(os.getenv("QD_CH", "1.5e-3"))
            cp_air = float(os.getenv("QD_CP_A", "1004.0"))
            rho_air = float(getattr(getattr(gcm, "hum_params", None), "rho_a", 1.2))
            B_land = float(os.getenv("QD_BOWEN_LAND", "0.7"))
            B_ocean = float(os.getenv("QD_BOWEN_OCEAN", "0.3"))
            SH_arr, _ = energy.boundary_layer_fluxes(gcm.T_s, T_a, gcm.u, gcm.v, land_mask, C_H=C_H, rho=rho_air, c_p=cp_air, B_land=B_land, B_ocean=B_ocean)
            LH_arr = getattr(gcm, "LH_last", 0.0)
            if np.isscalar(LH_arr):
                scalar_lh = np.asarray(LH_arr).item()
                LH_arr = np.full_like(gcm.T_s, float(scalar_lh))
            Q_net = SW_sfc - LW_sfc - SH_arr - LH_arr
            ocean.step(dt, gcm.u, gcm.v, Q_net=Q_net, ice_mask=(gcm.h_ice > 0.0))
            ocean_open = (land_mask == 0) & (gcm.h_ice <= 0.0)
            gcm.T_s = np.where(ocean_open, ocean.Ts, gcm.T_s)

        sim_time += dt
    t1 = time.perf_counter()

    per_step = (t1 - t0) / max(1, steps)
    mem_peak_mb = _memory_peak_mb()
    tracemalloc.stop()
    result: dict[str, Any] = {
        "schema_version": 1,
        "backend": {
            "jax_enabled": bool(JAX_IS_ENABLED()),
            "qd_use_jax": bool(USE_JAX),
            "qd_jax_platform": str(PLAT),
            "jax_backend": str(JAX_BACKEND()),
        },
        "config": {
            "nlat": int(nlat),
            "nlon": int(nlon),
            "steps": int(steps),
            "dt_seconds": float(dt),
            "with_ocean": bool(with_ocean),
            "warmup_steps": int(warmup),
        },
        "metrics": {
            "total_wall_seconds": float(t1 - t0),
            "per_step_seconds": float(per_step),
            "sim_days": float(sim_time / (2 * np.pi / constants.PLANET_OMEGA)),
            "memory_peak_mb": float(mem_peak_mb),
        },
    }
    metrics: dict[str, Any] = result["metrics"]
    print(
        f"[Benchmark] Total wall time: {float(metrics['total_wall_seconds']):.3f} s | "
        f"per-step: {float(metrics['per_step_seconds']):.6f} s | "
        f"sim_days={float(metrics['sim_days']):.3f} | "
        f"memory_peak={float(metrics['memory_peak_mb']):.2f} MB"
    )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlat", type=int, default=121)
    ap.add_argument("--nlon", type=int, default=240)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=300.0)
    ap.add_argument("--with-ocean", action="store_true", default=False)
    ap.add_argument("--output-json", type=str, default="")
    args = ap.parse_args()

    # Print header
    print("=== Qingdai GCM JAX Benchmark (P016 M1) ===")
    result = run_benchmark(args.nlat, args.nlon, args.steps, args.dt, args.with_ocean)
    if str(args.output_json).strip():
        with open(str(args.output_json).strip(), "w", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
        print(f"[Benchmark] Wrote JSON: {args.output_json}")


if __name__ == "__main__":
    main()
