from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pygcm.world import QingdaiWorld


def run_oo_regression(days: float = 5.0) -> dict:
    os.environ["QD_N_LAT"] = os.getenv("QD_N_LAT", "32")
    os.environ["QD_N_LON"] = os.getenv("QD_N_LON", "64")
    os.environ["QD_DT_SECONDS"] = os.getenv("QD_DT_SECONDS", "300")
    os.environ["QD_ECO_ENABLE"] = os.getenv("QD_ECO_ENABLE", "1")
    os.environ["QD_ECO_ALBEDO_COUPLE"] = os.getenv("QD_ECO_ALBEDO_COUPLE", "1")
    os.environ["QD_OO_DIAG_EVERY"] = os.getenv("QD_OO_DIAG_EVERY", "1000000")
    world = QingdaiWorld.create_default()
    t0 = time.perf_counter()
    world.run(duration_days=days)
    elapsed = time.perf_counter() - t0
    metrics = dict(world.m4_metrics)
    metrics["elapsed_seconds"] = elapsed
    metrics["days"] = days
    return metrics


def run_cli(mode: str, sim_days: float = 0.5) -> dict:
    env = os.environ.copy()
    env["QD_SIM_DAYS"] = str(sim_days)
    env["QD_PLOT_EVERY_DAYS"] = "0"
    env["QD_ENERGY_DIAG"] = "0"
    env["QD_WATER_DIAG"] = "0"
    env["QD_OCEAN_DIAG"] = "0"
    env["QD_HUMIDITY_DIAG"] = "0"
    env["QD_ECO_DIAG"] = "0"
    env["QD_ECO_ENABLE"] = "0"
    if mode == "oo":
        env["QD_USE_OO"] = "1"
        env["QD_USE_OO_STRICT"] = "1"
    else:
        env["QD_USE_OO"] = "0"
        env["QD_USE_OO_STRICT"] = "0"
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.run_simulation"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    lines = [x for x in proc.stdout.splitlines() if x.strip()]
    err_lines = [x for x in proc.stderr.splitlines() if x.strip()]
    return {
        "mode": mode,
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "tail": lines[-8:],
        "stderr_tail": err_lines[-8:],
    }


def main() -> None:
    oo_reg = run_oo_regression(days=float(os.getenv("P021_M4_OO_DAYS", "5.0")))
    oo_th = {
        "energy_mean_abs_toa": float(os.getenv("P021_M4_TOA_MAX", "90.0")),
        "energy_mean_abs_sfc": float(os.getenv("P021_M4_SFC_MAX", "450.0")),
        "energy_mean_abs_atm": float(os.getenv("P021_M4_ATM_MAX", "350.0")),
        "water_mean_abs_residual": float(os.getenv("P021_M4_WATER_MAX", "1e-8")),
    }
    oo_pass = (
        oo_reg["energy_mean_abs_toa"] <= oo_th["energy_mean_abs_toa"]
        and oo_reg["energy_mean_abs_sfc"] <= oo_th["energy_mean_abs_sfc"]
        and oo_reg["energy_mean_abs_atm"] <= oo_th["energy_mean_abs_atm"]
        and oo_reg["water_mean_abs_residual"] <= oo_th["water_mean_abs_residual"]
    )
    oo_bench = run_cli("oo", sim_days=float(os.getenv("P021_M4_BENCH_DAYS", "0.5")))
    legacy_bench = run_cli("legacy", sim_days=float(os.getenv("P021_M4_BENCH_DAYS", "0.5")))
    ratio = None
    if oo_bench["elapsed_seconds"] > 0:
        ratio = legacy_bench["elapsed_seconds"] / oo_bench["elapsed_seconds"]
    report = {
        "oo_regression": oo_reg,
        "oo_thresholds": oo_th,
        "oo_regression_pass": oo_pass,
        "benchmark": {
            "oo": oo_bench,
            "legacy": legacy_bench,
            "legacy_over_oo_ratio": ratio,
        },
    }
    out = Path(__file__).resolve().parents[1] / "projects" / "p021-m4-results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not oo_pass:
        raise SystemExit(2)
    if oo_bench["returncode"] != 0 or legacy_bench["returncode"] != 0:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
