#!/usr/bin/env python3
"""
Benchmark: DoubleBufferingArray vs. naive allocate/copy approach.

This script measures per-step time and memory behavior on simple update loops:
- DB mode: read -> compute -> write[:] -> swap()
- Naive mode: read -> compute -> allocate new array each step

It uses numpy by default. If JAX is available and --jax is set, it will run a JAX
version of the compute kernel (still respecting the DB contract by passing dba.read
and writing results back with dba.write[:] outside the jit function).

Usage:
  python3 -m scripts.benchmark_double_buffering --nlat 181 --nlon 360 --steps 200
  python3 -m scripts.benchmark_double_buffering --small
  python3 -m scripts.benchmark_double_buffering --jax --warmup 3 --steps 100

Results are printed to stdout.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except Exception:
    HAS_JAX = False

from pygcm.world.state import zeros_world_state
from pygcm.numerics.double_buffer import DoubleBufferingArray as DBA


@dataclass
class Result:
    mode: str
    steps: int
    wall_ms: float
    per_step_ms: float


def _compute_numpy(out_arr: np.ndarray, in_arr: np.ndarray, a: float, b: float) -> None:
    """
    Simple compute kernel: out = a * in + b
    out_arr is written in-place to avoid additional allocation.
    """
    np.multiply(in_arr, a, out=out_arr)
    np.add(out_arr, b, out=out_arr)


def _make_jax_kernel():
    @jax.jit
    def kernel(in_arr, a, b):
        return a * in_arr + b
    return kernel


def bench_db_numpy(nlat: int, nlon: int, steps: int) -> Result:
    ws = zeros_world_state((nlat, nlon), include_ocean=False, include_hydro=False)
    # seed read buffer
    ws.atmos.u.write[:] = 1.0
    ws.swap_all()

    a, b = 0.999, 0.123
    start = time.perf_counter()
    for _ in range(steps):
        # compute into write buffer in-place
        _compute_numpy(ws.atmos.u.write, ws.atmos.u.read, a, b)
        ws.swap_all()
    end = time.perf_counter()
    dt = (end - start) * 1000.0
    return Result(mode="db-numpy", steps=steps, wall_ms=dt, per_step_ms=dt / steps)


def bench_naive_numpy(nlat: int, nlon: int, steps: int) -> Result:
    arr = np.ones((nlat, nlon), dtype=np.float64)
    a, b = 0.999, 0.123
    start = time.perf_counter()
    for _ in range(steps):
        # allocate a new array every step (naive)
        arr = a * arr + b
    end = time.perf_counter()
    dt = (end - start) * 1000.0
    return Result(mode="naive-numpy", steps=steps, wall_ms=dt, per_step_ms=dt / steps)


def bench_db_jax(nlat: int, nlon: int, steps: int, warmup: int = 3) -> Result:
    if not HAS_JAX:
        raise RuntimeError("JAX not available. Install jax/jaxlib or run without --jax.")

    ws = zeros_world_state((nlat, nlon), include_ocean=False, include_hydro=False)
    ws.atmos.u.write[:] = 1.0
    ws.swap_all()

    kernel = _make_jax_kernel()
    a, b = 0.999, 0.123

    # Warmup to trigger JIT compilation
    for _ in range(max(0, warmup)):
        out = kernel(jnp.asarray(ws.atmos.u.read), a, b)
        ws.atmos.u.write[:] = np.asarray(out)
        ws.swap_all()

    start = time.perf_counter()
    for _ in range(steps):
        out = kernel(jnp.asarray(ws.atmos.u.read), a, b)
        ws.atmos.u.write[:] = np.asarray(out)
        ws.swap_all()
    end = time.perf_counter()
    dt = (end - start) * 1000.0
    return Result(mode=f"db-jax(warmup={warmup})", steps=steps, wall_ms=dt, per_step_ms=dt / steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nlat", type=int, default=181)
    ap.add_argument("--nlon", type=int, default=360)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--small", action="store_true", help="Use a small 61x120 grid.")
    ap.add_argument("--jax", action="store_true", help="Run JAX DB benchmark (requires jax).")
    ap.add_argument("--warmup", type=int, default=3, help="JAX warmup steps.")
    args = ap.parse_args()

    if args.small:
        args.nlat, args.nlon = 61, 120

    results: list[Result] = []

    # Naive numpy
    results.append(bench_naive_numpy(args.nlat, args.nlon, args.steps))
    # DB numpy
    results.append(bench_db_numpy(args.nlat, args.nlon, args.steps))
    # DB JAX (optional)
    if args.jax:
        if not HAS_JAX:
            print("JAX not available - skipping JAX benchmark.")
        else:
            results.append(bench_db_jax(args.nlat, args.nlon, args.steps, args.warmup))

    # Print summary
    print(f"Grid: {args.nlat}x{args.nlon}, steps={args.steps}")
    for r in results:
        print(f"[{r.mode}] total={r.wall_ms:.2f} ms, per-step={r.per_step_ms:.3f} ms")

    # Simple comparative hints
    if len(results) >= 2:
        naive = next(x for x in results if x.mode == "naive-numpy")
        dbnp = next(x for x in results if x.mode == "db-numpy")
        speedup = naive.per_step_ms / dbnp.per_step_ms if dbnp.per_step_ms > 0 else float("inf")
        print(f"DB (numpy) vs Naive speedup: {speedup:.2f}x (per-step)")


if __name__ == "__main__":
    main()
