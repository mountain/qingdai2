import importlib
import json
import sys


def test_benchmark_main_writes_json(monkeypatch, tmp_path):
    out = tmp_path / "bench.json"
    mod = importlib.import_module("scripts.benchmark_jax")

    def fake_run(nlat, nlon, steps, dt, with_ocean):
        return {
            "schema_version": 1,
            "backend": {"jax_enabled": False},
            "config": {"nlat": nlat, "nlon": nlon, "steps": steps, "dt_seconds": dt},
            "metrics": {
                "total_wall_seconds": 1.0,
                "per_step_seconds": 0.5,
                "sim_days": 0.1,
                "memory_peak_mb": 12.0,
            },
        }

    monkeypatch.setattr(mod, "run_benchmark", fake_run)
    monkeypatch.setattr(sys, "argv", ["benchmark_jax", "--steps", "2", "--output-json", str(out)])
    mod.main()
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert "metrics" in doc
    assert "memory_peak_mb" in doc["metrics"]


def test_jax_compat_allow_cpu_flag_importable(monkeypatch):
    monkeypatch.setenv("QD_USE_JAX", "1")
    monkeypatch.setenv("QD_JAX_ALLOW_CPU", "1")
    mod = importlib.import_module("pygcm.jax_compat")
    mod = importlib.reload(mod)
    assert isinstance(mod.is_enabled(), bool)
