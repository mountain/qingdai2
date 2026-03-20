import numpy as np
import pytest

from pygcm.world.diagnostics import (
    make_world_diagnostics_document,
    validate_world_diagnostics,
    world_diagnostics_from_dict,
    world_diagnostics_to_jsonable,
)


def test_validate_world_diagnostics_accepts_valid_doc():
    step = {
        "step": 2,
        "energy": {"toa_net": 1.0, "sfc_net": 2.0, "atm_net": 3.0},
        "water": {"evap_mean": 0.1, "precip_mean": 0.2, "runoff_mean": 0.3},
        "hydrology": {"runoff_mean": 0.3},
        "routing": {"steps": 2.0},
        "ecology": {"subdaily_calls": 2.0},
    }
    doc = {
        "schema_version": 1,
        "steps": 2,
        "summary": {
            "energy_mean_abs_toa": 1.0,
            "energy_mean_abs_sfc": 2.0,
            "energy_mean_abs_atm": 3.0,
            "water_mean_abs_residual": 4.0,
        },
        "last_step": step,
        "samples": [step],
    }
    validate_world_diagnostics(doc, expected_schema_version=1)


def test_validate_world_diagnostics_rejects_schema_mismatch():
    step = {
        "step": 1,
        "energy": {"toa_net": 1.0, "sfc_net": 2.0, "atm_net": 3.0},
        "water": {"evap_mean": 0.1, "precip_mean": 0.2, "runoff_mean": 0.3},
        "hydrology": {},
        "routing": {},
        "ecology": {},
    }
    doc = {
        "schema_version": 2,
        "steps": 1,
        "summary": {
            "energy_mean_abs_toa": 1.0,
            "energy_mean_abs_sfc": 2.0,
            "energy_mean_abs_atm": 3.0,
            "water_mean_abs_residual": 4.0,
        },
        "last_step": step,
        "samples": [step],
    }
    with pytest.raises(ValueError):
        validate_world_diagnostics(doc, expected_schema_version=1)


def test_world_diagnostics_strict_rejects_unknown_fields():
    step = {
        "step": 1,
        "energy": {"toa_net": 1.0, "sfc_net": 2.0, "atm_net": 3.0},
        "water": {"evap_mean": 0.1, "precip_mean": 0.2, "runoff_mean": 0.3},
        "hydrology": {},
        "routing": {},
        "ecology": {},
        "extra": 1,
    }
    doc = {
        "schema_version": 1,
        "steps": 1,
        "summary": {
            "energy_mean_abs_toa": 1.0,
            "energy_mean_abs_sfc": 2.0,
            "energy_mean_abs_atm": 3.0,
            "water_mean_abs_residual": 4.0,
        },
        "last_step": step,
        "samples": [step],
    }
    with pytest.raises(ValueError):
        world_diagnostics_from_dict(doc, expected_schema_version=1, strict=True)


def test_world_diagnostics_backward_compat_fills_missing_fields():
    legacy = {
        "schema_version": 1,
        "steps": 1,
        "summary": {"energy_mean_abs_toa": 1.0},
    }
    out = world_diagnostics_from_dict(
        legacy,
        expected_schema_version=1,
        strict=False,
        allow_backward_compat=True,
    )
    assert out.steps == 1
    assert out.summary.energy_mean_abs_sfc == 0.0


def test_make_world_diagnostics_document_returns_typed_contract():
    step = {
        "step": 1,
        "energy": {"toa_net": 1.0, "sfc_net": 2.0, "atm_net": 3.0},
        "water": {"evap_mean": 0.1, "precip_mean": 0.2, "runoff_mean": 0.3},
        "hydrology": {},
        "routing": {},
        "ecology": {},
    }
    doc = make_world_diagnostics_document(
        schema_version=1,
        steps=1,
        summary={
            "energy_mean_abs_toa": 1.0,
            "energy_mean_abs_sfc": 2.0,
            "energy_mean_abs_atm": 3.0,
            "water_mean_abs_residual": 4.0,
        },
        last_step=step,
        samples=[step],
    )
    assert doc.to_dict()["schema_version"] == 1


def test_world_diagnostics_to_jsonable_handles_ndarray():
    doc = {"x": np.array([[1.0, 2.0]])}
    out = world_diagnostics_to_jsonable(doc)
    assert out["x"] == [[1.0, 2.0]]
