import json

from pygcm.world import QingdaiWorld


def test_world_writes_metadata_json(monkeypatch, tmp_path):
    out = tmp_path / "oo-meta.json"
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_MLD_M", "60")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_METADATA_JSON", str(out))
    world = QingdaiWorld.create_default()
    world.run(n_steps=2)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["grid"]["n_lat"] == 12
    assert "params" in doc
    assert doc["params"]["parameter_groups"]["physics"]["mld_m"] == 60.0


def test_world_uses_configured_metadata_path(monkeypatch, tmp_path):
    out1 = tmp_path / "oo-meta-a.json"
    out2 = tmp_path / "oo-meta-b.json"
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_METADATA_JSON", str(out1))
    world = QingdaiWorld.create_default()
    monkeypatch.setenv("QD_OO_METADATA_JSON", str(out2))
    world.run(n_steps=1)
    assert out1.exists()
    assert not out2.exists()


def test_world_respects_metadata_enable_switch(monkeypatch, tmp_path):
    out = tmp_path / "oo-meta-disabled.json"
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_METADATA_ENABLE", "0")
    monkeypatch.setenv("QD_OO_METADATA_JSON", str(out))
    world = QingdaiWorld.create_default()
    world.run(n_steps=1)
    assert not out.exists()


def test_world_writes_world_diagnostics_json(monkeypatch, tmp_path):
    out = tmp_path / "oo-world-diag.json"
    monkeypatch.setenv("QD_N_LAT", "12")
    monkeypatch.setenv("QD_N_LON", "24")
    monkeypatch.setenv("QD_DT_SECONDS", "300")
    monkeypatch.setenv("QD_ECO_ENABLE", "0")
    monkeypatch.setenv("QD_OO_CONFIG_DIAG", "0")
    monkeypatch.setenv("QD_OO_DIAG_EVERY", "1")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_ENABLE", "1")
    monkeypatch.setenv("QD_OO_WORLD_DIAG_JSON", str(out))
    monkeypatch.setenv("QD_OO_WORLD_DIAG_SCHEMA_VERSION", "1")
    world = QingdaiWorld.create_default()
    world.run(n_steps=2)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["steps"] == 2
    assert "summary" in doc
    assert "samples" in doc
