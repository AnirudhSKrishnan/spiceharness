"""Pinned LM741 provenance, operating limits, and real simulator integration."""

import hashlib
import importlib.util
import io
import json
import shutil
import zipfile
from pathlib import Path

import pytest

from spiceharness.domain import DesignSpec
from spiceharness.scoring import part_counts
from spiceharness.simulator import MODEL_DIR, NgSpiceSimulator
from spiceharness.templates import make_candidate


def lm741_record():
    return json.loads((MODEL_DIR / "catalog.json").read_text())["LM741"]


def lm741_spec(**changes):
    values = {
        "topology": "buffer",
        "opamp_model": "LM741",
        "supply_positive_v": 12,
        "supply_negative_v": -12,
        "input_amplitude_v": 0.01,
        "input_frequency_hz": 1000,
        "load_ohm": 100000,
    }
    values.update(changes)
    return DesignSpec(**values)


def test_lm741_provenance_does_not_claim_validated_psrr():
    record = lm741_record()
    assert record["source_url"] == "https://www.ti.com/lit/zip/snom211"
    assert record["archive_entry"] == "lm741.lib"
    assert record["channels"] == 1
    assert record["model_psrr_validated"] is False
    assert record["supply_feedthrough_characterization"] == "unvalidated"


@pytest.mark.parametrize(
    "changes",
    [
        {"supply_positive_v": 9, "supply_negative_v": -9},
        {"supply_positive_v": 23, "supply_negative_v": -23},
        {"temperature_c": -1},
        {"temperature_c": 71},
    ],
)
def test_lm741_conservative_operating_envelope(changes):
    with pytest.raises(ValueError, match="operating condition exceeds catalog"):
        NgSpiceSimulator._check_operating_envelope(lm741_spec(**changes), {"LM741": lm741_record()})


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice is required")
@pytest.mark.skipif(not (MODEL_DIR / "LM741.lib").is_file(), reason="fetch the original LM741 model")
def test_real_pinned_lm741_op_ac_transient(tmp_path):
    circuit = make_candidate(lm741_spec())
    assert sum(part.kind == "OPAMP" for part in circuit.components) == 1
    result = NgSpiceSimulator(artifact_root=tmp_path).evaluate(circuit)
    assert result.status == "ok", result.errors
    assert result.model_checksums == {"LM741": lm741_record()["sha256"]}
    assert all(result.analyses[key]["points"] > 0 for key in ("op", "ac", "tran"))
    assert result.metrics["gain_at_input_frequency"] == pytest.approx(1, rel=0.01)
    assert result.metrics["transient_gain"] == pytest.approx(1, rel=0.01)
    assert 1e-3 < result.metrics["supply_current_a"] < 3e-3
    # A working signal response is not a validation of physical rail rejection.
    assert lm741_record()["model_psrr_validated"] is False


@pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice is required")
@pytest.mark.skipif(not (MODEL_DIR / "LM741.lib").is_file(), reason="fetch the original LM741 model")
def test_two_lm741_stages_use_two_physical_packages(tmp_path):
    circuit = make_candidate(lm741_spec(topology="active_lowpass_2", cutoff_hz=1000))
    amplifiers = [part for part in circuit.components if part.kind == "OPAMP"]
    assert len(amplifiers) == 2
    assert len({part.package for part in amplifiers}) == 2
    assert part_counts(circuit)[0] == len(circuit.components)
    result = NgSpiceSimulator(artifact_root=tmp_path).evaluate(circuit)
    assert result.status == "ok", result.errors
    assert result.metrics["cutoff_hz"] == pytest.approx(1000, rel=0.15)


def test_download_rejects_changed_archive_and_model_without_replacing_target(tmp_path, monkeypatch):
    module_spec = importlib.util.spec_from_file_location(
        "fetch_models", Path(__file__).resolve().parents[1] / "scripts" / "fetch_models.py"
    )
    fetcher = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(fetcher)
    original = b"original verified vendor bytes"
    target = tmp_path / "device.lib"
    target.write_bytes(original)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("device.lib", b"changed bytes")
    archive_bytes = payload.getvalue()
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(archive_bytes))
    record = {
        "path": "device.lib",
        "source_url": "https://www.ti.com/lit/zip/snom211",
        "archive_entry": "device.lib",
        "archive_sha256": "0" * 64,
        "sha256": hashlib.sha256(original).hexdigest(),
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch for device archive"):
        fetcher.fetch("device", record, tmp_path, force=True)
    assert target.read_bytes() == original
    record["archive_sha256"] = hashlib.sha256(archive_bytes).hexdigest()
    with pytest.raises(ValueError, match="SHA-256 mismatch for device:"):
        fetcher.fetch("device", record, tmp_path, force=True)
    assert target.read_bytes() == original
