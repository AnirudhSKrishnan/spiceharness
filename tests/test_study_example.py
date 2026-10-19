"""The committed recipes are typed inputs; the ordinary RC has an independent oracle."""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "experiments"


def test_example_specs_and_literal_lab_fixture():
    for path in EXAMPLES.rglob("*spec.json"):
        ExperimentSpec.model_validate_json(path.read_text())
    for path in EXAMPLES.rglob("*.json"):
        if not path.name.endswith("spec.json"):
            GeneralCircuitIR.model_validate_json(path.read_text())
    lab = EXAMPLES / "lab4-supply"
    spec = ExperimentSpec.model_validate_json((lab / "spec.json").read_text())
    original = GeneralCircuitIR.model_validate_json((lab / "original_t.json").read_text())
    parts = {part.ref: part for part in original.components}
    assert parts["PL1"].value == 7.95e-3
    assert parts["PL2"].value == 7.95e-6
    assert parts["PC1"].value == 6.4e-9
    assert len(original.components) == 9
    assert len(spec.baseline.components) == 3  # No silently added bypass capacitors.
    assert parts["RF"].value == 2000 and parts["RG"].value == 1000
    for noise in spec.noise_sources:
        assert noise.density * (noise.stop_hz - noise.start_hz) == pytest.approx(0.01 ** 2)
    assert spec.protected_refs == ("U1", "RF", "RG")


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_loaded_rc_saved_curve_matches_complex_oracle_and_direct_mode(tmp_path):
    module_spec = importlib.util.spec_from_file_location("study", ROOT / "scripts" / "run_lab4_study.py")
    study = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(study)
    manager = ExperimentManager(tmp_path)
    rc = EXAMPLES / "loaded-rc"
    record = study.run_comparison(manager, rc / "spec.json", [rc / "reference.json"])
    assert record["execution_mode"] == "direct"
    assert record["offline"] is True
    assert record["usage"] == []
    assert record["result"]["verified"] and record["result"]["feasible"]
    names, values = study.curve(record["candidates"][0], "ac", "signal")
    expected = study.loaded_rc_oracle(values[:, 0].real)
    measured = study.vector(names, values, "out")
    assert np.max(abs(measured - expected)) < 1e-7
    # Public evidence retains the frozen file and independent fresh verification.
    fixture = Path(record["result"]["artifacts"]["fixture"])
    assert json.loads(fixture.read_text()) == record["spec"]
    assert record["candidates"][1]["purpose"] == "verification"
    evaluator = ExperimentEvaluator()
    evaluator.validate(study.load_circuit(rc / "reference.json"), study.load_spec(rc / "spec.json"))
