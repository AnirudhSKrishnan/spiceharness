"""Regression tests for general-experiment validation and session edge cases."""
import shutil

import numpy as np
import pytest
from pydantic import ValidationError

from spiceharness import experiment_sessions, experiments
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import (
    ExperimentEvaluator,
    ExperimentSource,
    ExperimentSpec,
    GeneralCircuitIR,
    GeneralComponent,
    Measurement,
    MetricBound,
    VoltageProbe,
)

needs_ngspice = pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")


def divider(requirements=()):
    circuit = GeneralCircuitIR(components=(
        GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1000),))
    spec = ExperimentSpec(
        sources=(ExperimentSource(ref="v", nodes=("in", "0"), dc=1),),
        loads=(GeneralComponent(ref="RL", kind="R", nodes=("out", "0"), value=1000),),
        probes=(VoltageProbe(name="o", positive="out"),),
        measurements=(Measurement(name="dc", operator="op_voltage", probe="o"),),
        requirements=requirements)
    return circuit, spec


def test_gnd_alias_is_rejected_everywhere():
    for build in (
        lambda: GeneralComponent(ref="R9", kind="R", nodes=("out", "GND"), value=1),
        lambda: ExperimentSource(ref="v", nodes=("gnd", "in")),
        lambda: VoltageProbe(name="o", positive="gnd"),
        lambda: GeneralCircuitIR(components=(), connections=(("gnd", "x"),)),
    ):
        with pytest.raises(ValidationError, match="gnd"):
            build()


@pytest.mark.parametrize("extra,wires", [
    ((), (("in", "0"),)),
    ((GeneralComponent(ref="L1", kind="L", nodes=("in", "0"), value=1e-3),), ()),
    ((GeneralComponent(ref="L1", kind="L", nodes=("out", "0"), value=1e-3),
      GeneralComponent(ref="L2", kind="L", nodes=("out", "0"), value=1e-3)), ()),
])
def test_short_circuit_loops_are_rejected_before_simulating(tmp_path, extra, wires):
    circuit, spec = divider()
    bad = circuit.model_copy(update={"components": circuit.components + extra, "connections": wires})
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(bad, spec)
    assert result.status == "invalid" and "short-circuit loop" in result.errors[0]
    assert result.artifact_dir is None
    manager = ExperimentManager(tmp_path / "ws")
    run = manager.start_experiment(spec)["run_id"]
    with pytest.raises(ValueError, match="short-circuit loop"):
        manager.evaluate_experiment(run, bad)
    assert manager.get_experiment(run)["evaluation_count"] == 0


def test_lossy_inductor_and_load_resistor_are_not_loops(tmp_path):
    circuit, spec = divider()
    lossy = GeneralComponent(ref="L1", kind="L", nodes=("in", "0"), value=1e-3, series_resistance_ohm=1)
    ExperimentEvaluator().validate(circuit.model_copy(update={"components": circuit.components + (lossy,)}), spec)
    ExperimentEvaluator().validate(circuit.model_copy(update={"connections": (("out", "0"),)}), spec)


def test_parallel_voltage_sources_are_rejected_in_spec():
    _, spec = divider()
    second = ExperimentSource(ref="w", nodes=("0", "in"), dc=2)
    with pytest.raises(ValidationError, match="short-circuit loop"):
        ExperimentSpec.model_validate({**spec.model_dump(), "sources": [
            *spec.model_dump()["sources"], second.model_dump()]})


def test_candidate_cannot_reuse_fixture_load_reference():
    circuit, spec = divider()
    clash = GeneralComponent(ref="rl", kind="R", nodes=("out", "0"), value=5)
    with pytest.raises(ValueError, match="duplicate fixture load"):
        ExperimentEvaluator().validate(
            circuit.model_copy(update={"components": circuit.components + (clash,)}), spec)


@needs_ngspice
def test_nonfinite_simulator_samples_never_pass(tmp_path, monkeypatch):
    circuit, spec = divider((MetricBound(metric="dc", minimum=0.0),))
    real = experiments._read_raw

    def poisoned(path):
        names, data = real(path)
        data = data.copy()
        data[-1, -1] = np.nan
        return names, data

    monkeypatch.setattr(experiments, "_read_raw", poisoned)
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, spec)
    assert result.status == "invalid" and "non-finite" in result.errors[0]
    assert result.score.valid is False and result.score.feasible is False


@needs_ngspice
def test_completed_run_rejects_unknown_or_different_candidate(tmp_path):
    circuit, spec = divider()
    manager = ExperimentManager(tmp_path)
    run = manager.start_experiment(spec)["run_id"]
    first = manager.evaluate_experiment(run, circuit)["candidate_id"]
    other = circuit.model_copy(update={"components": (
        circuit.components[0].model_copy(update={"value": 2000}),)})
    second = manager.evaluate_experiment(run, other)["candidate_id"]
    done = manager.finalize_experiment(run, second)
    assert done["status"] == "completed"
    with pytest.raises(ValueError, match="not a saved evaluation"):
        manager.finalize_experiment(run, "nope")
    with pytest.raises(ValueError, match="different candidate"):
        manager.finalize_experiment(run, first)
    assert manager.finalize_experiment(run, second)["result"]["best_candidate_id"] == second
    assert manager.finalize_experiment(run)["status"] == "completed"


def test_search_timeout_leaves_a_full_verification_allowance(tmp_path, monkeypatch):
    circuit, spec = divider()
    manager = ExperimentManager(tmp_path)
    run = manager.start_experiment(spec)["run_id"]
    record = manager._read(run)
    record["elapsed_seconds"] = 159.0
    manager._save(record)
    seen = []

    class Spy:
        def __init__(self, artifact_root=None, timeout_s=20, **_):
            seen.append(timeout_s)

        def validate(self, *_):
            return {}

        def evaluate(self, *_):
            raise RuntimeError("stop")

    monkeypatch.setattr(experiments, "ExperimentEvaluator", Spy)
    with pytest.raises(RuntimeError):
        manager.evaluate_experiment(run, circuit)
    assert seen == [pytest.approx(180 - 159 - experiment_sessions.MAX_OPERATION_SECONDS)]
