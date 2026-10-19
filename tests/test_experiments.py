"""Independent closed-form and real ngspice checks for generic experiments."""
import math
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError

from spiceharness import experiments
from spiceharness.experiments import (
    ACSweep,
    ExperimentEvaluator,
    ExperimentSource,
    ExperimentSpec,
    GeneralCircuitIR,
    GeneralComponent,
    Measurement,
    MetricBound,
    NoiseSourcePSD,
    TransientAnalysis,
    VoltageProbe,
)


def rc_fixture(requirements=()):
    circuit = GeneralCircuitIR(components=(
        GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1000),
        GeneralComponent(ref="C1", kind="C", nodes=("out", "0"), value=1e-6),
    ))
    spec = ExperimentSpec(
        sources=(ExperimentSource(ref="input", nodes=("in", "0"), dc=2,
                                  waveform="sine", amplitude=1, frequency_hz=1000),),
        loads=(GeneralComponent(ref="load", kind="R", nodes=("out", "0"), value=1e12),),
        probes=(VoltageProbe(name="output", positive="out"),),
        sweep=ACSweep(start_hz=1, stop_hz=1e5, points_per_decade=200),
        transient=TransientAnalysis(start_s=0.02, stop_s=0.03, step_s=1e-6),
        measurements=(
            Measurement(name="dc", operator="op_voltage", probe="output"),
            Measurement(name="gain", operator="ac_magnitude", probe="output", source="input",
                        frequency_hz=1000),
            Measurement(name="cutoff", operator="ac_cutoff_3db", probe="output", source="input",
                        reference_frequency_hz=1, band_start_hz=1, band_stop_hz=1e5),
            Measurement(name="mean", operator="transient_mean", probe="output"),
            Measurement(name="amplitude", operator="transient_fundamental", probe="output",
                        fundamental_hz=1000),
            Measurement(name="thd", operator="transient_thd", probe="output", fundamental_hz=1000),
            Measurement(name="noise", operator="integrated_noise_rms", probe="output"),
        ), requirements=requirements,
        noise_sources=(NoiseSourcePSD(source="input", density=1e-12, start_hz=1, stop_hz=1e5),),
    )
    return circuit, spec


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_rc_closed_form_and_transient(tmp_path):
    circuit, spec = rc_fixture()
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, spec)
    assert result.status == "ok", result.errors
    gain = 1 / math.sqrt(1 + (2 * math.pi * 1000 * 0.001) ** 2)
    assert result.metrics["gain"] == pytest.approx(gain, rel=2e-4)
    assert result.metrics["amplitude"] == pytest.approx(gain, rel=2e-3)
    assert result.metrics["dc"] == pytest.approx(2, abs=1e-6)
    assert result.metrics["mean"] == pytest.approx(2, abs=1e-5)
    assert result.metrics["thd"] < 1e-3
    assert result.metrics["cutoff"] == pytest.approx(1 / (2 * math.pi * 0.001), rel=2e-4)
    variance = 1e-12 * (math.atan(2 * math.pi * 0.001 * 1e5)
                        - math.atan(2 * math.pi * 0.001)) / (2 * math.pi * 0.001)
    assert result.metrics["noise"] == pytest.approx(math.sqrt(variance), rel=1e-4)
    assert result.score.feasible is None
    assert result.score.rank_key is None
    assert result.fixture_fingerprint == spec.fingerprint
    assert "metrics.json" in result.artifacts
    assert list(Path(result.artifact_dir).glob("*.curves.csv"))


def test_raw_controls_bands_and_resource_bounds():
    with pytest.raises(ValidationError):
        ExperimentSource(ref="x\n.control", nodes=("a", "0"))
    with pytest.raises(ValidationError):
        GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1,
                         parameters={"expression": 1})
    with pytest.raises(ValidationError):
        ACSweep(start_hz=1e-20, stop_hz=1e9, points_per_decade=1000)
    with pytest.raises(ValidationError):
        TransientAnalysis(stop_s=1, start_s=0, step_s=1e-9)
    with pytest.raises(ValidationError):
        ExperimentSource(ref="x", nodes=("a", "0"), kind="I", dc=20)
    with pytest.raises(ValueError):
        ExperimentEvaluator(timeout_s=21)
    _, spec = rc_fixture()
    with pytest.raises(ValidationError, match="both band edges"):
        ExperimentSpec.model_validate(spec.model_copy(update={"measurements": (
            Measurement(name="x", operator="ac_max", probe="output", source="input",
                        band_start_hz=1),)}).model_dump())


def test_protected_host_and_fixture_tamper(tmp_path):
    circuit, spec = rc_fixture()
    frozen = spec.model_copy(update={"baseline": circuit, "protected_refs": ("R1",),
                                     "redesign_ports": ("out", "0")})
    changed = circuit.model_copy(update={"components": (
        circuit.components[0].model_copy(update={"value": 2000}), circuit.components[1])})
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(changed, frozen)
    assert result.status == "invalid"
    assert "protected" in result.errors[0]
    assert not result.score.valid
    interior = circuit.model_copy(update={"connections": (("in", "0"),)})
    assert ExperimentEvaluator().evaluate(interior, frozen).status == "invalid"
    bad = circuit.model_copy(update={"components": (
        circuit.components[0].model_copy(update={"nodes": ("in", "out\n.control")}),)})
    assert ExperimentEvaluator().evaluate(bad, spec).status == "invalid"
    assert spec.fingerprint != spec.model_copy(update={"assumptions": ("different",)}).fingerprint


def protected_wire_fixture():
    circuit = GeneralCircuitIR(components=(
        GeneralComponent(ref="R1", kind="R", nodes=("in", "a"), value=1000),
        GeneralComponent(ref="R2", kind="R", nodes=("b", "0"), value=1000),
    ), connections=(("a", "b"),))
    spec = ExperimentSpec(
        sources=(ExperimentSource(ref="input", nodes=("in", "0"), dc=1),),
        probes=(VoltageProbe(name="output", positive="b"),),
        measurements=(Measurement(name="dc", operator="op_voltage", probe="output"),),
        baseline=circuit, protected_refs=("R1", "R2"), redesign_ports=("in", "0"),
    )
    return circuit, spec


def test_protected_wires_require_explicit_ports_and_validated_components():
    circuit, spec = protected_wire_fixture()
    evaluator = ExperimentEvaluator()
    ambiguous = spec.model_copy(update={"redesign_ports": ()})
    with pytest.raises(ValidationError, match="explicit redesign_ports"):
        evaluator.validate(circuit, ambiguous)
    # A model_copy can inject invalid nested IR despite the frozen model contract.
    invalid_part = circuit.components[0].model_copy(update={"value": -1})
    invalid = circuit.model_copy(update={"components": (invalid_part, circuit.components[1])})
    with pytest.raises(ValidationError):
        evaluator.validate(invalid, spec)


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_protected_internal_wire_is_preserved_and_cannot_be_bypassed(tmp_path):
    circuit, spec = protected_wire_fixture()
    evaluator = ExperimentEvaluator(artifact_root=tmp_path)
    # Endpoint order has no electrical meaning; the retained host wire remains legal.
    retained = circuit.model_copy(update={"connections": (("b", "a"),)})
    baseline = evaluator.evaluate(retained, spec)
    assert baseline.status == "ok", baseline.errors
    assert baseline.metrics["dc"] == pytest.approx(0.5)
    bypass = circuit.model_copy(update={"connections": (("a", "0"), ("b", "0"))})
    rejected = evaluator.evaluate(bypass, spec)
    assert rejected.status == "invalid"
    assert "protected baseline wire" in rejected.errors[0]
    assert rejected.metrics == {}
    # Keeping the host wire cannot authorize adding an interior short either.
    shortcut = circuit.model_copy(update={"connections": (("a", "b"), ("b", "0"))})
    rejected = evaluator.evaluate(shortcut, spec)
    assert rejected.status == "invalid"
    assert "protected host interior" in rejected.errors[0]


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_declared_filter_port_wire_can_be_replaced(tmp_path):
    circuit, spec = protected_wire_fixture()
    exposed = spec.model_copy(update={"redesign_ports": ("a", "b", "0")})
    filtered = GeneralCircuitIR(components=(*circuit.components,
        GeneralComponent(ref="RF", kind="R", nodes=("a", "b"), value=1000)))
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(filtered, exposed)
    assert result.status == "ok", result.errors
    assert result.metrics["dc"] == pytest.approx(1 / 3)


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
@pytest.mark.parametrize("expired_during", ("parse", "archive"))
def test_deadline_stops_curve_exports_after_expensive_work(tmp_path, monkeypatch, expired_during):
    circuit, spec = protected_wire_fixture()
    clock = [0.0]
    monkeypatch.setattr(experiments.time, "monotonic", lambda: clock[0])
    original_read = experiments._read_raw
    original_archive = experiments.np.savez_compressed
    exports = []

    def read_raw(path):
        result = original_read(path)
        if expired_during == "parse":
            clock[0] = 21.0
        return result

    def archive(*args, **kwargs):
        exports.append("archive")
        original_archive(*args, **kwargs)
        if expired_during == "archive":
            clock[0] = 21.0

    def csv(*args, **kwargs):
        pytest.fail("CSV export must not run after the total deadline")

    monkeypatch.setattr(experiments, "_read_raw", read_raw)
    monkeypatch.setattr(experiments.np, "savez_compressed", archive)
    monkeypatch.setattr(experiments.np, "savetxt", csv)
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, spec)
    assert result.status == "timeout", result.errors
    assert not result.score.valid
    assert result.metrics == {}
    assert exports == ([] if expired_during == "parse" else ["archive"])


def test_intrinsic_loss_counts_one_explicit_resistor_counts(tmp_path):
    circuit, spec = rc_fixture()
    circuit = circuit.model_copy(update={"components": (
        circuit.components[0], circuit.components[1].model_copy(update={"series_resistance_ohm": 0.2}))})
    deck = ExperimentEvaluator().generate_netlist(circuit, spec, "op")
    assert "REXLOSS1" in deck
    assert "exinternal1" in deck
    result = ExperimentEvaluator(artifact_root=tmp_path, executable="not-a-real-ngspice").evaluate(circuit, spec)
    assert result.score.package_count == 2
    assert result.status == "unavailable"
    explicit = circuit.model_copy(update={"components": (*circuit.components,
        GeneralComponent(ref="Rextra", kind="R", nodes=("out", "0"), value=1e9, role="parasitic"))})
    result = ExperimentEvaluator(executable="not-a-real-ngspice").evaluate(explicit, spec)
    assert result.score.package_count == 3


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_passing_constraints_dominate_parts_and_missing_crossing_fails(tmp_path):
    circuit, spec = rc_fixture((MetricBound(metric="gain", maximum=0.2),))
    evaluator = ExperimentEvaluator(artifact_root=tmp_path)
    passing = evaluator.evaluate(circuit, spec)
    assert passing.score.feasible
    # A larger capacitor breaches the declared frequency crossing band only when
    # the crossing is absent; a resistor-only circuit never fabricates a cutoff.
    simpler = GeneralCircuitIR(components=(circuit.components[0],))
    failed = evaluator.evaluate(simpler, spec)
    assert failed.status == "invalid"
    assert not failed.score.valid
    assert passing.score.rank_key < failed.score.rank_key
    analysis = spec.model_copy(update={"measurements": (spec.measurements[1],),
                                       "noise_sources": (), "transient": None})
    failing = evaluator.evaluate(simpler, analysis)
    assert failing.status == "ok"
    assert failing.score.feasible is False
    assert passing.score.package_count > failing.score.package_count
    assert passing.score.rank_key < failing.score.rank_key


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_ideal_topology_connections_are_zero_parts(tmp_path):
    circuit = GeneralCircuitIR(connections=(("in", "out"),), components=())
    spec = ExperimentSpec(sources=(ExperimentSource(ref="input", nodes=("in", "0"), dc=2),),
                          loads=(GeneralComponent(ref="load", kind="R", nodes=("out", "0"), value=1e4),),
                          probes=(VoltageProbe(name="out", positive="out"),),
                          measurements=(Measurement(name="out", operator="op_voltage", probe="out"),))
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, spec)
    assert result.status == "ok", result.errors
    assert result.metrics["out"] == pytest.approx(2)
    assert result.score.package_count == 0
    with pytest.raises(ValidationError):
        GeneralCircuitIR(components=(), connections=(("a", "b"), ("b", "a")))


@pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
def test_real_741_offset_gain_and_independent_upstream_rail_injection(tmp_path):
    if not Path("src/spiceharness/data/models/LM741.lib").exists():
        pytest.skip("run scripts/fetch_models.py for real manufacturer model")
    circuit = GeneralCircuitIR(components=(
        GeneralComponent(ref="U1", kind="OPAMP", nodes=("in", "inv", "vcc", "vee", "out"), model="LM741"),
        GeneralComponent(ref="RF", kind="R", nodes=("out", "inv"), value=2000),
        GeneralComponent(ref="RG", kind="R", nodes=("inv", "0"), value=1000),
        GeneralComponent(ref="CP", kind="C", nodes=("vcc", "0"), value=10e-6, series_resistance_ohm=0.1),
        GeneralComponent(ref="CN", kind="C", nodes=("vee", "0"), value=10e-6, series_resistance_ohm=0.1),
    ))
    spec = ExperimentSpec(
        sources=(ExperimentSource(ref="signal", nodes=("in", "0"), dc=2,
                                  waveform="sine", amplitude=1, frequency_hz=1000),
                 ExperimentSource(ref="positive", nodes=("psrc", "0"), dc=12),
                 ExperimentSource(ref="negative", nodes=("nsrc", "0"), dc=-12)),
        loads=(GeneralComponent(ref="RP", kind="R", nodes=("psrc", "vcc"), value=10),
               GeneralComponent(ref="RN", kind="R", nodes=("nsrc", "vee"), value=10),
               GeneralComponent(ref="load", kind="R", nodes=("out", "0"), value=1e5)),
        probes=(VoltageProbe(name="output", positive="out"), VoltageProbe(name="prail", positive="vcc"),
                VoltageProbe(name="nrail", positive="vee")), allowed_models=("LM741",),
        sweep=ACSweep(start_hz=10, stop_hz=1e5),
        transient=TransientAnalysis(start_s=0.01, stop_s=0.02, step_s=1e-6),
        measurements=(Measurement(name="offset", operator="op_voltage", probe="output"),
                      Measurement(name="current", operator="op_current", source="positive"),
                      Measurement(name="gain", operator="ac_magnitude", source="signal", probe="output",
                                  frequency_hz=1000),
                      Measurement(name="amplitude", operator="transient_fundamental", probe="output",
                                  fundamental_hz=1000),
                      Measurement(name="positive_transfer", operator="ac_magnitude", source="positive",
                                  probe="prail", frequency_hz=10000),
                      Measurement(name="negative_transfer", operator="ac_magnitude", source="negative",
                                  probe="nrail", frequency_hz=10000),
                      Measurement(name="rail_noise", operator="integrated_noise_rms", probe="output")),
        noise_sources=(NoiseSourcePSD(source="positive", density=1e-10, start_hz=10, stop_hz=1e5),
                       NoiseSourcePSD(source="negative", density=1e-10, start_hz=10, stop_hz=1e5)),
    )
    result = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, spec)
    assert result.status == "ok", result.errors
    assert result.metrics["offset"] == pytest.approx(6, abs=0.03)
    assert result.metrics["gain"] == pytest.approx(3, rel=0.02)
    assert result.metrics["amplitude"] == pytest.approx(3, rel=0.02)
    assert result.metrics["current"] < 0
    assert 0 < result.metrics["positive_transfer"] < 0.3
    assert 0 < result.metrics["negative_transfer"] < 0.3
    assert math.isfinite(result.metrics["rail_noise"])
    assert result.model_checksums["LM741"]
    # A valid upstream supply cannot excuse violating the actual filtered-rail envelope.
    low_rails = spec.model_copy(update={"loads": (
        spec.loads[0].model_copy(update={"value": 10000}), spec.loads[1], spec.loads[2])})
    failed = ExperimentEvaluator(artifact_root=tmp_path).evaluate(circuit, low_rails)
    assert failed.status == "invalid"
    assert "span" in failed.errors[0]
