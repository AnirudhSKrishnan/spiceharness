"""Electrical and adversarial checks for the trusted evaluation boundary."""

from __future__ import annotations

import json
import math
import shutil

import pytest
from pydantic import ValidationError

from spiceharness.domain import Circuit, Component, DesignSpec, SimulationResult
from spiceharness.render import export_bom, render_schematic
from spiceharness.scoring import part_counts, score_result
from spiceharness.simulator import (
    MODEL_DIR,
    NgSpiceSimulator,
    expected_corner_labels,
    smoke_models,
    validate_circuit,
)
from spiceharness.templates import E12, E24, generate_candidates, make_candidate, standard_value


def spec(**changes) -> DesignSpec:
    values = {
        "topology": "passive_lowpass",
        "cutoff_hz": 1000,
        "gain": 1,
        "supply_positive_v": 12,
        "supply_negative_v": -12,
        "input_amplitude_v": 0.05,
        "input_frequency_hz": 100,
        "load_ohm": 100000,
    }
    values.update(changes)
    return DesignSpec(**values)


def measured(s: DesignSpec, **updates) -> SimulationResult:
    metrics = {
        "gain": 1.0,
        "gain_at_input_frequency": 1.0,
        "phase_deg": 0.0,
        "cutoff_hz": 1000.0,
        "thd": 0.001,
        "gain_compression": 0.001,
        "output_offset_v": 0.0,
        "supply_current_a": 0.001,
        "output_swing_v": 0.05,
        "transient_gain": 1.0,
    }
    metrics.update(updates)
    return SimulationResult(status="ok", metrics=metrics, spec_fingerprint=s.fingerprint)


def test_spec_requires_explicit_operating_conditions_and_valid_gain():
    with pytest.raises(ValidationError):
        DesignSpec(topology="buffer")
    with pytest.raises(ValidationError):
        spec(input_amplitude_v=float("nan"))
    with pytest.raises(ValidationError):
        spec(topology="inverting_amplifier", gain=2)
    with pytest.raises(ValidationError):
        spec(topology="unbounded custom expression")


def test_candidates_cannot_change_fixture_or_inject_spice():
    s = spec()
    c = make_candidate(s)
    with pytest.raises(ValidationError):
        Component(ref="R1\n.control", kind="R", nodes=("in", "out"), value=1)
    with pytest.raises(ValidationError):
        Component(ref="R1", kind="R", nodes=("in", "out"), value=1, parameters={"m": 100})
    with pytest.raises(ValidationError):
        Component(ref="V1", kind="V", nodes=("in", "out"), value=1)
    changed = c.model_copy(update={"spec": s.model_copy(update={"load_ohm": 1e12})})
    assert any("operating conditions" in e for e in validate_circuit(changed, s))
    assert c.spec.fingerprint == s.fingerprint
    assert not validate_circuit(c, s)


def test_package_count_cannot_hide_resistors_or_unrelated_devices():
    with pytest.raises(ValidationError):
        Component(ref="R1", kind="R", nodes=("in", "out"), value=1, package="U1")
    s = spec(topology="active_lowpass_2")
    lm = make_candidate(s)
    opa = make_candidate(s.model_copy(update={"opamp_model": "OPA1641"}))
    assert part_counts(opa)[0] == part_counts(lm)[0] + 1
    assert len(export_bom(lm)) == part_counts(lm)[0]
    assert sum(row["role"] == "decoupling" for row in export_bom(lm)) == 2


def test_floating_and_disconnected_load_rejected():
    s = spec()
    c = Circuit(
        topology=s.topology,
        spec=s,
        components=(
            Component(ref="R1", kind="R", nodes=("in", "0"), value=1000),
            Component(ref="R2", kind="R", nodes=("out", "0"), value=1000),
        ),
    )
    assert any("signal path" in e for e in validate_circuit(c, s))
    assert not score_result(s, c, measured(s)).valid


def test_missing_measurement_or_spec_identity_never_passes():
    s = spec()
    c = make_candidate(s)
    r = measured(s)
    missing = r.model_copy(update={"metrics": {"gain": 1.0}})
    assert not score_result(s, c, missing).valid
    assert not score_result(s, c, r.model_copy(update={"spec_fingerprint": None})).valid
    # A bypassed model validator must still be caught by the evaluator.
    bad = r.model_copy(update={"metrics": {**r.metrics, "gain": math.inf}})
    assert not score_result(s, c, bad).valid
    for state in ("timeout", "failed", "invalid", "unavailable"):
        assert not score_result(s, c, r.model_copy(update={"status": state})).feasible


def test_feasibility_always_dominates_component_count():
    s = spec(topology="lowpass")
    choices = generate_candidates(s)
    passive = next(c for c in choices if c.topology == "passive_lowpass")
    active = next(c for c in choices if c.topology == "active_lowpass_1")
    passing = score_result(s, active, measured(s))
    failing = score_result(s, passive, measured(s, gain=0.8))
    assert passing.feasible and not failing.feasible
    assert passing.rank_key < failing.rank_key
    assert passing.points >= 1000 > failing.points
    assert score_result(s, passive, measured(s)).rank_key < passing.rank_key


def test_polarity_and_clipping_are_mandatory_measurements():
    s = spec(topology="inverting_amplifier", gain=-2, cutoff_hz=None)
    c = make_candidate(s)
    wrong = score_result(s, c, measured(s, gain=2, phase_deg=0))
    assert not wrong.feasible and wrong.violations["nominal.gain_polarity"] > 0
    clipped = score_result(s, c, measured(s, gain=2, phase_deg=180, gain_compression=0.2))
    assert not clipped.feasible


def test_tolerance_corners_cannot_be_omitted():
    s = spec(corner_policy="tolerance")
    c = make_candidate(s)
    assert not score_result(s, c, measured(s)).valid


def test_unrealizable_or_unmeasured_spec_fields_are_rejected_up_front():
    # An "amplifier" family with 0 < gain < 1 used to pass validation and then crash
    # candidate generation with a cryptic "component value must be finite and positive".
    with pytest.raises(ValidationError, match="positive gain below one"):
        spec(topology="amplifier", gain=0.5, cutoff_hz=None)
    assert spec(topology="amplifier", gain=-0.5, cutoff_hz=None).gain == -0.5
    # Absurd gains produced a >1e15 ohm feedback resistor and a pydantic error at generation.
    with pytest.raises(ValidationError, match="1e9"):
        spec(topology="noninverting_amplifier", gain=1e13, cutoff_hz=None)
    with pytest.raises(ValidationError, match="1e15 ohm"):
        spec(topology="inverting_amplifier", gain=-1e7, source_resistance_ohm=1e9, cutoff_hz=None)
    assert generate_candidates(spec(topology="inverting_amplifier", gain=-1e9, cutoff_hz=None))
    # cutoff_hz is only ever measured for filters; on an amplifier every candidate was
    # scored invalid with "missing mandatory measurements: cutoff_hz".
    with pytest.raises(ValidationError, match="only for filter"):
        spec(topology="noninverting_amplifier", gain=10, cutoff_hz=50000)


def test_transistor_variants_span_both_emitter_voltages_in_every_family_slot():
    # In a mixed family the transistor stage sits at variants 6..11; the emitter-voltage
    # choice used the raw variant index so only half of its design points were ever reached.
    explicit = generate_candidates(spec(topology="common_emitter", gain=-5, cutoff_hz=None))
    mixed = [c for c in generate_candidates(spec(topology="amplifier", gain=-5, cutoff_hz=None))
             if c.topology == "common_emitter"]
    assert len(explicit) == len(mixed) == 6
    key = lambda c: tuple((p.ref, p.value) for p in c.components)
    assert {key(c) for c in mixed} == {key(c) for c in explicit}


def test_standard_values_and_interleaved_families():
    assert standard_value(980) == 1000
    assert standard_value(140e-9, E12) == pytest.approx(150e-9)
    s = spec(topology="lowpass", max_candidates=3)
    assert {c.topology for c in generate_candidates(s)} == {
        "passive_lowpass",
        "active_lowpass_1",
        "active_lowpass_2",
    }
    for c in generate_candidates(s):
        for p in c.components:
            if p.kind in ("R", "C"):
                series = E24 if p.kind == "R" else E12
                normalized = p.value / 10 ** math.floor(math.log10(p.value))
                assert any(math.isclose(normalized, x) for x in series)


def test_renderer_uses_actual_connections_and_counts_all_parts():
    c = make_candidate(spec(topology="noninverting_amplifier", gain=2, cutoff_hz=None))
    drawing = render_schematic(c)
    assert "Exact connection table:" in drawing
    assert all(p.ref in drawing for p in c.components)
    assert "decoupling" in drawing
    assert any(p.role == "unused_channel" for p in c.components)


requires_ngspice = pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice is not installed")
requires_models = pytest.mark.skipif(
    not (MODEL_DIR / "LM358.lib").exists(), reason="vendor models not fetched"
)


@requires_ngspice
def test_real_rc_measurements_and_reproducible_fixture(tmp_path):
    s = spec()
    c = make_candidate(s)
    runner = NgSpiceSimulator(artifact_root=tmp_path)
    r = runner.evaluate(c)
    assert r.status == "ok", r.errors
    assert r.metrics["cutoff_hz"] == pytest.approx(1000, rel=0.1)
    assert r.metrics["gain"] == pytest.approx(1, rel=0.02)
    assert score_result(s, c, r).feasible
    assert r.spec_fingerprint == s.fingerprint and r.fixture_fingerprint
    assert all(
        (tmp_path / r.artifact_dir / name).exists()
        for name in ("op.raw", "ac.raw", "tran.raw", "result.json")
    )
    again = runner.evaluate(c)
    assert again.fixture_fingerprint == r.fixture_fingerprint
    assert again.metrics == pytest.approx(r.metrics)


@requires_ngspice
@requires_models
def test_all_real_vendor_models_smoke(tmp_path):
    results = smoke_models(artifact_root=tmp_path)
    assert set(results) == set(json.loads((MODEL_DIR / "catalog.json").read_text()))
    for name, result in results.items():
        assert result.status == "ok", (name, result.errors)
        assert name in result.model_checksums
        assert all(result.analyses[key]["points"] > 0 for key in ("op", "ac", "tran"))


@requires_ngspice
@requires_models
def test_real_clipping_is_penalized(tmp_path):
    s = spec(topology="noninverting_amplifier", gain=10, cutoff_hz=None, input_amplitude_v=2)
    c = make_candidate(s, 2)
    r = NgSpiceSimulator(artifact_root=tmp_path).evaluate(c)
    # Failure to converge is also invalid, never a passing clipped amplifier.
    assert not score_result(s, c, r).feasible
    if r.status == "ok":
        assert r.metrics["gain_compression"] > 0.05 or r.metrics["thd"] > s.max_thd


@requires_ngspice
def test_real_tolerance_corner_results(tmp_path):
    s = spec(corner_policy="tolerance", cutoff_tolerance=0.2)
    c = make_candidate(s)
    r = NgSpiceSimulator(artifact_root=tmp_path).evaluate(c)
    assert r.status == "ok", r.errors
    assert set(r.analyses["corners"]) == expected_corner_labels(s, c)
    assert len(r.analyses["corners"]) == 8
    assert score_result(s, c, r).valid


def test_unavailable_binary_and_model_are_explicit(tmp_path):
    s = spec()
    c = make_candidate(s)
    assert NgSpiceSimulator(executable="missing_ngspice_123").evaluate(c).status == "unavailable"
    active = make_candidate(spec(topology="buffer", cutoff_hz=None))
    result = NgSpiceSimulator(model_catalog={}).evaluate(active)
    assert result.status == "invalid" and "unavailable" in result.errors[0]


def test_timeout_applies_to_whole_evaluation(monkeypatch, tmp_path):
    import subprocess

    def timeout(*args, **kwargs):
        assert kwargs["timeout"] <= 0.1
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr("spiceharness.simulator.subprocess.run", timeout)
    s = spec()
    r = NgSpiceSimulator(timeout_s=0.1, artifact_root=tmp_path).evaluate(make_candidate(s))
    assert r.status == "timeout"


def test_filter_polarity_and_explicit_order_conflicts_rejected():
    with pytest.raises(ValidationError):
        spec(gain=-1)
    with pytest.raises(ValidationError):
        spec(min_filter_order=2)
    with pytest.raises(ValidationError):
        spec(topology="common_emitter", gain=None, cutoff_hz=None)


@requires_ngspice
def test_real_stopband_outside_original_sweep_is_measured(tmp_path):
    s = spec(stopband_frequency_hz=1e7, max_stopband_gain=1e-3)
    c = make_candidate(s)
    r = NgSpiceSimulator(artifact_root=tmp_path).evaluate(c)
    assert r.status == "ok", r.errors
    assert r.metrics["stopband_gain"] < 1.1e-4
    assert score_result(s, c, r).feasible


@requires_ngspice
@requires_models
def test_actual_independent_resistor_tolerance_changes_gain(tmp_path):
    s = spec(
        topology="noninverting_amplifier",
        gain=2,
        cutoff_hz=None,
        gain_tolerance=0.01,
        resistor_tolerance=0.05,
        corner_policy="tolerance",
    )
    c = make_candidate(s, 2)
    r = NgSpiceSimulator(artifact_root=tmp_path, timeout_s=30).evaluate(c)
    assert r.status == "ok", r.errors
    result = score_result(s, c, r)
    assert result.valid and not result.feasible
    assert any(
        label.endswith(".gain") and value > 0 and label.startswith("RF.")
        for label, value in result.violations.items()
    )


@requires_ngspice
@requires_models
@pytest.mark.parametrize(
    "changes", [{"supply_positive_v": 20, "supply_negative_v": -20}, {"temperature_c": -10}]
)
def test_device_catalog_envelope_is_enforced(changes):
    s = spec(topology="buffer", cutoff_hz=None, **changes)
    r = NgSpiceSimulator().evaluate(make_candidate(s))
    assert r.status == "invalid" and "operating condition" in r.errors[0]
