"""Real-SPICE supply-filter comparison and immutable fixture boundaries."""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from spiceharness.domain import Component
from spiceharness.supply_filter import (
    SupplyFilterEvaluator,
    SupplyFilterSpec,
    analytical_response,
    make_supply_filter,
    rank_supply_filters,
    validate_supply_filter,
)


@pytest.fixture
def evaluator(tmp_path):
    assert shutil.which("ngspice"), "ngspice is required for real supply-filter tests"
    return SupplyFilterEvaluator(tmp_path)


def test_spec_frozen_and_resistor_assumption():
    spec = SupplyFilterSpec()
    with pytest.raises(ValidationError):
        spec.supply_v = 24
    with pytest.raises(ValidationError, match="resistor fixture"):
        SupplyFilterSpec(dc_current_a=0.01)
    with pytest.raises(ValidationError):
        SupplyFilterSpec(stopband_start_hz=1)
    assert spec.fingerprint != SupplyFilterSpec(source_r_ohm=10).fingerprint


@pytest.mark.parametrize("change", [
    {"input_node": "source"}, {"output_node": "vcc"},
    {"topology": "supply_lc\n.control"}, {"metadata": {"control": "quit"}},
])
def test_revalidate_copied_circuit_and_fixture_aliases(change):
    circuit = make_supply_filter("supply_rc").model_copy(update=change)
    with pytest.raises(ValueError):
        validate_supply_filter(circuit)
    assert SupplyFilterEvaluator().evaluate(circuit).status == "invalid"


@pytest.mark.parametrize("change", [
    {"value": -1}, {"model": "ideal"}, {"nodes": ("source", "out")},
    {"nodes": ("in", "dangling")}, {"role": "parasitic"},
    {"parameters": {"ac": 1}}, {"package": "fake"},
    {"ref": "R1\n.control"},
])
def test_revalidate_copied_component_and_exact_template(change):
    original = make_supply_filter("supply_rc")
    part = original.components[0].model_copy(update=change)
    circuit = original.model_copy(update={"components": (part, *original.components[1:])})
    with pytest.raises(ValueError):
        validate_supply_filter(circuit)


def test_no_unassociated_parasitics_or_dc_blocked_supply():
    original = make_supply_filter("supply_rc")
    extra = Component(ref="REXTRA", kind="R", nodes=("out", "0"), value=10, role="parasitic")
    with pytest.raises(ValueError):
        validate_supply_filter(original.model_copy(update={"components": (*original.components, extra)}))
    blocking = original.components[0].model_copy(update={"kind": "C"})
    with pytest.raises(ValueError):
        validate_supply_filter(original.model_copy(update={"components": (blocking, *original.components[1:])}))


def test_t_resonance_cannot_pass_despite_good_stopband(evaluator):
    circuit = make_supply_filter("supply_t", inductance_h=0.01, capacitance_f=100e-6, dcr_ohm=0.05)
    result = evaluator.evaluate(circuit)
    assert result.status == "ok", result.errors
    assert result.metrics["worst_stopband_attenuation_db"] > 30
    assert result.metrics["peak_gain_db"] > 18
    assert not result.feasible
    assert "peak_db" in result.violations
    assert result.physical_components_per_rail == 3
    assert result.modeled_components_per_rail == 6


def test_rc_dc_loss_is_measured_current_not_nominal_current(evaluator):
    circuit = make_supply_filter("supply_rc", resistance_ohm=22, capacitance_f=100e-6)
    result = evaluator.evaluate(circuit)
    assert result.status == "ok", result.errors
    assert result.feasible
    current = 12 / (4000 + 1 + 22)
    assert result.metrics["dc_load_current_a"] == pytest.approx(current, rel=1e-8)
    assert result.metrics["dc_drop_v"] == pytest.approx(current * 22, rel=1e-8)
    assert result.metrics["added_dc_loss_w_per_rail"] == pytest.approx(current**2 * 22, rel=1e-8)
    assert result.metrics["added_dc_loss_w_dual_rail"] == pytest.approx(2 * current**2 * 22, rel=1e-8)
    assert result.metrics["source_dc_loss_w_per_rail"] == pytest.approx(current**2, rel=1e-8)


def test_damped_lc_and_capacitor_only_keep_low_dc_loss(evaluator):
    lc = make_supply_filter(
        "supply_lc", inductance_h=0.01, capacitance_f=100e-6, dcr_ohm=0.05,
        damping_r_ohm=10, damping_c_f=1000e-6,
    )
    result = evaluator.evaluate(lc)
    cap = evaluator.evaluate(make_supply_filter("supply_capacitor", capacitance_f=4700e-6))
    assert result.status == cap.status == "ok"
    assert result.feasible and cap.feasible
    assert result.metrics["peak_gain_db"] < 1
    assert result.metrics["added_dc_loss_w_per_rail"] == pytest.approx(
        (12 / 4001.05) ** 2 * 0.05, rel=1e-7
    )
    assert cap.metrics["added_dc_loss_w_per_rail"] < 1e-10
    assert result.physical_components_per_rail == 4
    assert cap.physical_components_per_rail == 1
    assert rank_supply_filters([result, cap])[0] == cap


def test_feasible_dominates_parts_and_every_result_has_artifacts(evaluator):
    direct = evaluator.evaluate(make_supply_filter("supply_direct"))
    rc = evaluator.evaluate(make_supply_filter("supply_rc", resistance_ohm=22))
    assert direct.status == rc.status == "ok"
    assert not direct.feasible and rc.feasible
    assert rank_supply_filters([direct, rc])[0] == rc
    directory = Path(rc.artifact_dir)
    for name in ("spec.json", "circuit.json", "op.cir", "ac.cir", "op.raw", "ac.raw", "op.log", "ac.log", "result.json", "ac_metrics.json"):
        assert (directory / name).is_file()
    curves = json.loads((directory / "ac_metrics.json").read_text())
    assert len(curves["frequency_hz"]) >= 1000
    spectral = rc.spectral["1000"]
    assert spectral["load_noise_power_w_per_input_v_rms_squared"] == pytest.approx(
        spectral["output_noise_v_rms_per_input_v_rms"] ** 2 / 4000, rel=1e-7
    )
    assert json.loads((directory / "spec.json").read_text())["source_r_ohm"] == 1


def test_analytical_rc_matches_closed_form_including_esr():
    circuit = make_supply_filter("supply_rc", resistance_ohm=22, capacitance_f=100e-6, esr_ohm=0.1)
    frequency = np.array([100, 1000, 10000], dtype=float)
    shunt = 0.1 + 1 / (2j * np.pi * frequency * 100e-6)
    parallel = 1 / (1 / shunt + 1 / 4000)
    expected = parallel / (23 + parallel)
    assert np.allclose(analytical_response(circuit, SupplyFilterSpec(), frequency), expected, rtol=1e-10)


def test_unavailable_simulator_never_passes(tmp_path):
    result = SupplyFilterEvaluator(tmp_path, executable="missing-ngspice-for-test").evaluate(
        make_supply_filter("supply_capacitor", capacitance_f=4700e-6)
    )
    assert result.status == "unavailable"
    assert not result.feasible
    assert (Path(result.artifact_dir) / "result.json").exists()
