"""Native KiCad export is checked electrically through KiCad and real ngspice."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from spiceharness.experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR
from spiceharness.kicad_export import model_pin_names, render_kicad_schematic
from spiceharness.simulator import MODEL_DIR, NgSpiceSimulator, _read_raw

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLE = _ROOT / "examples/experiments/loaded-rc"
_KICAD = (os.environ.get("KICAD_CLI") or shutil.which("kicad-cli") or next(
    (str(path) for path in (
        Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"),
        Path("/Volumes/KiCad/KiCad/KiCad.app/Contents/MacOS/kicad-cli"),
    ) if path.is_file()), None))
_NATIVE = pytest.mark.skipif(not _KICAD or not shutil.which("ngspice"),
                            reason="actual KiCad CLI and ngspice are required for native electrical validation")


def _tree(content):
    """Small test-only S-expression reader, independent of exporter structure."""
    tokens = re.findall(r'"(?:\\.|[^"\\])*"|[()]|[^\s()]+', content)
    stack, root = [], None
    for token in tokens:
        if token == "(":
            item = []
            if stack:
                stack[-1].append(item)
            elif root is not None:
                raise AssertionError("more than one top-level expression")
            else:
                root = item
            stack.append(item)
        elif token == ")":
            stack.pop()
        else:
            stack[-1].append(json.loads(token) if token.startswith('"') else token)
    assert not stack
    return root


def _symbols(tree):
    return [item for item in tree if isinstance(item, list) and item[0] == "symbol"]


def _properties(symbol):
    return {item[1]: item[2] for item in symbol if isinstance(item, list) and item[0] == "property"}


def _general_fixture():
    return (json.loads((_EXAMPLE / "reference.json").read_text()),
            json.loads((_EXAMPLE / "spec.json").read_text()))


def _native_deck(tmp_path, circuit, spec, bindings=None):
    schematic = tmp_path / "candidate.kicad_sch"
    schematic.write_text(render_kicad_schematic(circuit, spec=spec, model_bindings=bindings))
    # A real adjacent project gives KiCad the project root, independently of cwd.
    (tmp_path / "candidate.kicad_pro").write_text("{}\n")
    deck = tmp_path / "native.cir"
    result = subprocess.run([_KICAD, "sch", "export", "netlist", "--format", "spice", "-o", str(deck),
                             str(schematic)], check=False, text=True, capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert deck.is_file(), result.stdout + result.stderr
    return deck


def _native_gain(tmp_path, deck):
    content = re.sub(r"^\.(?:op|ac|tran)\b[^\n]*\n", "", deck.read_text(), flags=re.MULTILINE)
    assert content.rstrip().endswith(".end")
    deck.write_text(content.rsplit(".end", 1)[0] + ".ac lin 1 1000 1000\n.save v(/out)\n.end\n")
    (tmp_path / ".spiceinit").write_text("set ngbehavior=ltpsa\nset filetype=ascii\n")
    result = subprocess.run(["ngspice", "-b", "-r", "native.raw", "-o", "native.log", str(deck)],
                            cwd=tmp_path, check=False, text=True, capture_output=True, timeout=30)
    log = (tmp_path / "native.log").read_text()
    assert result.returncode == 0, log
    variables, values = _read_raw(tmp_path / "native.raw")
    return abs(values[0, variables.index("v(/out)")])


def test_original_symbols_ordered_ir_and_model_name_mapping():
    circuit = {"name": "Named model pins", "components": [
        {"ref": "amp", "kind": "OPAMP", "nodes": ["plus", "minus", "vp", "vn", "out"],
         "model": "NamedAmp", "package": "amp"},
        {"ref": "transistor", "kind": "BJT", "nodes": ["c", "b", "e"], "model": "Device"},
        {"ref": "rectifier", "kind": "DIODE", "nodes": ["a", "k"], "model": "Diode"},
    ]}
    binding = {"NamedAmp": {"library": "models/amp.lib", "spice_name": "named_amp",
                            "pins": ["output", "positive_supply", "negative_supply", "noninverting", "inverting"],
                            "model_pins": ["28", "99", "50", "1", "2"]},
               "Device": {"library": "models/device.lib", "spice_name": "transistor"},
               "Diode": {"library": "models/diode.lib", "spice_name": "diode"}}
    rendered = render_kicad_schematic(circuit, model_bindings=binding)
    assert rendered == render_kicad_schematic(circuit, model_bindings=binding)
    tree = _tree(rendered)
    symbols = _symbols(tree)
    assert len(symbols) == 3
    properties = [_properties(symbol) for symbol in symbols]
    assert properties[0]["IR.OrderedNodes"] == '["plus", "minus", "vp", "vn", "out"]'
    assert properties[0]["Sim.Pins"] == "1=1 2=2 3=99 4=50 5=28"
    assert properties[0]["Sim.Library"] == "models/amp.lib"
    assert properties[1]["Sim.Pins"] == "1=C 2=B 3=E"
    assert properties[2]["Sim.Pins"] == "1=A 2=K"
    libraries = next(item for item in tree if isinstance(item, list) and item[0] == "lib_symbols")
    assert len(libraries) == 4
    assert not any(item[0] == "image" for item in tree if isinstance(item, list))


def test_model_headers_read_actual_names_and_continuations_without_executing():
    model = "* original model header\n.SUBCKT LM741 1 2\n+ 99 50 28 PARAMS: unused=1\n.ENDS\n"
    assert model_pin_names(model, "lm741") == ["1", "2", "99", "50", "28"]
    assert model_pin_names(".MODEL device PNP(IS=1e-14)", "device") == ["C", "B", "E"]
    assert model_pin_names(".MODEL diode D(IS=1e-14)", "diode") == ["A", "K"]
    with pytest.raises(ValueError, match="no supported model"):
        model_pin_names(model, "unavailable")


def test_missing_active_model_is_visible_and_cannot_silently_be_omitted():
    circuit = {"components": [{"ref": "amp", "kind": "OPAMP", "nodes": ["in", "out", "vp", "vn", "out"],
                               "model": "LM741"}]}
    rendered = render_kicad_schematic(circuit)
    symbol = _symbols(_tree(rendered))[0]
    properties = _properties(symbol)
    assert "MISSING_MODEL_LM741.lib" in properties["Sim.Library"]
    assert properties["Sim.Name"] == "LM741"
    assert properties["Sim.Device"] == "SUBCKT"
    assert ["exclude_from_sim", "no"] in symbol
    assert "MODEL UNBOUND" in rendered
    assert "must fail" in rendered


def test_native_export_rejects_unvalidated_ir_and_unsafe_model_paths():
    with pytest.raises(ValidationError):
        render_kicad_schematic({"components": [], "spice": ".control\nshell anything"})
    circuit = {"name": 'Name\n.control\n) (wire "injected")', "components": [
        {"ref": "amp", "kind": "OPAMP", "nodes": ["in", "out", "vp", "vn", "out"], "model": "LM741"},
    ]}
    # Names are quoted data, and control characters never create SPICE directives.
    tree = _tree(render_kicad_schematic(circuit))
    assert tree[0] == "kicad_sch"
    assert not any(item[0] == "wire" and item[1] == "injected" for item in tree if isinstance(item, list))
    with pytest.raises(ValueError, match="bundle-relative"):
        render_kicad_schematic(circuit, model_bindings={"LM741": {"library": "../outside.lib"}})


@_NATIVE
def test_kicad_native_rc_netlist_matches_real_evaluator_and_intrinsic_loss(tmp_path):
    circuit, spec = _general_fixture()
    spec["temperature_c"] = 42
    deck = _native_deck(tmp_path, circuit, spec)
    content = deck.read_text()
    assert re.search(r"^\.temp 42$", content, re.MULTILINE)
    assert ".options numdgt=15 reltol=1e-4 method=gear" in content
    assert re.search(r"^\.ac dec 200 10 ", content, re.MULTILINE)
    # Count elements independently from KiCad's actual emitted SPICE, including ESR.
    assert len([line for line in content.splitlines() if line.startswith("R")]) == 4
    assert len([line for line in content.splitlines() if line.startswith("C")]) == 1
    native_gain = _native_gain(tmp_path, deck)
    result = ExperimentEvaluator().evaluate(GeneralCircuitIR.model_validate(circuit), ExperimentSpec.model_validate(spec))
    assert result.status == "ok", result.errors
    assert native_gain == pytest.approx(result.metrics["gain_1khz"], rel=1e-8)
    properties = [_properties(symbol) for symbol in _symbols(_tree((tmp_path / "candidate.kicad_sch").read_text()))]
    intrinsic = next(p for p in properties if p["IR.Category"] == "Intrinsic loss")
    assert intrinsic["IR.Package"] == "C1"
    assert intrinsic["IR.Reference"] == "C1_intrinsic"


@_NATIVE
def test_native_zero_part_wire_is_real_connected_net_not_a_drawing_only(tmp_path):
    _, spec = _general_fixture()
    circuit = {"name": "Direct", "components": [], "connections": [["in", "out"]]}
    deck = _native_deck(tmp_path, circuit, spec)
    # KiCad is free to choose either connected label as the native net's name.
    content = deck.read_text()
    assert "/in /out" not in content
    alias = "/out" if "/out" in content else "/in"
    content = content.replace(alias, "/out")
    deck.write_text(content)
    assert _native_gain(tmp_path, deck) == pytest.approx(10000 / 10050, rel=1e-10)
    tree = _tree((tmp_path / "candidate.kicad_sch").read_text())
    assert len(_symbols(tree)) == 3  # all fixture symbols; zero DUT parts
    assert all(_properties(symbol)["IR.Category"] == "Bench fixture" for symbol in _symbols(tree))


@_NATIVE
@pytest.mark.skipif(not (MODEL_DIR / "LM741.lib").is_file(), reason="original LM741 vendor model required")
def test_original_lm741_native_model_binding_matches_evaluator(tmp_path):
    circuit = {"name": "Inverting amplifier", "components": [
        {"ref": "RIN", "kind": "R", "nodes": ["in", "minus"], "value": 1000},
        {"ref": "RF", "kind": "R", "nodes": ["out", "minus"], "value": 10000},
        {"ref": "AMP", "kind": "OPAMP", "nodes": ["0", "minus", "vp", "vn", "out"], "model": "LM741"},
    ]}
    spec = {"sources": [
        {"ref": "positive", "nodes": ["vp", "0"], "dc": 12},
        {"ref": "negative", "nodes": ["vn", "0"], "dc": -12},
        {"ref": "signal", "nodes": ["in", "0"], "dc": 0, "waveform": "sine", "amplitude": 0.01},
    ], "loads": [{"ref": "load", "kind": "R", "nodes": ["out", "0"], "value": 10000}],
        "probes": [{"name": "output", "positive": "out"}],
        "measurements": [{"name": "gain", "operator": "ac_magnitude", "probe": "output", "source": "signal",
                          "frequency_hz": 1000}], "allowed_models": ["LM741"]}
    entry = NgSpiceSimulator().catalog["LM741"]
    original = MODEL_DIR / entry["path"]
    (tmp_path / "models").mkdir()
    shutil.copyfile(original, tmp_path / "models/LM741.lib")
    bindings = {"LM741": {**entry, "library": "models/LM741.lib",
                           "model_pins": model_pin_names(original.read_text(), entry["spice_name"])}}
    deck = _native_deck(tmp_path, circuit, spec, bindings)
    content = deck.read_text()
    assert str(tmp_path / "models/LM741.lib") in content
    assert re.search(r"^XU\d+ 0 /minus /vp /vn /out LM741$", content, re.MULTILINE), content
    native_gain = _native_gain(tmp_path, deck)
    result = ExperimentEvaluator().evaluate(GeneralCircuitIR.model_validate(circuit), ExperimentSpec.model_validate(spec))
    assert result.status == "ok", result.errors
    assert native_gain == pytest.approx(result.metrics["gain"], rel=1e-7)


@_NATIVE
@pytest.mark.skipif(not all((MODEL_DIR / name).is_file() for name in ("2N3904.lib", "1N4148.lib")),
                    reason="original BJT and diode models required")
def test_real_bjt_diode_and_lossy_inductor_native_connectivity(tmp_path):
    circuit = {"components": [
        {"ref": "Lfeed", "kind": "L", "nodes": ["supply", "vcc"], "value": 0.01,
         "series_resistance_ohm": 1},
        {"ref": "Cdec", "kind": "C", "nodes": ["vcc", "0"], "value": 100e-6},
        {"ref": "Rcollector", "kind": "R", "nodes": ["vcc", "collector"], "value": 4700},
        {"ref": "Remitter", "kind": "R", "nodes": ["emitter", "0"], "value": 1000},
        {"ref": "Rbiasp", "kind": "R", "nodes": ["vcc", "base"], "value": 100000},
        {"ref": "Rbiasn", "kind": "R", "nodes": ["base", "0"], "value": 22000},
        {"ref": "Cin", "kind": "C", "nodes": ["in", "base"], "value": 1e-6},
        {"ref": "Cout", "kind": "C", "nodes": ["collector", "out"], "value": 1e-6},
        {"ref": "Qactive", "kind": "BJT", "nodes": ["collector", "base", "emitter"], "model": "2N3904"},
        {"ref": "Dclamp", "kind": "DIODE", "nodes": ["out", "0"], "model": "1N4148"},
    ]}
    spec = {"sources": [{"ref": "supply", "nodes": ["supply", "0"], "dc": 12},
                        {"ref": "signal", "nodes": ["in", "0"], "waveform": "sine", "amplitude": 0.001}],
            "loads": [{"ref": "load", "kind": "R", "nodes": ["out", "0"], "value": 10000}],
            "probes": [{"name": "output", "positive": "out"}],
            "measurements": [{"name": "gain", "operator": "ac_magnitude", "probe": "output", "source": "signal",
                              "frequency_hz": 1000}], "allowed_models": ["2N3904", "1N4148"]}
    (tmp_path / "models").mkdir()
    bindings = {}
    catalog = NgSpiceSimulator().catalog
    for model in spec["allowed_models"]:
        entry = catalog[model]
        original = MODEL_DIR / entry["path"]
        shutil.copyfile(original, tmp_path / "models" / original.name)
        bindings[model] = {**entry, "library": "models/" + original.name,
                           "model_pins": model_pin_names(original.read_text(), entry["spice_name"])}
    deck = _native_deck(tmp_path, circuit, spec, bindings)
    content = deck.read_text()
    assert re.search(r"^Q\w+ /collector /base /emitter Q2n3904", content, re.MULTILINE), content
    assert re.search(r"^D\w+ /out 0 D1N4148/TEMP", content, re.MULTILINE), content
    native_gain = _native_gain(tmp_path, deck)
    result = ExperimentEvaluator().evaluate(GeneralCircuitIR.model_validate(circuit), ExperimentSpec.model_validate(spec))
    assert result.status == "ok", result.errors
    assert native_gain == pytest.approx(result.metrics["gain"], rel=1e-7)


@_NATIVE
def test_native_pulse_and_current_fixture_waveforms(tmp_path):
    spec = {"sources": [
        {"ref": "pulse", "nodes": ["drive", "0"], "dc": 1, "waveform": "pulse", "amplitude": 2,
         "delay_s": 0.001, "rise_s": 20e-6, "fall_s": 20e-6, "width_s": 0.002, "period_s": 0.01},
        {"ref": "sink", "kind": "I", "nodes": ["out", "0"], "dc": 0.001},
    ], "loads": [{"ref": "load", "kind": "R", "nodes": ["out", "0"], "value": 10000}],
        "probes": [{"name": "output", "positive": "out"}],
        "transient": {"start_s": 0.01, "stop_s": 0.025, "step_s": 1e-6},
        "measurements": [{"name": "mean", "operator": "transient_mean", "probe": "output"}]}
    circuit = {"components": [{"ref": "R1", "kind": "R", "nodes": ["drive", "out"], "value": 1000}]}
    deck = _native_deck(tmp_path, circuit, spec)
    content = deck.read_text()
    assert re.search(r"^I\d+ /out 0 DC ", content, re.MULTILINE), content
    assert "PULSE( 1 3 " in content
    assert re.search(r"^\.tran \S+ 0.025\S* 0 \S+", content, re.MULTILINE), content
    (tmp_path / ".spiceinit").write_text("set ngbehavior=ltpsa\nset filetype=ascii\n")
    result = subprocess.run(["ngspice", "-b", "-r", "pulse.raw", "-o", "pulse.log", str(deck)],
                            cwd=tmp_path, check=False, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, (tmp_path / "pulse.log").read_text()
    names, values = _read_raw(tmp_path / "pulse.raw")
    time, voltage = values[:, names.index("time")].real, values[:, names.index("v(/out)")].real
    # Current flows from out to ground; three pulses prove polarity and repetition.
    assert np.interp([0.0015, 0.0115, 0.0215], time, voltage) == pytest.approx([2 / 1.1] * 3, abs=1e-7)
    assert np.interp([0.006, 0.016], time, voltage) == pytest.approx([0, 0], abs=1e-7)


@_NATIVE
def test_native_legacy_fixture_matches_clamped_source_and_case_insensitive_spice_nodes(tmp_path):
    circuit = {"name": "Legacy lowpass", "topology": "passive_lowpass", "input_node": "in", "output_node": "out",
               "components": [{"ref": "R1", "kind": "R", "nodes": ["IN", "OUT"], "value": 1000},
                              {"ref": "C1", "kind": "C", "nodes": ["OUT", "0"], "value": 100e-9}],
               "spec": {"topology": "passive_lowpass", "supply_positive_v": 12, "supply_negative_v": -12,
                        "input_amplitude_v": 0.2, "input_frequency_hz": 1000, "source_resistance_ohm": 0,
                        "load_ohm": 10000, "cutoff_hz": 1600}}
    deck = _native_deck(tmp_path, circuit, None)
    content = deck.read_text()
    assert re.search(r"^\.tran ", content, re.MULTILINE)
    assert re.search(r"^V\d+ /vcc 0 DC 12 AC 0", content, re.MULTILINE)
    assert re.search(r"^V\d+ /vee 0 DC -12 AC 0", content, re.MULTILINE)
    assert re.search(r"^V\d+ /source 0 DC 0 SIN\([^\n]*AC 1", content, re.MULTILINE)
    gain = _native_gain(tmp_path, deck)
    shunt = 1 / (1 / 10000 + 2j * np.pi * 1000 * 100e-9)
    assert gain == pytest.approx(abs(shunt / (1000 + 1e-6 + shunt)), rel=1e-10)
