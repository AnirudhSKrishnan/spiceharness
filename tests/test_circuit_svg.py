"""Saved IR must remain electrically explicit when presented as a schematic."""

from copy import deepcopy
from xml.etree import ElementTree as ET

from spiceharness.circuit_svg import render_circuit_svg

_NS = {"svg": "http://www.w3.org/2000/svg"}


def _texts(element):
    return ["".join(item.itertext()) for item in element.findall(".//svg:text", _NS)]


def _cards(svg, name):
    return [group for group in svg.findall(".//svg:g", _NS) if group.get("class") == name]


def test_all_components_ordered_pins_and_intrinsic_losses_are_visible():
    circuit = {"name": "Complete amplifier", "components": [
        {"ref": "R1", "kind": "R", "nodes": ["input", "filtered"], "value": 2200},
        {"ref": "C1", "kind": "C", "nodes": ["filtered", "0"], "value": 100e-9,
         "series_resistance_ohm": 0.2, "role": "decoupling"},
        {"ref": "L1", "kind": "L", "nodes": ["supply", "vcc"], "value": 7.95e-3,
         "series_resistance_ohm": 3},
        {"ref": "U1A", "kind": "OPAMP", "nodes": ["filtered", "feedback", "vcc", "vee", "out"],
         "model": "LM358", "package": "U1"},
        {"ref": "U1B", "kind": "OPAMP", "nodes": ["0", "spare", "vcc", "vee", "spare"],
         "model": "LM358", "package": "U1", "role": "spare"},
        {"ref": "Q1", "kind": "BJT", "nodes": ["collector", "base", "emitter"], "model": "2N3904"},
        {"ref": "D1", "kind": "DIODE", "nodes": ["anode", "cathode"], "model": "1N4148"},
    ], "connections": [["feedback", "out"]]}
    original = deepcopy(circuit)
    result = render_circuit_svg(circuit)
    svg = ET.fromstring(result)
    cards = _cards(svg, "dut-card")
    assert len(cards) == 7
    assert circuit == original
    assert render_circuit_svg(circuit) == result
    all_text = _texts(svg)
    assert "DUT · 7 IR components · 6 package groups" in all_text
    assert "R · 2.2 kΩ" in all_text
    assert "C · 100 nF" in all_text
    assert "L · 7.95 mH" in all_text
    assert any("ESR: 200 mΩ" in text and "no extra part" in text for text in all_text)
    assert any("DCR: 3 Ω" in text for text in all_text)
    assert any("Role: decoupling" in text for text in all_text)
    assert any("Role: spare" in text for text in all_text)
    # Distinctive nets catch reordered inputs, supplies, output, and transistor pins.
    for line in ("1. + = filtered", "2. - = feedback", "3. V+ = vcc", "4. V- = vee", "5. OUT = out"):
        assert line in _texts(cards[3])
    for line in ("1. C = collector", "2. B = base", "3. E = emitter"):
        assert line in _texts(cards[5])
    assert "1. A = anode" in _texts(cards[6])
    assert "2. K = cathode" in _texts(cards[6])
    # Passive glyphs contain actual vector strokes, not only component tables.
    for card in cards[:3]:
        assert len(card.findall(".//svg:path", _NS)) >= 3
    assert "Explicit ideal connections · zero additional parts" in all_text
    assert "Matching net labels are connected; 0 is the reference net." in all_text


def test_zero_part_candidate_and_explicit_general_fixture_are_separate():
    circuit = {"name": "Direct connection", "components": [], "connections": [["input", "out"]]}
    spec = {"sources": [
        {"ref": "signal", "kind": "V", "nodes": ["input", "0"], "dc": -1,
         "waveform": "sine", "amplitude": 0.5, "frequency_hz": 2000},
        {"ref": "injected_current", "kind": "I", "nodes": ["out", "0"], "dc": 0.001},
    ], "loads": [{"ref": "RLOAD", "kind": "R", "nodes": ["out", "0"], "value": 10000}]}
    svg = ET.fromstring(render_circuit_svg(circuit, spec=spec))
    assert not _cards(svg, "dut-card")
    fixtures = _cards(svg, "fixture-card")
    assert len(fixtures) == 3
    all_text = _texts(svg)
    assert "DUT · 0 IR components · 0 package groups" in all_text
    assert "input" in all_text and "out" in all_text
    assert "Bench fixture · frozen sources and loads" in all_text
    assert "V · DC -1 V" in _texts(fixtures[0])
    assert "Sine: 500 mV peak · 2 kHz" in _texts(fixtures[0])
    assert "1. + = input" in _texts(fixtures[0])
    assert "2. - = 0" in _texts(fixtures[0])
    assert "1. from = out" in _texts(fixtures[1])
    assert "2. to = 0" in _texts(fixtures[1])
    assert "R · 10 kΩ" in _texts(fixtures[2])
    assert not svg.findall(".//svg:script", _NS)
    assert not svg.findall(".//*[@href]")


def test_pulse_fixture_uses_period_and_retains_edge_timing():
    svg = ET.fromstring(render_circuit_svg({"components": []}, spec={"sources": [
        {"ref": "step", "kind": "V", "nodes": ["stimulus", "0"], "dc": 1,
         "waveform": "pulse", "amplitude": 2, "period_s": 0.01, "width_s": 0.002,
         "delay_s": 0.001, "rise_s": 1e-6, "fall_s": 2e-6},
    ]}))
    text = _texts(svg)
    assert "Pulse: Δ2 V · period 10 ms · width 2 ms" in text
    assert "Delay 1 ms · rise 1 µs · fall 2 µs" in text


def test_legacy_spec_is_a_fixture_summary_and_does_not_invent_ir_parts():
    circuit = {"name": "Legacy circuit", "components": [], "input_node": "drive", "output_node": "sense",
               "spec": {"input_amplitude_v": 0.2, "input_frequency_hz": 1000, "load_ohm": 4700,
                        "source_resistance_ohm": 0, "supply_positive_v": 12, "supply_negative_v": -12}}
    svg = ET.fromstring(render_circuit_svg(circuit))
    assert not _cards(svg, "dut-card")
    assert not _cards(svg, "fixture-card")
    text = _texts(svg)
    assert "Legacy evaluator fixture — summary, outside Circuit IR" in text
    assert any("0 Ω feeding input net drive" in item for item in text)
    assert "Load: 4.7 kΩ between output net sense and 0." in text
    assert "Supply conditions: vcc 12 V; vee -12 V; referenced to 0." in text


def test_untrusted_legacy_strings_are_only_text_not_svg_or_html():
    malicious = '</text><script>alert("x")</script><image href="https://evil" onload="alert(1)">'
    circuit = {"name": malicious, "components": [
        {"ref": malicious, "kind": "OPAMP", "nodes": [malicious, "b", "p", "n", "o"],
         "model": malicious, "package": malicious, "role": malicious},
    ], "connections": [[malicious, "out"]]}
    result = render_circuit_svg(circuit, spec={"sources": [
        {"ref": malicious, "kind": "V", "nodes": [malicious, "0"], "waveform": malicious},
    ]})
    svg = ET.fromstring(result)
    assert malicious not in result
    assert malicious in _texts(svg)
    assert not svg.findall(".//svg:script", _NS)
    assert not svg.findall(".//svg:image", _NS)
    for element in svg.iter():
        assert not any(key.lower().startswith("on") or key in ("href", "src") for key in element.attrib)
    # XML-illegal controls also cannot corrupt the document handed to the browser.
    ET.fromstring(render_circuit_svg({"name": "old\x00name", "components": []}))


def test_maximum_ir_size_keeps_every_part_and_autosizes():
    circuit = {"components": [{"ref": f"R{i}", "kind": "R", "nodes": [f"n{i}", f"n{i + 1}"],
                                "value": 1000} for i in range(200)]}
    svg = ET.fromstring(render_circuit_svg(circuit))
    assert len(_cards(svg, "dut-card")) == 200
    assert "R199" in _texts(svg)
    assert "2. 2 = n200" in _texts(svg)
    _, _, width, height = map(int, svg.attrib["viewBox"].split())
    assert width >= 640
    assert height > 2000
    assert int(svg.attrib["height"]) == height
