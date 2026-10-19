"""KiCad schematic import: round trip, synthetic edits, frozen fixtures, rescoring and kicad-cli agreement."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from typer.testing import CliRunner

from spiceharness.cli import app
from spiceharness.domain import DesignSpec
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import ExperimentSpec, GeneralCircuitIR
from spiceharness.kicad_export import _POSITIONS, _TWO_PINS
from spiceharness.kicad_import import (
    MAX_DIAGONAL_WIRES,
    KiCadImportError,
    import_kicad_bytes,
    import_kicad_schematic,
    parse_sexpr,
    parse_si_value,
)
from spiceharness.spice_bundle import export_spice_bundle
from spiceharness.storage import RunStore

ROOT = Path(__file__).resolve().parents[1]
LAB4 = ROOT / "examples/experiments/lab4-supply"
RC = ROOT / "examples/experiments/loaded-rc"
runner = CliRunner()
_KICAD = (os.environ.get("KICAD_CLI") or shutil.which("kicad-cli") or next(
    (str(p) for p in (Path("/Applications/KiCad/KiCad.app/Contents/MacOS/kicad-cli"),) if p.is_file()), None))
real_spice = pytest.mark.skipif(not shutil.which("ngspice"), reason="ngspice is required")
needs_kicad = pytest.mark.skipif(not _KICAD, reason="kicad-cli is not installed")


# ----------------------------------------------------------------------------- fixtures


def _compare(workspace: Path, directory: Path, spec: Path) -> dict:
    result = runner.invoke(app, ["experiment", "compare", "--spec", str(spec), "--candidates", str(directory),
                                 "-w", str(workspace), "--json"])
    assert result.stdout.lstrip().startswith("{"), result.output
    return json.loads(result.stdout)


def _export(workspace: Path, run_id: str, destination: Path) -> Path:
    run = ExperimentManager(workspace).get_experiment(run_id)
    return export_spice_bundle(run, destination, workspace=workspace).parent


@pytest.fixture(scope="module")
def lab4(tmp_path_factory):
    if not shutil.which("ngspice"):
        pytest.skip("ngspice is required")
    base = tmp_path_factory.mktemp("lab4")
    workspace = base / "ws"
    run = _compare(workspace, LAB4, LAB4 / "spec.json")
    export = _export(workspace, run["run_id"], base / "export")
    return {"workspace": workspace, "run": run, "export": export, "spec": run["spec"]}


@pytest.fixture
def rc(tmp_path):
    """Loaded-RC reference as a standalone export folder (no simulation needed)."""
    from spiceharness.kicad_export import render_kicad_schematic

    circuit = json.loads((RC / "reference.json").read_text())
    spec = json.loads((RC / "spec.json").read_text())
    folder = tmp_path / "rc"
    folder.mkdir()
    (folder / "candidate.kicad_sch").write_text(render_kicad_schematic(circuit, spec=spec))
    (folder / "circuit.json").write_text(json.dumps(circuit))
    return folder, spec, circuit


# ----------------------------------------------------------------------------- text-edit helpers


def _lines(folder: Path) -> list[str]:
    return (folder / "candidate.kicad_sch").read_text().splitlines()


def _save(folder: Path, lines: list[str]) -> Path:
    path = folder / "candidate.kicad_sch"
    path.write_text("\n".join(lines) + "\n")
    return path


def _block(lines: list[str], ir_ref: str) -> tuple[int, int]:
    """(first line, line holding (instances) of the symbol whose IR.Reference is ir_ref."""
    start = None
    for i, line in enumerate(lines):
        if line.startswith("(symbol (lib_id"):
            start = i
        if line.startswith(f'(property "IR.Reference" "{ir_ref}"'):
            assert start is not None
            end = next(j for j in range(i, len(lines)) if lines[j].startswith("(instances"))
            return start, end
    raise AssertionError(f"no symbol {ir_ref}")


def _set(lines: list[str], ir_ref: str, name: str, value: str) -> None:
    start, end = _block(lines, ir_ref)
    for i in range(start, end):
        if lines[i].startswith(f'(property "{name}" '):
            lines[i] = re.sub(rf'^(\(property "{name}" )"[^"]*"', lambda m: f'{m[1]}"{value}"', lines[i])
            return
    raise AssertionError(f"no property {name} on {ir_ref}")


def _stub_lines(lines: list[str], ir_ref: str, count: int) -> range:
    _, end = _block(lines, ir_ref)
    return range(end + 1, end + 1 + 2 * count)


def _relabel(lines: list[str], ir_ref: str, pin: int, text: str) -> None:
    index = _stub_lines(lines, ir_ref, pin)[-1]
    assert lines[index].startswith("(label ")
    lines[index] = re.sub(r'^\(label "[^"]*"', f'(label "{text}"', lines[index])


def _lib_positions(kind: str) -> list[tuple[float, float]]:
    return [(x, y) for x, y, _, _ in _POSITIONS.get(kind, _TWO_PINS)]


def _sheet_point(lib: tuple[float, float], rotation: int, mirror: str | None, origin: tuple[float, float]):
    """Independent statement of KiCad's rule: rotate CCW (+Y up), mirror, then flip onto the sheet."""
    cos, sin = {0: (1, 0), 90: (0, 1), 180: (-1, 0), 270: (0, -1)}[rotation]
    x, y = lib[0] * cos - lib[1] * sin, lib[0] * sin + lib[1] * cos
    if mirror == "x":
        y = -y
    if mirror == "y":
        x = -x
    return round(origin[0] + x, 4), round(origin[1] - y, 4)


def _orient(lines: list[str], ir_ref: str, kind: str, rotation: int, mirror: str | None) -> None:
    """Rotate/mirror a symbol and re-route its stub wires so the labels follow the moved pins."""
    start, _ = _block(lines, ir_ref)
    match = re.search(r"\(at ([\d.]+) ([\d.]+) 0\)", lines[start])
    origin = (float(match[1]), float(match[2]))
    lines[start] = lines[start].replace(match[0], f"(at {match[1]} {match[2]} {rotation})"
                                        + (f" (mirror {mirror})" if mirror else ""))
    pins = _lib_positions(kind)
    stubs = _stub_lines(lines, ir_ref, len(pins))
    texts = [re.match(r'\(label "([^"]*)"', lines[stubs[2 * i + 1]])[1] for i in range(len(pins))]
    for i, (lib, text) in enumerate(zip(pins, texts, strict=True)):
        x, y = _sheet_point(lib, rotation, mirror, origin)
        dx, dy = x - origin[0], y - origin[1]
        end_x, end_y = (x + (5.08 if dx >= 0 else -5.08), y) if abs(dx) >= abs(dy) else (
            x, y + (5.08 if dy >= 0 else -5.08))
        lines[stubs[2 * i]] = (f"(wire (pts (xy {x} {y}) (xy {end_x} {end_y})) (stroke (width 0) "
                               f"(type default)) (uuid 00000000-0000-4000-8000-{rotation:04d}{i:08d}))")
        lines[stubs[2 * i + 1]] = (f'(label "{text}" (at {round(end_x, 4)} {round(end_y, 4)} 0) '
                                   "(effects (font (size 0.9 0.9)) (justify left)) "
                                   f"(uuid 00000000-0000-4000-8001-{rotation:04d}{i:08d}))")


_DEVICE_R = '''(symbol "Device:R" (pin_numbers (hide yes)) (pin_names (offset 0)) (exclude_from_sim no) (in_bom yes) (on_board yes)
(property "Reference" "R" (at 2.032 0 90) (effects (font (size 1.27 1.27))))
(property "Value" "R" (at 0 0 90) (effects (font (size 1.27 1.27))))
(property "Footprint" "" (at -1.778 0 90) (effects (font (size 1.27 1.27)) (hide yes)))
(property "Datasheet" "~" (at 0 0 0) (effects (font (size 1.27 1.27)) (hide yes)))
(symbol "R_0_1" (rectangle (start -1.016 -2.54) (end 1.016 2.54) (stroke (width 0.254) (type default)) (fill (type none))))
(symbol "R_1_1"
(pin passive line (at 0 3.81 270) (length 1.27) (name "~" (effects (font (size 1.27 1.27)))) (number "1" (effects (font (size 1.27 1.27)))))
(pin passive line (at 0 -3.81 90) (length 1.27) (name "~" (effects (font (size 1.27 1.27)))) (number "2" (effects (font (size 1.27 1.27)))))
))'''


def _add_stock(lines: list[str], lib_id: str, ref: str, value: str, x: float, y: float, nets: tuple[str, str]):
    """Add a KiCad stock vertical passive with its definition embedded, wired by labels like a real edit."""
    kind = lib_id.split(":")[1][0]
    definition = _DEVICE_R.replace('"Device:R"', f'"{lib_id}"').replace('"R_0_1"', f'"{lib_id.split(":")[1]}_0_1"') \
        .replace('"R_1_1"', f'"{lib_id.split(":")[1]}_1_1"')
    if not any(f'(symbol "{lib_id}"' in line for line in lines):
        index = next(i for i, line in enumerate(lines) if line.startswith("(lib_symbols"))
        lines[index] = lines[index].replace("(lib_symbols ", "(lib_symbols " + definition + "\n", 1)
    block = [(f'(symbol (lib_id "{lib_id}") (at {x} {y} 0) (unit 1) (exclude_from_sim no) (in_bom yes) (on_board yes) '
              f'(dnp no) (uuid 00000000-0000-4000-8002-{abs(hash(ref)) % 10**12:012d})'),
             f'(property "Reference" "{ref}" (at {x + 3} {y} 90) (effects (font (size 1.27 1.27))))',
             f'(property "Value" "{value}" (at {x - 3} {y} 90) (effects (font (size 1.27 1.27))))',
             f'(pin "1" (uuid 00000000-0000-4000-8003-{abs(hash(ref)) % 10**12:012d}))',
             f'(pin "2" (uuid 00000000-0000-4000-8004-{abs(hash(ref)) % 10**12:012d}))',
             (f'(instances (project "candidate" (path "/00000000-0000-4000-8000-000000000000" (reference "{ref}") '
              '(unit 1)))))')]
    for number, (px, py, net) in enumerate(((x, y - 3.81, nets[0]), (x, y + 3.81, nets[1])), 1):
        sign = -1 if number == 1 else 1
        block.append(f"(wire (pts (xy {px} {py}) (xy {px} {py + sign * 5.08})) (stroke (width 0) (type default)) "
                     f"(uuid 00000000-0000-4000-8005-{number:012d}{abs(hash(ref)) % 1000:03d}))")
        block.append(f'(label "{net}" (at {px} {round(py + sign * 5.08, 4)} 0) (effects (font (size 1 1))) '
                     f"(uuid 00000000-0000-4000-8006-{number:012d}{abs(hash(ref)) % 1000:03d}))")
    end = next(i for i, line in enumerate(lines) if line.startswith("(sheet_instances"))
    lines[end:end] = block
    assert kind in "RCL"


def _rc_import(folder: Path, spec: dict):
    return import_kicad_schematic(folder / "candidate.kicad_sch", spec=spec)


# ----------------------------------------------------------------------------- parser and numbers


def test_sexpr_parser_is_bounded_and_strict():
    tree = parse_sexpr('(a "b \\"q\\" \\\\" (c 1 2) "")')
    assert tree[1] == 'b "q" \\' and tree[2] == ["c", "1", "2"] and tree[3] == ""
    for bad in ("(a", "a)", '(a "unterminated)', "(a) (b)", "", "(" * 100 + ")" * 100):
        with pytest.raises(KiCadImportError):
            parse_sexpr(bad)


@pytest.mark.parametrize("text,kind,value", [
    ("2.2k", "R", 2200), ("4k7", "R", 4700), ("100u", "C", 1e-4), ("100µ", "C", 1e-4), ("100μF", "C", 1e-4),
    ("10n", "C", 1e-8), ("1Meg", "R", 1e6), ("1MEG", "R", 1e6), ("1meg", "R", 1e6), ("1m", "L", 1e-3),
    ("2R2", "R", 2.2), ("1k5", "R", 1500), ("100kΩ", "R", 1e5), ("47uF", "C", 47e-6), ("10mH", "L", 1e-2),
    ("1e-6", "C", 1e-6), ("0.0022000000000000001", "C", 0.0022), ("2.2 k", "R", 2200), ("100R", "R", 100),
    ("1p", "C", 1e-12), ("3.3", "R", 3.3),
])
def test_si_value_rule(text, kind, value):
    assert parse_si_value(text, kind)[0] == pytest.approx(value, rel=1e-12)


def test_bare_m_follows_kicad_10_and_warns():
    mega, notes = parse_si_value("1M", "R")
    assert mega == 1e6 and "mega" in notes[0] and "Meg" in notes[0]
    milli, notes = parse_si_value("5m", "R")
    assert milli == 5e-3 and "milli" in notes[0]
    assert parse_si_value("1Meg", "R")[1] == []
    for bad in ("10F", "abc", "1x", "5mil", ""):
        with pytest.raises(ValueError):
            parse_si_value(bad, "C")


# ----------------------------------------------------------------------------- round trip


@real_spice
def test_round_trip_every_exported_candidate_is_exact(lab4):
    folders = sorted(path for path in lab4["export"].iterdir() if (path / "candidate.kicad_sch").is_file())
    assert len(folders) == len(lab4["run"]["candidates"]) == 9
    spec = ExperimentSpec.model_validate(lab4["spec"])
    for folder in folders:
        original = json.loads((folder / "circuit.json").read_text())
        result = import_kicad_schematic(folder / "candidate.kicad_sch", spec=lab4["spec"])
        assert result.ok and result.fixture_ok is True and result.original_found
        assert result.circuit == original, folder.name
        assert GeneralCircuitIR.model_validate(result.circuit).fingerprint == GeneralCircuitIR.model_validate(
            original).fingerprint
        assert result.changes == [] and result.warnings == []
        assert len(result.sha256) == 64
    assert spec.protected_refs  # the protected-part comparison inside validate() accepted every candidate


@real_spice
def test_round_trip_without_circuit_json_keeps_ir_up_to_order(lab4):
    folder = next(path for path in lab4["export"].iterdir() if (path / "candidate.kicad_sch").is_file())
    original = json.loads((folder / "circuit.json").read_text())
    alone = folder.parent / "alone"
    alone.mkdir()
    shutil.copy(folder / "candidate.kicad_sch", alone / "candidate.kicad_sch")
    result = import_kicad_schematic(alone / "candidate.kicad_sch", spec=lab4["spec"])
    assert result.ok and result.changes == [] and not result.original_found
    key = lambda part: part["ref"]
    assert sorted(result.circuit["components"], key=key) == sorted(original["components"], key=key)
    assert sorted(map(sorted, result.circuit["connections"])) == sorted(map(sorted, original["connections"]))


# ----------------------------------------------------------------------------- edits


def test_value_edit_is_reported(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "R1", "Value", "2.2k")
    _set(lines, "R1", "Sim.Params", "r=2.2k")
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.ok and result.circuit["components"][0]["value"] == 2200
    assert [c["text"] for c in result.changes] == ["R1.value: 1 kΩ -> 2.2 kΩ"]


def test_value_field_only_edit_is_honored_with_original_but_conflict_is_error_without(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "R1", "Value", "4k7")
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.circuit["components"][0]["value"] == 4700
    assert any("only the Value field" in warning for warning in result.warnings)
    (folder / "circuit.json").unlink()
    with pytest.raises(KiCadImportError, match="disagree"):
        _rc_import(folder, spec)


def test_bare_M_warns_on_resistor_and_reads_mega(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "R1", "Value", "1M")
    _set(lines, "R1", "Sim.Params", "r=1M")
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.circuit["components"][0]["value"] == 1e6
    assert any("bare 'M'" in warning for warning in result.warnings)


@real_spice
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("mirror", [None, "x", "y"])
def test_rotated_and_mirrored_symbols_keep_pin_assignment(lab4, tmp_path, rotation, mirror):
    folder = next(path for path in sorted(lab4["export"].iterdir()) if (path / "candidate.kicad_sch").is_file())
    work = tmp_path / "w"
    shutil.copytree(folder, work, ignore=shutil.ignore_patterns("*.raw", "*.npz", "*.csv", "*.svg"))
    lines = _lines(work)
    _orient(lines, "U1", "OPAMP", rotation, mirror)
    _save(work, lines)
    result = import_kicad_schematic(work / "candidate.kicad_sch", spec=lab4["spec"])
    assert result.ok and result.circuit == json.loads((work / "circuit.json").read_text())


def test_rotated_resistor_and_rewired_pin(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _orient(lines, "R1", "R", 90, "x")
    _relabel(lines, "R1", 2, "tap")  # R1 pin 2 now lands on a new net, C1 pin 1 stays on "out"
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.circuit["components"][0]["nodes"] == ["in", "tap"]
    assert result.validation_error and "tap" in result.validation_error  # dangling: evaluator rejects it
    assert any("R1 pin 2: net out -> tap" in c["text"] for c in result.changes)
    lines = _lines(folder)
    _relabel(lines, "R1", 2, "out")
    _save(folder, lines)
    assert _rc_import(folder, spec).changes == []


def test_added_stock_resistor_with_embedded_definition(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _add_stock(lines, "Device:R", "R9", "10k", 200, 150, ("out", "0"))
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.ok, result.problems()
    added = result.circuit["components"][-1]
    assert (added["ref"], added["kind"], added["value"], added["nodes"], added["package"]) == (
        "R9", "R", 10000.0, ["out", "0"], None)
    assert [c["type"] for c in result.changes] == ["added"]


def test_stock_small_variants_and_unknown_symbols(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _add_stock(lines, "Device:C_Small", "C9", "10n", 200, 150, ("in", "0"))
    _save(folder, lines)
    assert _rc_import(folder, spec).circuit["components"][-1]["value"] == pytest.approx(1e-8)
    lines = _lines(folder)
    lines = [line.replace("Device:C_Small", "Device:C_Polarized") for line in lines]
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match="not supported"):
        _rc_import(folder, spec)


@pytest.mark.parametrize("edit,message", [
    (lambda t: t.replace('(lib_id "SpiceHarness:R")', '(lib_id "Device:D")', 1), "Device:D"),
    (lambda t: t.replace("(sheet_instances", '(sheet (at 10 10) (size 20 20))\n(sheet_instances'), "hierarchical"),
    (lambda t: t.replace("(sheet_instances", '(bus (pts (xy 1 1) (xy 5 1)))\n(sheet_instances'), "bus"),
    (lambda t: t.replace("(sheet_instances", '(hierarchical_label "x" (at 1 1 0))\n(sheet_instances'), "hierarchical"),
    (lambda t: t.replace("(sheet_instances", '(frobnicate 1)\n(sheet_instances'), "frobnicate"),
    (lambda t: t.replace("(sheet_instances", '(label "Bad-Name" (at 1 1 0))\n(sheet_instances'), "not a valid node name"),
    (lambda t: t.replace("(unit 1)", "(unit 2)", 1), "multi-unit"),
])
def test_unsupported_content_is_a_clear_error(rc, edit, message):
    folder, spec, _ = rc
    path = folder / "candidate.kicad_sch"
    path.write_text(edit(path.read_text()))
    with pytest.raises(KiCadImportError, match=message):
        _rc_import(folder, spec)


def test_unknown_model_and_ideal_substitution_are_errors(tmp_path):
    from spiceharness.kicad_export import render_kicad_schematic

    circuit = {"name": "buf", "components": [
        {"ref": "U1", "kind": "OPAMP", "nodes": ["a", "b", "vcc", "vee", "b"], "model": "LM741"},
        {"ref": "R1", "kind": "R", "nodes": ["a", "0"], "value": 1000}]}
    path = tmp_path / "candidate.kicad_sch"
    path.write_text(render_kicad_schematic(circuit))
    assert import_kicad_schematic(path).circuit["components"][0]["model"] == "LM741"
    text = path.read_text().replace('"LM741"', '"TL072"').replace('"MISSING_MODEL_LM741', '"x')
    path.write_text(text)
    with pytest.raises(KiCadImportError, match="trusted catalog"):
        import_kicad_schematic(path)


def test_no_connect_flag_on_a_used_pin_and_missing_connection(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    start, _ = _block(lines, "R1")
    x, y = (float(v) for v in re.search(r"\(at ([\d.]+) ([\d.]+) 0\)", lines[start]).groups())
    lines.insert(len(lines) - 1, f"(no_connect (at {x - 7.62} {y}) (uuid 00000000-0000-4000-8007-000000000001))")
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match="no-connect"):
        _rc_import(folder, spec)
    lines = _lines(folder)
    stub = _stub_lines(lines, "R1", 1)
    del lines[stub[0]:stub[0] + 2]
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match="not connected|no-connect"):
        _rc_import(folder, spec)


# ----------------------------------------------------------------------------- fixtures stay frozen


def test_edited_fixture_is_rejected_precisely(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "RLOAD", "Value", "5000")
    _set(lines, "RLOAD", "Sim.Params", "r=5000")
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.fixture_ok is False and not result.ok
    assert "fixture RLOAD" in result.fixture_errors[0] and "frozen load is 10 kΩ" in result.fixture_errors[0]
    with pytest.raises(KiCadImportError, match="RLOAD"):
        result.require_ok()


def test_edited_source_waveform_and_rewired_fixture_are_rejected(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "signal", "Sim.Params", "dc=1 ac=0 ampl=0.9 f=1000 td=0")
    _relabel(lines, "RSOURCE", 2, "out")
    _save(folder, lines)
    result = _rc_import(folder, spec)
    text = "\n".join(result.fixture_errors)
    assert result.fixture_ok is False
    assert "fixture signal" in text and "ampl" in text
    assert "fixture RSOURCE pin 2 is on net 'out' but the frozen bench requires 'in'" in text


def test_deleted_and_extra_fixture_are_rejected(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    start, end = _block(lines, "RLOAD")
    del lines[start:end + 1 + 6]
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.fixture_ok is False
    assert any("RLOAD was deleted" in item for item in result.fixture_errors)
    lines = _lines(folder)
    _set(lines, "R1", "IR.Category", "Bench fixture")
    _save(folder, lines)
    assert any("R1 is not part of the frozen bench" in item for item in _rc_import(folder, spec).fixture_errors)


def test_new_source_symbol_is_rejected(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    start, _ = _block(lines, "R1")
    lines[start] = lines[start].replace("SpiceHarness:R", "SpiceHarness:V")
    for i in range(start, _block(lines, "R1")[1]):
        if lines[i].startswith('(property "IR.Category"'):
            lines[i] = ""
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match="frozen bench fixtures"):
        _rc_import(folder, spec)


def test_shorted_source_is_rejected_by_the_existing_validator(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    # A pin-less wire across the source's nodes: exporter-style ideal wire "source" -- "0".
    lines.insert(len(lines) - 1, '(wire (pts (xy 20 330) (xy 60 330)) (stroke (width 0) (type default)) '
                                 '(uuid 00000000-0000-4000-8008-000000000001))')
    lines.insert(len(lines) - 1, '(label "source" (at 20 330 0) (effects (font (size 1 1))) (uuid 00000000-0000-4000-8009-000000000001))')
    lines.insert(len(lines) - 1, '(label "0" (at 60 330 0) (effects (font (size 1 1))) (uuid 00000000-0000-4000-8009-000000000002))')
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.circuit["connections"] == [["source", "0"]]
    assert not result.ok and "short-circuit" in result.validation_error


def test_wiring_that_joins_fixture_nets_is_reported(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    label = _stub_lines(lines, "C1_intrinsic", 1)[-1]
    x, y = (float(v) for v in re.search(r"\(label \"out\" \(at ([\d.]+) ([\d.]+)", lines[label]).groups())
    lines[label + 1:label + 1] = [_wire(x, y, x + 5, y),
                                  f'(label "in" (at {x + 5} {y} 0) (effects (font (size 1 1))))']
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match="distinct terminals") as info:  # R1 now sits across a short
        _rc_import(folder, spec)
    assert "were joined with it" in str(info.value) and "frozen bench requires" in str(info.value)


# ----------------------------------------------------------------------------- connectivity rules


def test_label_connection_round_trip_add_and_remove(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    end = next(i for i, line in enumerate(lines) if line.startswith("(sheet_instances"))
    lines[end:end] = ['(wire (pts (xy 25 340) (xy 65 340)) (stroke (width 0) (type default)) (uuid 00000000-0000-4000-8008-000000000002))',
                      '(label "in" (at 25 340 0) (effects (font (size 1 1))) (uuid 00000000-0000-4000-8009-000000000003))',
                      '(label "tap" (at 65 340 0) (effects (font (size 1 1))) (uuid 00000000-0000-4000-8009-000000000004))']
    _save(folder, lines)
    result = _rc_import(folder, spec)
    assert result.circuit["connections"] == [["in", "tap"]]
    assert result.changes[0]["type"] == "connection_added"
    assert "unconnected modeled terminal" in (result.validation_error or "")


def _two_resistors(extra: list[str]) -> str:
    from spiceharness.kicad_export import _library_symbol

    def part(ref: str, x: float, y: float) -> str:
        return (f'(symbol (lib_id "SpiceHarness:R") (at {x} {y} 0) (unit 1) (uuid 00000000-0000-4000-8000-0000000000{ref[-1]}0)\n'
                f'(property "Reference" "{ref}" (at 0 0 0)) (property "Value" "1k" (at 0 0 0)))')
    return ("(kicad_sch (version 20250114) (generator x)\n(lib_symbols " + _library_symbol("R") + ")\n"
            + part("R1", 42.38, 50) + "\n" + part("R2", 77.62, 50) + "\n" + "\n".join(extra) + "\n)\n")


def _wire(x1, y1, x2, y2) -> str:
    return f"(wire (pts (xy {x1} {y1}) (xy {x2} {y2})) (stroke (width 0) (type default)))"


_LABELS = ['(label "L" (at 34.76 50 0))', '(label "M" (at 85.24 50 0))']
_DOWN = [_wire(50, 50, 50, 60), _wire(70, 50, 70, 60)]
_BELOW = _wire(40, 60, 80, 60)
_CROSS = [_wire(50, 50, 50, 70), _wire(70, 50, 70, 70)]


@pytest.mark.parametrize("extra,joined", [
    ([_wire(50, 50, 60, 50), _wire(60, 50, 70, 50)], True),                       # end to end
    ([_wire(50, 50, 60, 50)], False),                                              # does not reach
    ([*_DOWN, _BELOW], False),                                                     # T without junction (KiCad 10)
    ([*_DOWN, _BELOW, "(junction (at 50 60))", "(junction (at 70 60))"], True),    # T with junctions
    ([*_CROSS, _BELOW], False),                                                    # crossing without junction
    ([*_CROSS, _BELOW, "(junction (at 50 60))", "(junction (at 70 60))"], True),   # crossing with junctions
    ([_wire(50, 50, 60, 50), _wire(60, 50, 60, 50), _wire(70, 50, 60, 50)], True),  # zero-length wire is inert
    (['(label "X" (at 50 50 0))', '(label "X" (at 70 50 0))'], True),              # labels on pins, same text
    (['(label "X" (at 50 50 0))', '(label "Y" (at 70 50 0))'], False),
    (['(global_label "X" (shape input) (at 50 50 0))', '(label "X" (at 70 50 0))'], True),
])
def test_wire_joining_rules(tmp_path, extra, joined):
    path = tmp_path / "j.kicad_sch"
    path.write_text(_two_resistors([*_LABELS, *extra]))
    try:
        result = import_kicad_schematic(path)
    except KiCadImportError as exc:
        assert not joined and "not connected to anything" in str(exc)
        return
    assert (result.pin_nets[("R1", "2")] == result.pin_nets[("R2", "1")]) is joined


def test_missed_t_junction_is_warned_about(tmp_path):
    path = tmp_path / "j.kicad_sch"
    path.write_text(_two_resistors([*_LABELS, *_DOWN, _BELOW, '(label "A" (at 50 50 0))', '(label "A" (at 70 50 0))']))
    result = import_kicad_schematic(path)
    assert any("touches the middle of a wire without a junction" in w for w in result.warnings)


def test_power_symbol_and_ground_aliases_are_node_zero(tmp_path):
    power = ('(symbol "power:GND" (power global) (pin_names (offset 0) (hide yes))\n'
             '(property "Value" "GND" (at 0 0 0))\n(symbol "GND_1_1" (pin power_in line (at 0 0 270) (length 0) '
             '(name "~") (number "1"))))')
    text = _two_resistors(['(label "L" (at 34.76 50 0))', '(label "M" (at 85.24 50 0))', '(label "Gnd" (at 70 50 0))',
                           ('(symbol (lib_id "power:GND") (at 50 50 0) (unit 1) (uuid 00000000-0000-4000-8000-000000000099)\n'
                            '(property "Reference" "#PWR01" (at 0 0 0)) (property "Value" "GND" (at 0 0 0)))')])
    path = tmp_path / "p.kicad_sch"
    path.write_text(text.replace("(lib_symbols ", "(lib_symbols " + power + "\n", 1))
    result = import_kicad_schematic(path)
    assert result.pin_nets[("R1", "2")] == result.pin_nets[("R2", "1")] == "0"


def test_labels_differing_only_in_case_are_an_error(tmp_path):
    path = tmp_path / "c.kicad_sch"
    path.write_text(_two_resistors([*_LABELS, '(label "Out" (at 50 50 0))', '(label "out" (at 70 50 0))']))
    with pytest.raises(KiCadImportError, match="differ only in case"):
        import_kicad_schematic(path)


# ----------------------------------------------------------------------------- rescore CLI and MCP


@real_spice
def test_rescore_cli_end_to_end_with_real_ngspice(tmp_path):
    workspace = tmp_path / "ws"
    manager = ExperimentManager(workspace)
    spec = ExperimentSpec.model_validate_json((RC / "spec.json").read_text())
    started = manager.start_experiment(spec, execution_mode="direct")
    manager.evaluate_experiment(started["run_id"], GeneralCircuitIR.model_validate_json((RC / "reference.json").read_text()))
    manager.finalize_experiment(started["run_id"])
    run_id = started["run_id"]
    before = (manager.directory / run_id / "run.json").read_bytes()
    export = _export(workspace, run_id, tmp_path / "export")
    candidate = next(path for path in export.iterdir() if (path / "candidate.kicad_sch").is_file()
                     and json.loads((path / "evidence.json").read_text())["purpose"] == "search")
    lines = _lines(candidate)
    _set(lines, "R1", "Value", "2.2k")
    _set(lines, "R1", "Sim.Params", "r=2.2k")
    _save(candidate, lines)
    result = runner.invoke(app, ["experiment", "rescore-kicad", run_id[:8], str(candidate / "candidate.kicad_sch"),
                                 "-w", str(workspace), "--json"])
    report = json.loads(result.stdout)
    assert result.exit_code == 1 and report["passed"] is False  # 2.2k breaks the frozen cutoff band
    assert report["new_run_id"] != run_id and report["original_run_id"] == run_id
    assert report["valid"] is True and report["feasible"] is False and report["verified"] is True
    assert [c["text"] for c in report["changes"]] == ["R1.value: 1 kΩ -> 2.2 kΩ"]
    assert {row["metric"]: row["passed"] for row in report["checks"]}["cutoff_hz"] is False
    assert report["placement"]["of"] == 2 and report["original_winner"]["name"] == "loaded_rc_reference"
    new = manager.get_experiment(report["new_run_id"])
    assert new["status"] == "completed" and new["result"]["verified"] is True
    assert new["fixture_fingerprint"] == manager.get_experiment(run_id)["fixture_fingerprint"]
    assert (manager.directory / run_id / "run.json").read_bytes() == before  # original run untouched
    provenance = json.loads((manager.directory / report["new_run_id"] / "kicad_import.json").read_text())
    assert provenance["original_run_id"] == run_id and provenance["schematic_sha256"] == report["schematic"]["sha256"]
    assert any(event["type"] == "kicad_rescore_import" for event in new["events"])
    text = runner.invoke(app, ["experiment", "rescore-kicad", run_id, str(candidate / "candidate.kicad_sch"),
                               "-w", str(workspace)])
    assert "On the original leaderboard this edit would place" in text.output and "FAIL" in text.output
    assert "R1.value: 1 kΩ -> 2.2 kΩ" in text.output and "does NOT meet" in text.output
    # an unedited re-import passes the frozen requirements
    lines = _lines(candidate)
    _set(lines, "R1", "Value", "1000")
    _set(lines, "R1", "Sim.Params", "r=1000")
    _save(candidate, lines)
    ok = runner.invoke(app, ["experiment", "rescore-kicad", run_id, str(candidate / "candidate.kicad_sch"),
                             "-w", str(workspace)])
    assert ok.exit_code == 0 and "the edited circuit passes" in ok.output


def test_rescore_rejects_edited_fixture_without_creating_a_run(tmp_path, rc):
    folder, spec, _ = rc
    workspace = tmp_path / "ws"
    manager = ExperimentManager(workspace)
    run = manager.start_experiment(ExperimentSpec.model_validate(spec), execution_mode="direct")
    lines = _lines(folder)
    _set(lines, "RLOAD", "Sim.Params", "r=1")
    _set(lines, "RLOAD", "Value", "1")
    _save(folder, lines)
    result = runner.invoke(app, ["experiment", "rescore-kicad", run["run_id"], str(folder / "candidate.kicad_sch"),
                                 "-w", str(workspace)])
    assert result.exit_code == 1 and "RLOAD" in result.output and "nothing was started" in result.output
    assert len(list(manager.directory.iterdir())) == 1


def test_rescore_refuses_template_design_runs(tmp_path, rc):
    folder, _, _ = rc
    run_id = RunStore(tmp_path / ".spiceharness" / "runs.sqlite3").create_run(DesignSpec.model_validate(
        json.loads((ROOT / "examples/lowpass.json").read_text())))
    result = runner.invoke(app, ["experiment", "rescore-kicad", run_id, str(folder / "candidate.kicad_sch"),
                                 "-w", str(tmp_path)])
    assert result.exit_code == 1 and "only general experiment runs are supported" in result.output


@real_spice
async def test_mcp_import_tool_is_read_only_and_path_safe(tmp_path):
    workspace = tmp_path / "ws"
    manager = ExperimentManager(workspace)
    spec = ExperimentSpec.model_validate_json((RC / "spec.json").read_text())
    started = manager.start_experiment(spec, execution_mode="direct")
    manager.evaluate_experiment(started["run_id"], GeneralCircuitIR.model_validate_json((RC / "reference.json").read_text()))
    manager.finalize_experiment(started["run_id"])
    run_id = started["run_id"]
    export = _export(workspace, run_id, tmp_path / "elsewhere" / "export")  # outside the workspace
    candidate = next(path for path in export.iterdir() if (path / "candidate.kicad_sch").is_file())
    lines = _lines(candidate)
    _set(lines, "R1", "Value", "2.2k")
    _set(lines, "R1", "Sim.Params", "r=2.2k")
    _save(candidate, lines)
    inside = workspace / "copy.kicad_sch"
    shutil.copy(candidate / "candidate.kicad_sch", inside)
    (workspace / "link.kicad_sch").symlink_to(candidate / "candidate.kicad_sch")
    stranger = tmp_path / "stranger.kicad_sch"
    shutil.copy(candidate / "candidate.kicad_sch", stranger)
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "spiceharness.mcp_server", "--workspace", str(workspace)],
                                   env={"PATH": os.environ["PATH"]})
    async with Client(params) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert tools["import_kicad_schematic"].annotations.read_only_hint is True
        guide = await client.read_resource("spiceharness://experiments")
        assert "import_kicad_schematic" in guide.contents[0].text
        for good in (candidate / "candidate.kicad_sch", inside):
            reply = await client.call_tool("import_kicad_schematic", {"run_id": run_id, "path": str(good)})
            assert not reply.is_error, reply
            data = reply.structured_content
            assert data["fixture_ok"] is True and data["circuit"]["components"][0]["value"] == 2200
            assert len(data["sha256"]) == 64
            if good != inside:  # only the export folder carries circuit.json for the change list
                assert data["changes"][0]["text"] == "R1.value: 1 kΩ -> 2.2 kΩ" and data["original_circuit_found"]
        for bad in (stranger, workspace / "link.kicad_sch", workspace / ".." / "stranger.kicad_sch",
                    workspace / "missing.kicad_sch", candidate / "circuit.json"):
            assert (await client.call_tool("import_kicad_schematic", {"run_id": run_id, "path": str(bad)})).is_error
        assert (await client.call_tool("import_kicad_schematic", {"run_id": "f" * 32, "path": str(inside)})).is_error
    assert not list(manager.directory.glob("*/kicad_import.json"))  # read-only: no run was created


# ----------------------------------------------------------------------------- kicad-cli agreement


def _kicad_partition(path: Path):
    """KiCad's own connectivity: net membership of (reference, pin), via a temporary on_board copy."""
    import tempfile

    from spiceharness.kicad_import import _kid, _kids

    with tempfile.TemporaryDirectory() as temporary:
        sch = Path(temporary) / "x.kicad_sch"
        sch.write_text(path.read_text().replace("(on_board no)", "(on_board yes)"))
        (Path(temporary) / "x.kicad_pro").write_text("{}")
        out = Path(temporary) / "x.net"
        done = subprocess.run([_KICAD, "sch", "export", "netlist", "--format", "kicadsexpr", "-o", str(out), str(sch)],
                              check=False, capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stdout + done.stderr
        nets: dict[str, set] = {}
        stack = [parse_sexpr(out.read_text())]
        while stack:
            for child in stack.pop():
                if isinstance(child, list):
                    if child and child[0] == "net":
                        name = re.sub(r"^/", "", _kid(child, "name")[1]).casefold()
                        name = "0" if name in ("gnd", "0") else name
                        nets.setdefault(name, set()).update(
                            (_kid(node, "ref")[1], _kid(node, "pin")[1]) for node in _kids(child, "node"))
                    else:
                        stack.append(child)
    return {frozenset(members) for members in nets.values()}


def _ours(result):
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in result.circuit["connections"]:
        parent[find(a)] = find(b)
    groups: dict[str, set] = {}
    for pin, node in result.pin_nets.items():
        groups.setdefault(find(node), set()).add(pin)
    return {frozenset(members) for members in groups.values()}


@needs_kicad
@real_spice
def test_importer_nets_match_kicad_cli_for_every_exported_candidate(lab4):
    folders = sorted(path for path in lab4["export"].iterdir() if (path / "candidate.kicad_sch").is_file())
    for folder in folders:
        result = import_kicad_schematic(folder / "candidate.kicad_sch", spec=lab4["spec"])
        assert _kicad_partition(folder / "candidate.kicad_sch") == _ours(result), folder.name


@needs_kicad
@real_spice
def test_importer_matches_kicad_cli_after_kicad_resave_and_synthetic_edits(lab4, rc, tmp_path):
    # KiCad rewrites the file (sorted items, new syntax); connectivity must be unchanged.
    folder = next(path for path in sorted(lab4["export"].iterdir()) if (path / "candidate.kicad_sch").is_file())
    saved = tmp_path / "saved"
    shutil.copytree(folder, saved, ignore=shutil.ignore_patterns("*.raw", "*.npz", "*.csv", "*.svg", "*.log"))
    subprocess.run([_KICAD, "sch", "upgrade", "--force", str(saved / "candidate.kicad_sch")], check=True,
                   capture_output=True, timeout=60)
    assert "(generator \"eeschema\")" in (saved / "candidate.kicad_sch").read_text()
    resaved = import_kicad_schematic(saved / "candidate.kicad_sch", spec=lab4["spec"])
    assert resaved.ok and resaved.circuit == json.loads((folder / "circuit.json").read_text())
    assert _kicad_partition(saved / "candidate.kicad_sch") == _ours(resaved)
    # orientation sweep on the op-amp
    for rotation in (0, 90, 180, 270):
        for mirror in (None, "x", "y"):
            work = tmp_path / f"o{rotation}{mirror}"
            shutil.copytree(folder, work, ignore=shutil.ignore_patterns("*.raw", "*.npz", "*.csv", "*.svg", "*.log"))
            lines = _lines(work)
            _orient(lines, "U1", "OPAMP", rotation, mirror)
            _save(work, lines)
            result = import_kicad_schematic(work / "candidate.kicad_sch", spec=lab4["spec"])
            assert _kicad_partition(work / "candidate.kicad_sch") == _ours(result), (rotation, mirror)
    # value edits, an added stock resistor and a rewire on the small circuit
    small, spec, _ = rc
    lines = _lines(small)
    _set(lines, "R1", "Value", "4k7")
    _set(lines, "R1", "Sim.Params", "r=4k7")
    _add_stock(lines, "Device:R", "R9", "1M", 200, 150, ("out", "0"))
    _orient(lines, "C1", "C", 270, "y")
    _save(small, lines)
    result = _rc_import(small, spec)
    assert result.ok and _kicad_partition(small / "candidate.kicad_sch") == _ours(result)


@needs_kicad
def test_values_agree_with_kicad_spice_export(rc, tmp_path):
    folder, spec, _ = rc
    lines = _lines(folder)
    values = {"R1": "4k7", "C1": "47µF"}
    for ref, value in values.items():
        _set(lines, ref, "Value", value)
        _set(lines, ref, "Sim.Params", f"{'r' if ref[0] == 'R' else 'c'}={value}")
    _add_stock(lines, "Device:R", "R9", "1M", 200, 150, ("out", "0"))
    _add_stock(lines, "Device:R", "R8", "2R2", 230, 150, ("out", "0"))
    _add_stock(lines, "Device:C_Small", "C9", "10n", 260, 150, ("out", "0"))
    _save(folder, lines)
    result = _rc_import(folder, spec)
    deck = tmp_path / "x.cir"
    (folder / "candidate.kicad_pro").write_text("{}")
    subprocess.run([_KICAD, "sch", "export", "netlist", "--format", "spice", "-o", str(deck),
                    str(folder / "candidate.kicad_sch")], check=True, capture_output=True, timeout=60)
    seen = 0
    for line in deck.read_text().splitlines():
        fields = line.split()
        if fields and fields[0][0] in "RCL" and fields[0] in result.symbol_values:
            kicad_value, _ = parse_si_value(fields[-1], fields[0][0])
            assert kicad_value == pytest.approx(result.symbol_values[fields[0]], rel=1e-12), line
            seen += 1
    assert seen == len(result.symbol_values) >= 6


# ----------------------------------------------------------------------------- Sim.Pins consistency


@pytest.fixture
def opamp(tmp_path):
    """Exported op-amp circuit whose LM741 binding carries the exporter's Sim.Pins mapping."""
    from spiceharness.kicad_export import render_kicad_schematic
    from spiceharness.simulator import MODEL_DIR

    circuit = json.loads((LAB4 / "direct.json").read_text())
    spec = json.loads((LAB4 / "spec.json").read_text())
    catalog = json.loads((MODEL_DIR / "catalog.json").read_text())
    folder = tmp_path / "op"
    folder.mkdir()

    def render(model: str, pins: list[str]) -> None:
        circuit["components"][0]["model"] = model
        binding = {**catalog[model], "library": "model_0.lib", "model_pins": pins}
        (folder / "candidate.kicad_sch").write_text(
            render_kicad_schematic(circuit, spec=spec, model_bindings={model: binding}))

    render("LM741", ["1", "2", "99", "50", "28"])
    (folder / "circuit.json").write_text(json.dumps(circuit))
    return folder, spec, render


def _pins_edit(folder: Path, value: str) -> None:
    lines = _lines(folder)
    _set(lines, "U1", "Sim.Pins", value)
    _save(folder, lines)


def test_untouched_export_with_sim_pins_imports_cleanly(opamp):
    folder, spec, _ = opamp
    assert "Sim.Pins" in (folder / "candidate.kicad_sch").read_text()
    assert _rc_import(folder, spec).require_ok().circuit["components"][0]["model"] == "LM741"


@pytest.mark.parametrize("value", ["1=2 2=1 3=99 4=50 5=28", "1=1 2=2", "1=1 2=2 3=99 4=50 4=28",
                                   "1=1 2=2 3=99 4=50 5=28 6=7", "1=1 1=2 3=99 4=50 5=28", ""])
def test_edited_sim_pins_on_opamp_are_a_hard_error(opamp, value):
    folder, spec, _ = opamp
    _pins_edit(folder, value)
    with pytest.raises(KiCadImportError, match=r"U1: Sim\.Pins"):
        _rc_import(folder, spec)


def test_swapped_inputs_message_names_expected_and_found(opamp):
    folder, spec, _ = opamp
    _pins_edit(folder, "1=2 2=1 3=99 4=50 5=28")
    with pytest.raises(KiCadImportError) as info:
        _rc_import(folder, spec)
    assert "1=1 2=2 3=99 4=50 5=28" in str(info.value) and "found '1=2 2=1" in str(info.value)


def test_stale_sim_pins_after_model_change_are_rejected(opamp, tmp_path):
    from spiceharness.simulator import MODEL_DIR

    folder, spec, render = opamp
    render("LM358", ["3", "2", "8", "4", "1"])  # a model with other pin names, then change back by hand
    lines = _lines(folder)
    _set(lines, "U1", "Sim.Pins", "1=1 2=2 3=99 4=50 5=28")  # stale LM741 mapping
    _save(folder, lines)
    expected_names = (MODEL_DIR / "LM358.lib").exists()
    if expected_names:
        with pytest.raises(KiCadImportError, match=r"U1: Sim\.Pins"):
            _rc_import(folder, spec)


def test_two_terminal_sim_pins_accept_either_order_but_not_garbage(rc):
    folder, spec, _ = rc
    lines = _lines(folder)
    _set(lines, "R1", "Sim.Pins", "1=2 2=1")
    _save(folder, lines)
    assert _rc_import(folder, spec).ok
    _set(lines, "R1", "Sim.Pins", "1=1")
    _save(folder, lines)
    with pytest.raises(KiCadImportError, match=r"Sim\.Pins"):
        _rc_import(folder, spec)


# ----------------------------------------------------------------------------- bounded import time


def _wires_only(lines: list[str]) -> bytes:
    body = "\n".join(lines)
    return (f'(kicad_sch (version 20250114) (generator "eeschema") (uuid "u") (paper "A4") (lib_symbols)\n'
            f'{body})').encode()


def test_many_collinear_wires_import_quickly():
    import time

    wires = [f"(wire (pts (xy {2 * i} 0) (xy {2 * i + 1} 0)) (stroke (width 0) (type default)))"
             for i in range(20000)]
    data = _wires_only(wires)
    best = float("inf")
    for _ in range(2):  # best of two: a loaded CI machine must not turn this into a flaky test
        start = time.perf_counter()
        import_kicad_bytes(data)
        best = min(best, time.perf_counter() - start)
    assert best < 2.0


def test_too_many_diagonal_wires_are_rejected_quickly():
    import time

    count = MAX_DIAGONAL_WIRES + 1
    wires = [f"(wire (pts (xy {i} 0) (xy {i + 1} 1)) (stroke (width 0) (type default)))" for i in range(count)]
    start = time.perf_counter()
    with pytest.raises(KiCadImportError, match="diagonal wires"):
        import_kicad_bytes(_wires_only(wires))
    assert time.perf_counter() - start < 2.0


def test_import_time_budget_is_enforced(monkeypatch):
    monkeypatch.setattr("spiceharness.kicad_import.IMPORT_TIME_BUDGET_S", -1.0)
    wires = [f"(wire (pts (xy {2 * i} 0) (xy {2 * i + 1} 0)) (stroke (width 0) (type default)))"
             for i in range(300)]
    junctions = [f"(junction (at {2 * i + 0.5} 0))" for i in range(300)]
    with pytest.raises(KiCadImportError, match="time budget"):
        import_kicad_bytes(_wires_only(wires + junctions))
