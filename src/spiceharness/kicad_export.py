"""Editable, self-contained KiCad simulation schematics from validated IR.

The embedded symbols are original drawings with canonical IR pin numbering.
References in this simulation view are unique per IR channel; IR.Package retains
physical package association. Exporting a schematic never changes saved scores.

Format and model-field references (KiCad 10):
https://dev-docs.kicad.org/en/file-formats/sexpr-schematic/
https://dev-docs.kicad.org/en/file-formats/sexpr-intro/
https://docs.kicad.org/10.0/en/eeschema/eeschema.html#simulator
Sim.Pins name semantics were checked against KiCad's sim_model_serializer.cpp
and sim_model_subckt.cpp in the official KiCad source mirror.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import PurePosixPath
from uuid import NAMESPACE_URL, uuid5

from .circuit_svg import _engineering
from .domain import Circuit, DesignSpec
from .experiments import ExperimentSource, ExperimentSpec, GeneralCircuitIR

_CANONICAL = {
    "OPAMP": ("noninverting", "inverting", "positive_supply", "negative_supply", "output"),
    "BJT": ("collector", "base", "emitter"),
    "DIODE": ("anode", "cathode"),
}
_PIN_NAMES = {"R": ("1", "2"), "C": ("1", "2"), "L": ("1", "2"),
              "OPAMP": ("+", "-", "V+", "V-", "OUT"), "BJT": ("C", "B", "E"),
              "DIODE": ("A", "K"), "V": ("+", "-"), "I": ("+", "-")}
_POSITIONS = {
    "OPAMP": ((-10.16, 2.54, 0, 5.08), (-10.16, -2.54, 0, 5.08),
              (-2.54, 7.62, 270, 3.81), (-2.54, -7.62, 90, 3.81), (10.16, 0, 180, 5.08)),
    "BJT": ((2.54, 7.62, 270, 3.81), (-10.16, 0, 0, 7.62), (2.54, -7.62, 90, 3.81)),
}
_TWO_PINS = ((-7.62, 0, 0, 2.54), (7.62, 0, 180, 2.54))


def _q(value: object) -> str:
    # Keep user text on one line so it cannot become a schematic SPICE directive.
    value = "".join(char if ord(char) >= 32 else " " for char in str(value))
    return json.dumps(value, ensure_ascii=False)


def _n(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".") if value else "0"


def _id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, "spiceharness:kicad:" + value))


def _effects(size: float = 1.0, *, hidden: bool = False, left: bool = False) -> str:
    return (f"(effects (font (size {_n(size)} {_n(size)}))"
            + (" (justify left)" if left else "") + (" hide" if hidden else "") + ")")


def _property(key: str, value: object, x: float, y: float, *, hidden: bool = True) -> str:
    return f"(property {_q(key)} {_q(value)} (at {_n(x)} {_n(y)} 0) {_effects(hidden=hidden, left=True)})"


def _poly(points: list[tuple[float, float]]) -> str:
    pts = " ".join(f"(xy {_n(x)} {_n(y)})" for x, y in points)
    return f"(polyline (pts {pts}) (stroke (width 0.254) (type default)) (fill (type none)))"


def _symbol_graphic(kind: str) -> str:
    if kind == "R":
        return _poly([(-5.08, 0), (-4.445, 1.27), (-3.175, -1.27), (-1.905, 1.27),
                      (-0.635, -1.27), (0.635, 1.27), (1.905, -1.27), (3.175, 1.27),
                      (4.445, -1.27), (5.08, 0)])
    if kind == "C":
        return "".join(_poly(points) for points in ([(-5.08, 0), (-1.27, 0)],
                                                  [(-1.27, -3.81), (-1.27, 3.81)],
                                                  [(1.27, -3.81), (1.27, 3.81)], [(1.27, 0), (5.08, 0)]))
    if kind == "L":
        return "".join(f"(arc (start {_n(x)} 0) (mid {_n(x + 1.27)} 1.8) "
                       f"(end {_n(x + 2.54)} 0) (stroke (width 0.254) (type default)) (fill (type none)))"
                       for x in (-5.08, -2.54, 0, 2.54))
    if kind == "OPAMP":
        return _poly([(-5.08, 5.08), (5.08, 0), (-5.08, -5.08), (-5.08, 5.08)])
    if kind == "BJT":
        return ('(circle (center 0 0) (radius 5.08) (stroke (width 0.254) (type default)) '
                '(fill (type none)))' + "".join(_poly(points) for points in (
                    [(-2.54, -2.54), (-2.54, 2.54)], [(-2.54, 1.27), (2.54, 3.81)],
                    [(-2.54, -1.27), (2.54, -3.81)])))
    if kind == "DIODE":
        return "".join(_poly(points) for points in (
            [(-5.08, 0), (-2.54, 0)], [(-2.54, 2.54), (2.54, 0), (-2.54, -2.54), (-2.54, 2.54)],
            [(2.54, -2.54), (2.54, 2.54)], [(2.54, 0), (5.08, 0)]))
    body = ('(circle (center 0 0) (radius 3.81) (stroke (width 0.254) (type default)) '
            '(fill (type none)))' + _poly([(-5.08, 0), (-3.81, 0)]) + _poly([(3.81, 0), (5.08, 0)]))
    if kind == "V":
        return body + _poly([(-2.54, 0), (-0.635, 0)]) + _poly([(-1.5875, -0.9525), (-1.5875, 0.9525)]) + _poly([(0.635, 0), (2.54, 0)])
    return body + _poly([(-2.54, 0), (2.54, 0), (1.27, 0.9525)]) + _poly([(2.54, 0), (1.27, -0.9525)])


def _library_symbol(kind: str) -> str:
    pins = _PIN_NAMES[kind]
    positions = _POSITIONS.get(kind, _TWO_PINS)
    definition = [(f"(symbol {_q('SpiceHarness:' + kind)} (pin_names (offset 0.635)) "
                   "(in_bom yes) (on_board no)")]
    for key, value in (("Reference", {"OPAMP": "U", "BJT": "Q", "DIODE": "D"}.get(kind, kind)),
                       ("Value", kind), ("Footprint", ""), ("Datasheet", "")):
        definition.append(_property(key, value, 0, 0))
    definition.append(f"(symbol {_q(kind + '_0_1')} {_symbol_graphic(kind)})")
    definition.append(f"(symbol {_q(kind + '_1_1')}")
    for i, (name, (x, y, angle, length)) in enumerate(zip(pins, positions), start=1):
        electrical = "passive"
        if kind == "OPAMP":
            electrical = "input" if i <= 2 else "power_in" if i <= 4 else "output"
        definition.append(f"(pin {electrical} line (at {_n(x)} {_n(y)} {angle}) (length {_n(length)}) "
                          f"(name {_q(name)} {_effects(0.9)}) (number {_q(i)} {_effects(0.8)}))")
    definition.append("))")
    return "\n".join(definition)


def model_pin_names(model_text: str, spice_name: str) -> list[str]:
    """Read names, without executing code, from an already trusted model file.

    KiCad Sim.Pins maps symbol numbers to model *names*. The model names may be
    1/2/99/50/28, for example, rather than the five canonical IR pin numbers.
    """
    logical = []
    for line in model_text.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+") and logical:
            logical[-1] += " " + line[1:]
        else:
            logical.append(line)
    for line in logical:
        fields = line.split()
        if len(fields) < 3 or fields[1].casefold() != spice_name.casefold():
            continue
        if fields[0].lower() == ".subckt":
            pins = []
            for token in fields[2:]:
                if token.lower() in ("params:", "param:") or "=" in token or token.startswith((";", "$")):
                    break
                if not re.fullmatch(r"[A-Za-z0-9_+.-]+", token):
                    raise ValueError("unsupported model pin token")
                pins.append(token)
            if not pins or len(set(pins)) != len(pins):
                raise ValueError("model subcircuit pins must be nonempty and unique")
            return pins
        if fields[0].lower() == ".model":
            device = fields[2].split("(", 1)[0].upper()
            if device in ("NPN", "PNP"):
                return ["C", "B", "E"]
            if device == "D":
                return ["A", "K"]
    raise ValueError(f"no supported model definition found for {spice_name}")


def _model_properties(part: dict, bindings: dict) -> tuple[dict, str | None]:
    model = part["model"]
    binding = bindings.get(model)
    missing = {"Sim.Device": "SUBCKT", "Sim.Name": model,
               "Sim.Library": f"models/MISSING_MODEL_{model}.lib"}
    if not binding:
        return missing, f"MODEL UNBOUND: {model}; simulation must fail until the original model is supplied"
    library = str(binding.get("library", binding.get("path", "")))
    path = PurePosixPath(library)
    if not library or path.is_absolute() or ".." in path.parts or re.search(r'[\\\r\n"]', library):
        raise ValueError("KiCad model libraries must be safe bundle-relative paths")
    spice_name = binding.get("spice_name", model)
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,95}", spice_name):
        raise ValueError("unsafe model name")
    canonical = _CANONICAL[part["kind"]]
    order = binding.get("pins", canonical)
    if all(isinstance(pin, int) for pin in order):
        indices = list(order)
    else:
        try:
            indices = [canonical.index(pin) for pin in order]
        except ValueError as exc:
            raise ValueError("catalog model pins do not match canonical IR pins") from exc
    if sorted(indices) != list(range(len(canonical))):
        raise ValueError("catalog model pin order must be a complete permutation")
    names = binding.get("model_pins")
    if names is None and part["kind"] != "OPAMP":
        names = {"BJT": ["C", "B", "E"], "DIODE": ["A", "K"]}[part["kind"]]
    if names is None:
        return missing, f"MODEL UNBOUND: {model}; actual subcircuit pin names unavailable"
    if len(names) != len(indices) or any(not re.fullmatch(r"[A-Za-z0-9_+.-]+", name) for name in names):
        raise ValueError("model pin names do not match catalog pin count")
    mapping = sorted((index + 1, name) for index, name in zip(indices, names))
    return {"Sim.Library": library, "Sim.Name": spice_name,
            "Sim.Pins": " ".join(f"{index}={name}" for index, name in mapping)}, None


def _source_properties(source: dict, *, ac: bool) -> dict:
    waveform = source["waveform"]
    params = {"dc": source["dc"], "ac": 1 if ac else 0}
    if waveform == "sine":
        params.update(ampl=source["amplitude"], f=source["frequency_hz"], td=source["delay_s"])
    if waveform == "pulse":
        prefix = "y"
        params = {prefix + "1": source["dc"], prefix + "2": source["dc"] + source["amplitude"],
                  "td": source["delay_s"], "tr": source["rise_s"], "tf": source["fall_s"],
                  "tw": source["width_s"], "per": source["period_s"], "ac": 1 if ac else 0}
    return {"Sim.Device": source["kind"], "Sim.Type": {"dc": "DC", "sine": "SIN", "pulse": "PULSE"}[waveform],
            "Sim.Pins": "1=+ 2=-", "Sim.Params": " ".join(f"{key}={value:.17g}" for key, value in params.items())}


def _validated(circuit: dict, spec: dict | None) -> tuple[dict, dict | None, list[dict]]:
    if "topology" in circuit:
        ir = Circuit.model_validate(circuit)
        frozen = DesignSpec.model_validate(spec) if spec is not None else ir.spec
        data = ir.model_dump(mode="json")
        fixture = []
        if frozen:
            values = frozen.model_dump(mode="json")
            for ref, nodes, dc, wave, ampl in (
                ("positive_supply", ("vcc", "0"), frozen.supply_positive_v, "dc", 0),
                ("negative_supply", ("vee", "0"), frozen.supply_negative_v, "dc", 0),
                ("input_source", ("source", "0"), frozen.input_offset_v, "sine", frozen.input_amplitude_v),
            ):
                fixture.append(ExperimentSource(ref=ref, nodes=nodes, dc=dc, waveform=wave,
                                                amplitude=ampl, frequency_hz=frozen.input_frequency_hz).model_dump())
            fixture.extend((
                {"ref": "source_impedance", "kind": "R", "nodes": ["source", ir.input_node.lower()],
                 "value": max(frozen.source_resistance_ohm, 1e-6)},
                {"ref": "output_load", "kind": "R", "nodes": [ir.output_node.lower(), "0"], "value": frozen.load_ohm},
            ))
            return data, values, fixture
        return data, None, fixture
    ir = GeneralCircuitIR.model_validate(circuit)
    frozen = ExperimentSpec.model_validate(spec) if spec is not None else None
    return (ir.model_dump(mode="json"), frozen.model_dump(mode="json") if frozen else None,
            [*(source.model_dump() for source in frozen.sources), *(part.model_dump() for part in frozen.loads)]
            if frozen else [])


def _default_analysis(ir: dict, spec: dict) -> str:
    """Choose one evaluator-owned default; other saved analysis decks stay separate."""
    if "topology" in ir:
        period = 1 / spec["input_frequency_hz"]
        settle = max(12 * period, 12 / (2 * math.pi * (spec.get("cutoff_hz") or spec["input_frequency_hz"])))
        return f".tran {period / 160:.17g} {settle + 8 * period:.17g} {settle:.17g} {period / 80:.17g}"
    measurements = spec["measurements"]
    if spec.get("transient") and any(m["operator"].startswith("transient_") for m in measurements):
        tran = spec["transient"]
        return f".tran {tran['step_s']:.17g} {tran['stop_s']:.17g} 0 {tran['step_s']:.17g}"
    if any(m["operator"].startswith("ac_") or m["operator"] == "integrated_noise_rms" for m in measurements):
        sweep = spec["sweep"]
        stop = sweep["stop_hz"] * 10 ** (1 / sweep["points_per_decade"])
        return f".ac dec {sweep['points_per_decade']} {sweep['start_hz']:.17g} {stop:.17g}"
    return ".op"


def render_kicad_schematic(circuit: dict, *, spec: dict | None = None,
                           model_bindings: dict | None = None, project_name: str = "candidate") -> str:
    """Return editable .kicad_sch text with embedded original symbol drawings.

    ``model_bindings`` is a trusted catalog map augmented with bundle-relative
    ``library`` paths and actual ``model_pins`` from :func:`model_pin_names`.
    Missing active models remain visible with an explicitly missing library;
    simulation must fail until bound, never omit or idealize them. Fixtures and intrinsic loss
    resistors are excluded from the schematic BOM. Saved score/package counts
    remain authoritative, including shared opamp channels.
    """
    ir, frozen, fixture = _validated(circuit, spec)
    model_bindings = model_bindings or {}
    root = _id(json.dumps([ir, frozen, model_bindings, project_name], sort_keys=True))
    objects = []
    refs: dict[str, int] = {}
    used_nodes = {node.lower() for p in [*ir["components"], *fixture] for node in p["nodes"]}
    used_nodes.update(node.lower() for pair in ir.get("connections", ()) for node in pair)
    for index, original in enumerate(ir["components"]):
        part = dict(original)
        # SPICE is case-insensitive, whereas KiCad label connectivity is not.
        part["nodes"] = [node.lower() for node in original["nodes"]]
        part["_category"] = "DUT"
        part["_original_nodes"] = list(original["nodes"])
        loss = part.get("series_resistance_ohm", 0)
        if loss:
            internal = f"SH_INTRINSIC_{index + 1}"
            while internal.lower() in used_nodes:
                internal += "_"
            used_nodes.add(internal.lower())
            part["nodes"][0] = internal
            objects.append({"ref": f"{part['ref']}_intrinsic", "kind": "R", "nodes": [original["nodes"][0], internal],
                            "value": loss, "_category": "Intrinsic loss", "_parent": part["ref"],
                            "package": part.get("package") or part["ref"]})
        objects.append(part)
    for part in fixture:
        objects.append({**part, "_category": "Bench fixture"})
    columns = min(8, max(2, math.ceil(math.sqrt(max(len(objects), 1)))))
    rows = math.ceil(len(objects) / columns)
    cell_w, cell_h = 110, 54
    page_width = max(297, columns * cell_w + 35)
    page_height = max(210, rows * cell_h + 125 + len(ir.get("connections", ())) * 8)
    text = [f"(kicad_sch (version 20250114) (generator spiceharness) (uuid {root})",
            f'(paper "User" {page_width} {page_height})',
            f'(title_block (title {_q(ir.get("name", "Circuit"))}) (comment 1 "Validated Circuit IR simulation view"))',
            '(lib_symbols ' + "\n".join(_library_symbol(kind) for kind in sorted({p["kind"] for p in objects})) + ')']
    serial = 0

    def uid(category: str) -> str:
        nonlocal serial
        serial += 1
        return _id(f"{root}:{serial}:{category}")

    def note(value: str, x: float, y: float, *, size: float = 1.1) -> None:
        text.append(f"(text {_q(value)} (at {_n(x)} {_n(y)} 0) {_effects(size, left=True)} (uuid {uid('text')}))")

    def label(value: str, x: float, y: float, *, to_left: bool = False) -> None:
        effects = _effects(0.9, left=True)
        if to_left:
            effects = effects.replace("justify left", "justify right")
        text.append(f"(label {_q(value)} (at {_n(x)} {_n(y)} 0) {effects} (uuid {uid('label')}))")

    def wire(x1: float, y1: float, x2: float, y2: float) -> None:
        text.append(f"(wire (pts (xy {_n(x1)} {_n(y1)}) (xy {_n(x2)} {_n(y2)})) "
                    f"(stroke (width 0) (type default)) (uuid {uid('wire')}))")

    packages = {str(p.get("package") or p["ref"]).casefold() for p in ir["components"]}
    note(f"Circuit: {ir.get('name', 'candidate')}", 15, 12, size=1.8)
    note(f"DUT: {len(ir['components'])} IR components / {len(packages)} package groups. "
         "Bench fixtures and intrinsic loss elements are marked separately.", 15, 18)
    note("Matching net labels connect. Every symbol pin is canonical IR order; model mapping uses actual model pin names.", 15, 24)
    note("One symbol per IR channel. IR.Package groups physical devices; saved evaluator scores remain authoritative.", 15, 30)
    if "topology" in ir:
        note("SPICE net names use lowercase labels; original pin spelling is retained in IR.OrderedNodes.", 15, 36)
    first_ac = "input_source" if "topology" in ir else (
        fixture[0]["ref"] if fixture and fixture[0]["kind"] in ("V", "I") else None)
    if frozen and "measurements" in frozen:
        first_ac = next((m["source"] for m in frozen["measurements"]
                         if m["operator"].startswith("ac_") and m.get("source")), first_ac)
        if not any(m["operator"].startswith("ac_") for m in frozen["measurements"]) and frozen["noise_sources"]:
            first_ac = frozen["noise_sources"][0]["source"]
        note(f"Native AC excitation: {first_ac}. Saved per-source evaluator decks cover other excitations.", 15, 36)
    if frozen:
        # Only evaluator-owned directives derived from validated typed settings.
        note(f".temp {frozen['temperature_c']:.17g}", 15, 42, size=0.85)
        note(".options numdgt=15 reltol=1e-4 method=gear", 15, 47, size=0.85)
        note(_default_analysis(ir, frozen), 15, 52, size=0.85)
        note("Native default analysis. Saved per-source decks and noise integration remain the evaluator evidence.",
             15, 57, size=0.85)
    for i, part in enumerate(objects):
        kind = part["kind"]
        x = 55 + (i % columns) * cell_w
        y = 85 + (i // columns) * cell_h
        prefix = {"OPAMP": "U", "BJT": "Q", "DIODE": "D"}.get(kind, kind)
        refs[prefix] = refs.get(prefix, 0) + 1
        ref = f"{prefix}{refs[prefix]}"
        native_id = uid("symbol")
        category = part["_category"]
        model = part.get("model")
        warning = None
        if model:
            props, warning = _model_properties(part, model_bindings)
            display = model
        elif kind in ("V", "I"):
            props = _source_properties(part, ac=part["ref"].casefold() == str(first_ac).casefold())
            display = f"{part['dc']:.17g}"
        else:
            display = f"{part['value']:.17g}"
            props = {"Sim.Device": kind, "Sim.Type": "", "Sim.Pins": "1=1 2=2",
                     "Sim.Params": f"{kind.lower()}={display}"}
        text.append(f'(symbol (lib_id "SpiceHarness:{kind}") (at {_n(x)} {_n(y)} 0) (unit 1) '
                    '(exclude_from_sim no) '
                    f'(in_bom {"yes" if category == "DUT" else "no"}) (on_board no) (dnp no) (uuid {native_id})')
        text.extend((_property("Reference", ref, x - 5, y - 20, hidden=False),
                     _property("Value", display, x + 5, y - 20, hidden=False),
                     _property("Footprint", "", x, y), _property("Datasheet", "", x, y),
                     _property("IR.Reference", part["ref"], x, y), _property("IR.Category", category, x, y),
                     _property("IR.Package", part.get("package") or part.get("_parent") or part["ref"], x, y),
                     _property("IR.OrderedNodes", json.dumps(part.get("_original_nodes", part["nodes"])), x, y)))
        for key, value in props.items():
            text.append(_property(key, value, x, y))
        for number in range(1, len(part["nodes"]) + 1):
            text.append(f'(pin "{number}" (uuid {uid("pin")}))')
        text.append(f'(instances (project {_q(project_name)} (path "/{root}" (reference {_q(ref)}) (unit 1)))))')
        for node, (px, py, angle, _) in zip(part["nodes"], _POSITIONS.get(kind, _TWO_PINS)):
            # Library coordinates use +Y up; schematic sheet coordinates +Y down.
            end_x = x + px + (-5.08 if angle == 0 else 5.08 if angle == 180 else 0)
            end_y = y - py + (-5.08 if angle == 270 else 5.08 if angle == 90 else 0)
            wire(x + px, y - py, end_x, end_y)
            label(node, end_x, end_y, to_left=angle == 0)
        value = model or _engineering(part.get("value", part.get("dc", 0)),
                                      {"R": "Ω", "C": "F", "L": "H", "V": "V", "I": "A"}.get(kind, ""))
        note(f"{category}: {part['ref']} / {value}", x - 40, y + 19, size=0.95)
        if category == "Intrinsic loss":
            note(f"Intrinsic ESR/DCR of {part['_parent']}; same physical component, no extra part.", x - 40, y + 23, size=0.8)
        elif kind == "BJT":
            note("C/B/E; device polarity follows the named model (arrow omitted).", x - 40, y + 23, size=0.8)
        elif kind in ("V", "I"):
            note(f"Waveform: {part['waveform']}; parameters are editable in Simulation Model.", x - 40, y + 23, size=0.8)
        else:
            note(f"Package: {part.get('package') or part['ref']}; role: {part.get('role', 'signal')}", x - 40, y + 23, size=0.8)
        if warning:
            note(warning, x - 40, y + 27, size=0.8)
    conn_y = rows * cell_h + 74
    for index, (a, b) in enumerate(ir.get("connections", ())):
        cy = conn_y + index * 8
        wire(25, cy, 65, cy)
        label(a, 25, cy, to_left=True)
        label(b, 65, cy)
        note("Explicit ideal IR connection (zero parts)", 75, cy, size=0.9)
    if not fixture:
        note("No saved fixture supplied; only DUT connectivity is shown.", 15, page_height - 25)
    text.append('(sheet_instances (path "/" (page "1")))\n)')
    return "\n".join(text) + "\n"
