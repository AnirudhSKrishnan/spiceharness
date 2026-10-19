"""Deterministic, net-labelled SVG views of persisted circuit IR.

This is a connectivity view, not a placement or a netlist generator. Every IR
component and ordered pin is shown; only explicit IR connections become wires
between nets. Matching net labels denote electrical connections throughout.
"""

from __future__ import annotations

import math
from html import escape

_WIDTH = 640
_HEIGHT = 360
_GAP = 20
_PINS = {
    "R": ("1", "2"),
    "C": ("1", "2"),
    "L": ("1", "2"),
    "DIODE": ("A", "K"),
    "OPAMP": ("+", "-", "V+", "V-", "OUT"),
    "BJT": ("C", "B", "E"),
    "V": ("+", "-"),
    "I": ("from", "to"),
}


def _escape(value: object) -> str:
    # Persisted legacy dictionaries need not have passed today's IR validators.
    # XML 1.0 cannot represent some control characters, even as entities.
    clean = "".join(
        char if char in "\t\n\r" or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF else "�"
        for char in str(value)
    )
    return escape(clean, quote=True)


def _text(x: int, y: int, value: object, *, css: str = "label", anchor: str = "start",
          width: int = 600) -> str:
    # Long identifiers remain complete; fitting prevents overlap with other pins.
    fit = f' textLength="{width}" lengthAdjust="spacingAndGlyphs"' if len(str(value)) * 7 > width else ""
    return (f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}"{fit}>'
            f'{_escape(value)}</text>')


def _engineering(value: object, unit: str = "") -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return f"{value} {unit}".strip()
    if value == 0:
        return f"0 {unit}".strip()
    for exponent, prefix in ((12, "T"), (9, "G"), (6, "M"), (3, "k"), (0, ""),
                             (-3, "m"), (-6, "µ"), (-9, "n"), (-12, "p"), (-15, "f")):
        if abs(value) >= 10**exponent or exponent == -15:
            return f"{value / 10**exponent:.4g} {prefix}{unit}".strip()
    return str(value)


def _line(x1: int, y1: int, x2: int, y2: int) -> str:
    return f'<path class="wire" d="M{x1} {y1} L{x2} {y2}"/>'


def _terminal(x: int, y: int, label: object, *, side: str) -> str:
    positions = {
        "left": (x - 10, y + 4, "end", 225),
        "right": (x + 10, y + 4, "start", 225),
        "top": (x, y - 8, "middle", 400),
        "bottom": (x, y + 17, "middle", 400),
    }
    tx, ty, anchor, width = positions[side]
    return (f'<circle cx="{x}" cy="{y}" r="3" class="terminal"/>'
            + _text(tx, ty, label, css="net", anchor=anchor, width=width))


def _glyph(kind: str, nodes: list) -> str:
    """Pin positions follow canonical IR order, never catalog SPICE ordering."""
    result = []
    if kind == "OPAMP":
        result.append('<path class="symbol" d="M282 106 L360 150 L282 194 Z"/>')
        positions = ((250, 130, "left"), (250, 170, "left"), (310, 94, "top"),
                     (310, 208, "bottom"), (392, 150, "right"))
        result.extend((_line(250, 130, 282, 130), _line(250, 170, 282, 170),
                       _line(310, 94, 310, 122), _line(310, 178, 310, 208),
                       _line(360, 150, 392, 150)))
        result.extend((_text(290, 135, "+", css="pin"), _text(290, 175, "−", css="pin"),
                       _text(320, 110, "V+", css="small"), _text(320, 204, "V−", css="small")))
    elif kind == "BJT":
        result.extend(('<circle cx="320" cy="150" r="43" class="symbol"/>',
                       ('<path class="wire" d="M302 125 V175 M250 150 H302 '
                        'M302 138 L338 116 V96 M302 162 L338 184 V208"/>')))
        positions = ((338, 96, "top"), (250, 150, "left"), (338, 208, "bottom"))
        result.extend((_text(344, 119, "C", css="small"), _text(286, 143, "B", css="small"),
                       _text(344, 190, "E", css="small")))
    else:
        positions = ((250, 150, "left"), (390, 150, "right"))
        result.extend((_line(250, 150, 280, 150), _line(360, 150, 390, 150)))
        if kind == "R":
            result.append('<path class="wire" d="M280 150 L286 138 L298 162 L310 138 '
                          'L322 162 L334 138 L346 162 L354 138 L360 150"/>')
        elif kind == "C":
            result.append('<path class="wire" d="M280 150 H310 M310 125 V175 '
                          'M330 125 V175 M330 150 H360"/>')
        elif kind == "L":
            result.append('<path class="wire" d="M280 150 a10 14 0 0 1 20 0 '
                          'a10 14 0 0 1 20 0 a10 14 0 0 1 20 0 a10 14 0 0 1 20 0"/>')
        elif kind == "DIODE":
            result.extend(('<path class="symbol" d="M290 126 L338 150 L290 174 Z"/>',
                           '<path class="wire" d="M280 150 H290 M338 124 V176 M338 150 H360"/>',
                           _text(282, 193, "A", css="small"), _text(344, 193, "K", css="small")))
        elif kind in ("V", "I"):
            result.extend(('<circle cx="320" cy="150" r="30" class="symbol"/>',
                           _line(280, 150, 290, 150), _line(350, 150, 360, 150)))
            if kind == "V":
                result.extend((_text(301, 156, "+", css="pin"), _text(328, 156, "−", css="pin")))
            else:
                result.extend((_line(302, 150, 337, 150),
                               '<path class="terminal" d="M337 150 L326 144 L326 156 Z"/>'))
        else:
            result.extend(('<rect x="280" y="125" width="80" height="50" class="symbol"/>',
                           _text(320, 155, "?", css="pin", anchor="middle")))
    for node, (x, y, side) in zip(nodes, positions):
        result.append(_terminal(x, y, node, side=side))
    return "".join(result)


def _source_description(part: dict) -> tuple[str, str, str]:
    unit = "A" if part.get("kind") == "I" else "V"
    dc = _engineering(part.get("dc", 0), unit)
    waveform = part.get("waveform", "dc")
    if waveform == "sine":
        return (f"DC {dc}", (f"Sine: {_engineering(part.get('amplitude', 0), unit)} peak · "
                            f"{_engineering(part.get('frequency_hz', 1000), 'Hz')}"),
                f"Delay: {_engineering(part.get('delay_s', 0), 's')}")
    if waveform == "pulse":
        return (f"DC {dc}", (f"Pulse: Δ{_engineering(part.get('amplitude', 0), unit)} · "
                            f"period {_engineering(part.get('period_s', 0.001), 's')} · "
                            f"width {_engineering(part.get('width_s', 0.0005), 's')}"),
                (f"Delay {_engineering(part.get('delay_s', 0), 's')} · "
                 f"rise {_engineering(part.get('rise_s', 1e-6), 's')} · "
                 f"fall {_engineering(part.get('fall_s', 1e-6), 's')}"))
    return f"DC {dc}", f"Waveform: {waveform}", ""


def _card(part: dict, x: int, y: int, *, fixture: bool, height: int) -> str:
    kind = str(part.get("kind", "unknown"))
    nodes = list(part.get("nodes", ()))
    pins = _PINS.get(kind, ())
    ref = part.get("ref", "unnamed")
    if fixture and kind in ("V", "I"):
        description, meta, foot = _source_description(part)
    else:
        value = part.get("model") or _engineering(part.get("value"), {"R": "Ω", "C": "F", "L": "H"}.get(kind, ""))
        description = str(value)
        meta = f"Package: {part.get('package') or ref} · Role: {part.get('role', 'signal')}"
        loss = part.get("series_resistance_ohm", 0)
        foot = (f"Intrinsic {'ESR' if kind == 'C' else 'DCR'}: {_engineering(loss, 'Ω')} "
                "(same component; no extra part)" if loss else "")
        if kind == "BJT":
            foot = "C/B/E as declared; polarity follows device model (arrow omitted)."
    result = [f'<g transform="translate({x} {y})" class="{"fixture-card" if fixture else "dut-card"}">',
              f'<rect width="{_WIDTH}" height="{height}" rx="10" class="card"/>',
              '<title>' + _escape(f"{ref}: {kind}; " + "; ".join(
                  f"{i + 1} {pins[i] if i < len(pins) else '?'} = {node}" for i, node in enumerate(nodes)
              )) + '</title>',
              _text(20, 27, ref, css="reference"), _text(20, 49, f"{kind} · {description}"),
              _text(20, 69, meta, css="small"), _glyph(kind, nodes),
              _text(20, 239, "Ordered IR pins → net labels", css="small")]
    for index, node in enumerate(nodes):
        pin = pins[index] if index < len(pins) else "undeclared pin type"
        result.append(_text(20, 259 + index * 18, f"{index + 1}. {pin} = {node}", css="net"))
    if len(nodes) != len(pins):
        foot = f"Unrecognized kind or pin count: all {len(nodes)} declared nodes listed above."
    if foot:
        result.append(_text(20, height - 15, foot, css="small"))
    result.append('</g>')
    return "".join(result)


def _grid(parts: list[dict], y: int, *, fixture: bool, columns: int) -> tuple[str, int]:
    if not parts:
        return "", y
    height = max(_HEIGHT, 290 + max(len(p.get("nodes", ())) for p in parts) * 18)
    result = [_card(part, 24 + (i % columns) * (_WIDTH + _GAP), y + (i // columns) * (height + _GAP),
                    fixture=fixture, height=height) for i, part in enumerate(parts)]
    return "".join(result), y + math.ceil(len(parts) / columns) * (height + _GAP)


def _legacy_fixture(spec: dict, circuit: dict, y: int, width: int) -> tuple[str, int]:
    """Summarize legacy evaluator conditions without fabricating fixture IR."""
    result = [_text(24, y + 25, "Legacy evaluator fixture — summary, outside Circuit IR", css="section")]
    lines = [
        (f"Signal source: DC {_engineering(spec.get('input_offset_v', 0), 'V')}; "
         f"sine {_engineering(spec.get('input_amplitude_v'), 'V')} peak at "
         f"{_engineering(spec.get('input_frequency_hz'), 'Hz')}."),
        (f"Requested source resistance: {_engineering(spec.get('source_resistance_ohm', 50), 'Ω')} "
         f"feeding input net {circuit.get('input_node', 'in')}."),
        (f"Load: {_engineering(spec.get('load_ohm'), 'Ω')} between output net "
         f"{circuit.get('output_node', 'out')} and 0."),
        (f"Supply conditions: vcc {_engineering(spec.get('supply_positive_v'), 'V')}; "
         f"vee {_engineering(spec.get('supply_negative_v', 0), 'V')}; referenced to 0."),
        "These evaluator sources and loads are bench fixtures, excluded from DUT component counts.",
    ]
    for index, line in enumerate(lines):
        result.append(_text(24, y + 53 + index * 23, line, width=width - 48))
    return "".join(result), y + 175


def render_circuit_svg(circuit: dict, *, spec: dict | None = None) -> str:
    """Render GeneralCircuitIR or legacy Circuit JSON without changing evidence.

    ``spec`` may be the frozen ExperimentSpec (explicit sources and loads) or a
    legacy DesignSpec (fixture summary). Circuit.spec is used if none is passed.
    All dynamic content is escaped as XML text; attributes contain only renderer
    constants or coordinates. No models, scripts, external assets or AI are used.
    """
    parts = list(circuit.get("components", ()))
    spec = spec if spec is not None else circuit.get("spec")
    fixtures = []
    if spec and "sources" in spec:
        fixtures = [*spec.get("sources", ()), *spec.get("loads", ())]
    columns = 2 if max(len(parts), len(fixtures)) > 1 else 1
    width = 48 + columns * _WIDTH + (columns - 1) * _GAP
    packages = {str(part.get("package") or part.get("ref", "unnamed")).lower() for part in parts}
    content = [_text(24, 33, circuit.get("name", "Circuit"), css="heading", width=width - 48),
               _text(24, 59, "Matching net labels are connected; 0 is the reference net.", width=width - 48),
               _text(24, 81, "Every IR pin is shown in order; glyphs do not imply extra wires.",
                     css="small", width=width - 48),
               _text(24, 111, f"DUT · {len(parts)} IR components · {len(packages)} package groups",
                     css="section", width=width - 48)]
    y = 130
    if parts:
        grid, y = _grid(parts, y, fixture=False, columns=columns)
        content.append(grid)
    else:
        content.append(_text(24, y + 23, "No DUT components (zero-part connectivity is shown below)."))
        y += 55
    connections = circuit.get("connections", ())
    if connections:
        content.append(_text(24, y + 24, "Explicit ideal connections · zero additional parts", css="section"))
        y += 48
        for pair in connections:
            a, b = pair
            content.extend((_text(234, y + 5, a, css="net", anchor="end", width=210),
                            _line(250, y, 390, y), _terminal(250, y, "", side="left"),
                            _terminal(390, y, b, side="right")))
            y += 34
        y += 18
    if spec and "sources" in spec:
        content.extend((_text(24, y + 24, "Bench fixture · frozen sources and loads", css="section"),
                        _text(24, y + 48, "Separate from DUT components and package counts.", css="small")))
        y += 65
        grid, y = _grid(fixtures, y, fixture=True, columns=columns)
        content.append(grid)
    elif spec:
        summary, y = _legacy_fixture(spec, circuit, y, width)
        content.append(summary)
    else:
        content.append(_text(24, y + 23, "Fixture unavailable: this view shows only saved DUT connectivity."))
        y += 45
    content.append(_text(24, y + 16, "All IR parts are shown, including decoupling and spare channels. "
                         "Package groups follow IR labels.", css="small", width=width - 48))
    height = y + 40
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="sh-circuit" role="img" '
            f'viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
            '<title>' + _escape(circuit.get("name", "Circuit")) + ' — exact circuit connectivity</title>'
            '<desc>Deterministic component schematic with ordered pins, named nets, explicit ideal '
            'connections, and separately identified bench fixtures. Matching net labels are connected.</desc>'
            '<style>.sh-circuit{background:#f7f9fc;color:#172b4d;font-family:system-ui,sans-serif}'
            '.sh-circuit text{fill:#172b4d;font-size:13px}'
            '.sh-circuit .heading{font-size:22px;font-weight:700}'
            '.sh-circuit .section{font-size:16px;font-weight:650}'
            '.sh-circuit .reference{font-size:17px;font-weight:700}'
            '.sh-circuit .small{font-size:11px;fill:#52637a}'
            '.sh-circuit .net{font-family:ui-monospace,monospace;font-size:12px;fill:#164e76}'
            '.sh-circuit .pin{font-size:18px}'
            '.sh-circuit .card{fill:white;stroke:#cad4e1;stroke-width:1.2}'
            '.sh-circuit .fixture-card .card{fill:#fff8ec;stroke:#bf8b3d;stroke-dasharray:5 3}'
            '.sh-circuit .wire,.sh-circuit .symbol{fill:none;stroke:#172b4d;stroke-width:2;'
            'stroke-linecap:round;stroke-linejoin:round}'
            '.sh-circuit .terminal{fill:#164e76;stroke:none}</style>' + "".join(content) + '</svg>')
