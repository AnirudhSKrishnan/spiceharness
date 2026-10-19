"""Read an edited KiCad simulation schematic back into validated general Circuit IR.

The importer is the inverse of :mod:`kicad_export`.  It never evaluates anything: it
parses the schematic text with a small bounded s-expression reader, derives connectivity
from the drawn geometry (not from the ``IR.OrderedNodes`` hints, because the user may have
rewired), and rebuilds a :class:`GeneralCircuitIR`.  Scores, rankings and measurements are
produced only by the evaluator afterwards, under the frozen specification.

Connectivity rules were checked against ``kicad-cli`` 10.0.7 (see ``docs/kicad-import.md``):
wire ends, pins and labels that share a coordinate connect; a pin, wire end or crossing
wire that merely touches another wire's interior connects only through a junction; labels
(local, global and power-symbol values) with identical text connect; a power symbol is
named by its Value field.  Library coordinates are +Y up, sheet coordinates +Y down.

Value rule (KiCad 10 behaviour, not plain SPICE): ``M`` is mega and ``m`` is milli;
``Meg``/``MEG``/``meg`` are mega.  Bare ``M`` is accepted with a warning.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import stat
from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .circuit_svg import _engineering
from .experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR
from .simulator import MODEL_DIR

MAX_BYTES = 16 * 1024 * 1024
MAX_DEPTH = 48
MAX_ITEMS = 60000
_NM = 1_000_000  # coordinates are snapped to integer nanometres before matching


class KiCadImportError(ValueError):
    """The schematic cannot be turned into supported Circuit IR (message lists every problem)."""


@dataclass
class ImportResult:
    circuit: dict[str, Any]
    fixture_ok: bool | None
    fixture_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    changes: list[dict[str, Any]] = field(default_factory=list)
    original_found: bool = False
    validation_error: str | None = None
    sha256: str = ""
    path: str | None = None
    pin_nets: dict[tuple[str, str], str] = field(default_factory=dict)  # (KiCad ref, pin) -> node, before folding
    symbol_values: dict[str, float] = field(default_factory=dict)  # KiCad ref -> R/C/L value as read

    @property
    def ok(self) -> bool:
        """True when the fixture is untouched (or unchecked) and the evaluator accepts the circuit."""
        return self.fixture_ok is not False and self.validation_error is None

    def problems(self) -> list[str]:
        found = list(self.fixture_errors)
        if self.validation_error:
            found.append(f"circuit rejected by the evaluator: {self.validation_error}")
        return found

    def require_ok(self) -> ImportResult:
        if not self.ok:
            raise KiCadImportError("\n".join(f"- {item}" for item in self.problems()))
        return self


# --------------------------------------------------------------------------- s-expressions


class _Atom(str):
    """Unquoted token; quoted strings stay plain ``str``."""

    __slots__ = ()


_SCAN = re.compile(r'\s*(?:(\()|(\))|"((?:\\.|[^"\\])*)"|([^\s()"]+))', re.DOTALL)
_ESCAPES = {"n": "\n", "r": "\r", "t": "\t"}


def _unescape(text: str) -> str:
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), text, flags=re.DOTALL)


def parse_sexpr(text: str) -> list:
    """Parse one s-expression without recursion or eval; bounded depth and size."""
    if len(text) > MAX_BYTES:
        raise KiCadImportError("schematic exceeds the 16 MiB import limit")
    stack: list[list] = []
    root: list | None = None
    pos, items = 0, 0
    while True:
        match = _SCAN.match(text, pos)
        if match is None:
            if text[pos:].strip():
                line = text.count("\n", 0, pos) + 1
                raise KiCadImportError(f"malformed schematic text near line {line} (unterminated string?)")
            break
        pos = match.end()
        items += 1
        if items > 4_000_000:
            raise KiCadImportError("schematic has too many tokens")
        if match.group(1):
            if len(stack) >= MAX_DEPTH:
                raise KiCadImportError(f"schematic nesting is deeper than {MAX_DEPTH}")
            node: list = []
            if stack:
                stack[-1].append(node)
            elif root is not None:
                raise KiCadImportError("schematic has more than one top-level expression")
            else:
                root = node
            stack.append(node)
        elif match.group(2):
            if not stack:
                raise KiCadImportError("unbalanced ')' in schematic")
            stack.pop()
        else:
            if not stack:
                raise KiCadImportError("token outside the top-level expression")
            stack[-1].append(_unescape(match.group(3)) if match.group(3) is not None else _Atom(match.group(4)))
    if stack:
        raise KiCadImportError("unbalanced '(' in schematic (file truncated?)")
    if root is None:
        raise KiCadImportError("empty schematic")
    return root


def _head(node: Any) -> str | None:
    return node[0] if isinstance(node, list) and node and isinstance(node[0], _Atom) else None


def _kids(node: list, head: str) -> list[list]:
    return [child for child in node if _head(child) == head]


def _kid(node: list, head: str) -> list | None:
    return next((child for child in node if _head(child) == head), None)


def _text(node: list, head: str, default: str | None = None) -> str | None:
    child = _kid(node, head)
    return child[1] if child is not None and len(child) > 1 and isinstance(child[1], str) else default


def _flag(node: list, head: str) -> bool:
    child = _kid(node, head)
    if child is None:
        return False
    return len(child) < 2 or str(child[1]).lower() in ("yes", "true", "1")


def _at(node: list) -> tuple[float, float, float]:
    at = _kid(node, "at")
    try:
        return float(at[1]), float(at[2]), float(at[3]) if len(at) > 3 else 0.0
    except (TypeError, ValueError, IndexError):
        raise KiCadImportError("an item has a malformed (at x y) position") from None


def _iu(value: float) -> int:
    return round(value * _NM)


# --------------------------------------------------------------------------- library symbols


@dataclass
class _LibPin:
    number: str
    x: float
    y: float
    unit: int
    style: int


@dataclass
class _Lib:
    name: str
    pins: list[_LibPin]
    power: bool
    extends: str | None
    max_unit: int


def _parse_lib_symbols(node: list | None) -> dict[str, _Lib]:
    libs: dict[str, _Lib] = {}
    for symbol in _kids(node or [], "symbol"):
        if len(symbol) < 2 or not isinstance(symbol[1], str):
            continue
        pins: list[_LibPin] = []
        max_unit = 1

        def collect(parent: list, unit: int, style: int, pins: list[_LibPin] = pins) -> None:
            for pin in _kids(parent, "pin"):
                number = _kid(pin, "number")
                x, y, _ = _at(pin)
                pins.append(_LibPin(str(number[1]) if number and len(number) > 1 else "", x, y, unit, style))

        collect(symbol, 0, 0)
        for sub in _kids(symbol, "symbol"):
            sub_name = sub[1] if len(sub) > 1 and isinstance(sub[1], str) else ""
            found = re.search(r"_(\d+)_(\d+)$", sub_name)
            unit, style = (int(found[1]), int(found[2])) if found else (0, 0)
            max_unit = max(max_unit, unit)
            collect(sub, unit, style)
        extends = _text(symbol, "extends")
        libs[symbol[1]] = _Lib(symbol[1], pins, _kid(symbol, "power") is not None, extends, max_unit)
    return libs


def _lib_pins(libs: dict[str, _Lib], name: str, depth: int = 0) -> list[_LibPin]:
    lib = libs.get(name)
    if lib is None:
        return []
    if not lib.pins and lib.extends and depth < 8:
        return _lib_pins(libs, lib.extends, depth + 1)
    return lib.pins


# --------------------------------------------------------------------------- symbols

_STOCK = {
    **dict.fromkeys(("Device:R", "Device:R_Small", "Device:R_US", "Device:R_Small_US"), "R"),
    **dict.fromkeys(("Device:C", "Device:C_Small"), "C"),
    **dict.fromkeys(("Device:L", "Device:L_Small", "Device:L_US", "Device:L_Small_US"), "L"),
}
_OWN = {f"SpiceHarness:{kind}": kind for kind in ("R", "C", "L", "V", "I", "OPAMP", "BJT", "DIODE")}
_SUPPORTED = {**_OWN, **_STOCK}
_PIN_COUNT = {"R": 2, "C": 2, "L": 2, "V": 2, "I": 2, "OPAMP": 5, "BJT": 3, "DIODE": 2}
_PASSIVE_PARAM = {"R": "r", "C": "c", "L": "l"}
_ACTIVE = ("OPAMP", "BJT", "DIODE")
_KNOWN_TOP = {
    "version", "generator", "generator_version", "uuid", "paper", "title_block", "lib_symbols", "junction",
    "no_connect", "wire", "polyline", "label", "global_label", "symbol", "text", "text_box", "image",
    "rectangle", "circle", "arc", "bezier", "rule_area", "table", "group", "embedded_fonts", "embedded_files",
    "sheet_instances", "symbol_instances", "bus_alias", "page", "uuid_list", "title",
}
_REJECTED_TOP = {
    "sheet": "hierarchical sheets are not supported (draw a single flat sheet)",
    "bus": "buses are not supported (use individual wires and labels)",
    "bus_entry": "buses are not supported (use individual wires and labels)",
    "hierarchical_label": "hierarchical labels are not supported (use a flat sheet with local or global labels)",
    "netclass_flag": "net-class directive labels are not supported",
    "directive_label": "directive labels are not supported",
}


@dataclass
class _Sym:
    index: int
    lib_id: str
    kind: str
    kicad_ref: str
    props: dict[str, str]
    pins: dict[str, tuple[int, int]]  # pin number -> snapped sheet point
    category: str | None
    ir_ref: str | None
    dnp: bool
    excluded: bool

    @property
    def label(self) -> str:
        return f"{self.kicad_ref} ({self.lib_id})"


def _pin_point(x: float, y: float, rotation: int, mirror: tuple[str, ...], ox: float, oy: float) -> tuple[int, int]:
    """Library pin position -> sheet point: rotate CCW (+Y up), mirror, then flip Y onto the sheet."""
    for _ in range(rotation // 90):
        x, y = -y, x
    if "x" in mirror:
        y = -y
    if "y" in mirror:
        x = -x
    return _iu(ox + x), _iu(oy - y)


# --------------------------------------------------------------------------- numbers

_NUM = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_POWERS = {"t": 12, "g": 9, "k": 3, "u": -6, "n": -9, "p": -12, "f": -15}
_UNITS = {
    "R": {"", "r", "ohm", "ohms"}, "C": {"", "f", "farad", "farads"}, "L": {"", "h", "henry", "henries"},
    "V": {"", "v", "volt", "volts"}, "I": {"", "a", "amp", "amps"}, None: {"", "s", "hz", "v", "a"},
}


def parse_si_value(text: str, kind: str | None = None) -> tuple[float, list[str]]:
    """Parse ``2.2k``/``4k7``/``100u``/``10n``/``1Meg``/``2.2µF``; returns (value, warnings).

    KiCad 10 convention: ``M`` is mega, ``m`` is milli, ``Meg`` (any case) is mega.
    """
    original = text
    s = re.sub(r"\s+", "", text.replace("µ", "u").replace("μ", "u").replace("Ω", "ohm").replace("\u2126", "ohm"))
    if not s:
        raise ValueError("empty value")
    notes: list[str] = []

    def scaled(number: str, power: int) -> float:
        try:
            return float(Decimal(number) * (Decimal(10) ** power))
        except InvalidOperation:
            raise ValueError(f"cannot read number {original!r}") from None

    rkm = re.fullmatch(r"(\d+)(Meg|MEG|meg|[RkKMGTtmunpf])(\d+)", s)
    if rkm:
        letter = rkm[2]
        power = 0 if letter == "R" else 6 if letter.lower() == "meg" else 6 if letter == "M" else (
            -3 if letter == "m" else _POWERS[letter.lower()])
        if letter == "M":
            notes.append(f"'{original}': bare 'M' read as mega (KiCad 10 convention; plain SPICE reads M as milli; "
                         "write 'Meg' to be unambiguous)")
        return scaled(f"{rkm[1]}.{rkm[3]}", power), notes
    plain = re.fullmatch(f"({_NUM})(.*)", s)
    if plain is None:
        raise ValueError(f"cannot read value {original!r}")
    number, rest = plain[1], plain[2]
    power = 0
    if rest[:3].lower() == "meg":
        power, rest = 6, rest[3:]
    elif rest[:3].lower() == "mil":
        raise ValueError(f"'{original}': mil is not a supported unit")
    elif rest[:1] == "M":
        power, rest = 6, rest[1:]
        notes.append(f"'{original}': bare 'M' read as mega (KiCad 10 convention; plain SPICE reads M as milli; "
                     "write 'Meg' to be unambiguous)")
    elif rest[:1] == "m":
        power, rest = -3, rest[1:]
        if kind == "R":
            notes.append(f"'{original}': lowercase 'm' on a resistor is milli-ohms; use 'M' or 'Meg' for megohms")
    elif rest[:1] == "F":
        raise ValueError(f"'{original}': ambiguous 'F' (SPICE femto vs farad); write 'f' for femto or the plain number")
    elif rest[:1].lower() in _POWERS:
        power, rest = _POWERS[rest[:1].lower()], rest[1:]
    if rest.lower() not in _UNITS.get(kind, _UNITS[None]):
        raise ValueError(f"cannot read value {original!r} (unrecognised suffix {rest!r})")
    return scaled(number, power), notes


def _same(a: float | None, b: float | None) -> bool:
    return a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def _params(text: str | None) -> dict[str, str]:
    found: dict[str, str] = {}
    if not text:
        return found
    leftover = re.sub(r'(\w+)\s*=\s*("[^"]*"|\S+)', lambda m: found.__setitem__(m[1].lower(), m[2].strip('"')) or "", text)
    if leftover.strip():
        raise ValueError(f"cannot read Sim.Params {text!r}")
    return found


# --------------------------------------------------------------------------- geometry


class _UF:
    def __init__(self, size: int = 0):
        self.parent = list(range(size))

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class _Wires:
    """Axis-indexed wires for 'does this point lie on a wire's interior' queries."""

    def __init__(self, wires: list[tuple[int, int, int, int]]):
        self.wires, self.h, self.v, self.d = wires, defaultdict(list), defaultdict(list), []
        for index, (ax, ay, bx, by) in enumerate(wires):
            if ay == by:
                self.h[ay].append((min(ax, bx), max(ax, bx), index))
            elif ax == bx:
                self.v[ax].append((min(ay, by), max(ay, by), index))
            else:
                self.d.append(index)

    def interior(self, point: tuple[int, int]) -> list[int]:
        x, y = point
        found = [i for lo, hi, i in self.h.get(y, ()) if lo < x < hi]
        found += [i for lo, hi, i in self.v.get(x, ()) if lo < y < hi]
        for i in self.d:
            ax, ay, bx, by = self.wires[i]
            if ((bx - ax) * (y - ay) - (by - ay) * (x - ax) == 0 and min(ax, bx) < x < max(ax, bx)
                    and min(ay, by) <= y <= max(ay, by)):
                found.append(i)
        return found


# --------------------------------------------------------------------------- model catalog


def _catalog() -> dict[str, dict]:
    return json.loads((MODEL_DIR / "catalog.json").read_text())


def _model_key(catalog: dict, name: str | None) -> str | None:
    if not name:
        return None
    if name in catalog:
        return name
    lowered = name.casefold()
    for key, entry in catalog.items():
        if key.casefold() == lowered or str(entry.get("spice_name", "")).casefold() == lowered:
            return key
    return None


# --------------------------------------------------------------------------- the importer


def _spec_dict(spec: Any) -> dict | None:
    if spec is None:
        return None
    return spec.model_dump(mode="json") if hasattr(spec, "model_dump") else dict(spec)


def read_schematic_bytes(path: Path | str, limit: int = MAX_BYTES) -> bytes:
    path = Path(path).expanduser()
    try:
        info = path.stat()
    except OSError as exc:
        raise KiCadImportError(f"cannot read {path}: {exc.strerror or exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise KiCadImportError(f"{path} is not a regular file")
    if info.st_size > limit:
        raise KiCadImportError(f"{path.name} is {info.st_size} bytes; the import limit is {limit}")
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise KiCadImportError(f"{path.name} exceeds the import limit of {limit} bytes")
    return data


def import_kicad_schematic(path: Path | str, *, spec: dict | None = None) -> ImportResult:
    """Import ``path`` (a ``.kicad_sch``); the adjacent ``circuit.json`` is used for the change diff."""
    path = Path(path).expanduser()
    return import_kicad_bytes(read_schematic_bytes(path), original_dir=path.parent, spec=spec, source=str(path))


def import_kicad_bytes(data: bytes, *, original_dir: Path | None = None, spec: dict | None = None,
                       source: str | None = None) -> ImportResult:
    return _Importer(data, original_dir, _spec_dict(spec), source).run()


class _Importer:
    def __init__(self, data: bytes, original_dir: Path | None, spec: dict | None, source: str | None):
        self.data, self.source, self.spec_dict = data, source, spec
        self.original_dir = original_dir
        self.warnings: list[str] = []
        self.problems: list[str] = []
        self.fixture_errors: list[str] = []
        self.catalog = _catalog()
        self.spec = ExperimentSpec.model_validate(spec) if spec is not None else None
        self.original: GeneralCircuitIR | None = None
        self.baseline: dict[str, Any] = {}
        self.merged: dict[str, set[str]] = {}

    # ---- helpers
    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    def problem(self, message: str) -> None:
        if message not in self.problems:
            self.problems.append(message)

    def run(self) -> ImportResult:
        sha = hashlib.sha256(self.data).hexdigest()
        try:
            text = self.data.decode("utf-8")
        except UnicodeDecodeError:
            raise KiCadImportError("schematic is not valid UTF-8 text (is this a KiCad 6+ .kicad_sch file?)") from None
        root = parse_sexpr(text)
        if _head(root) != "kicad_sch":
            raise KiCadImportError("not a KiCad schematic: the top-level expression is not (kicad_sch ...)")
        self._load_original()
        top_names = {_head(child) for child in root[1:] if isinstance(child, list)}
        for name, why in _REJECTED_TOP.items():
            if name in top_names:
                self.problem(f"unsupported schematic content: {why}")
        for name in sorted(n for n in top_names if n and n not in _KNOWN_TOP and n not in _REJECTED_TOP):
            self.problem(f"unsupported schematic item ({name} ...); remove it or save with KiCad 10")
        total = sum(1 for child in root if isinstance(child, list))
        if total > MAX_ITEMS:
            raise KiCadImportError(f"schematic has {total} items; the import limit is {MAX_ITEMS}")
        libs = _parse_lib_symbols(_kid(root, "lib_symbols"))
        syms, power_pins = self._symbols(root, libs)
        wires, junctions, labels, no_connects = self._drawing(root)
        if self.problems:  # unknown symbols etc. make connectivity meaningless
            raise KiCadImportError("\n".join(f"- {item}" for item in self.problems))
        nets = self._connectivity(syms, power_pins, wires, junctions, labels, no_connects)
        result = self._build(root, syms, nets)
        result.sha256, result.path = sha, self.source
        return result

    def _load_original(self) -> None:
        if self.original_dir is None:
            return
        try:
            file = Path(self.original_dir) / "circuit.json"
            if file.is_file() and not file.is_symlink() and file.stat().st_size <= 4 * 1024 * 1024:
                self.original = GeneralCircuitIR.model_validate_json(file.read_text())
        except (OSError, ValueError):
            self.warn("circuit.json next to the schematic could not be read; no change diff is available")
        if self.spec is not None and self.spec.baseline is not None:
            protected = {ref.lower() for ref in self.spec.protected_refs}
            self.baseline = {part.ref.lower(): part for part in self.spec.baseline.components
                             if part.ref.lower() in protected}

    # ---- symbols
    def _symbols(self, root: list, libs: dict[str, _Lib]) -> tuple[list[_Sym], list[tuple[str, tuple[int, int]]]]:
        syms: list[_Sym] = []
        power: list[tuple[str, tuple[int, int]]] = []
        for node in root:
            if _head(node) != "symbol" or _kid(node, "lib_id") is None:
                continue
            lib_id = _text(node, "lib_id") or ""
            lib_name = _text(node, "lib_name") or lib_id
            props = {p[1]: p[2] for p in _kids(node, "property") if len(p) > 2 and isinstance(p[1], str)
                     and isinstance(p[2], str)}
            instances = _kid(node, "instances")
            ref = None
            if instances is not None:
                for found in _walk(instances, "reference"):
                    ref = found[1]
                    break
            ref = ref or props.get("Reference", "?")
            lib = libs.get(lib_name) or libs.get(lib_id)
            is_power = (lib is not None and lib.power) or lib_id.startswith("power:")
            x, y, angle = _at(node)
            if angle % 90 != 0:
                self.problem(f"{ref}: rotation {angle:g} degrees is not a multiple of 90")
                continue
            mirror = tuple(str(m).lower() for m in (_kid(node, "mirror") or [])[1:])
            unit_node = _kid(node, "unit")
            unit = int(unit_node[1]) if unit_node and len(unit_node) > 1 and str(unit_node[1]).isdigit() else 1
            raw_pins = _lib_pins(libs, lib_name) or _lib_pins(libs, lib_id)
            if is_power:
                name = props.get("Value") or ""
                for pin in raw_pins:
                    power.append((name, _pin_point(pin.x, pin.y, int(angle) % 360, mirror, x, y)))
                if not raw_pins:
                    self.problem(f"power symbol {ref} ({lib_id}) has no pin in lib_symbols")
                continue
            kind = _SUPPORTED.get(lib_id)
            if kind is None:
                self.problem(
                    f"{ref}: symbol '{lib_id}' is not supported. Supported: SpiceHarness:R/C/L/V/I/OPAMP/BJT/DIODE, "
                    "KiCad Device:R/C/L (and R_Small, C_Small, L_Small, R_US, L_US), and power:GND. "
                    "Polarized capacitors, other devices, sheets and subcircuits cannot be represented")
                continue
            if lib is None:
                self.problem(f"{ref}: symbol '{lib_id}' has no definition in lib_symbols; re-save the schematic in KiCad")
                continue
            if lib.max_unit > 1 or unit != 1:
                self.problem(f"{ref}: multi-unit symbol '{lib_id}' (unit {unit}) is not supported")
                continue
            style = int(_text(node, "body_style") or _text(node, "convert") or 1)
            pins = {p.number: p for p in raw_pins if p.unit in (0, unit) and p.style in (0, style)}
            expected = {str(i) for i in range(1, _PIN_COUNT[kind] + 1)}
            if set(pins) != expected:
                self.problem(f"{ref}: symbol '{lib_id}' has pins {sorted(pins)}; expected {sorted(expected)}")
                continue
            category = props.get("IR.Category")
            if category not in (None, "DUT", "Intrinsic loss", "Bench fixture"):
                self.problem(f"{ref}: unknown IR.Category {category!r}")
                continue
            if kind in ("V", "I") and category != "Bench fixture":
                self.problem(f"{ref}: sources are frozen bench fixtures and cannot be added or moved into the circuit")
                continue
            points = {number: _pin_point(pin.x, pin.y, int(angle) % 360, mirror, x, y) for number, pin in pins.items()}
            ir_ref = props.get("IR.Reference")
            syms.append(_Sym(len(syms), lib_id, kind, ref, props, points, category, ir_ref,
                             _flag(node, "dnp"), _flag(node, "exclude_from_sim")))
        for sym in syms:
            if "?" in sym.kicad_ref and sym.ir_ref is None:
                self.problem(f"{sym.label}: annotate the schematic first (reference {sym.kicad_ref!r})")
        return syms, power

    # ---- wires, labels
    def _drawing(self, root: list):
        wires, junctions, labels, no_connects = [], [], [], []
        for node in root:
            head = _head(node)
            if head == "wire":
                pts = [(_iu(float(p[1])), _iu(float(p[2]))) for p in _kids(_kid(node, "pts") or [], "xy")]
                if len(pts) != 2:
                    self.problem("a wire does not have exactly two points")
                elif pts[0] != pts[1]:
                    wires.append((*pts[0], *pts[1]))
            elif head == "junction":
                x, y, _ = _at(node)
                junctions.append((_iu(x), _iu(y)))
            elif head in ("label", "global_label"):
                x, y, _ = _at(node)
                labels.append((str(node[1]), (_iu(x), _iu(y)), head))
            elif head == "no_connect":
                x, y, _ = _at(node)
                no_connects.append((_iu(x), _iu(y)))
            elif head == "text" and len(node) > 1 and isinstance(node[1], str) and node[1].lstrip().startswith("."):
                body = node[1].strip().split()[0].lower()
                if body not in (".temp", ".options", ".tran", ".ac", ".op"):
                    self.warn(f"SPICE directive text {node[1].strip()[:40]!r} is ignored; "
                              "analyses and options are owned by the frozen experiment")
        return wires, junctions, labels, no_connects

    # ---- connectivity (stage 1: (symbol, pin) -> net; stage 2 builds IR)
    def _connectivity(self, syms, power_pins, wires, junctions, labels, no_connects):
        uf = _UF()
        pin_ids: list[tuple[int, str] | None] = []
        point_of: list[tuple[int, int]] = []
        at_point: dict[tuple[int, int], list[int]] = defaultdict(list)
        pin_element: dict[tuple[int, str], int] = {}
        power_names: dict[int, str] = {}

        def new(point: tuple[int, int]) -> int:
            element = uf.add()
            point_of.append(point)
            at_point[point].append(element)
            return element

        for sym in syms:
            for number, point in sym.pins.items():
                pin_element[(sym.index, number)] = new(point)
                pin_ids.append((sym.index, number))
        n_real = len(pin_ids)
        for name, point in power_pins:
            element = new(point)
            pin_ids.append(None)
            power_names[element] = name
        wire_ids = []
        for ax, ay, bx, by in wires:
            a = new((ax, ay))
            b = new((bx, by))
            uf.union(a, b)
            wire_ids.append(a)
        label_ids = [(new(point), text, kind) for text, point, kind in labels]
        junction_ids = [new(point) for point in junctions]
        for members in at_point.values():
            for other in members[1:]:
                uf.union(members[0], other)
        index = _Wires(wires)
        wire_element = {i: wire_ids[i] for i in range(len(wires))}
        pin_points = {point_of[e] for e in range(len(point_of)) if e < n_real or e in power_names}
        for element, point in zip(junction_ids, junctions, strict=True):
            for w in index.interior(point):
                uf.union(element, wire_element[w])
        for element, _, _ in label_ids:
            point = point_of[element]
            if point not in pin_points:  # KiCad attaches a label sitting on a pin to the pin only
                for w in index.interior(point):
                    uf.union(element, wire_element[w])
        junction_set = set(junctions)
        reported = 0
        touch = [(point_of[e], f"pin {syms[pin_ids[e][0]].kicad_ref}.{pin_ids[e][1]}") for e in range(n_real)]
        touch += [((ax, ay), "a wire end") for ax, ay, _, _ in wires]
        touch += [((bx, by), "a wire end") for _, _, bx, by in wires]
        for point, what in touch:
            if point not in junction_set and index.interior(point) and reported < 8:
                reported += 1
                self.warn(f"{what} at ({point[0] / _NM:g}, {point[1] / _NM:g}) touches the middle of a wire "
                          "without a junction; KiCad does not connect it (add a junction if intended)")
        # names
        raw_names: dict[int, list[tuple[str, tuple[int, int], int]]] = defaultdict(list)
        order = 0
        for element, text, _ in label_ids:
            raw_names[uf.find(element)].append((text, point_of[element], order))
            order += 1
        for element, name in power_names.items():
            raw_names[uf.find(element)].append((name, point_of[element], order))
            order += 1
        clusters: dict[int, list[int]] = defaultdict(list)
        for e in range(n_real):
            clusters[uf.find(e)].append(e)
        no_connect_points = set(no_connects)
        for e in range(n_real):
            if point_of[e] in no_connect_points:
                members = [m for m in at_point[point_of[e]]]
                cluster = clusters[uf.find(e)]
                sym = syms[pin_ids[e][0]]
                if len(cluster) > 1 or raw_names.get(uf.find(e)) or len(members) > 1:
                    self.problem(f"{sym.kicad_ref} pin {pin_ids[e][1]} carries a no-connect flag but is wired; "
                                 "remove the flag or the wire")
                else:
                    self.problem(f"{sym.kicad_ref} pin {pin_ids[e][1]} is flagged no-connect, but every modeled "
                                 "terminal must be connected")
        # canonical names
        def canon(text: str) -> str | None:
            t = text.strip()
            if t.casefold() in ("0", "gnd"):
                return "0"
            if re.fullmatch(r"[A-Za-z0-9_]{1,48}", t) is None or t.lower().startswith("exinternal"):
                self.problem(f"net/label name {text!r} is not a valid node name (use letters, digits and "
                             "underscore, up to 48 characters; ground is 0 or GND)")
                return None
            return t.lower()

        spellings: dict[str, set[str]] = defaultdict(set)
        for members in raw_names.values():
            for text, _, _ in members:
                if text.strip().casefold() not in ("0", "gnd"):
                    spellings[text.strip().casefold()].add(text.strip())
        for variants in spellings.values():
            if len(variants) > 1:
                self.problem("labels " + ", ".join(repr(v) for v in sorted(variants)) + " differ only in case; KiCad "
                             "treats them as different nets but SpiceHarness nodes are case-insensitive. Rename one")
        names_uf: dict[str, str] = {}

        def nfind(name: str) -> str:
            names_uf.setdefault(name, name)
            while names_uf[name] != name:
                names_uf[name] = names_uf[names_uf[name]]
                name = names_uf[name]
            return name

        first_seen: dict[str, int] = {}
        cluster_names: dict[int, list[str]] = {}
        for root_id, members in raw_names.items():
            canonical = []
            for text, _, seq in sorted(members, key=lambda m: m[2]):
                name = canon(text)
                if name is None:
                    continue
                first_seen.setdefault(name, seq)
                nfind(name)
                if name not in canonical:
                    canonical.append(name)
            cluster_names[root_id] = canonical
        merged: dict[str, set[str]] = defaultdict(set)
        for root_id, cluster in clusters.items():
            names = cluster_names.get(root_id, [])
            for other in names[1:]:
                a, b = nfind(names[0]), nfind(other)
                if a != b:
                    names_uf[max(a, b)] = min(a, b)
        priority = set()
        if self.spec is not None:
            for source in self.spec.sources:
                priority.update(source.nodes)
            for load in self.spec.loads:
                priority.update(load.nodes)
            priority.update(p.positive for p in self.spec.probes)
            priority.update(p.negative for p in self.spec.probes)
        groups: dict[str, list[str]] = defaultdict(list)
        for name in list(names_uf):
            groups[nfind(name)].append(name)
        group_name: dict[str, str] = {}
        for root_name, members in groups.items():
            members.sort(key=lambda m: first_seen.get(m, 10**9))
            choice = ("0" if "0" in members else
                      next((m for m in members if m in priority), members[0]))
            group_name[root_name] = choice
            if len(members) > 1:
                merged[choice] = set(members)
                self.warn("nets " + ", ".join(repr(m) for m in members) + f" are joined by wiring through component "
                          f"pins; they are one node '{choice}'")
        self.merged = merged
        used = set(group_name.values()) | priority | {"0"}
        generated = 0
        pin_node: dict[tuple[int, str], str] = {}
        cluster_node: dict[int, str] = {}
        for root_id in sorted(clusters, key=lambda r: min(clusters[r])):
            names = cluster_names.get(root_id, [])
            if names:
                cluster_node[root_id] = group_name[nfind(names[0])]
                continue
            if len(clusters[root_id]) == 1:
                sym_index, number = pin_ids[clusters[root_id][0]]
                self.problem(f"{syms[sym_index].kicad_ref} pin {number} is not connected to anything "
                             "(wire it to a net or add a label)")
            while True:
                generated += 1
                name = f"net{generated}"
                if name not in used:
                    break
            used.add(name)
            cluster_node[root_id] = name
        for e in range(n_real):
            pin_node[pin_ids[e]] = cluster_node[uf.find(e)]
        # connections: pinless clusters that carry two or more names are exporter-style ideal wires
        connections = []
        for root_id, names in cluster_names.items():
            if root_id in clusters:
                continue
            ordered = [m for m in sorted(raw_names[root_id], key=lambda m: (m[1][0], m[1][1]))]
            mapped = []
            for text, _, _ in ordered:
                name = canon(text)
                if name is not None:
                    node = group_name[nfind(name)]
                    if node not in mapped:
                        mapped.append(node)
            if len(mapped) > 1:
                top = min((p[1], p[0]) for _, p, _ in raw_names[root_id])
                connections.append((top, mapped))
        connections.sort(key=lambda item: item[0])
        return {"pin_node": pin_node, "connections": [m for _, m in connections]}

    # ---- stage 2: IR
    def _build(self, root: list, syms: list[_Sym], nets: dict) -> ImportResult:
        pin_node: dict[tuple[int, str], str] = nets["pin_node"]
        by_sym = {s.index: [pin_node[(s.index, str(i))] for i in range(1, _PIN_COUNT[s.kind] + 1)] for s in syms}
        pin_nets = {(s.kicad_ref, str(i + 1)): node for s in syms for i, node in enumerate(by_sym[s.index])}
        for s in syms:
            if s.dnp or s.excluded:
                flag = "DNP" if s.dnp else "exclude-from-simulation"
                if s.category == "Bench fixture":
                    self.fixture_errors.append(
                        f"fixture {s.ir_ref or s.kicad_ref} is marked {flag}; the frozen bench fixture cannot be removed")
                else:
                    self.problem(f"{s.label} is marked {flag}; delete the symbol to remove it from the circuit")
            if s.props.get("Sim.Enable", "1").strip().lower() in ("0", "false", "no") and s.category != "Bench fixture":
                self.problem(f"{s.label} has Sim.Enable=0; delete the symbol to remove it from the circuit")
        # fixtures
        fixtures = [s for s in syms if s.category == "Bench fixture"]
        fixture_ok: bool | None = None
        if self.spec is not None:
            self._check_fixtures(fixtures, by_sym)
            fixture_ok = not self.fixture_errors
        elif fixtures:
            self.warn("no frozen specification supplied; bench fixtures were not checked")
        # intrinsic loss -> parent
        parts = [s for s in syms if s.category in (None, "DUT")]
        intrinsic = [s for s in syms if s.category == "Intrinsic loss"]
        by_ir = defaultdict(list)
        for s in parts:
            by_ir[(s.ir_ref or s.kicad_ref).casefold()].append(s)
        for group in by_ir.values():
            if len(group) > 1:
                shown = ", ".join(s.kicad_ref for s in group)
                self.problem(f"component reference {group[0].ir_ref or group[0].kicad_ref!r} is used by several "
                             f"symbols ({shown}); a copied symbol keeps its IR.Reference, so delete the IR.* "
                             "properties of the copy or give it a unique IR.Reference")
        members = defaultdict(list)
        for s in syms:
            for i, node in enumerate(by_sym[s.index], 1):
                members[node].append((s.index, i))
        symbol_values: dict[str, float] = {}
        folded: dict[int, tuple[float, int, str]] = {}
        parent_of: dict[int, _Sym] = {}
        for res in intrinsic:
            parent = self._parent(res, by_ir)
            if parent is None:
                self.problem(f"intrinsic-loss resistor {res.ir_ref or res.kicad_ref} has no matching parent "
                             f"(IR.Package {res.props.get('IR.Package')!r}); restore the parent or delete this resistor")
                continue
            if parent.kind not in ("C", "L"):
                self.problem(f"intrinsic-loss resistor {res.ir_ref} belongs to {parent.kicad_ref} which is not a "
                             "capacitor or inductor")
                continue
            if parent.index in folded:
                self.problem(f"{parent.ir_ref or parent.kicad_ref} has more than one intrinsic-loss resistor")
                continue
            choice = None
            for rp in (2, 1):
                private = by_sym[res.index][rp - 1]
                for pp in (1, 2):
                    if by_sym[parent.index][pp - 1] == private and sorted(members[private]) == sorted(
                            [(res.index, rp), (parent.index, pp)]):
                        choice = (rp, pp)
                        break
                if choice:
                    break
            if choice is None:
                self.problem(f"intrinsic-loss resistor {res.ir_ref or res.kicad_ref} is no longer in series with "
                             f"{parent.ir_ref or parent.kicad_ref} on a private net; restore the series wiring or "
                             "delete the resistor (a deleted ESR/DCR is treated as zero loss)")
                continue
            rp, pp = choice
            value = self._passive_value(res, "R", self._original_series(parent))
            if value is None or value < 0:
                continue
            symbol_values[res.kicad_ref] = value
            external = by_sym[res.index][2 - rp]
            folded[parent.index] = (value, pp, external)
            parent_of[res.index] = parent
        # components
        order = {part.ref.lower(): i for i, part in enumerate(self.original.components)} if self.original else {}
        built = []
        refs_seen: dict[str, str] = {}
        fixture_refs = {s.ir_ref.casefold() for s in fixtures if s.ir_ref}
        if self.spec is not None:
            fixture_refs |= {p.ref.casefold() for p in self.spec.loads}
        for s in parts:
            ref = s.ir_ref or s.kicad_ref
            if re.fullmatch(r"[A-Za-z0-9_]{1,48}", ref) is None:
                self.problem(f"{s.label}: reference {ref!r} is not a simple identifier")
                continue
            if ref.casefold() in refs_seen or ref.casefold() in fixture_refs:
                if ref.casefold() in fixture_refs:
                    self.problem(f"{s.label}: reference {ref!r} duplicates a frozen fixture reference")
                else:
                    self.problem(f"duplicate component reference {ref!r}")
                continue
            refs_seen[ref.casefold()] = ref
            nodes = list(by_sym[s.index])
            base = self.baseline.get(ref.casefold()) or (
                next((p for p in self.original.components if p.ref.casefold() == ref.casefold()), None)
                if self.original else None)
            if base is not None and base.kind != s.kind:
                base = None
            part: dict[str, Any] = {"ref": ref, "kind": s.kind, "nodes": nodes, "value": None, "model": None,
                                    "package": None, "role": base.role if base else "signal", "parameters": {},
                                    "series_resistance_ohm": 0.0}
            if s.kind in _ACTIVE:
                part["model"] = self._model(s, base.model if base else None)
                if part["model"] is None:
                    continue
            else:
                part["value"] = self._passive_value(s, s.kind, base.value if base else None)
                if part["value"] is None:
                    continue
                symbol_values[s.kicad_ref] = part["value"]
            if s.index in folded:
                loss, pp, external = folded[s.index]
                other = nodes[2 - pp]
                part["nodes"] = [external, other]  # IR keeps the series loss on the first terminal
                part["series_resistance_ohm"] = loss
                if pp == 2:
                    self.warn(f"{ref}: terminals swapped so the series loss stays on the first IR terminal")
            package = (s.props.get("IR.Package") or "").strip()
            if base is not None and (base.package or base.ref).casefold() == (package or ref).casefold():
                part["package"] = base.package
            elif package and (package.casefold() != ref.casefold() or s.kind == "OPAMP"):
                part["package"] = package
            built.append((order.get(ref.casefold(), 10**6 + s.index), part))
        built.sort(key=lambda item: item[0])
        # connections
        connections: list[list[str]] = []
        loop = _UF()
        index_of: dict[str, int] = {}
        pairs = []
        for names in nets["connections"]:
            pairs.extend(itertools.pairwise(names))
        if self.original:
            original_pairs = [tuple(pair) for pair in self.original.connections]
            sorted_orig = {tuple(sorted(p)): i for i, p in enumerate(original_pairs)}
            pairs.sort(key=lambda p: sorted_orig.get(tuple(sorted(p)), 10**6))
            pairs = [original_pairs[sorted_orig[tuple(sorted(p))]] if tuple(sorted(p)) in sorted_orig else p
                     for p in pairs]
        for a, b in pairs:
            ia = index_of.setdefault(a, loop.add())
            ib = index_of.setdefault(b, loop.add())
            if loop.find(ia) == loop.find(ib):
                self.warn(f"redundant wire between '{a}' and '{b}' (already connected) was dropped")
                continue
            loop.union(ia, ib)
            connections.append([a, b])
        if self.problems:
            raise KiCadImportError("\n".join(f"- {item}" for item in self.problems))
        name = _text(_kid(root, "title_block") or [], "title") or (self.original.name if self.original else "candidate")
        raw = {"name": name[:200], "components": [part for _, part in built], "connections": connections}
        try:
            circuit = GeneralCircuitIR.model_validate(raw)
        except ValidationError as exc:
            detail = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:6])
            raise KiCadImportError(f"- imported circuit is not valid Circuit IR: {detail}") from exc
        validation = None
        if self.spec is not None and fixture_ok:
            try:
                ExperimentEvaluator().validate(circuit, self.spec)
            except (ValueError, KeyError, OSError) as exc:
                validation = str(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)
        dumped = circuit.model_dump(mode="json")
        return ImportResult(circuit=dumped, fixture_ok=fixture_ok, fixture_errors=list(self.fixture_errors),
                            warnings=list(self.warnings), changes=_diff(self.original, dumped),
                            original_found=self.original is not None, validation_error=validation,
                            pin_nets=pin_nets, symbol_values=symbol_values)

    def _original_series(self, parent: _Sym) -> float | None:
        ref = (parent.ir_ref or parent.kicad_ref).casefold()
        base = self.baseline.get(ref) or (next((p for p in self.original.components if p.ref.casefold() == ref), None)
                                          if self.original else None)
        return base.series_resistance_ohm if base is not None else None

    def _parent(self, res: _Sym, by_ir: dict[str, list[_Sym]]) -> _Sym | None:
        wanted = (res.props.get("IR.Package") or "").strip().casefold()
        if not wanted and res.ir_ref and res.ir_ref.endswith("_intrinsic"):
            wanted = res.ir_ref[: -len("_intrinsic")].casefold()
        group = by_ir.get(wanted, [])
        return group[0] if len(group) == 1 else None

    # ---- values
    def _passive_value(self, s: _Sym, kind: str, original: float | None) -> float | None:
        ref = s.ir_ref or s.kicad_ref
        shown = s.props.get("Value", "").strip()
        sim: float | None = None
        disp: float | None = None
        if s.props.get("Sim.Library") or s.props.get("Sim.Name"):
            self.problem(f"{ref}: a {kind} with a Sim.Library/Sim.Name behavioural model is not supported")
            return None
        device = s.props.get("Sim.Device")
        if device and device != kind:
            self.problem(f"{ref}: Sim.Device {device!r} does not match a {kind}")
            return None
        try:
            params = _params(s.props.get("Sim.Params"))
            extra = set(params) - {_PASSIVE_PARAM[kind]}
            if extra:
                self.problem(f"{ref}: unsupported Sim.Params {sorted(extra)}; only {_PASSIVE_PARAM[kind]}= is read")
                return None
            if _PASSIVE_PARAM[kind] in params:
                sim, notes = parse_si_value(params[_PASSIVE_PARAM[kind]], kind)
                for note in notes:
                    self.warn(f"{ref} Sim.Params: {note}")
        except ValueError as exc:
            self.problem(f"{ref}: {exc}")
            return None
        if shown and shown not in ("~",) and shown.casefold() != kind.casefold():
            try:
                disp, notes = parse_si_value(shown, kind)
                for note in notes:
                    self.warn(f"{ref} Value: {note}")
            except ValueError as exc:
                if sim is None:
                    self.problem(f"{ref}: {exc}")
                    return None
                self.warn(f"{ref}: Value field {shown!r} is not a number; using Sim.Params")
        value = self._reconcile(ref, "value", sim, disp, original, shown, s.props.get("Sim.Params", ""))
        if value is None:
            if sim is None and disp is None:
                self.problem(f"{ref}: no value; set the Value field (for example 2.2k) or Sim.Params")
            return None
        if not value > 0 or not math.isfinite(value):
            self.problem(f"{ref}: value must be positive")
            return None
        return value

    def _reconcile(self, ref: str, what: str, sim: float | None, disp: float | None, original: float | None,
                   shown: str, params: str) -> float | None:
        if sim is None or disp is None or _same(sim, disp):
            return sim if sim is not None else disp
        if original is not None:
            if _same(sim, original):
                self.warn(f"{ref}: only the Value field {shown!r} was edited (Sim.Params {params!r} still has the "
                          "original); using the edited Value. Keep both fields in step to match KiCad's simulator")
                return disp
            if _same(disp, original):
                return sim
        self.problem(f"{ref}: Value field {shown!r} ({disp:.6g}) and Sim.Params {params!r} ({sim:.6g}) disagree "
                     "and cannot be told apart from the original; KiCad's simulator uses Sim.Params. "
                     "Edit both fields (or delete Sim.Params) so they agree")
        return None

    def _model(self, s: _Sym, original: str | None) -> str | None:
        ref = s.ir_ref or s.kicad_ref
        shown = s.props.get("Value", "").strip()
        simname = s.props.get("Sim.Name", "").strip()
        placeholder = {"", "~", s.kind.casefold(), "opamp", "bjt", "diode"}
        known = ", ".join(sorted(k for k, v in self.catalog.items() if v.get("kind") == s.kind)) or "none"
        value_key = _model_key(self.catalog, shown) if shown.casefold() not in placeholder else None
        sim_key = _model_key(self.catalog, simname)
        if simname and sim_key is None:
            self.problem(f"{ref}: model {simname!r} is not in the trusted catalog (known {s.kind}: {known}); "
                         "no ideal or custom substitute is accepted")
            return None
        if shown.casefold() not in placeholder and value_key is None:
            self.problem(f"{ref}: model {shown!r} is not in the trusted catalog (known {s.kind}: {known}); "
                         "no ideal or custom substitute is accepted")
            return None
        key = sim_key or value_key
        if key is None:
            self.problem(f"{ref}: choose a catalog model (known {s.kind}: {known}) in Value or Sim.Name")
            return None
        if sim_key and value_key and sim_key != value_key:
            if original in (sim_key, value_key):
                key = value_key if original == sim_key else sim_key
                self.warn(f"{ref}: Value {shown!r} and Sim.Name {simname!r} name different models; using the "
                          f"edited one, {key}")
            else:
                self.problem(f"{ref}: Value {shown!r} and Sim.Name {simname!r} name different catalog models")
                return None
        if self.catalog[key].get("kind") != s.kind:
            self.problem(f"{ref}: model {key} is a {self.catalog[key].get('kind')}, not a {s.kind}")
            return None
        library = s.props.get("Sim.Library", "")
        if library and not re.fullmatch(r"(?:.*/)?(?:model_\d+|[A-Za-z0-9_]+)\.lib", library):
            self.warn(f"{ref}: Sim.Library {library!r} is ignored; models come only from the trusted catalog")
        return key

    # ---- fixture check
    def _check_fixtures(self, fixtures: list[_Sym], by_sym: dict[int, list[str]]) -> None:
        from .kicad_export import _source_properties

        assert self.spec is not None
        expected: dict[str, tuple[str, Any]] = {s.ref.casefold(): ("source", s) for s in self.spec.sources}
        expected.update({p.ref.casefold(): ("load", p) for p in self.spec.loads})
        found: dict[str, _Sym] = {}
        for s in fixtures:
            key = (s.ir_ref or s.kicad_ref).casefold()
            if key in found:
                self.fixture_errors.append(f"fixture {s.ir_ref or s.kicad_ref} appears twice in the schematic")
            elif key not in expected:
                self.fixture_errors.append(
                    f"fixture symbol {s.ir_ref or s.kicad_ref} is not part of the frozen bench; the bench cannot be extended")
            else:
                found[key] = s
        merged = self.merged
        for key, (role, frozen) in expected.items():
            sym = found.get(key)
            if sym is None:
                self.fixture_errors.append(f"fixture {role} {frozen.ref} was deleted; the bench is frozen")
                continue
            name = frozen.ref
            kind = frozen.kind
            if sym.kind != kind:
                self.fixture_errors.append(f"fixture {name} was changed from {kind} to {sym.kind}")
                continue
            for i, want in enumerate(frozen.nodes):
                got = by_sym[sym.index][i]
                if got != want:
                    joined = [m for choice, ms in merged.items() if choice == got for m in ms if m != got]
                    extra = f" (nets {', '.join(repr(m) for m in sorted(joined))} were joined with it)" if joined else ""
                    self.fixture_errors.append(
                        f"fixture {name} pin {i + 1} is on net '{got}' but the frozen bench requires '{want}'{extra}")
            try:
                if role == "source":
                    self._check_source(sym, frozen, _source_properties)
                else:
                    self._check_load(sym, frozen)
            except ValueError as exc:
                self.fixture_errors.append(f"fixture {name}: {exc}")

    def _check_source(self, sym: _Sym, frozen, source_properties) -> None:
        want = source_properties(frozen.model_dump(), ac=False)
        props = sym.props
        if props.get("Sim.Type") != want["Sim.Type"]:
            raise ValueError(f"waveform type was changed ({props.get('Sim.Type')!r}, frozen {want['Sim.Type']!r})")
        if props.get("Sim.Pins", "").split() != want["Sim.Pins"].split():
            raise ValueError("Sim.Pins (source polarity) was changed")
        have, expect = _params(props.get("Sim.Params")), _params(want["Sim.Params"])
        have.pop("ac", None)
        expect.pop("ac", None)
        if set(have) != set(expect):
            raise ValueError(f"Sim.Params keys {sorted(have)} differ from frozen {sorted(expect)}")
        for key, text in expect.items():
            frozen_value = float(text)
            got, _ = parse_si_value(have[key])
            if not _same(got, frozen_value) and not (got == 0 == frozen_value):
                raise ValueError(f"{key} is {have[key]} but the frozen value is {frozen_value:.17g}")
        shown = props.get("Value", "").strip()
        if shown:
            try:
                got, _ = parse_si_value(shown, "V" if frozen.kind == "V" else "I")
            except ValueError:
                return
            if not _same(got, frozen.dc) and not (got == 0 == frozen.dc):
                raise ValueError(f"Value field shows {shown} but the frozen DC level is {frozen.dc:.17g}")

    def _check_load(self, sym: _Sym, frozen) -> None:
        props = sym.props
        values = []
        try:
            params = _params(props.get("Sim.Params"))
        except ValueError as exc:
            raise ValueError(str(exc)) from None
        if set(params) - {"r"}:
            raise ValueError(f"unsupported Sim.Params {sorted(set(params) - {'r'})}")
        if "r" in params:
            values.append(("Sim.Params", parse_si_value(params["r"], "R")[0]))
        shown = props.get("Value", "").strip()
        if shown and shown.casefold() != "r":
            try:
                values.append(("Value", parse_si_value(shown, "R")[0]))
            except ValueError as exc:
                raise ValueError(str(exc)) from None
        if not values:
            raise ValueError("has no value")
        for where, value in values:
            if not _same(value, frozen.value):
                raise ValueError(f"{where} gives {_engineering(value, 'Ω')} but the frozen load is "
                                 f"{_engineering(frozen.value, 'Ω')}")


def _walk(node: list, head: str):
    """Yield every sub-list with the given head, depth first (bounded by the parser depth)."""
    stack = [node]
    while stack:
        current = stack.pop(0)
        for child in current:
            if isinstance(child, list):
                if _head(child) == head:
                    yield child
                stack.append(child)


# --------------------------------------------------------------------------- diff


def _unit(kind: str) -> str:
    return {"R": "Ω", "C": "F", "L": "H"}.get(kind, "")


def _diff(original: GeneralCircuitIR | None, new: dict[str, Any]) -> list[dict[str, Any]]:
    if original is None:
        return []
    old = {p.ref.casefold(): p.model_dump(mode="json") for p in original.components}
    now = {p["ref"].casefold(): p for p in new["components"]}
    changes: list[dict[str, Any]] = []
    for key, part in old.items():
        if key not in now:
            changes.append({"type": "removed", "ref": part["ref"], "text": f"removed {part['ref']} ({part['kind']})"})
    for key, part in now.items():
        if key not in old:
            what = part.get("model") or _engineering(part["value"], _unit(part["kind"]))
            changes.append({"type": "added", "ref": part["ref"], "kind": part["kind"],
                            "nodes": part["nodes"], "text": f"added {part['ref']} ({part['kind']} {what}) "
                                                            f"between {' / '.join(part['nodes'])}"})
            continue
        before = old[key]
        for fld in ("value", "model", "series_resistance_ohm", "package", "role"):
            if before.get(fld) != part.get(fld):
                a, b = before.get(fld), part.get(fld)
                if fld in ("value", "series_resistance_ohm") and isinstance(a, (int, float)) and isinstance(b, (int, float)):
                    shown = f"{_engineering(a, _unit(part['kind']) if fld == 'value' else 'Ω')} -> " \
                            f"{_engineering(b, _unit(part['kind']) if fld == 'value' else 'Ω')}"
                else:
                    shown = f"{a} -> {b}"
                changes.append({"type": "changed", "ref": part["ref"], "field": fld, "old": a, "new": b,
                                "text": f"{part['ref']}.{fld}: {shown}"})
        for pin, (a, b) in enumerate(zip(before["nodes"], part["nodes"], strict=False), 1):
            if a != b:
                changes.append({"type": "changed", "ref": part["ref"], "field": f"pin{pin}", "old": a, "new": b,
                                "text": f"{part['ref']} pin {pin}: net {a} -> {b}"})
    old_wires = {tuple(sorted(w)) for w in original.connections}
    new_wires = {tuple(sorted(w)) for w in new["connections"]}
    for wire in sorted(old_wires - new_wires):
        changes.append({"type": "connection_removed", "pair": list(wire), "text": f"removed ideal wire {wire[0]} - {wire[1]}"})
    for wire in sorted(new_wires - old_wires):
        changes.append({"type": "connection_added", "pair": list(wire), "text": f"added ideal wire {wire[0]} - {wire[1]}"})
    return changes
