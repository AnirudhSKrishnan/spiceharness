"""Terminal views and a BOM derived directly from circuit connectivity."""

from __future__ import annotations

import csv
import io
from collections import defaultdict

from .domain import Circuit


def format_value(value: float | None, kind: str = "") -> str:
    if value is None:
        return ""
    suffix = {"R": "Ω", "C": "F", "L": "H"}.get(kind, "")
    for exponent, prefix in (
        (12, "T"),
        (9, "G"),
        (6, "M"),
        (3, "k"),
        (0, ""),
        (-3, "m"),
        (-6, "µ"),
        (-9, "n"),
        (-12, "p"),
        (-15, "f"),
    ):
        scaled = value / (10**exponent)
        if scaled >= 0.999 or exponent == -15:
            return f"{scaled:.4g} {prefix}{suffix}".strip()
    return str(value)


def export_bom(circuit: Circuit) -> list[dict]:
    """One row per physical package, including bias and decoupling parts."""
    grouped = defaultdict(list)
    for p in circuit.components:
        grouped[p.package or p.ref].append(p)
    result = []
    for package, parts in grouped.items():
        p = parts[0]
        result.append(
            {
                "reference": package,
                "references": ", ".join(x.ref for x in parts),
                "kind": p.kind,
                "value": p.value,
                "display_value": p.model or format_value(p.value, p.kind),
                "model": p.model,
                "quantity": 1,
                "role": p.role,
                "channels_used": len(parts),
            }
        )
    return result


def export_bom_csv(circuit: Circuit) -> str:
    rows = export_bom(circuit)
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()


def connection_table(circuit: Circuit) -> str:
    rows = [("Ref", "Device / value", "Connections (ordered pins)")]
    for p in circuit.components:
        description = p.model or format_value(p.value, p.kind)
        if p.kind == "OPAMP":
            pins = "+, −, V+, V−, OUT"
        elif p.kind == "BJT":
            pins = "C, B, E"
        elif p.kind == "DIODE":
            pins = "A, K"
        else:
            pins = "1, 2"
        rows.append((p.ref, description, f"{', '.join(p.nodes)}  [{pins}]"))
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    return "\n".join(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows
    )


def render_schematic(circuit: Circuit) -> str:
    """Render every component as an exact connectivity glyph, never inferred wires."""
    lines = [
        f"{circuit.name}  ({circuit.topology})",
        f"Input: {circuit.input_node}   Output: {circuit.output_node}",
        "Fixture: source + source impedance → input; output → declared load → 0",
        "Ground: 0   Supply rails: vcc / vee",
        "",
    ]
    symbols = {"R": "[▭]", "C": "┤├", "L": "∿∿", "DIODE": "▷│"}
    for p in circuit.components:
        label = f"{p.ref} {p.model or format_value(p.value, p.kind)}"
        if p.kind in symbols:
            lines.append(
                f"{p.nodes[0]:>12} ──{symbols[p.kind]}── {p.nodes[1]:<12} {label}"
                + ("  (decoupling)" if p.role == "decoupling" else "")
            )
        elif p.kind == "OPAMP":
            plus, minus, vp, vn, out = p.nodes
            lines.extend(
                [
                    f"{plus:>12} ──(+)╲",
                    f"{minus:>12} ──(−) ▷── {out}   {label} [{p.package or p.ref}]",
                    f"                 ╱     rails: {vp}, {vn}",
                ]
            )
        elif p.kind == "BJT":
            c, b, e = p.nodes
            lines.append(f"{b:>12} ──B [ {label} ] C──{c}  E──{e}")
    lines.extend(
        [
            "",
            "Exact connection table:",
            connection_table(circuit),
            "",
            f"BOM: {len(export_bom(circuit))} physical packages; evaluator source, supplies, source impedance, and load are test fixtures.",
        ]
    )
    return "\n".join(lines)


render_circuit = render_schematic
