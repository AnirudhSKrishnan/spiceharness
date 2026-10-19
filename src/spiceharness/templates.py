"""Bounded templates using real devices and explicit external components."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .domain import Circuit, Component, DesignSpec, normalize_spec

E24 = (
    1,
    1.1,
    1.2,
    1.3,
    1.5,
    1.6,
    1.8,
    2,
    2.2,
    2.4,
    2.7,
    3,
    3.3,
    3.6,
    3.9,
    4.3,
    4.7,
    5.1,
    5.6,
    6.2,
    6.8,
    7.5,
    8.2,
    9.1,
)
E12 = (1, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2)


def standard_value(value: float, series: tuple[float, ...] = E24) -> float:
    """Nearest E24 value; logarithmic distance avoids decade-boundary bias."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError("component value must be finite and positive")
    decade = math.floor(math.log10(value))
    return min(
        (x * 10**d for d in (decade - 1, decade, decade + 1) for x in series),
        key=lambda x: abs(math.log(x / value)),
    )


def shortlist_templates(spec: DesignSpec) -> list[str]:
    spec = normalize_spec(spec)
    if spec.topology in ("lowpass", "highpass"):
        suffix = spec.topology
        return ([f"passive_{suffix}", f"active_{suffix}_1"] if spec.min_filter_order == 1 else []) + [
            f"active_{suffix}_2"
        ]
    if spec.topology == "amplifier":
        if spec.gain < 0:
            return ["inverting_amplifier", "common_emitter"]
        return (["buffer", "emitter_follower"] if spec.gain == 1 else []) + ["noninverting_amplifier"]
    return [spec.topology]


def make_candidate(spec: DesignSpec, variant: int = 0) -> Circuit:
    spec = normalize_spec(spec)
    parts: list[Component] = []
    catalog = json.loads((Path(__file__).parent / "data" / "models" / "catalog.json").read_text())
    channels = int(catalog.get(spec.opamp_model, {}).get("channels", 1))

    def add(ref: str, kind: str, nodes: tuple[str, ...], value: float | None = None, **kwargs) -> None:
        parts.append(Component(ref=ref, kind=kind, nodes=nodes, value=value, **kwargs))

    def resistor(ref: str, a: str, b: str, v: float, role: str = "signal") -> None:
        add(ref, "R", (a, b), standard_value(v), role=role)

    def capacitor(ref: str, a: str, b: str, v: float, role: str = "signal") -> None:
        add(ref, "C", (a, b), standard_value(v, E12), role=role)

    def amp(ref: str, plus: str, minus: str, out: str) -> None:
        package = "U" + str(sum(p.kind == "OPAMP" for p in parts) // channels + 1)
        add(ref, "OPAMP", (plus, minus, "vcc", "vee", out), model=spec.opamp_model, package=package)

    families = shortlist_templates(spec)
    topology = families[(variant // 6) % len(families)]
    rbase = (1e3, 2.2e3, 4.7e3, 10e3, 22e3, 47e3)[variant % 6]
    if topology in ("passive_lowpass", "passive_highpass"):
        fc = spec.cutoff_hz
        if topology.endswith("lowpass"):
            resistor("R1", "in", "out", rbase)
            effective = (
                (rbase + spec.source_resistance_ohm)
                * spec.load_ohm
                / (rbase + spec.source_resistance_ohm + spec.load_ohm)
            )
            capacitor("C1", "out", "0", 1 / (2 * math.pi * fc * effective))
        else:
            capacitor(
                "C1",
                "in",
                "out",
                1
                / (
                    2
                    * math.pi
                    * fc
                    * (spec.source_resistance_ohm + rbase * spec.load_ohm / (rbase + spec.load_ohm))
                ),
            )
            resistor("R1", "out", "0", rbase)
    elif topology in ("common_emitter", "emitter_follower"):
        span = spec.supply_positive_v - spec.supply_negative_v
        current = (0.0005, 0.001, 0.002)[variant % 3]
        ve = span * (0.1 if variant % 6 < 3 else 0.15)
        re = ve / current
        vb = spec.supply_negative_v + ve + 0.65
        divider_current = current / 10
        resistor("RB1", "vcc", "base", max((spec.supply_positive_v - vb) / divider_current, 100))
        resistor("RB2", "base", "vee", max((vb - spec.supply_negative_v) / divider_current, 100))
        capacitor("CIN", "in", "base", max(10 / (2 * math.pi * spec.input_frequency_hz * 1000), 1e-6))
        resistor("RE", "emitter", "vee", re)
        if topology == "common_emitter":
            target = abs(spec.gain or -5)
            desired = target * (re + 0.026 / current)
            rc = min(desired / max(1 - desired / spec.load_ohm, 0.1), span * 0.5 / current)
            resistor("RC", "vcc", "collector", rc)
            collector = "collector"
            output = "collector"
        else:
            collector = "vcc"
            output = "emitter"
        add("Q1", "BJT", (collector, "base", "emitter"), model=spec.transistor_model)
        capacitor(
            "COUT", output, "out", max(10 / (2 * math.pi * spec.input_frequency_hz * spec.load_ohm), 1e-6)
        )
        resistor("RDC", "out", "0", 1e6, role="bias")
        capacitor("CDEC1", "vcc", "0", 100e-9, role="decoupling")
        if spec.supply_negative_v != 0:
            capacitor("CDEC2", "vee", "0", 100e-9, role="decoupling")
    else:
        # Bipolar rails reference the signal to ground. On single supply, this
        # real divider is a counted part of the DUT; it is never an ideal source.
        reference = "0"
        if spec.supply_negative_v >= 0:
            reference = "vref"
            resistor("RBIAS1", "vcc", reference, 10e3, "bias")
            resistor("RBIAS2", reference, "vee", 10e3, "bias")
            capacitor("CBIAS", reference, "vee", 10e-6, "bias")
        if topology == "buffer":
            amp("U1A", "in", "out", "out")
        elif topology == "inverting_amplifier":
            resistor("RIN", "in", "inv", rbase)
            resistor(
                "RF", "out", "inv", abs(spec.gain) * (standard_value(rbase) + spec.source_resistance_ohm)
            )
            amp("U1A", reference, "inv", "out")
        elif topology == "noninverting_amplifier":
            if spec.gain == 1:
                amp("U1A", "in", "out", "out")
            else:
                resistor("RG", "inv", reference, rbase)
                resistor("RF", "out", "inv", (spec.gain - 1) * standard_value(rbase))
                amp("U1A", "in", "inv", "out")
        elif topology.startswith("active_"):
            order = int(topology[-1])
            high = "highpass" in topology
            # Two buffered real poles have an exact overall -3 dB frequency.
            pole = (
                spec.cutoff_hz
                * ((math.sqrt(math.sqrt(2) - 1)) if high else (1 / math.sqrt(math.sqrt(2) - 1)))
                if order == 2
                else spec.cutoff_hz
            )
            incoming = "in"
            for stage in range(1, order + 1):
                node = f"filter{stage}"
                outgoing = "out" if stage == order else "stage1"
                resistor(f"R{stage}", node, reference, rbase) if high else resistor(
                    f"R{stage}", incoming, node, rbase
                )
                capacitor(
                    f"C{stage}", incoming, node, 1 / (2 * math.pi * pole * rbase)
                ) if high else capacitor(f"C{stage}", node, reference, 1 / (2 * math.pi * pole * rbase))
                amp(f"U1{chr(64 + stage)}", node, outgoing, outgoing)
                incoming = outgoing
            target = abs(spec.gain or 1)
            if target > 1:
                # Set the final buffer's noninverting gain with explicit parts.
                last = parts[-1]
                parts[-1] = last.model_copy(update={"nodes": (last.nodes[0], "inv", "vcc", "vee", "out")})
                resistor("RG", "inv", reference, rbase)
                resistor("RF", "out", "inv", (target - 1) * standard_value(rbase))
        else:
            raise ValueError(f"unsupported template {topology}")
        capacitor("CDEC1", "vcc", "0", 100e-9, role="decoupling")
        if spec.supply_negative_v != 0:
            capacitor("CDEC2", "vee", "0", 100e-9, role="decoupling")
        # Populate and terminate every channel of the last physical package.
        # Packaging must not hide an unpowered channel or its supply current.
        used_channels = sum(p.kind == "OPAMP" for p in parts)
        for spare in range((-used_channels) % channels):
            add(
                f"USPARE{spare + 1}",
                "OPAMP",
                (reference, f"spare_out{spare + 1}", "vcc", "vee", f"spare_out{spare + 1}"),
                model=spec.opamp_model,
                package="U" + str((used_channels - 1) // channels + 1),
                role="unused_channel",
            )
    return Circuit(
        name=f"{topology}_{variant:02d}",
        topology=topology,
        components=tuple(parts),
        spec=spec,
        metadata={
            "template": topology,
            "variant": variant,
            "filter_order": int(topology[-1]) if topology.startswith("active_") else 1,
        },
    )


def generate_candidates(spec: DesignSpec) -> list[Circuit]:
    spec = normalize_spec(spec)
    candidates = []
    seen = set()
    # Interleave topologies so a small budget still compares package counts.
    variants = [
        family * 6 + variant for variant in range(6) for family in range(len(shortlist_templates(spec)))
    ]
    for variant in variants[: spec.max_candidates]:
        candidate = make_candidate(spec, variant)
        key = tuple((p.kind, p.nodes, p.value, p.model) for p in candidate.components)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def seed_candidates(spec: DesignSpec, family: str | None = None) -> list[Circuit]:
    return [c for c in generate_candidates(spec) if family is None or c.topology == family]
