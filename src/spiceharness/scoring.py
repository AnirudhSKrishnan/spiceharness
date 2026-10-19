"""Deterministic feasibility-first scoring. No language model participates."""

from __future__ import annotations

import math

from .domain import Circuit, DesignSpec, Scorecard, SimulationResult
from .simulator import expected_corner_labels, validate_circuit


def part_counts(circuit: Circuit) -> tuple[int, int]:
    packages = {p.package or p.ref for p in circuit.components}
    active = {p.package or p.ref for p in circuit.components if p.kind in ("OPAMP", "BJT", "DIODE")}
    return len(packages), len(active)


def score_result(spec: DesignSpec, circuit: Circuit, result: SimulationResult) -> Scorecard:
    packages, active = part_counts(circuit)
    explanations = []
    required = {
        "gain",
        "gain_at_input_frequency",
        "phase_deg",
        "thd",
        "gain_compression",
        "output_offset_v",
        "supply_current_a",
        "output_swing_v",
        "transient_gain",
    }
    if spec.cutoff_hz is not None:
        required.add("cutoff_hz")
    if spec.stopband_frequency_hz is not None:
        required.add("stopband_gain")
    if spec.max_output_offset_v is not None:
        required.add("output_offset_v")
    sets = {"nominal": result.metrics}
    structural = validate_circuit(circuit, spec)
    valid = result.status == "ok" and not structural
    explanations.extend(structural)
    if result.spec_fingerprint != spec.fingerprint:
        valid = False
        explanations.append("simulation belongs to a different specification")
    if spec.corner_policy == "tolerance":
        corners = result.analyses.get("corners", {})
        if set(corners) != expected_corner_labels(spec, circuit):
            valid = False
            explanations.append("mandatory tolerance cases are missing")
        for label, entry in corners.items():
            if entry.get("status") != "ok":
                valid = False
                explanations.append(f"{label}: simulator did not succeed")
            sets[label] = entry.get("metrics", {})
    for label, metrics in sets.items():
        missing = required - set(metrics)
        if missing:
            valid = False
            explanations.append(f"{label}: missing mandatory measurements: {', '.join(sorted(missing))}")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in metrics.values()):
            valid = False
            explanations.append(f"{label}: nonfinite or invalid measurement")
    if not valid:
        explanations.extend(result.errors)
        return Scorecard(
            valid=False,
            feasible=False,
            violations={"invalid_evaluation": 1e12},
            max_violation=1e12,
            sum_violation=1e12,
            package_count=packages,
            active_count=active,
            rank_key=(2, 1e12, 1e12, packages, active),
            explanation=explanations,
            metrics={
                k: v for k, v in result.metrics.items() if isinstance(v, (int, float)) and math.isfinite(v)
            },
        )
    violations = {}
    margins = []
    for label, metrics in sets.items():

        def upper(name: str, value: float, limit: float, scale: float, label: str = label) -> None:
            violation = max(0.0, (value - limit) / max(scale, 1e-15))
            violations[f"{label}.{name}"] = violation
            margins.append(max(0.0, min(1.0, (limit - value) / max(scale, 1e-15))))

        def target(name: str, value: float, desired: float, tolerance: float) -> None:
            upper(name, abs(value - desired), abs(desired) * tolerance, abs(desired) * tolerance)

        desired = abs(spec.gain) if spec.gain is not None else 1.0
        target("gain", metrics["gain"], desired, spec.gain_tolerance)
        if spec.cutoff_hz is not None:
            target("cutoff_hz", metrics["cutoff_hz"], spec.cutoff_hz, spec.cutoff_tolerance)
        if spec.max_stopband_gain is not None:
            upper("stopband_gain", metrics["stopband_gain"], spec.max_stopband_gain, spec.max_stopband_gain)
        upper("thd", metrics["thd"], spec.max_thd, spec.max_thd)
        upper("gain_compression", metrics["gain_compression"], 0.05, 0.05)
        if spec.max_supply_current_a is not None:
            upper(
                "supply_current_a",
                metrics["supply_current_a"],
                spec.max_supply_current_a,
                spec.max_supply_current_a,
            )
        if spec.max_output_offset_v is not None:
            upper(
                "output_offset_v",
                abs(metrics["output_offset_v"]),
                spec.max_output_offset_v,
                max(spec.max_output_offset_v, 0.001),
            )
        if spec.min_output_swing_v is not None:
            upper(
                "minimum_output_swing_v",
                spec.min_output_swing_v - metrics["output_swing_v"],
                0,
                spec.min_output_swing_v,
            )
        if spec.gain is not None and "pass" not in spec.topology:
            expected = 180 if spec.gain < 0 else 0
            distance = abs((metrics["phase_deg"] - expected + 180) % 360 - 180)
            upper("gain_polarity", distance, 45, 45)
    if spec.max_parts is not None:
        violations["package_count"] = max(0.0, (packages - spec.max_parts) / spec.max_parts)
    vmax = max(violations.values(), default=0.0)
    vsum = sum(violations.values())
    feasible = vmax == 0
    margin = min(margins, default=0)
    power = result.metrics.get(
        "supply_power_w",
        result.metrics["supply_current_a"] * (spec.supply_positive_v - spec.supply_negative_v),
    )
    key = (
        (0.0, float(packages), -margin, power)
        if feasible
        else (1.0, vmax, vsum, float(packages), float(active))
    )
    breakdown = {
        "feasibility": 1000.0 if feasible else 0.0,
        "simplicity": 100 / (1 + packages) if feasible else 0.0,
        "robustness": 0.0001 * margin if feasible else 0.0,
        "power": 0.000001 / (1 + power) if feasible else 0.0,
        "progress": 0.0 if feasible else 999 / (1 + vmax + vsum),
    }
    explanations.extend(
        f"{metric} violates its bound by {amount:.4g} tolerance units"
        for metric, amount in violations.items()
        if amount > 0
    )
    return Scorecard(
        valid=True,
        feasible=feasible,
        violations=violations,
        max_violation=vmax,
        sum_violation=vsum,
        package_count=packages,
        active_count=active,
        rank_key=key,
        explanation=explanations,
        metrics=result.metrics,
        points=sum(breakdown.values()),
        points_breakdown=breakdown,
    )


score = score_result


def rank_candidates(
    candidates: list[tuple[Circuit, SimulationResult]], spec: DesignSpec
) -> list[tuple[Circuit, SimulationResult, Scorecard]]:
    return sorted(
        ((c, r, score_result(spec, c, r)) for c, r in candidates), key=lambda item: item[2].rank_key
    )
