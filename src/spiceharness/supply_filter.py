"""A dedicated, deterministic passive supply-filter study using real ngspice.

The load is a resistor approximation, not a UA741 model. One positive rail is
measured; reported dual-rail losses assume identical independent ± rails.
AC results are a sampled nominal screen, not exhaustive worst-case analysis.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from .domain import Circuit, Component, StrictModel
from .simulator import _read_raw

FIXTURE_VERSION = "supply-filter-resistor-fixture-v1"
TOPOLOGIES = {"supply_direct", "supply_capacitor", "supply_rc", "supply_lc", "supply_t", "supply_pi"}


class SupplyFilterSpec(StrictModel):
    supply_v: float = Field(default=12, gt=0, le=60)
    source_r_ohm: float = Field(default=1, gt=0, le=1e6)
    dc_current_a: float = Field(default=0.003, gt=0, le=1)
    load_ac_r_ohm: float = Field(default=4000, gt=0, le=1e9)
    sweep_start_hz: float = Field(default=10, gt=0)
    sweep_stop_hz: float = Field(default=1e6, gt=0, le=1e9)
    points_per_decade: int = Field(default=200, ge=50, le=2000)
    stopband_start_hz: float = Field(default=1000, gt=0)
    stopband_stop_hz: float = Field(default=100000, gt=0)
    min_stopband_attenuation_db: float = Field(default=20, ge=0, le=300)
    max_peak_db: float = Field(default=3, ge=0, le=100)
    max_dc_drop_v: float = Field(default=0.1, ge=0, le=60)
    spectral_frequencies_hz: tuple[float, ...] = (100, 1000, 10000, 100000)

    @model_validator(mode="after")
    def coherent(self) -> SupplyFilterSpec:
        if not self.sweep_start_hz <= self.stopband_start_hz < self.stopband_stop_hz <= self.sweep_stop_hz:
            raise ValueError("stopband must lie inside the increasing sweep")
        if not math.isclose(self.supply_v / self.load_ac_r_ohm, self.dc_current_a, rel_tol=1e-9):
            raise ValueError("resistor fixture requires dc_current_a = supply_v / load_ac_r_ohm")
        if not self.spectral_frequencies_hz or any(
            not math.isfinite(f) or not self.sweep_start_hz <= f <= self.sweep_stop_hz
            for f in self.spectral_frequencies_hz
        ):
            raise ValueError("spectral frequencies must lie in the measured sweep")
        return self

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


class SupplyFilterResult(StrictModel):
    name: str
    topology: str
    status: Literal["ok", "invalid", "failed", "timeout", "unavailable"]
    feasible: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)
    spectral: dict[str, dict[str, float]] = Field(default_factory=dict)
    violations: dict[str, float] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    physical_components_per_rail: int = 0
    modeled_components_per_rail: int = 0
    artifact_dir: str | None = None
    spec_fingerprint: str | None = None
    fixture_fingerprint: str | None = None
    simulator_version: str | None = None


def _values(value: float | tuple[float, ...], count: int) -> tuple[float, ...]:
    result = (value,) * count if isinstance(value, (int, float)) else tuple(value)
    if len(result) != count:
        raise ValueError(f"expected {count} values")
    return result


def make_supply_filter(
    topology: str,
    *,
    name: str | None = None,
    resistance_ohm: float = 10,
    inductance_h: float | tuple[float, ...] = 0.01,
    capacitance_f: float | tuple[float, ...] = 100e-6,
    dcr_ohm: float | tuple[float, ...] = 1,
    esr_ohm: float | tuple[float, ...] = 0.1,
    damping_r_ohm: float | None = None,
    damping_c_f: float | None = None,
) -> Circuit:
    """Build validated fixed templates; ESR/DCR are explicit modeled resistors.

    All L/C require positive specified ESR/DCR. The capacitor/direct templates
    use an explicit 1 µΩ link approximation, with its loss retained in metrics.
    Optional series R-C shunt damping consumes two additional physical parts.
    """
    if topology not in TOPOLOGIES:
        raise ValueError("unsupported supply-filter topology")
    parts = []

    def part(ref, kind, a, b, value, role="signal"):
        parts.append(Component(ref=ref, kind=kind, nodes=(a, b), value=value, role=role))

    def inductor(index, a, b, value, dcr):
        node = f"l{index}_dcr"
        part(f"L{index}", "L", a, node, value)
        part(f"RDCR_L{index}", "R", node, b, dcr, "parasitic")

    def capacitor(index, a, value, esr):
        node = f"c{index}_esr"
        part(f"C{index}", "C", a, node, value)
        part(f"RESR_C{index}", "R", node, "0", esr, "parasitic")

    nc = 0 if topology == "supply_direct" else 2 if topology == "supply_pi" else 1
    nl = 2 if topology == "supply_t" else 1 if topology in {"supply_lc", "supply_pi"} else 0
    cs, esrs = _values(capacitance_f, nc), _values(esr_ohm, nc)
    ls, dcrs = _values(inductance_h, nl), _values(dcr_ohm, nl)
    if topology in {"supply_direct", "supply_capacitor"}:
        part("RLINK", "R", "in", "out", 1e-6, "baseline_link")
    elif topology == "supply_rc":
        part("R1", "R", "in", "out", resistance_ohm)
    elif topology == "supply_t":
        inductor(1, "in", "mid", ls[0], dcrs[0])
        inductor(2, "mid", "out", ls[1], dcrs[1])
    else:
        inductor(1, "in", "out", ls[0], dcrs[0])
    if nc:
        capacitor(1, "mid" if topology == "supply_t" else "in" if nc == 2 else "out", cs[0], esrs[0])
    if nc == 2:
        capacitor(2, "out", cs[1], esrs[1])
    if (damping_r_ohm is None) != (damping_c_f is None):
        raise ValueError("damping resistor and capacitor must be supplied together")
    if damping_r_ohm is not None:
        if topology not in {"supply_lc", "supply_t", "supply_pi"}:
            raise ValueError("damping is supported only for inductive filters")
        part("RDAMP", "R", "out", "damp", damping_r_ohm)
        part("CDAMP", "C", "damp", "0", damping_c_f)
    return Circuit(name=name or topology, topology=topology, components=tuple(parts))


def validate_supply_filter(circuit: Circuit) -> Circuit:
    """Revalidate even model_copy/model_construct inputs and require exact shapes."""
    circuit = Circuit.model_validate(circuit.model_dump(mode="python"))
    if circuit.input_node != "in" or circuit.output_node != "out" or circuit.spec or circuit.metadata:
        raise ValueError("supply candidates cannot change terminals, specification, or fixture metadata")
    if circuit.topology not in TOPOLOGIES:
        raise ValueError("unsupported supply-filter topology")
    p = {part.ref: part for part in circuit.components}

    def values(prefix, count, default):
        return tuple(p[f"{prefix}{i}"].value for i in range(1, count + 1)) if count else default

    nl = 2 if circuit.topology == "supply_t" else 1 if circuit.topology in {"supply_lc", "supply_pi"} else 0
    nc = 0 if circuit.topology == "supply_direct" else 2 if circuit.topology == "supply_pi" else 1
    try:
        expected = make_supply_filter(
            circuit.topology,
            name=circuit.name,
            resistance_ohm=p["R1"].value if "R1" in p else 10,
            inductance_h=values("L", nl, 0.01),
            dcr_ohm=values("RDCR_L", nl, 1),
            capacitance_f=values("C", nc, 100e-6),
            esr_ohm=values("RESR_C", nc, 0.1),
            damping_r_ohm=p["RDAMP"].value if "RDAMP" in p else None,
            damping_c_f=p["CDAMP"].value if "CDAMP" in p else None,
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("candidate is missing required template components") from exc
    if sorted(circuit.components, key=lambda x: x.ref) != sorted(expected.components, key=lambda x: x.ref):
        raise ValueError("candidate differs from the validated passive template or parasitic associations")
    return circuit


def analytical_response(circuit: Circuit, spec: SupplyFilterSpec, frequencies: np.ndarray) -> np.ndarray:
    """Independent complex nodal analysis of the same validated passive IR."""
    circuit = validate_supply_filter(circuit)
    spec = SupplyFilterSpec.model_validate(spec.model_dump(mode="python"))
    nodes = sorted({n for p in circuit.components for n in p.nodes} - {"0"})
    index = {n: i for i, n in enumerate(nodes)}
    result = []
    for frequency in frequencies:
        if not math.isfinite(frequency) or frequency <= 0:
            raise ValueError("AC frequencies must be finite and positive")
        matrix = np.zeros((len(nodes), len(nodes)), complex)
        rhs = np.zeros(len(nodes), complex)
        matrix[index["in"], index["in"]] += 1 / spec.source_r_ohm
        matrix[index["out"], index["out"]] += 1 / spec.load_ac_r_ohm
        rhs[index["in"]] = 1 / spec.source_r_ohm
        for part in circuit.components:
            w = 2 * math.pi * frequency
            y = 1 / part.value if part.kind == "R" else 1j * w * part.value if part.kind == "C" else 1 / (1j * w * part.value)
            a, b = part.nodes
            if a != "0":
                matrix[index[a], index[a]] += y
            if b != "0":
                matrix[index[b], index[b]] += y
            if a != "0" and b != "0":
                matrix[index[a], index[b]] -= y
                matrix[index[b], index[a]] -= y
        result.append(np.linalg.solve(matrix, rhs)[index["out"]])
    return np.array(result)


class SupplyFilterEvaluator:
    def __init__(self, artifact_root: str | Path | None = None, executable: str = "ngspice", timeout_s: float = 20):
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("timeout_s must be positive and at most 300")
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.executable = executable
        self.timeout_s = timeout_s

    @staticmethod
    def generate_netlist(circuit: Circuit, spec: SupplyFilterSpec, analysis: str) -> str:
        circuit = validate_supply_filter(circuit)
        spec = SupplyFilterSpec.model_validate(spec.model_dump(mode="python"))
        lines = [
            "spiceharness resistor-load supply-filter fixture",
            f"* {FIXTURE_VERSION} spec {spec.fingerprint}",
            ".options numdgt=15 reltol=1e-8",
            f"VFIX source 0 DC {spec.supply_v:.17g} AC 1",
            f"RFIXSOURCE source in {spec.source_r_ohm:.17g}",
            f"RFIXLOAD out 0 {spec.load_ac_r_ohm:.17g}",
        ]
        for part in circuit.components:
            lines.append(f"{part.ref} {' '.join(part.nodes)} {part.value:.17g}")
        if analysis == "op":
            lines.append(".op")
        elif analysis == "ac":
            lines.append(f".ac dec {spec.points_per_decade} {spec.sweep_start_hz:.17g} {spec.sweep_stop_hz:.17g}")
        else:
            raise ValueError("only evaluator-owned op and ac analyses are supported")
        nodes = sorted({n for part in circuit.components for n in part.nodes} - {"0"})
        lines.extend([".save v(source) i(VFIX) " + " ".join(f"v({n})" for n in nodes), ".end"])
        return "\n".join(lines) + "\n"

    def evaluate(self, circuit: Circuit, spec: SupplyFilterSpec | None = None) -> SupplyFilterResult:
        base = {"name": circuit.name, "topology": circuit.topology}
        try:
            circuit = validate_supply_filter(circuit)
            spec = SupplyFilterSpec.model_validate((spec or SupplyFilterSpec()).model_dump(mode="python"))
        except (ValueError, TypeError, AttributeError) as exc:
            return SupplyFilterResult(status="invalid", errors=(str(exc),), **base)
        base.update(
            spec_fingerprint=spec.fingerprint,
            fixture_fingerprint=hashlib.sha256((FIXTURE_VERSION + spec.fingerprint).encode()).hexdigest(),
            modeled_components_per_rail=len(circuit.components),
            physical_components_per_rail=sum(p.role not in {"parasitic", "baseline_link"} for p in circuit.components),
        )
        if self.artifact_root:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="supply-filter-", dir=self.artifact_root))
        base["artifact_dir"] = str(directory)
        (directory / "spec.json").write_text(spec.model_dump_json(indent=2))
        (directory / "circuit.json").write_text(circuit.model_dump_json(indent=2))
        (directory / ".spiceinit").write_text("set filetype=ascii\n")
        data = {}
        try:
            if not shutil.which(self.executable):
                result = SupplyFilterResult(status="unavailable", errors=("ngspice executable unavailable",), **base)
            else:
                for analysis in ("op", "ac"):
                    (directory / f"{analysis}.cir").write_text(self.generate_netlist(circuit, spec, analysis))
                    process = subprocess.run(
                        [self.executable, "-b", "-r", f"{analysis}.raw", f"{analysis}.cir"],
                        cwd=directory, env=dict(os.environ, HOME=str(directory), SPICE_ASCIIRAWFILE="1"),
                        capture_output=True, text=True, timeout=self.timeout_s, check=False,
                    )
                    log = process.stdout + "\n" + process.stderr
                    (directory / f"{analysis}.log").write_text(log)
                    if process.returncode or re.search(r"fatal error|singular matrix|doanalyses:|simulation interrupted", log, re.IGNORECASE):
                        raise ValueError(f"{analysis} failed: {log[-1500:]}")
                    names, samples = _read_raw(directory / f"{analysis}.raw")
                    data[analysis] = {name: samples[:, i] for i, name in enumerate(names)}
                version = re.search(r"^Command:\s*(.*)$", (directory / "op.raw").read_text(), re.MULTILINE)
                base["simulator_version"] = version.group(1).strip() if version else "ngspice"
                result = self._measure(circuit, spec, data, base, directory)
        except subprocess.TimeoutExpired as exc:
            result = SupplyFilterResult(status="timeout", errors=(str(exc),), **base)
        except (ValueError, OSError, KeyError, IndexError, np.linalg.LinAlgError) as exc:
            result = SupplyFilterResult(status="failed", errors=(str(exc),), **base)
        (directory / "result.json").write_text(result.model_dump_json(indent=2))
        return result

    @staticmethod
    def _measure(circuit, spec, data, base, directory):
        op, ac = data["op"], data["ac"]
        f = ac["frequency"].real
        if len(f) < 2 or not np.all(np.diff(f) > 0):
            raise ValueError("invalid AC frequency scale")
        if f[0] > spec.sweep_start_hz * (1 + 1e-9) or f[-1] < spec.sweep_stop_hz * (1 - 1e-9):
            raise ValueError("AC sweep is incomplete")
        gain = ac["v(out)"] / ac["v(source)"]
        reference = analytical_response(circuit, spec, f)
        if not np.allclose(gain, reference, rtol=2e-5, atol=1e-8):
            raise ValueError("ngspice AC response disagrees with independent nodal analysis")
        db = 20 * np.log10(np.maximum(np.abs(gain), 1e-300))
        band = (f >= spec.stopband_start_hz * (1 - 1e-9)) & (f <= spec.stopband_stop_hz * (1 + 1e-9))
        if not np.any(band):
            raise ValueError("no measured stopband samples")

        def voltage(dataset, node):
            return np.zeros(len(dataset["v(out)"])) if node == "0" else dataset[f"v({node})"]

        dc_loss = 0.0
        ac_power = np.zeros(len(f))
        for part in circuit.components:
            if part.kind == "R":
                dc_loss += float(abs(voltage(op, part.nodes[0])[0] - voltage(op, part.nodes[1])[0]) ** 2 / part.value)
                ac_power += abs(voltage(ac, part.nodes[0]) - voltage(ac, part.nodes[1])) ** 2 / part.value
        metrics = {
            "dc_input_v": float(op["v(in)"][0].real),
            "dc_output_v": float(op["v(out)"][0].real),
            "dc_load_current_a": float(op["v(out)"][0].real / spec.load_ac_r_ohm),
            "dc_drop_v": float((op["v(in)"][0] - op["v(out)"][0]).real),
            "added_dc_loss_w_per_rail": dc_loss,
            "added_dc_loss_w_dual_rail": 2 * dc_loss,
            "source_dc_loss_w_per_rail": float(abs(op["i(vfix)"][0]) ** 2 * spec.source_r_ohm),
            "peak_gain_db": float(np.max(db)),
            "peak_frequency_hz": float(f[np.argmax(db)]),
            "worst_stopband_attenuation_db": float(-np.max(db[band])),
            "worst_stopband_frequency_hz": float(f[band][np.argmax(db[band])]),
        }
        violations = {
            k: float(v) for k, v in {
                "dc_drop_v": max(0, metrics["dc_drop_v"] - spec.max_dc_drop_v),
                "peak_db": max(0, metrics["peak_gain_db"] - spec.max_peak_db),
                "stopband_db": max(0, spec.min_stopband_attenuation_db - metrics["worst_stopband_attenuation_db"]),
            }.items() if v > 0
        }
        spectral = {}
        for frequency in spec.spectral_frequencies_hz:
            def interpolate(values, frequency=frequency):
                return float(np.interp(math.log(frequency), np.log(f), values))

            spectral[f"{frequency:g}"] = {
                "attenuation_db": -interpolate(db),
                "output_noise_v_rms_per_input_v_rms": interpolate(np.abs(gain)),
                "load_noise_power_w_per_input_v_rms_squared": interpolate(abs(gain) ** 2 / spec.load_ac_r_ohm),
                "filter_noise_dissipation_w_per_input_v_rms_squared": interpolate(ac_power),
                "source_noise_dissipation_w_per_input_v_rms_squared": interpolate(
                    abs(ac["v(source)"] - ac["v(in)"]) ** 2 / spec.source_r_ohm
                ),
            }
        # AC=1 is an amplitude transfer; these power coefficients are evaluated
        # for 1 Vrms noise by |H|²/R, not incorrectly treating it as 1 Vpeak.
        curves = {
            "frequency_hz": f.tolist(), "gain_real": gain.real.tolist(), "gain_imag": gain.imag.tolist(),
            "gain_db": db.tolist(), "filter_noise_dissipation_w_per_input_v_rms_squared": ac_power.tolist(),
            "load_noise_power_w_per_input_v_rms_squared": (abs(gain) ** 2 / spec.load_ac_r_ohm).tolist(),
            "output_noise_v_rms_per_input_v_rms": abs(gain).tolist(),
            "source_noise_dissipation_w_per_input_v_rms_squared": (
                abs(ac["v(source)"] - ac["v(in)"]) ** 2 / spec.source_r_ohm
            ).tolist(),
        }
        (directory / "ac_metrics.json").write_text(json.dumps(curves, indent=2, allow_nan=False))
        return SupplyFilterResult(status="ok", feasible=not violations, metrics=metrics, spectral=spectral,
                                  violations=violations, **base)


def rank_supply_filters(results: list[SupplyFilterResult]) -> list[SupplyFilterResult]:
    """All checks dominate loss, attenuation, then physical component count."""
    def key(result):
        return (
            0 if result.status == "ok" and result.feasible else 1 if result.status == "ok" else 2,
            sum(result.violations.values()) if not result.feasible else 0,
            result.metrics.get("added_dc_loss_w_dual_rail", math.inf),
            -result.metrics.get("worst_stopband_attenuation_db", -math.inf),
            result.physical_components_per_rail,
            result.name,
        )

    return sorted(results, key=key)
