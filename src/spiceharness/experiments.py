"""Topology-independent, bounded experiments emitted exclusively from typed IR.

AC extrema are nominal sampled screens. Noise is independent, externally
injected, band-limited source noise, not device noise or a worst-case proof.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from enum import StrEnum
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from .domain import Circuit, Component, StrictModel
from .simulator import NgSpiceSimulator, _read_raw

POLICY = "general-experiment-v1;reltol=1e-4;gear;log-frequency-interpolation;sampled-extrema;independent-PSD"
_TOKEN = r"[A-Za-z0-9_]{1,48}"


def _hash(value: StrictModel) -> str:
    return hashlib.sha256(value.model_dump_json().encode()).hexdigest()


def _token(value: str) -> str:
    if not re.fullmatch(_TOKEN, value) or value.lower().startswith("exinternal"):
        raise ValueError("expected a simple identifier outside the evaluator namespace")
    return value


def _node(value: str) -> str:
    """Net names must not alias ngspice ground (``gnd`` is silently node 0)."""
    _token(value)
    if value.lower() == "gnd":
        raise ValueError("net name 'gnd' aliases ground in ngspice; use node 0")
    return value


def _has_short_loop(edges) -> bool:
    """True when ideal zero-volt branches (V sources, wires, lossless L) close a loop."""
    parent: dict[str, str] = {}

    def root(node):
        while parent.get(node, node) != node:
            node = parent[node]
        return node

    for a, b in edges:
        ra, rb = root(a), root(b)
        if ra == rb:
            return True
        parent[ra] = rb
    return False


class GeneralComponent(Component):
    series_resistance_ohm: float = Field(default=0, ge=0, le=1e9)

    @model_validator(mode="after")
    def intrinsic_loss(self) -> GeneralComponent:
        if self.series_resistance_ohm and self.kind not in ("L", "C"):
            raise ValueError("intrinsic series resistance is supported only for L/C")
        _token(self.ref)
        for node in self.nodes:
            _node(node)
        return self

    @field_validator("nodes")
    @classmethod
    def lower_nodes(cls, nodes: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(node.lower() for node in nodes)


class GeneralCircuitIR(StrictModel):
    name: str = Field(default="candidate", max_length=200)
    components: tuple[GeneralComponent, ...] = Field(max_length=200)
    connections: tuple[tuple[str, str], ...] = Field(default=(), max_length=200)

    @model_validator(mode="after")
    def unique_refs(self) -> GeneralCircuitIR:
        refs = [part.ref.lower() for part in self.components]
        if len(refs) != len(set(refs)):
            raise ValueError("component references must be unique ignoring case")
        groups = {}
        for a, b in self.connections:
            _node(a)
            _node(b)
            if a != a.lower() or b != b.lower():
                raise ValueError("wire nodes must be lowercase")
            def root(node):
                while node in groups:
                    node = groups[node]
                return node
            ar, br = root(a), root(b)
            if ar == br:
                raise ValueError("ideal wire connections must not form loops or self-connections")
            groups[ar] = br
        # Reuse the package grouping rules without the topology-specific validator.
        Circuit(topology="general", components=self.components)
        return self

    @property
    def fingerprint(self) -> str:
        return _hash(self)


class ExperimentSource(StrictModel):
    ref: str
    kind: Literal["V", "I"] = "V"
    nodes: tuple[str, str]
    dc: float = Field(default=0, ge=-100, le=100)
    waveform: Literal["dc", "sine", "pulse"] = "dc"
    amplitude: float = Field(default=0, ge=0, le=100)
    frequency_hz: float = Field(default=1000, gt=0, le=1e8)
    delay_s: float = Field(default=0, ge=0, le=100)
    rise_s: float = Field(default=1e-6, gt=0, le=100)
    fall_s: float = Field(default=1e-6, gt=0, le=100)
    width_s: float = Field(default=0.0005, gt=0, le=100)
    period_s: float = Field(default=0.001, gt=0, le=100)

    @model_validator(mode="after")
    def safe(self) -> ExperimentSource:
        _token(self.ref)
        for node in self.nodes:
            _node(node)
        if self.nodes[0].lower() == self.nodes[1].lower():
            raise ValueError("source terminals must differ")
        if abs(self.dc) + self.amplitude > (10 if self.kind == "I" else 100):
            raise ValueError("source exceeds bounded voltage/current envelope")
        if self.waveform == "pulse" and self.width_s + self.rise_s + self.fall_s > self.period_s:
            raise ValueError("pulse width plus edges must fit in period")
        return self

    @field_validator("nodes")
    @classmethod
    def lower_nodes(cls, nodes: tuple[str, str]) -> tuple[str, str]:
        return tuple(node.lower() for node in nodes)


class VoltageProbe(StrictModel):
    name: str
    positive: str
    negative: str = "0"

    @field_validator("name", "positive", "negative")
    @classmethod
    def safe(cls, value: str, info) -> str:
        return (_token(value) if info.field_name == "name" else _node(value)).lower()


class ACSweep(StrictModel):
    start_hz: float = Field(default=10, gt=0, le=1e9)
    stop_hz: float = Field(default=1e6, gt=0, le=1e9)
    points_per_decade: int = Field(default=100, ge=10, le=1000)

    @model_validator(mode="after")
    def bounded(self) -> ACSweep:
        if self.stop_hz <= self.start_hz:
            raise ValueError("AC sweep must increase")
        if math.log10(self.stop_hz / self.start_hz) * self.points_per_decade > 20000:
            raise ValueError("AC sweep exceeds 20000 samples")
        return self


class TransientAnalysis(StrictModel):
    stop_s: float = Field(default=0.02, gt=0, le=100)
    step_s: float = Field(default=1e-6, gt=0, le=1)
    start_s: float = Field(default=0.01, ge=0, le=100)

    @model_validator(mode="after")
    def bounded(self) -> TransientAnalysis:
        if self.start_s >= self.stop_s or self.stop_s / self.step_s > 200000:
            raise ValueError("transient requires increasing window and at most 200000 nominal steps")
        return self


class MeasurementOperator(StrEnum):
    OP_VOLTAGE = "op_voltage"
    OP_CURRENT = "op_current"
    AC_MAGNITUDE = "ac_magnitude"
    AC_MIN = "ac_min"
    AC_MAX = "ac_max"
    AC_CUTOFF_3DB = "ac_cutoff_3db"
    TRANSIENT_MEAN = "transient_mean"
    TRANSIENT_MIN = "transient_min"
    TRANSIENT_MAX = "transient_max"
    TRANSIENT_FUNDAMENTAL = "transient_fundamental"
    TRANSIENT_THD = "transient_thd"
    INTEGRATED_NOISE_RMS = "integrated_noise_rms"


class Measurement(StrictModel):
    name: str
    operator: MeasurementOperator
    probe: str | None = None
    source: str | None = None
    frequency_hz: float | None = Field(default=None, gt=0, le=1e9)
    band_start_hz: float | None = Field(default=None, gt=0, le=1e9)
    band_stop_hz: float | None = Field(default=None, gt=0, le=1e9)
    reference_frequency_hz: float | None = Field(default=None, gt=0, le=1e9)
    crossing: Literal["falling", "rising"] = "falling"
    fundamental_hz: float | None = Field(default=None, gt=0, le=1e8)
    harmonics: int = Field(default=10, ge=2, le=30)

    @field_validator("name", "probe", "source")
    @classmethod
    def safe(cls, value: str | None) -> str | None:
        return _token(value).lower() if value is not None else None


class MetricBound(StrictModel):
    metric: str
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def ordered(self) -> MetricBound:
        if self.minimum is None and self.maximum is None:
            raise ValueError("a requirement must supply at least one bound")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("requirement bounds must increase")
        return self


class NoiseSourcePSD(StrictModel):
    """One-sided constant PSD in V²/Hz (V source) or A²/Hz (I source)."""
    source: str
    density: float = Field(ge=0, le=1e12)
    start_hz: float = Field(gt=0, le=1e9)
    stop_hz: float = Field(gt=0, le=1e9)
    correlation: Literal["independent"] = "independent"

    @model_validator(mode="after")
    def ordered(self) -> NoiseSourcePSD:
        if self.start_hz >= self.stop_hz:
            raise ValueError("noise band must increase")
        return self

    @field_validator("source")
    @classmethod
    def safe(cls, value: str) -> str:
        return _token(value).lower()


class SourceProvenance(StrictModel):
    source: str = Field(min_length=1, max_length=2000)
    page: str | None = Field(default=None, max_length=200)
    note: str = Field(default="", max_length=2000)


class ExperimentSpec(StrictModel):
    sources: tuple[ExperimentSource, ...] = Field(min_length=1, max_length=16)
    loads: tuple[GeneralComponent, ...] = Field(default=(), max_length=32)
    probes: tuple[VoltageProbe, ...] = Field(min_length=1, max_length=32)
    sweep: ACSweep = Field(default_factory=ACSweep)
    transient: TransientAnalysis | None = None
    measurements: tuple[Measurement, ...] = Field(min_length=1, max_length=64)
    requirements: tuple[MetricBound, ...] = Field(default=(), max_length=64)
    baseline: GeneralCircuitIR | None = None
    protected_refs: tuple[str, ...] = ()
    redesign_ports: tuple[str, ...] = ()
    allowed_models: tuple[str, ...] = ()
    noise_sources: tuple[NoiseSourcePSD, ...] = Field(default=(), max_length=16)
    temperature_c: float = Field(default=27, ge=-55, le=150)
    provenance: tuple[SourceProvenance, ...] = ()
    assumptions: tuple[str, ...] = ()
    numerical_policy: Literal[POLICY] = POLICY

    @model_validator(mode="after")
    def coherent(self) -> ExperimentSpec:
        sources = {source.ref.lower(): source for source in self.sources}
        probes = {probe.name: probe for probe in self.probes}
        names = {measurement.name for measurement in self.measurements}
        if len(sources) != len(self.sources) or len(probes) != len(self.probes):
            raise ValueError("source/probe names must be unique ignoring case")
        if len(names) != len(self.measurements):
            raise ValueError("measurement names must be unique")
        if _has_short_loop(s.nodes for s in self.sources if s.kind == "V"):
            raise ValueError("voltage sources form a short-circuit loop")
        if len({part.ref.lower() for part in self.loads}) != len(self.loads):
            raise ValueError("load references must be unique")
        if any(part.kind != "R" or part.series_resistance_ohm for part in self.loads):
            raise ValueError("fixture loads must be plain fixed resistors")
        if any(bound.metric not in names for bound in self.requirements):
            raise ValueError("requirement references an undeclared measurement")
        if self.protected_refs and self.baseline is None:
            raise ValueError("protected references require a frozen baseline")
        if self.protected_refs and self.baseline.connections and not self.redesign_ports:
            raise ValueError("protected baselines with wires require explicit redesign_ports")
        baseline_refs = {part.ref.lower() for part in self.baseline.components} if self.baseline else set()
        if any(ref.lower() not in baseline_refs for ref in self.protected_refs):
            raise ValueError("protected component is absent from baseline")
        for noise in self.noise_sources:
            if noise.source not in sources:
                raise ValueError("noise references an unknown source")
            self._coverage(noise.start_hz, noise.stop_hz)
        if len({noise.source for noise in self.noise_sources}) != len(self.noise_sources):
            raise ValueError("specify one PSD per independent source")
        ac_sources = set()
        for measurement in self.measurements:
            op = measurement.operator
            if op == MeasurementOperator.OP_CURRENT:
                if measurement.source not in sources or sources[measurement.source].kind != "V":
                    raise ValueError("OP current requires a named voltage source")
            elif measurement.probe not in probes:
                raise ValueError("measurement requires a named differential voltage probe")
            if op.value.startswith("ac_"):
                if measurement.source not in sources:
                    raise ValueError("AC measurement requires a named excitation source")
                ac_sources.add(measurement.source)
                if op == MeasurementOperator.AC_MAGNITUDE:
                    self._coverage(measurement.frequency_hz)
                else:
                    if measurement.band_stop_hz is None:
                        raise ValueError("AC band measurement requires both band edges")
                    self._coverage(measurement.band_start_hz, measurement.band_stop_hz)
                if op == MeasurementOperator.AC_CUTOFF_3DB:
                    self._coverage(measurement.reference_frequency_hz)
            if op.value.startswith("transient_"):
                if self.transient is None:
                    raise ValueError("transient measurement requires a transient analysis")
                if op in (MeasurementOperator.TRANSIENT_FUNDAMENTAL, MeasurementOperator.TRANSIENT_THD):
                    f = measurement.fundamental_hz
                    if f is None or (self.transient.stop_s - self.transient.start_s) * f < 2:
                        raise ValueError("harmonic measurements require at least two fundamental periods")
                    if self.transient.step_s * f * measurement.harmonics > 0.05:
                        raise ValueError("transient step must resolve highest fitted harmonic with 20 samples")
            if op == MeasurementOperator.INTEGRATED_NOISE_RMS and not self.noise_sources:
                raise ValueError("noise measurement requires explicit independent source PSDs")
        ac_sources.update(noise.source for noise in self.noise_sources)
        points = math.ceil(math.log10(self.sweep.stop_hz / self.sweep.start_hz)
                           * self.sweep.points_per_decade) + 3
        if len(ac_sources) * points > 100000:
            raise ValueError("combined AC experiment exceeds 100000 samples")
        transient_measurements = [m for m in self.measurements if m.operator.value.startswith("transient_")]
        if transient_measurements:
            count = math.ceil((self.transient.stop_s - self.transient.start_s) / self.transient.step_s) + 1
            work = sum(count * (2 * m.harmonics + 1) if m.operator in (
                MeasurementOperator.TRANSIENT_FUNDAMENTAL, MeasurementOperator.TRANSIENT_THD)
                       else count for m in transient_measurements)
            if work > 3000000:
                raise ValueError("transient measurement work exceeds bounded analysis budget")
            for source in self.sources:
                if source.waveform == "sine" and source.frequency_hz * self.transient.step_s > 0.05:
                    raise ValueError("transient step must resolve every sine source with 20 samples")
                if source.waveform == "pulse" and self.transient.step_s > min(
                        source.rise_s, source.fall_s, source.width_s) / 10:
                    raise ValueError("transient step must resolve every pulse edge/width with 10 samples")
        return self

    def _coverage(self, low: float | None, high: float | None = None) -> None:
        if low is None or not self.sweep.start_hz <= low <= self.sweep.stop_hz:
            raise ValueError("measurement frequency/band must lie inside AC sweep")
        if high is not None and not low < high <= self.sweep.stop_hz:
            raise ValueError("measurement band must increase inside AC sweep")

    @property
    def fingerprint(self) -> str:
        return _hash(self)


class ExperimentScore(StrictModel):
    valid: bool
    feasible: bool | None
    violations: dict[str, float] = Field(default_factory=dict)
    package_count: int
    rank_key: tuple[float, ...] | None


class ExperimentResult(StrictModel):
    status: Literal["ok", "invalid", "failed", "timeout", "unavailable"]
    metrics: dict[str, float] = Field(default_factory=dict)
    errors: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    artifact_dir: str | None = None
    model_checksums: dict[str, str] = Field(default_factory=dict)
    fixture_fingerprint: str | None = None
    circuit_fingerprint: str | None = None
    simulator_version: str | None = None
    score: ExperimentScore


_CANONICAL = {
    "OPAMP": ("noninverting", "inverting", "positive_supply", "negative_supply", "output"),
    "BJT": ("collector", "base", "emitter"),
    "DIODE": ("anode", "cathode"),
}


def _package_count(circuit: GeneralCircuitIR) -> int:
    return len({(part.package or part.ref).lower() for part in circuit.components})


def _score(circuit: GeneralCircuitIR, spec: ExperimentSpec, metrics: dict[str, float],
           valid: bool) -> ExperimentScore:
    count = _package_count(circuit)
    violations = {}
    for index, bound in enumerate(spec.requirements):
        value = metrics.get(bound.metric)
        if value is None or not math.isfinite(value):
            valid = False
            continue
        delta = max(0.0, (bound.minimum - value) if bound.minimum is not None else 0,
                    (value - bound.maximum) if bound.maximum is not None else 0)
        scale = max(abs(bound.minimum or 0), abs(bound.maximum or 0), 1e-12)
        violations[f"{bound.metric}:{index}"] = delta / scale
    feasible = valid and not any(violations.values()) if spec.requirements else None
    rank = ((0 if feasible else 1 if valid else 2), max(violations.values(), default=0),
            sum(violations.values()), count) if spec.requirements else None
    return ExperimentScore(valid=valid, feasible=feasible, violations=violations,
                           package_count=count, rank_key=rank)


class ExperimentEvaluator:
    def __init__(self, artifact_root: str | Path | None = None, executable: str = "ngspice",
                 timeout_s: float = 20, model_catalog: dict | str | Path | None = None,
                 *, lean: bool = False, simulator_version: str | None = None):
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 20:
            raise ValueError("experiment timeout must be positive and at most 20 seconds total")
        self.timeout_s = timeout_s
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.executable = executable
        # Lean mode (default off) is for Monte Carlo samples: same validation, deck
        # generation, parsing, measurement and scoring path, but no artifacts are kept.
        self.lean = lean
        self.simulator_version = simulator_version
        self.simulator = NgSpiceSimulator(model_catalog=model_catalog)

    def validate(self, circuit: GeneralCircuitIR, spec: ExperimentSpec) -> dict:
        # model_copy updates bypass Pydantic validation, including nested components.
        circuit = GeneralCircuitIR.model_validate(circuit.model_dump())
        spec = ExperimentSpec.model_validate(spec.model_dump())
        fixed = {part.ref.lower(): part for part in spec.baseline.components} if spec.baseline else {}
        actual = {part.ref.lower(): part for part in circuit.components}
        for ref in spec.protected_refs:
            if actual.get(ref.lower()) != fixed[ref.lower()]:
                raise ValueError(f"protected component {ref} changed or was removed")
        protected = {ref.lower() for ref in spec.protected_refs}
        host_nodes = {node for ref, part in fixed.items() if ref in protected for node in part.nodes}
        original_ports = {node for ref, part in fixed.items() if ref not in protected for node in part.nodes}
        ports = set(spec.redesign_ports) if spec.redesign_ports else original_ports
        frozen_wires = {tuple(sorted(wire)) for wire in spec.baseline.connections
                        if not set(wire) <= ports} if protected else set()
        candidate_wires = {tuple(sorted(wire)) for wire in circuit.connections}
        if not frozen_wires <= candidate_wires:
            raise ValueError("protected baseline wire changed or was removed")
        forbidden = host_nodes - ports - {"0"}
        for part in circuit.components:
            if part.ref.lower() not in protected and forbidden.intersection(part.nodes):
                raise ValueError(f"{part.ref}: redesign connects to a protected host interior node")
        if any(forbidden.intersection(wire) for wire in candidate_wires - frozen_wires):
            raise ValueError("redesign wire connects to a protected host interior node")
        if any(part.model and part.model not in spec.allowed_models for part in circuit.components):
            raise ValueError("candidate active model is outside the frozen allowlist")
        if {part.ref.lower() for part in circuit.components} & {part.ref.lower() for part in spec.loads}:
            raise ValueError("candidate component references must not duplicate fixture load references")
        if _has_short_loop([*(s.nodes for s in spec.sources if s.kind == "V"), *circuit.connections,
                            *(p.nodes for p in circuit.components
                              if p.kind == "L" and not p.series_resistance_ohm)]):
            raise ValueError("ideal voltage sources, wires and lossless inductors form a short-circuit loop")
        nodes = [node for part in (*circuit.components, *spec.loads) for node in part.nodes]
        nodes += [node for source in spec.sources for node in source.nodes]
        nodes += [node for wire in circuit.connections for node in wire]
        for node in set(nodes) - {"0"}:
            if nodes.count(node) < 2:
                raise ValueError(f"unconnected modeled terminal {node}")
        if "0" not in nodes:
            raise ValueError("experiment requires a ground reference")
        graph = {}
        for terminals in [part.nodes for part in (*circuit.components, *spec.loads)] + [
                source.nodes for source in spec.sources] + list(circuit.connections):
            for node in terminals:
                graph.setdefault(node, set()).update(terminals)
        visited, pending = set(), ["0"]
        while pending:
            node = pending.pop()
            if node not in visited:
                visited.add(node)
                pending.extend(graph.get(node, set()) - visited)
        if set(nodes) - visited:
            raise ValueError("circuit has a disconnected island")
        for probe in spec.probes:
            if probe.positive not in nodes or probe.negative not in nodes:
                raise ValueError(f"probe {probe.name} references an absent node")
        normalized = tuple(Component.model_validate({key: value for key, value in part.model_dump().items()
                                                    if key != "series_resistance_ohm"}).model_copy(
            update={"package": part.package.lower() if part.package else None, "ref": part.ref.lower()})
                           for part in circuit.components)
        models = self.simulator._models(Circuit(topology="general", components=normalized))
        for name, entry in models.items():
            if not entry.get("sha256") or not entry.get("source_url"):
                raise ValueError(f"{name}: trusted model needs checksum and source provenance")
            original = self.simulator.catalog[name]
            if not original.get("sha256"):
                raise ValueError(f"{name}: catalog must pin the model checksum")
            canonical = _CANONICAL[entry["kind"].upper()]
            pins = entry.get("pins")
            if pins is None or not (tuple(sorted(pins)) == tuple(range(len(canonical)))
                                    if pins and isinstance(pins[0], int)
                                    else set(pins) == set(canonical) and len(pins) == len(canonical)):
                raise ValueError(f"{name}: catalog must map every canonical pin exactly once")
            for key, lower in (("min_temperature_c", True), ("max_temperature_c", False)):
                if key in entry and ((spec.temperature_c < entry[key]) if lower
                                     else (spec.temperature_c > entry[key])):
                    raise ValueError(f"{name}: temperature exceeds catalog envelope")
        vectors = len(set(nodes) - {"0"}) + len(spec.sources) + 1
        ac_count = len({m.source for m in spec.measurements if m.operator.value.startswith("ac_")}
                       | {noise.source for noise in spec.noise_sources})
        samples = ac_count * (math.ceil(math.log10(spec.sweep.stop_hz / spec.sweep.start_hz)
                                       * spec.sweep.points_per_decade) + 3)
        if spec.transient and any(m.operator.value.startswith("transient_") for m in spec.measurements):
            samples += math.ceil(spec.transient.stop_s / spec.transient.step_s) + 1
        if samples * vectors > 5000000:
            raise ValueError("experiment exceeds bounded simulator data budget")
        return models

    def generate_netlist(self, circuit: GeneralCircuitIR, spec: ExperimentSpec, analysis: str,
                         source: str | None = None, model_files: dict | None = None) -> str:
        circuit = GeneralCircuitIR.model_validate(circuit.model_dump())
        spec = ExperimentSpec.model_validate(spec.model_dump())
        models = self.validate(circuit, spec)
        if analysis not in ("op", "ac", "tran"):
            raise ValueError("unsupported evaluator-owned analysis")
        if analysis == "ac" and source not in {item.ref.lower() for item in spec.sources}:
            raise ValueError("AC requires a known excitation source")
        lines = ["spiceharness general experiment", f"* fixture {spec.fingerprint}",
                 f".temp {spec.temperature_c:.17g}", ".options numdgt=15 reltol=1e-4 method=gear"]
        for index, item in enumerate(spec.sources):
            value = f"DC {item.dc:.17g} AC {int(item.ref.lower() == source)}"
            if item.waveform == "sine":
                value += f" SIN({item.dc:.17g} {item.amplitude:.17g} {item.frequency_hz:.17g} {item.delay_s:.17g})"
            if item.waveform == "pulse":
                value += (f" PULSE({item.dc:.17g} {item.dc + item.amplitude:.17g} {item.delay_s:.17g} "
                          f"{item.rise_s:.17g} {item.fall_s:.17g} {item.width_s:.17g} {item.period_s:.17g})")
            lines.append(f"{item.kind}EX{index} {' '.join(item.nodes)} {value}")
        for index, (a, b) in enumerate(circuit.connections):
            lines.append(f"VEXWIRE{index} {a} {b} DC 0 AC 0")
        for index, (name, entry) in enumerate(sorted(models.items())):
            filename = str((model_files or {}).get(name, entry["path"]))
            if any(character in filename for character in ('"', '\n', '\r')):
                raise ValueError("unsafe model path")
            lines.append(f'.include "{filename}"')
        for index, part in enumerate((*circuit.components, *spec.loads)):
            nodes = part.nodes
            if part.model:
                entry = models[part.model]
                canonical = _CANONICAL[part.kind]
                pins = entry["pins"]
                nodes = tuple(nodes[i] for i in pins) if isinstance(pins[0], int) else tuple(
                    dict(zip(canonical, nodes))[pin] for pin in pins)
                value = entry["spice_name"]
            else:
                value = f"{part.value:.17g}"
            if part.series_resistance_ohm:
                internal = f"exinternal{index}"
                lines.append(f"REXLOSS{index} {nodes[0]} {internal} {part.series_resistance_ohm:.17g}")
                nodes = (internal, nodes[1])
            prefix = {"OPAMP": "X", "BJT": "Q", "DIODE": "D"}.get(part.kind, part.kind)
            lines.append(f"{prefix}EXP{index} {' '.join(nodes)} {value}")
        if analysis == "op":
            lines.append(".op")
        elif analysis == "ac":
            sweep = spec.sweep
            # ngspice may omit a non-grid stop; extend one step to cover exact requested boundaries.
            stop = sweep.stop_hz * 10 ** (1 / sweep.points_per_decade)
            lines.append(f".ac dec {sweep.points_per_decade} {sweep.start_hz:.17g} {stop:.17g}")
        else:
            if spec.transient is None:
                raise ValueError("transient settings are required")
            tran = spec.transient
            lines.append(f".tran {tran.step_s:.17g} {tran.stop_s:.17g} 0 {tran.step_s:.17g}")
        saved_nodes = sorted({node for part in (*circuit.components, *spec.loads) for node in part.nodes}
                             | {node for item in spec.sources for node in item.nodes}
                             | {node for wire in circuit.connections for node in wire} - {"0"})
        saved_nodes = [node for node in saved_nodes if node != "0"]
        currents = [f"i(VEX{i})" for i, item in enumerate(spec.sources) if item.kind == "V"]
        lines.extend([".save " + " ".join([f"v({node})" for node in saved_nodes] + currents), ".end"])
        return "\n".join(lines) + "\n"

    @staticmethod
    def _voltage(names: list[str], data: np.ndarray, positive: str, negative: str = "0") -> np.ndarray:
        def node(name):
            return np.zeros(len(data)) if name == "0" else data[:, names.index(f"v({name})")]
        return node(positive) - node(negative)

    def _envelope(self, circuit: GeneralCircuitIR, spec: ExperimentSpec, models: dict,
                  names: list[str], data: np.ndarray) -> None:
        for part in circuit.components:
            if part.kind not in ("OPAMP", "BJT"):
                continue
            entry = models[part.model]
            span = (self._voltage(names, data, part.nodes[2], part.nodes[3]).real
                    if part.kind == "OPAMP" else
                    np.abs(self._voltage(names, data, part.nodes[0], part.nodes[2]).real))
            for key, lower in (("min_supply_span_v", True), ("max_supply_span_v", False)):
                if key in entry and np.any(span < entry[key] if lower else span > entry[key]):
                    raise ValueError(f"{part.ref}: measured supply span exceeds catalog {key}")
        # BJT envelope is conservative collector/emitter stress, not complete thermal/SOA validation.

    def evaluate(self, circuit: GeneralCircuitIR, spec: ExperimentSpec) -> ExperimentResult:
        deadline = time.monotonic() + self.timeout_s
        directory = None
        models = {}
        metrics = {}
        version = None
        status, errors = "ok", ()
        try:
            circuit = GeneralCircuitIR.model_validate(circuit.model_dump())
            spec = ExperimentSpec.model_validate(spec.model_dump())
            models = self.validate(circuit, spec)
            if not shutil.which(self.executable):
                raise FileNotFoundError("ngspice executable was not found")
            if self.artifact_root:
                self.artifact_root.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix="experiment-", dir=self.artifact_root))
            if not self.lean:
                (directory / "circuit.json").write_text(circuit.model_dump_json(indent=2))
                (directory / "fixture.json").write_text(spec.model_dump_json(indent=2))
            (directory / ".spiceinit").write_text("set ngbehavior=ltpsa\nset filetype=ascii\n")
            model_files = {}
            for index, (name, entry) in enumerate(sorted(models.items())):
                filename = f"model_{index}.lib"
                shutil.copyfile(entry["path"], directory / filename)
                model_files[name] = filename
            environment = dict(os.environ, HOME=str(directory), SPICE_ASCIIRAWFILE="1")
            if self.lean:
                version = self.simulator_version
            else:
                version_run = subprocess.run([self.executable, "--version"], capture_output=True, text=True,
                                            check=False, timeout=max(0.001, deadline - time.monotonic()),
                                            env=environment)
                version = version_run.stdout.strip()[:2000]
            ac_sources = {m.source for m in spec.measurements if m.operator.value.startswith("ac_")}
            if any(m.operator == MeasurementOperator.INTEGRATED_NOISE_RMS for m in spec.measurements):
                ac_sources.update(noise.source for noise in spec.noise_sources)
            analyses = [("op", None)] + [("ac", source) for source in sorted(ac_sources)]
            if any(m.operator.value.startswith("transient_") for m in spec.measurements):
                analyses.append(("tran", None))
            datasets = {}
            for index, (analysis, source) in enumerate(analyses):
                label = f"{index}_{analysis}" + (f"_{source}" if source else "")
                deck, raw, log = directory / f"{label}.cir", directory / f"{label}.raw", directory / f"{label}.log"
                deck.write_text(self.generate_netlist(circuit, spec, analysis, source, model_files))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
                run = subprocess.run([self.executable, "-b", "-r", raw.name, "-o", log.name, deck.name],
                                     cwd=directory, env=environment, capture_output=True, text=True, check=False,
                                     timeout=remaining)
                (directory / f"{label}.console.txt").write_text(run.stdout + run.stderr)
                if run.returncode or not raw.exists():
                    raise RuntimeError(f"ngspice failed for {label}; inspect simulator log")
                log_text = log.read_text(errors="replace") if log.exists() else ""
                if re.search(r"(timestep too small|fatal error|simulation interrupted|doAnalyses:.*failed)",
                             log_text + run.stderr, re.IGNORECASE):
                    raise RuntimeError(f"ngspice reported an incomplete analysis for {label}")
                if raw.stat().st_size > 128 * 1024 * 1024:
                    raise ValueError("simulator raw output exceeds 128 MiB resource budget")
                names, data = _read_raw(raw)
                if not np.all(np.isfinite(data)):
                    raise ValueError(f"simulator produced non-finite samples for {label}")
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
                if analysis in ("ac", "tran") and (len(data) < 2 or np.any(np.diff(data[:, 0].real) <= 0)):
                    raise ValueError("simulator samples must have a strictly increasing analysis axis")
                if len(data) > 250000:
                    raise ValueError("simulator exceeded bounded output sample count")
                datasets[(analysis, source)] = names, data
                if time.monotonic() > deadline:
                    raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
                if not self.lean:
                    np.savez_compressed(directory / f"{label}.curves.npz", names=np.array(names), data=data)
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
                    columns = np.column_stack([array for column in data.T for array in (column.real, column.imag)])
                    headers = ",".join(f"{name}_{part}" for name in names for part in ("real", "imag"))
                    if time.monotonic() > deadline:
                        raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
                    np.savetxt(directory / f"{label}.curves.csv", columns, delimiter=",", header=headers)
                if analysis in ("op", "tran"):
                    self._envelope(circuit, spec, models, names, data)
            metrics = self._measure(spec, datasets, deadline)
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
        except subprocess.TimeoutExpired:
            status, errors = "timeout", ("total experiment wall-time budget exceeded",)
        except FileNotFoundError as exc:
            status, errors = "unavailable", (str(exc),)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            status, errors = "invalid", (str(exc),)
        except (OSError, RuntimeError) as exc:
            status, errors = "failed", (str(exc),)
        except BaseException:
            if self.lean and directory:
                shutil.rmtree(directory, ignore_errors=True)
            raise
        lean_directory = directory if self.lean else None
        if lean_directory:
            directory = None
        result = ExperimentResult(status=status, metrics=metrics, errors=errors,
                                  model_checksums={name: entry["sha256"] for name, entry in models.items()},
                                  fixture_fingerprint=spec.fingerprint, circuit_fingerprint=circuit.fingerprint,
                                  artifact_dir=str(directory) if directory else None, simulator_version=version,
                                  score=_score(circuit, spec, metrics, status == "ok"),
                                  artifacts=tuple(sorted([path.name for path in directory.iterdir()] + ["metrics.json"]))
                                  if directory else ())
        if directory:
            (directory / "metrics.json").write_text(result.model_dump_json(indent=2))
        if lean_directory:
            shutil.rmtree(lean_directory, ignore_errors=True)
        return result

    def _measure(self, spec: ExperimentSpec, datasets: dict, deadline: float = math.inf) -> dict[str, float]:
        probes = {probe.name: probe for probe in spec.probes}
        metrics = {}

        def voltage(analysis, source, probe):
            names, data = datasets[(analysis, source)]
            return self._voltage(names, data, probe.positive, probe.negative)

        def frequency(source):
            names, data = datasets[("ac", source)]
            return data[:, names.index("frequency")].real

        def interp(f, values, target):
            if target < f[0] * (1 - 1e-10) or target > f[-1] * (1 + 1e-10):
                raise ValueError("simulator sweep does not cover required measurement")
            return np.interp(np.log(target), np.log(f), values)

        def band(f, values, low, high):
            mask = (f > low) & (f < high)
            return np.r_[low, f[mask], high], np.r_[interp(f, values, low), values[mask],
                                                       interp(f, values, high)]

        for measurement in spec.measurements:
            if time.monotonic() > deadline:
                raise subprocess.TimeoutExpired(self.executable, self.timeout_s)
            op = measurement.operator
            probe = probes.get(measurement.probe)
            if op == MeasurementOperator.OP_CURRENT:
                names, data = datasets[("op", None)]
                index = next(i for i, source in enumerate(spec.sources)
                             if source.ref.lower() == measurement.source)
                value = data[0, names.index(f"i(vex{index})")].real
            elif op == MeasurementOperator.OP_VOLTAGE:
                value = voltage("op", None, probe)[0].real
            elif op.value.startswith("ac_"):
                f = frequency(measurement.source)
                magnitude = np.abs(voltage("ac", measurement.source, probe))
                if op == MeasurementOperator.AC_MAGNITUDE:
                    value = interp(f, magnitude, measurement.frequency_hz)
                else:
                    fs, ys = band(f, magnitude, measurement.band_start_hz, measurement.band_stop_hz)
                    if op == MeasurementOperator.AC_MIN:
                        value = min(ys)
                    elif op == MeasurementOperator.AC_MAX:
                        value = max(ys)
                    else:
                        target = interp(f, magnitude, measurement.reference_frequency_hz) / math.sqrt(2)
                        if target <= 0:
                            raise ValueError("3dB cutoff requires a positive reference response")
                        crossings = []
                        for i in range(len(fs) - 1):
                            a, b = ys[i], ys[i + 1]
                            crosses = a > target >= b if measurement.crossing == "falling" else a < target <= b
                            if crosses:
                                if min(a, b) <= 0:
                                    fraction = (target - a) / (b - a)
                                else:
                                    fraction = math.log(target / a) / math.log(b / a)
                                crossings.append(math.exp(math.log(fs[i]) + fraction * math.log(fs[i + 1] / fs[i])))
                        if not crossings:
                            raise ValueError(f"{measurement.name}: no declared-direction 3dB crossing in band")
                        value = crossings[0]
            elif op == MeasurementOperator.INTEGRATED_NOISE_RMS:
                power = 0.0
                for noise in spec.noise_sources:
                    f = frequency(noise.source)
                    gain_squared = np.abs(voltage("ac", noise.source, probe)) ** 2
                    fs, ys = band(f, gain_squared, noise.start_hz, noise.stop_hz)
                    power += float(np.trapezoid(ys * noise.density, fs))
                value = math.sqrt(power)
            else:
                names, data = datasets[("tran", None)]
                t = data[:, names.index("time")].real
                y = voltage("tran", None, probe).real
                tran = spec.transient
                if t[0] > tran.start_s or t[-1] < tran.stop_s * (1 - 1e-9):
                    raise ValueError("transient does not cover measurement window")
                # Fixed uniform resampling prevents adaptive solver density from biasing means/fits.
                count = math.ceil((tran.stop_s - tran.start_s) / tran.step_s) + 1
                ts = np.linspace(tran.start_s, tran.stop_s, count)
                ys = np.interp(ts, t, y)
                if op == MeasurementOperator.TRANSIENT_MEAN:
                    value = np.trapezoid(ys, ts) / (tran.stop_s - tran.start_s)
                elif op == MeasurementOperator.TRANSIENT_MIN:
                    value = min(ys)
                elif op == MeasurementOperator.TRANSIENT_MAX:
                    value = max(ys)
                else:
                    columns = [np.ones(count)]
                    for harmonic in range(1, measurement.harmonics + 1):
                        angle = 2 * np.pi * harmonic * measurement.fundamental_hz * ts
                        columns.extend((np.sin(angle), np.cos(angle)))
                    coefficients = np.linalg.lstsq(np.column_stack(columns), ys, rcond=None)[0]
                    amplitudes = np.hypot(coefficients[1::2], coefficients[2::2])
                    if op == MeasurementOperator.TRANSIENT_FUNDAMENTAL:
                        value = amplitudes[0]
                    else:
                        if amplitudes[0] <= 1e-15:
                            raise ValueError("THD requires a nonzero fitted fundamental")
                        value = np.linalg.norm(amplitudes[1:]) / amplitudes[0]
            if not math.isfinite(value):
                raise ValueError(f"{measurement.name}: nonfinite metric")
            metrics[measurement.name] = float(value)
        return metrics
