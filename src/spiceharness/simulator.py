"""Immutable fixtures and isolated ngspice batch execution.

Only the evaluator emits SPICE instructions. Candidate circuits cannot supply
sources, includes, control statements, expressions, or simulator commands.
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
import time
from pathlib import Path
from typing import Any

import numpy as np

from .domain import Circuit, Component, DesignSpec, SimulationResult
from .templates import shortlist_templates

MODEL_DIR = Path(__file__).parent / "data" / "models"
FIXTURE_VERSION = "spiceharness-fixture-v1"


def tolerance_cases(spec: DesignSpec, circuit: Circuit) -> dict[str, tuple[Component, ...]]:
    """Four correlated cases plus independent one-part-at-a-time limits.

    These are deterministic sensitivity screens, not an exhaustive proof over
    every combination of component tolerances.
    """
    cases = {}
    for label, rs, cs in (("low", -1, -1), ("high", 1, 1), ("mixed", 1, -1), ("mixed_inverse", -1, 1)):
        cases[label] = tuple(
            p.model_copy(
                update={
                    "value": p.value
                    * (1 + (rs * spec.resistor_tolerance if p.kind == "R" else cs * spec.capacitor_tolerance))
                }
            )
            if p.kind in ("R", "C")
            else p
            for p in circuit.components
        )
    for index, part in enumerate(circuit.components):
        tolerance = (
            spec.resistor_tolerance
            if part.kind == "R"
            else spec.capacitor_tolerance
            if part.kind == "C"
            else 0
        )
        if tolerance:
            for label, sign in (("low", -1), ("high", 1)):
                parts = list(circuit.components)
                parts[index] = part.model_copy(update={"value": part.value * (1 + sign * tolerance)})
                cases[f"{part.ref}.{label}"] = tuple(parts)
    return cases


def expected_corner_labels(spec: DesignSpec, circuit: Circuit) -> set[str]:
    return set(tolerance_cases(spec, circuit)) if spec.corner_policy == "tolerance" else set()


def validate_circuit(circuit: Circuit, spec: DesignSpec | None = None) -> list[str]:
    errors = []
    spec = spec or circuit.spec
    if spec is None:
        return ["an explicit DesignSpec is required"]
    nodes = [n.lower() for p in circuit.components for n in p.nodes]
    for node in (circuit.input_node, circuit.output_node):
        if node.lower() not in nodes:
            errors.append(f"terminal {node} is disconnected")
    if circuit.input_node.lower() == circuit.output_node.lower():
        errors.append("input and output must be distinct nodes")
    if circuit.input_node.lower() in {"0", "vcc", "vee", "source"} or circuit.output_node.lower() in {
        "0",
        "vcc",
        "vee",
        "source",
    }:
        errors.append("signal terminals cannot alias fixture ground, source, or rails")
    if "source" in nodes:
        errors.append("source is an evaluator-owned node")
    if circuit.topology not in shortlist_templates(spec):
        errors.append("candidate topology differs from the frozen specification")
    if spec.min_filter_order == 2 and "pass" in circuit.topology and not circuit.topology.endswith("_2"):
        errors.append("candidate does not implement the required second filter order")
    if circuit.spec is not None and circuit.spec.fingerprint != spec.fingerprint:
        errors.append("candidate operating conditions differ from the frozen specification")
    for node in set(nodes) - {"0", "vcc", "vee", circuit.input_node.lower(), circuit.output_node.lower()}:
        if nodes.count(node) < 2:
            errors.append(f"floating terminal: {node}")
    # Signal reachability excludes bench ground/rails: a disconnected output
    # load cannot masquerade as a signal connection through the common ground.
    graph = {}
    for p in circuit.components:
        signal = [
            n.lower()
            for n in (p.nodes[:2] + p.nodes[4:] if p.kind == "OPAMP" else p.nodes)
            if n.lower() not in ("0", "vcc", "vee")
        ]
        for n in signal:
            graph.setdefault(n, set()).update(x for x in signal if x != n)
    visited = set()
    pending = [circuit.input_node.lower()]
    while pending:
        n = pending.pop()
        if n not in visited:
            visited.add(n)
            pending.extend(graph.get(n, set()) - visited)
    if circuit.output_node.lower() not in visited:
        errors.append("no DUT signal path connects input to output")
    active = [p for p in circuit.components if p.kind in ("OPAMP", "BJT")]
    for part in active:
        if part.kind == "OPAMP" and tuple(n.lower() for n in part.nodes[2:4]) != ("vcc", "vee"):
            errors.append(f"{part.ref}: opamp supply pins must use evaluator vcc/vee rails")
    if active:
        for rail in ("vcc", "vee") if spec.supply_negative_v else ("vcc",):
            if not any(
                p.kind == "C"
                and p.role == "decoupling"
                and rail in p.nodes
                and "0" in p.nodes
                and p.value >= 10e-9
                for p in circuit.components
            ):
                errors.append(f"missing mandatory decoupling capacitor on {rail}")
    return errors


def _read_raw(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse ngspice ASCII raw data, preserving complex AC vectors."""
    data = path.read_text(errors="strict")
    if "Values:" not in data or "Variables:" not in data:
        raise ValueError("ngspice did not produce ASCII raw data")
    header, values = data.split("Values:", 1)
    count = int(re.search(r"No\. Variables:\s*(\d+)", header).group(1))
    points = int(re.search(r"No\. Points:\s*(\d+)", header).group(1))
    variables = header.split("\nVariables:", 1)[1].strip().splitlines()
    names = [line.split()[1].lower() for line in variables[:count]]
    tokens = values.split()
    if len(tokens) != points * (count + 1):
        raise ValueError("raw data has an incomplete sample table")
    result = np.empty((points, count), dtype=complex)
    for i in range(points):
        offset = i * (count + 1)
        if int(tokens[offset]) != i:
            raise ValueError("invalid raw sample index")
        for j, token in enumerate(tokens[offset + 1 : offset + count + 1]):
            fields = token.split(",")
            result[i, j] = complex(float(fields[0]), float(fields[1]) if len(fields) > 1 else 0)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite simulator samples")
    return names, result


class NgSpiceSimulator:
    def __init__(
        self,
        model_catalog: dict[str, Any] | str | Path | None = None,
        timeout_s: float = 20,
        artifact_root: str | Path | None = None,
        executable: str = "ngspice",
    ) -> None:
        if not math.isfinite(timeout_s) or not 0 < timeout_s <= 300:
            raise ValueError("timeout_s must be between 0 and 300")
        self.timeout_s = timeout_s
        self.executable = executable
        self.artifact_root = Path(artifact_root) if artifact_root else None
        self.model_dir = MODEL_DIR
        if isinstance(model_catalog, dict):
            self.catalog = model_catalog
        else:
            path = Path(model_catalog) if model_catalog else MODEL_DIR / "catalog.json"
            self.model_dir = path.parent
            self.catalog = json.loads(path.read_text()) if path.exists() else {}
        if "models" in self.catalog:
            entries = self.catalog["models"]
            self.catalog = {e["name"]: e for e in entries} if isinstance(entries, list) else entries

    def _models(self, circuit: Circuit) -> dict[str, dict[str, Any]]:
        models = {}
        packages = {}
        for part in circuit.components:
            if not part.model:
                continue
            if part.model not in self.catalog:
                raise ValueError(f"model {part.model} unavailable in trusted catalog; run model fetch")
            entry = dict(self.catalog[part.model])
            path = Path(entry.get("path", entry.get("file", "")))
            if not path.is_absolute():
                path = self.model_dir / path
            if not path.is_file():
                raise ValueError(f"model file unavailable for {part.model}: {path.name}")
            content = path.read_bytes()
            checksum = hashlib.sha256(content).hexdigest()
            expected = entry.get("sha256")
            if expected and checksum != expected:
                raise ValueError(f"model checksum mismatch for {part.model}")
            text = content.decode(errors="replace")
            if re.search(r"^\s*\.(control|include|inc|lib|shell|exec)\b", text, re.IGNORECASE | re.MULTILINE):
                raise ValueError(f"model {part.model} contains unsupported external commands or includes")
            spice_name = entry.get("spice_name", part.model)
            if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_./-]{0,95}", spice_name):
                raise ValueError("unsafe catalog model name")
            if entry.get("kind", part.kind).upper() != part.kind:
                raise ValueError(f"model {part.model} is not a {part.kind}")
            entry.update(path=path, sha256=checksum, spice_name=spice_name)
            models[part.model] = entry
            package = part.package or part.ref
            existing = packages.setdefault(package, [])
            existing.append(part.model)
            if len(set(existing)) != 1 or len(existing) > int(entry.get("channels", 1)):
                raise ValueError(f"invalid physical package sharing for {package}")
        for package, names in packages.items():
            entry = models[names[0]]
            if len(names) != int(entry.get("channels", 1)):
                raise ValueError(
                    f"{package}: all physical package channels must be present, including safely terminated spare channels"
                )
        return models

    @staticmethod
    def _check_operating_envelope(spec: DesignSpec, models: dict[str, dict[str, Any]]) -> None:
        span = spec.supply_positive_v - spec.supply_negative_v
        for name, entry in models.items():
            for key, value, lower in (
                ("min_supply_span_v", span, True),
                ("max_supply_span_v", span, False),
                ("min_temperature_c", spec.temperature_c, True),
                ("max_temperature_c", spec.temperature_c, False),
            ):
                if key in entry and ((value < entry[key]) if lower else (value > entry[key])):
                    raise ValueError(f"{name}: operating condition exceeds catalog {key}={entry[key]}")

    def generate_netlist(
        self,
        circuit: Circuit,
        analysis: str = "ac",
        spec: DesignSpec | None = None,
        model_files: dict[str, str] | None = None,
    ) -> str:
        spec = spec or circuit.spec
        if spec is None:
            raise ValueError("specification is required")
        errors = validate_circuit(circuit, spec)
        if errors:
            raise ValueError("; ".join(errors))
        models = self._models(circuit)
        self._check_operating_envelope(spec, models)
        lines = [
            "spiceharness evaluator-owned fixture",
            f"* {FIXTURE_VERSION} spec {spec.fingerprint}",
            f".temp {spec.temperature_c:.12g}",
            ".options numdgt=15 method=gear reltol=1e-4",
            f"VFIXP vcc 0 DC {spec.supply_positive_v:.12g}",
            f"VFIXN vee 0 DC {spec.supply_negative_v:.12g}",
            f"VFIXIN source 0 DC {spec.input_offset_v:.12g} AC 1 SIN({spec.input_offset_v:.12g} {spec.input_amplitude_v:.12g} {spec.input_frequency_hz:.12g})",
            f"RFIXSOURCE source {circuit.input_node} {max(spec.source_resistance_ohm, 1e-6):.12g}",
            f"RFIXLOAD {circuit.output_node} 0 {spec.load_ohm:.12g}",
        ]
        for index, (name, entry) in enumerate(sorted(models.items())):
            filename = (model_files or {}).get(name, str(entry["path"]))
            if any(c in filename for c in ('"', "\n", "\r")):
                raise ValueError("unsafe model path")
            lines.append(f'.include "{filename}"')
        prefix = {"OPAMP": "X", "BJT": "Q", "DIODE": "D"}
        canonical = {
            "OPAMP": ["noninverting", "inverting", "positive_supply", "negative_supply", "output"],
            "BJT": ["collector", "base", "emitter"],
            "DIODE": ["anode", "cathode"],
        }
        for index, part in enumerate(circuit.components):
            nodes = part.nodes
            if part.model:
                entry = models[part.model]
                pins = entry.get("pins", canonical[part.kind])
                if pins and isinstance(pins[0], int):
                    nodes = tuple(nodes[i] for i in pins)
                else:
                    mapping = dict(zip(canonical[part.kind], nodes))
                    try:
                        nodes = tuple(mapping[p] for p in pins)
                    except KeyError as exc:
                        raise ValueError(f"invalid catalog pin order for {part.model}: {pins}") from exc
                value = entry["spice_name"]
            else:
                value = f"{part.value:.12g}"
            lines.extend(
                [
                    f"* {part.ref}",
                    f"{prefix.get(part.kind, part.kind)}DUT{index} " + " ".join(nodes) + f" {value}",
                ]
            )
        if analysis == "op":
            lines.append(".op")
        elif analysis == "ac":
            anchor = spec.cutoff_hz or spec.input_frequency_hz
            frequencies = [anchor, spec.input_frequency_hz]
            if spec.stopband_frequency_hz is not None:
                frequencies.append(spec.stopband_frequency_hz)
            low = min(frequencies) / 1000
            high = max(frequencies) * 1000
            lines.append(f".ac dec 80 {low:.12g} {high:.12g}")
        elif analysis == "tran":
            period = 1 / spec.input_frequency_hz
            settle = max(12 * period, 12 / (2 * math.pi * (spec.cutoff_hz or spec.input_frequency_hz)))
            stop = settle + 8 * period
            lines.append(f".tran {period / 160:.12g} {stop:.12g} {settle:.12g} {period / 80:.12g}")
        else:
            raise ValueError("analysis must be op, ac, or tran")
        lines.extend(
            [f".save v(source) v({circuit.input_node}) v({circuit.output_node}) i(VFIXP) i(VFIXN)", ".end"]
        )
        return "\n".join(lines) + "\n"

    def evaluate(self, circuit: Circuit, spec: DesignSpec | None = None) -> SimulationResult:
        deadline = time.monotonic() + self.timeout_s
        spec = spec or circuit.spec
        if spec is None:
            return SimulationResult(status="invalid", errors=["specification required"])
        errors = validate_circuit(circuit, spec)
        if errors:
            return SimulationResult(status="invalid", errors=errors, spec_fingerprint=spec.fingerprint)
        if not shutil.which(self.executable):
            return SimulationResult(
                status="unavailable",
                errors=["ngspice executable was not found"],
                spec_fingerprint=spec.fingerprint,
            )
        try:
            models = self._models(circuit)
            self._check_operating_envelope(spec, models)
        except (ValueError, OSError, TypeError) as exc:
            return SimulationResult(status="invalid", errors=[str(exc)], spec_fingerprint=spec.fingerprint)
        if self.artifact_root:
            self.artifact_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="spiceharness-", dir=self.artifact_root))
        # HOME is scoped to the child process, so PSpice compatibility does not
        # read or overwrite the user's personal ngspice configuration.
        (directory / ".spiceinit").write_text("set ngbehavior=ltpsa\nset filetype=ascii\n")
        model_files = {}
        for index, (name, entry) in enumerate(sorted(models.items())):
            filename = f"model_{index}.lib"
            shutil.copyfile(entry["path"], directory / filename)
            model_files[name] = filename
        environment = dict(os.environ, HOME=str(directory), SPICE_ASCIIRAWFILE="1")
        logs = {}
        datasets = {}
        analyses = {}
        fixture_hash = hashlib.sha256((FIXTURE_VERSION + spec.fingerprint).encode()).hexdigest()
        base = {
            "artifact_dir": str(directory),
            "model_checksums": {n: e["sha256"] for n, e in models.items()},
            "spec_fingerprint": spec.fingerprint,
            "fixture_fingerprint": fixture_hash,
        }
        try:
            for analysis in ("op", "ac", "tran"):
                netlist = self.generate_netlist(circuit, analysis, spec, model_files)
                (directory / f"{analysis}.cir").write_text(netlist)
                command = [self.executable, "-b", "-r", f"{analysis}.raw", f"{analysis}.cir"]
                process = subprocess.run(
                    command,
                    cwd=directory,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=max(0.001, deadline - time.monotonic()),
                    check=False,
                )
                log = process.stdout + "\n" + process.stderr
                logs[analysis] = log
                (directory / f"{analysis}.log").write_text(log)
                if process.returncode != 0 or re.search(
                    r"(fatal error|timestep too small|singular matrix|simulation interrupted|doanalyses:)",
                    log,
                    re.IGNORECASE,
                ):
                    raise ValueError(f"{analysis} simulation failed: {log[-1800:]}")
                names, data = _read_raw(directory / f"{analysis}.raw")
                if analysis == "op":
                    version = re.search(
                        r"^Command:\s*(.*)$", (directory / "op.raw").read_text(), re.MULTILINE
                    )
                    base["simulator_version"] = version.group(1).strip() if version else "ngspice"
                datasets[analysis] = (names, data)
                analyses[analysis] = {
                    "points": len(data),
                    "variables": names,
                    "raw_file": str(directory / f"{analysis}.raw"),
                }
            metrics = self._metrics(spec, circuit, datasets)
            result = SimulationResult(status="ok", metrics=metrics, analyses=analyses, logs=logs, **base)
            if spec.corner_policy == "tolerance":
                corner_results = {}
                for label, parts in tolerance_cases(spec, circuit).items():
                    corner_spec = spec.model_copy(update={"corner_policy": "nominal"})
                    corner = circuit.model_copy(update={"components": parts, "spec": corner_spec})
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired([self.executable], self.timeout_s)
                    runner = NgSpiceSimulator(
                        self.catalog,
                        timeout_s=min(remaining, 300),
                        artifact_root=directory,
                        executable=self.executable,
                    )
                    runner.model_dir = self.model_dir
                    measured = runner.evaluate(corner, corner_spec)
                    corner_results[label] = {
                        "status": measured.status,
                        "metrics": measured.metrics,
                        "errors": measured.errors,
                        "artifact_dir": measured.artifact_dir,
                    }
                result = result.model_copy(update={"analyses": {**analyses, "corners": corner_results}})
            (directory / "result.json").write_text(result.model_dump_json(indent=2))
            return result
        except subprocess.TimeoutExpired as exc:

            def decoded(value):
                return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")

            logs["timeout"] = decoded(exc.stdout) + "\n" + decoded(exc.stderr)
            (directory / "timeout.log").write_text(logs["timeout"])
            return SimulationResult(
                status="timeout",
                errors=[f"{exc.cmd[0]} exceeded {self.timeout_s:g}s timeout"],
                logs=logs,
                analyses=analyses,
                **base,
            )
        except (ValueError, OSError, IndexError, KeyError, TypeError, UnicodeDecodeError) as exc:
            return SimulationResult(status="failed", errors=[str(exc)], logs=logs, analyses=analyses, **base)

    @staticmethod
    def _metrics(spec: DesignSpec, circuit: Circuit, datasets: dict) -> dict[str, float]:
        def column(analysis: str, name: str) -> np.ndarray:
            names, data = datasets[analysis]
            return data[:, names.index(name.lower())]

        f = column("ac", "frequency").real
        if len(f) < 2 or not np.all(np.diff(f) > 0):
            raise ValueError("AC frequency scale must be strictly increasing")

        def interpolate(frequency: float, values: np.ndarray) -> float:
            if not f[0] <= frequency <= f[-1]:
                raise ValueError(f"requested {frequency:g} Hz lies outside measured AC frequency range")
            return float(np.interp(math.log(frequency), np.log(f), values))

        transfer = column("ac", f"v({circuit.output_node})") / column("ac", "v(source)")
        magnitude = np.abs(transfer)
        if not np.isfinite(magnitude).all() or np.max(magnitude) <= 1e-12:
            raise ValueError("output has no finite measurable AC response")
        input_gain = interpolate(spec.input_frequency_hz, magnitude)
        high = "highpass" in circuit.topology
        is_filter = "pass" in circuit.topology
        if is_filter:
            # A decade clear of the specified cutoff avoids measuring an opamp's
            # GBW peak as passband gain; endpoint reference is deterministic.
            reference = spec.cutoff_hz * (10 if high else 0.01)
            gain = interpolate(reference, magnitude)
        else:
            gain = input_gain
        metrics = {
            "gain": gain,
            "gain_at_input_frequency": input_gain,
            "phase_deg": float(
                interpolate(spec.input_frequency_hz, np.unwrap(np.angle(transfer))) * 180 / math.pi
            ),
            "output_offset_v": float(column("op", f"v({circuit.output_node})")[0].real),
            "total_rail_current_a": float(
                abs(column("op", "i(vfixp)")[0].real) + abs(column("op", "i(vfixn)")[0].real)
            ),
            "supply_power_w": float(
                max(
                    0,
                    -spec.supply_positive_v * column("op", "i(vfixp)")[0].real
                    - spec.supply_negative_v * column("op", "i(vfixn)")[0].real,
                )
            ),
        }
        metrics["supply_current_a"] = metrics["supply_power_w"] / (
            spec.supply_positive_v - spec.supply_negative_v
        )
        if is_filter:
            level = gain / math.sqrt(2)
            crossings = np.where((magnitude[:-1] - level) * (magnitude[1:] - level) <= 0)[0]
            directional = [i for i in crossings if (magnitude[i + 1] > magnitude[i]) == high]
            if not directional:
                raise ValueError("no -3 dB cutoff crossing was measured")
            i = min(directional, key=lambda i: abs(math.log(f[i] / spec.cutoff_hz)))
            alpha = (level - magnitude[i]) / (magnitude[i + 1] - magnitude[i])
            metrics["cutoff_hz"] = float(math.exp(math.log(f[i]) + alpha * math.log(f[i + 1] / f[i])))
        if spec.stopband_frequency_hz is not None:
            metrics["stopband_gain"] = interpolate(spec.stopband_frequency_hz, magnitude)
        t = column("tran", "time").real
        y = column("tran", f"v({circuit.output_node})").real
        # Fit a DC term and ten harmonics on the last six complete cycles. A
        # uniform resample prevents adaptive timestep density biasing the fit.
        start = max(t[0], t[-1] - 6 / spec.input_frequency_hz)
        grid = np.linspace(start, t[-1], 1920, endpoint=False)
        wave = np.interp(grid, t, y)
        omega = 2 * math.pi * spec.input_frequency_hz * grid
        matrix = np.column_stack(
            [np.ones(len(grid))] + [fn(k * omega) for k in range(1, 11) for fn in (np.sin, np.cos)]
        )
        coefficients = np.linalg.lstsq(matrix, wave, rcond=None)[0]
        harmonics = np.hypot(coefficients[1::2], coefficients[2::2])
        if harmonics[0] < 1e-12:
            raise ValueError("no measurable transient fundamental")
        metrics.update(
            thd=float(np.linalg.norm(harmonics[1:]) / harmonics[0]),
            output_swing_v=float((np.max(wave) - np.min(wave)) / 2),
            transient_gain=float(harmonics[0] / spec.input_amplitude_v),
            transient_dc_v=float(coefficients[0]),
        )
        # Saturation also reduces fundamental gain without always producing
        # large harmonics; require small/large signal gain agreement.
        metrics["gain_compression"] = abs(metrics["transient_gain"] / input_gain - 1)
        if not all(math.isfinite(x) for x in metrics.values()):
            raise ValueError("nonfinite extracted metric")
        return metrics


def simulate(spec: DesignSpec, candidate: Circuit, **kwargs) -> SimulationResult:
    return NgSpiceSimulator(**kwargs).evaluate(candidate, spec)


def smoke_models(**kwargs) -> dict[str, SimulationResult]:
    """Exercise every trusted device in op, AC, and transient, without fake models."""
    from .templates import make_candidate

    runner = NgSpiceSimulator(**kwargs)
    results = {}
    common = {
        "supply_positive_v": 12,
        "supply_negative_v": -12,
        "input_amplitude_v": 0.01,
        "input_frequency_hz": 1000,
        "load_ohm": 100000,
    }
    for name, entry in runner.catalog.items():
        kind = entry["kind"]
        if kind == "OPAMP":
            spec = DesignSpec(topology="buffer", opamp_model=name, **common)
            circuit = make_candidate(spec)
        elif kind == "BJT" and name == "2N3904":
            spec = DesignSpec(topology="emitter_follower", transistor_model=name, **common)
            circuit = make_candidate(spec)
        elif kind == "BJT":
            spec = DesignSpec(topology="emitter_follower", transistor_model=name, **common)
            circuit = Circuit(
                name=f"smoke_{name}",
                topology=spec.topology,
                spec=spec,
                components=(
                    Component(ref="Q1", kind="BJT", nodes=("vee", "base", "emitter"), model=name),
                    Component(ref="RB1", kind="R", nodes=("vcc", "base"), value=100000),
                    Component(ref="RB2", kind="R", nodes=("base", "vee"), value=100000),
                    Component(ref="RE", kind="R", nodes=("vcc", "emitter"), value=10000),
                    Component(ref="CIN", kind="C", nodes=("in", "base"), value=1e-6),
                    Component(ref="COUT", kind="C", nodes=("emitter", "out"), value=1e-6),
                    Component(ref="RDC", kind="R", nodes=("out", "0"), value=1e6),
                    Component(ref="CDEC1", kind="C", nodes=("vcc", "0"), value=100e-9, role="decoupling"),
                    Component(ref="CDEC2", kind="C", nodes=("vee", "0"), value=100e-9, role="decoupling"),
                ),
            )
        elif kind == "DIODE":
            spec = DesignSpec(topology="buffer", input_offset_v=0.7, **common)
            circuit = Circuit(
                name=f"smoke_{name}",
                topology="buffer",
                spec=spec,
                components=(
                    Component(ref="D1", kind="DIODE", nodes=("in", "out"), model=name),
                    Component(ref="R1", kind="R", nodes=("out", "0"), value=10000),
                ),
            )
        else:
            results[name] = SimulationResult(status="invalid", errors=[f"no smoke fixture for {kind}"])
            continue
        results[name] = runner.evaluate(circuit, spec)
    return results
