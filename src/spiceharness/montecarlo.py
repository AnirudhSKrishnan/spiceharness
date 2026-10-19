"""Monte Carlo tolerance study for saved general-experiment candidates.

A study is a post-ranking robustness estimate. It perturbs only candidate-owned
plain R/C/L values (and optionally their series resistance) around a saved,
valid candidate, runs one real ngspice evaluation per sample through the
existing evaluator in lean mode, and stores the evidence as a sidecar under the
run directory. It never edits ``run.json``, candidates, ranking or verification.

The result is a statistical estimate, not a worst-case proof. No model API is
called and no raw SPICE input is accepted: every deck comes from validated IR.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator

from .domain import StrictModel
from .experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR
from .storage import encoded, utcnow

SAMPLER_VERSION = "mc-sha256-v1"
SCHEMA = "spiceharness.montecarlo/1"
NOTICE = ("Statistical estimate over sampled R/L/C tolerances of candidate-owned parts under the frozen "
          "fixture; not a worst-case proof. Excludes host/protected-part tolerance, device-model variation, "
          "temperature and source tolerance.")
ANALYSIS_ONLY = "Not scored (analysis only, no acceptance requirements)"
SCORED = "scored under the frozen requirements"
STOPPED_STATES = ("aborted", "nominal_mismatch", "identity_mismatch")
WILSON_Z = 1.959963984540054
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
FAIL_FAST_SAMPLES = 10
MAX_SAMPLES = 500
_PHI_LOW, _PHI_HIGH = NormalDist().cdf(-3.0), NormalDist().cdf(3.0)


class MonteCarloRequest(StrictModel):
    """Science parameters define the study; execution parameters only bound the work."""

    candidate_id: str | None = Field(default=None, max_length=64)
    samples: int = Field(default=100, ge=1, le=MAX_SAMPLES)
    seed: int = Field(default=1, ge=0, le=2**63 - 1)
    distribution: Literal["uniform", "gaussian"] = "uniform"
    resistor_tolerance: float = Field(default=0.01, ge=0, le=0.5)
    capacitor_tolerance: float = Field(default=0.05, ge=0, le=0.5)
    inductor_tolerance: float = Field(default=0.10, ge=0, le=0.5)
    series_resistance_tolerance: float = Field(default=0.0, ge=0, le=0.5)
    part_tolerances: dict[str, float] = Field(default_factory=dict)
    max_seconds: float = Field(default=120, gt=0, le=300)
    chunk_seconds: float = Field(default=20, gt=0, le=60)
    workers: int = Field(default=1, ge=1, le=8)
    sample_timeout_s: float = Field(default=20, gt=0, le=20)
    keep_worst_artifacts: bool = True

    @field_validator("part_tolerances")
    @classmethod
    def lowercase_overrides(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for ref, tolerance in value.items():
            key = ref.lower()
            if key in normalized:
                raise ValueError(f"part tolerance for {ref} is given twice")
            if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) \
                    or not 0 <= tolerance <= 0.5:
                raise ValueError(f"part tolerance for {ref} must lie in [0, 0.5]")
            normalized[key] = float(tolerance)
        return normalized

    def tolerance_for(self, kind: str, ref: str) -> float:
        override = self.part_tolerances.get(ref.lower())
        if override is not None:
            return override
        return {"R": self.resistor_tolerance, "C": self.capacitor_tolerance,
                "L": self.inductor_tolerance}[kind]

    def identity(self, *, run_id: str, candidate_id: str, fixture_fingerprint: str,
                 circuit_fingerprint: str) -> dict[str, Any]:
        return {"run_id": run_id, "candidate_id": candidate_id, "fixture_fingerprint": fixture_fingerprint,
                "circuit_fingerprint": circuit_fingerprint, "sampler_version": SAMPLER_VERSION,
                "seed": self.seed, "distribution": self.distribution,
                "tolerances": [self.resistor_tolerance, self.capacitor_tolerance, self.inductor_tolerance],
                "part_tolerances": dict(sorted(self.part_tolerances.items())),
                "series_resistance_tolerance": self.series_resistance_tolerance}

    def study_id(self, **identity: str) -> str:
        canonical = json.dumps(self.identity(**identity), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def validate_against(self, circuit: GeneralCircuitIR, spec: ExperimentSpec) -> None:
        """Reject overrides naming unknown, protected, fixture or non-R/C/L parts; never ignore them."""
        protected = {ref.lower() for ref in spec.protected_refs}
        loads = {part.ref.lower() for part in spec.loads}
        parts = {part.ref.lower(): part for part in circuit.components}
        for ref in self.part_tolerances:
            if ref in protected:
                raise ValueError(f"{ref}: protected/host parts are never perturbed")
            if ref in loads:
                raise ValueError(f"{ref}: fixture loads are never perturbed")
            if ref not in parts:
                raise ValueError(f"{ref}: unknown part; overrides must name a candidate R, C or L")
            if parts[ref].kind not in ("R", "C", "L"):
                raise ValueError(f"{ref}: only plain R, C and L values are perturbed (not {parts[ref].kind})")
        if not perturbable_parts(circuit, spec.protected_refs):
            raise ValueError("candidate has no perturbable parts (no unprotected R, C or L)")

    def assumptions(self, circuit: GeneralCircuitIR, protected: tuple[str, ...]) -> list[str]:
        shape = ("uniform within the tolerance" if self.distribution == "uniform"
                 else "gaussian with sigma = tolerance/3, truncated at the tolerance")
        parts = perturbable_parts(circuit, protected)
        lines = [f"resistor {self.resistor_tolerance:.4g} fractional tolerance, {shape}",
                 f"capacitor {self.capacitor_tolerance:.4g} fractional tolerance, {shape}",
                 f"inductor {self.inductor_tolerance:.4g} fractional tolerance, {shape}"]
        if self.series_resistance_tolerance > 0:
            lines.append(f"series resistance (ESR/DCR) of candidate L/C varied by "
                         f"{self.series_resistance_tolerance:.4g}, multiplicative")
        else:
            lines.append("series resistance (ESR/DCR) held at its stated value")
        for ref, tolerance in sorted(self.part_tolerances.items()):
            lines.append(f"{ref} override {tolerance:.4g}")
        lines.append("perturbed parts: " + ", ".join(sorted(part.ref.lower() for part in parts)))
        lines.append("independent symmetric draws per part; "
                     f"sampler {SAMPLER_VERSION}, seed {self.seed}")
        if protected:
            lines.append("protected/host parts held at nominal: " + ", ".join(sorted(r.lower() for r in protected)))
        lines.append("fixture loads, sources, device models and temperature held at nominal")
        return lines


# --------------------------------------------------------------------------- sampler


def unit_draw(seed: int, index: int, ref: str, field: str = "value") -> float:
    """Stateless uniform draw strictly inside (0, 1); independent of N, order and workers."""
    digest = hashlib.sha256(f"{SAMPLER_VERSION}|{seed}|{index}|{ref.lower()}|{field}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def multiplier(u: float, tolerance: float, distribution: str) -> float:
    if tolerance == 0:
        return 1.0
    if distribution == "uniform":
        return 1 + tolerance * (2 * u - 1)
    if distribution == "gaussian":
        z = NormalDist().inv_cdf(_PHI_LOW + u * (_PHI_HIGH - _PHI_LOW))
        return 1 + tolerance * z / 3
    raise ValueError(f"unknown distribution {distribution!r}")


def perturbable_parts(circuit: GeneralCircuitIR, protected: tuple[str, ...] | list[str]) -> list[Any]:
    held = {ref.lower() for ref in protected}
    return [part for part in circuit.components
            if part.kind in ("R", "C", "L") and part.value is not None and part.ref.lower() not in held]


def sample_multipliers(request: MonteCarloRequest, circuit: GeneralCircuitIR, index: int,
                       protected: tuple[str, ...] | list[str] = ()) -> dict[str, dict[str, float]]:
    """``{ref: {"value": m, "esr": m}}`` for perturbable parts only; tolerance 0 gives exactly 1.0."""
    result: dict[str, dict[str, float]] = {}
    for part in perturbable_parts(circuit, protected):
        ref = part.ref.lower()
        entry = {"value": multiplier(unit_draw(request.seed, index, ref, "value"),
                                     request.tolerance_for(part.kind, ref), request.distribution)}
        if part.kind in ("L", "C") and part.series_resistance_ohm > 0:
            entry["esr"] = multiplier(unit_draw(request.seed, index, ref, "esr"),
                                      request.series_resistance_tolerance, request.distribution)
        result[ref] = entry
    return result


def _scaled(value: float, factor: float) -> float:
    return value if factor == 1.0 else float(f"{value * factor:.12g}")


def perturb_circuit(circuit: GeneralCircuitIR, multipliers: dict[str, dict[str, float]]) -> GeneralCircuitIR:
    """Rebuild through validation (``model_copy`` would skip it)."""
    data = circuit.model_dump()
    for part in data["components"]:
        factors = multipliers.get(part["ref"].lower())
        if not factors:
            continue
        part["value"] = _scaled(part["value"], factors.get("value", 1.0))
        if "esr" in factors:
            part["series_resistance_ohm"] = _scaled(part["series_resistance_ohm"], factors["esr"])
    return GeneralCircuitIR.model_validate(data)


def sample_values(circuit: GeneralCircuitIR, multipliers: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    perturbed = {part.ref.lower(): part for part in perturb_circuit(circuit, multipliers).components}
    values: dict[str, dict[str, float]] = {}
    for ref, factors in multipliers.items():
        entry = {"value": perturbed[ref].value}
        if "esr" in factors:
            entry["esr"] = perturbed[ref].series_resistance_ohm
        values[ref] = entry
    return values


# --------------------------------------------------------------------------- statistics


def wilson_interval(k: int, n: int) -> tuple[float, float] | None:
    if n <= 0:
        return None
    z2 = WILSON_Z ** 2
    p = k / n
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    low = 0.0 if k == 0 else max(0.0, centre - half)
    high = 1.0 if k == n else min(1.0, centre + half)
    return low, high


def _fraction(passing: int, attempted: int) -> dict[str, Any]:
    interval = wilson_interval(passing, attempted)
    return {"passing": passing, "attempted": attempted,
            "fraction": passing / attempted if attempted else None,
            "wilson95": list(interval) if interval else None}


def _margin(bound: dict[str, Any], value: float) -> float:
    scale = max(abs(bound.get("minimum") or 0), abs(bound.get("maximum") or 0), 1e-12)
    margins = []
    if bound.get("minimum") is not None:
        margins.append(value - bound["minimum"])
    if bound.get("maximum") is not None:
        margins.append(bound["maximum"] - value)
    return min(margins) / scale


def _statistics(values: list[float], indices: list[int], nominal: float | None) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    low, high = float(array.min()), float(array.max())
    if low == high:
        edges, counts = [low, high], [len(array)]
    else:
        histogram, edge = np.histogram(array, bins=20, range=(low, high))
        edges, counts = [float(x) for x in edge], [int(x) for x in histogram]
    return {"n": len(array), "nominal": nominal, "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0, "min": low, "max": high,
            "percentiles": {str(p): float(np.percentile(array, p)) for p in PERCENTILES},
            "histogram": {"edges": edges, "counts": counts},
            "min_sample": indices[int(np.argmin(array))], "max_sample": indices[int(np.argmax(array))]}


def _sensitivity(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    ok = [sample for sample in samples if sample["status"] == "ok"]
    if len(ok) < 20:
        return {}
    columns: dict[str, list[float]] = {}
    for ref in sorted({ref for sample in ok for ref in sample["multipliers"]}):
        for field in ("value", "esr"):
            if all(field in sample["multipliers"].get(ref, {}) for sample in ok):
                columns[ref if field == "value" else f"{ref}.esr"] = [
                    sample["multipliers"][ref][field] for sample in ok]
    result: dict[str, list[dict[str, Any]]] = {}
    for metric in sorted({name for sample in ok for name in sample["metrics"]}):
        y = np.array([sample["metrics"].get(metric, math.nan) for sample in ok], dtype=float)
        if not np.all(np.isfinite(y)) or float(y.std()) == 0:
            continue
        rows = []
        for ref, column in columns.items():
            x = np.array(column)
            if float(x.std()) == 0:
                continue
            rows.append({"ref": ref, "pearson": float(np.corrcoef(x, y)[0, 1])})
        rows.sort(key=lambda row: (-abs(row["pearson"]), row["ref"]))
        if rows:
            result[metric] = rows[:3]
    return result


def summarize(samples: list[dict[str, Any]], spec: ExperimentSpec, nominal: dict[str, Any] | None
              ) -> dict[str, Any]:
    """Counts, yield, per-requirement and per-measurement statistics for a contiguous prefix."""
    analysis_only = not spec.requirements
    attempted = len(samples)
    ok = [sample for sample in samples if sample["status"] == "ok"]
    status_counts = Counter(sample["status"] for sample in samples)
    reasons = Counter(sample["error"] for sample in samples if sample["status"] != "ok" and sample.get("error"))
    counts = {"attempted": attempted, "ok": len(ok), "invalid": status_counts.get("invalid", 0),
              "timeout": status_counts.get("timeout", 0),
              "failed": attempted - len(ok) - status_counts.get("invalid", 0) - status_counts.get("timeout", 0),
              "invalid_reasons": dict(reasons)}
    nominal_metrics = (nominal or {}).get("metrics") or {}
    requirements = []
    yield_summary = None
    if not analysis_only and attempted > 0:
        for index, bound in enumerate(spec.requirements):
            key = f"{bound.metric}:{index}"
            passing = sum(1 for sample in ok if sample["violations"].get(key) == 0)
            margins = [(_margin(bound.model_dump(), sample["metrics"][bound.metric]), sample["i"])
                       for sample in ok if bound.metric in sample["metrics"]]
            worst_margin = min(margins, default=None)
            requirements.append({"metric": bound.metric, "index": index, "minimum": bound.minimum,
                                 "maximum": bound.maximum, **_fraction(passing, attempted),
                                 "min_margin": worst_margin[0] if worst_margin else None,
                                 "min_margin_sample": worst_margin[1] if worst_margin else None})
        passing = sum(1 for sample in ok if sample["feasible"] is True)
        yield_summary = _fraction(passing, attempted)
        limiting = min(requirements, key=lambda row: (row["passing"], row["index"]), default=None)
        yield_summary["limiting_requirement"] = (
            limiting["metric"] if limiting and passing < attempted else None)
    measurements = {}
    for name in sorted({metric for sample in ok for metric in sample["metrics"]}):
        pairs = [(sample["metrics"][name], sample["i"]) for sample in ok if name in sample["metrics"]]
        measurements[name] = _statistics([p[0] for p in pairs], [p[1] for p in pairs], nominal_metrics.get(name))
    worst = None
    if not analysis_only and attempted > 0:
        violating = [(max(sample["violations"].values(), default=0), sample) for sample in ok]
        violating = [(value, sample) for value, sample in violating if value > 0]
        if violating:
            top = max(violating, key=lambda pair: (pair[0], -pair[1]["i"]))
            sample = top[1]
            worst = {"index": sample["i"], "max_violation": top[0],
                     "violated": [key.rsplit(":", 1)[0] for key, value in sample["violations"].items() if value > 0],
                     "multipliers": sample["multipliers"], "values": sample["values"],
                     "metrics": sample["metrics"], "circuit": None, "artifact_dir": None, "reverified": False}
    invalid = [{"index": s["i"], "status": s["status"], "error": s.get("error")}
               for s in samples if s["status"] != "ok"][:5]
    return {"measured": attempted > 0, "counts": counts, "yield": yield_summary, "requirements": requirements,
            "measurements": measurements, "sensitivity": _sensitivity(samples), "worst_sample": worst,
            "invalid_samples": invalid}


# --------------------------------------------------------------------------- storage


class MonteCarloStore:
    """Files of one study; ``study.json`` is atomically replaced, ``samples.jsonl`` is append-only."""

    def __init__(self, run_directory: Path, study_id: str):
        self.root = run_directory / "montecarlo" / study_id
        self.study_id = study_id

    @property
    def study_path(self) -> Path:
        return self.root / "study.json"

    @property
    def samples_path(self) -> Path:
        return self.root / "samples.jsonl"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def cancel_path(self) -> Path:
        return self.root / "study.cancel"

    @property
    def worst(self) -> Path:
        return self.root / "worst"

    def _refuse_link(self, path: Path) -> Path:
        if path.is_symlink():
            raise ValueError(f"Monte Carlo storage must not contain symlinks: {path.name}")
        return path

    def prepare(self) -> None:
        parent = self.root.parent
        for directory in (parent.parent, parent, self.root):
            if directory.is_symlink():
                raise ValueError("Monte Carlo storage must be a real directory, not a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("study.json", "samples.jsonl", "study.lock", "study.cancel", "scratch", "worst"):
            self._refuse_link(self.root / name)
        self.scratch.mkdir(exist_ok=True)

    def _open(self, path: Path, flags: int, mode: str):
        """Open without following a symlink at the final component."""
        self._refuse_link(path)
        descriptor = os.open(path, flags | os.O_NOFOLLOW, 0o644)
        return os.fdopen(descriptor, mode)

    @contextmanager
    def lock(self) -> Iterator[None]:
        with self._open(self.root / "study.lock", os.O_WRONLY | os.O_CREAT | os.O_APPEND, "a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("Another Monte Carlo chunk for this study is active; retry shortly") from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def load(self) -> dict[str, Any] | None:
        self._refuse_link(self.study_path)
        if not self.study_path.is_file():
            return None
        with self._open(self.study_path, os.O_RDONLY, "r") as handle:
            return json.load(handle)

    def save(self, study: dict[str, Any]) -> None:
        study["updated_at"] = utcnow()
        self._refuse_link(self.study_path)
        temporary = self.root / f"study.{uuid.uuid4().hex}.tmp"
        with self._open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, "w") as handle:
            handle.write(encoded(study))
        temporary.replace(self.study_path)

    def read_samples(self) -> list[dict[str, Any]]:
        """Return the contiguous committed prefix; a torn trailing line is dropped and repaired."""
        self._refuse_link(self.samples_path)
        if not self.samples_path.is_file():
            return []
        samples, good = [], 0
        with self._open(self.samples_path, os.O_RDONLY, "rb") as handle:
            data = handle.read()
        for line in data.splitlines(keepends=True):
            try:
                row = json.loads(line)
            except ValueError:
                break
            if not isinstance(row, dict) or row.get("i") != len(samples) or not line.endswith(b"\n"):
                break
            samples.append(row)
            good += len(line)
        if good != len(data):
            with self._open(self.samples_path, os.O_RDWR, "r+b") as handle:
                handle.truncate(good)
        return samples

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self._open(self.samples_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, "a") as handle:
            for row in rows:
                handle.write(encoded(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def ngspice_version(executable: str = "ngspice") -> str:
    """The ngspice banner, queried on every study call so an upgrade is noticed (samples never spawn it)."""
    if not shutil.which(executable):
        raise ValueError("ngspice executable was not found; Monte Carlo runs real SPICE")
    run = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False, timeout=20)
    return run.stdout.strip()[:2000]


# --------------------------------------------------------------------------- execution


@dataclass(frozen=True)
class StudyContext:
    run_id: str
    candidate_id: str
    candidate_name: str
    circuit: GeneralCircuitIR
    spec: ExperimentSpec
    saved_metrics: dict[str, float]
    simulator_version: str | None = None


EvaluatorFactory = Callable[[bool], Any]


def _close(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-15)


def _metrics_match(a: dict[str, float], b: dict[str, float], rel: float = 1e-9) -> bool:
    return set(a) == set(b) and all(_close(a[name], b[name], rel) for name in a)


def _record(index: int, multipliers: dict, values: dict, result: Any, seconds: float) -> dict[str, Any]:
    score = result.score
    return {"i": index, "status": result.status, "valid": bool(score.valid), "feasible": score.feasible,
            "multipliers": multipliers, "values": values, "metrics": dict(result.metrics),
            "violations": dict(score.violations), "error": "; ".join(result.errors) or None,
            "seconds": round(seconds, 4)}


def default_factory(store: MonteCarloStore, context: StudyContext) -> EvaluatorFactory:
    def make(lean: bool) -> ExperimentEvaluator:
        if lean:
            return ExperimentEvaluator(artifact_root=store.scratch, lean=True,
                                       simulator_version=context.simulator_version)
        return ExperimentEvaluator(artifact_root=store.worst, simulator_version=context.simulator_version)
    return make


def _new_study(store: MonteCarloStore, context: StudyContext, request: MonteCarloRequest) -> dict[str, Any]:
    analysis_only = not context.spec.requirements
    return {"schema": SCHEMA, "study_id": store.study_id, "run_id": context.run_id,
            "candidate_id": context.candidate_id, "candidate_name": context.candidate_name,
            "created_at": utcnow(), "updated_at": utcnow(),
            "identity": {"fixture_fingerprint": context.spec.fingerprint,
                         "circuit_fingerprint": context.circuit.fingerprint,
                         "sampler_version": SAMPLER_VERSION, "simulator_version": context.simulator_version,
                         "model_checksums": {}},
            "request": request.model_dump(mode="json"), "state": "running",
            "progress": {"target": request.samples, "completed": 0, "elapsed_seconds": 0.0, "chunks": 0},
            "analysis_only": analysis_only, "scoring": ANALYSIS_ONLY if analysis_only else SCORED,
            "notice": NOTICE, "assumptions": request.assumptions(context.circuit, context.spec.protected_refs),
            "nominal": None, "error": None, "measured": False, "counts": None, "yield": None, "requirements": [], "measurements": {},
            "sensitivity": {}, "worst_sample": None, "invalid_samples": [], "repro_check": None}


def run_chunk(store: MonteCarloStore, context: StudyContext, request: MonteCarloRequest, *,
              evaluator_factory: EvaluatorFactory | None = None) -> dict[str, Any]:
    """Sample for at most ``chunk_seconds`` and persist the committed contiguous prefix.

    Repeat the call while ``state == "running"``. The same science parameters resume the same
    study; raising ``samples`` enlarges it as an exact prefix extension.
    """
    request.validate_against(context.circuit, context.spec)
    store.prepare()
    with store.lock():
        factory = evaluator_factory or default_factory(store, context)
        study = store.load() or _new_study(store, context, request)
        samples = store.read_samples()
        study["request"] = request.model_dump(mode="json")
        study["progress"]["target"] = request.samples
        if study["state"] in STOPPED_STATES:
            return study
        if study["identity"].get("simulator_version") != context.simulator_version:
            study["state"] = "identity_mismatch"
            study["error"] = (f"ngspice version changed since this study started "
                              f"({study['identity'].get('simulator_version')!r} to "
                              f"{context.simulator_version!r}); samples are not pooled. "
                              "Use a different seed to start a new study.")
            _finish(study, store, context, request, samples, factory, new_samples=False)
            store.save(study)
            return study
        if study["state"] == "complete":
            study["state"] = "running"
        protected = tuple(context.spec.protected_refs)
        started = time.monotonic()

        def chunk_elapsed() -> float:
            return time.monotonic() - started

        elapsed_before = float(study["progress"]["elapsed_seconds"])
        state = "running"
        if study["nominal"] is None and not _run_nominal(study, context, request, factory):
            state = "nominal_mismatch"
        local = threading.local()
        new_total = 0

        def evaluate(index: int, allowance: float) -> dict[str, Any]:
            evaluator = getattr(local, "evaluator", None)
            if evaluator is None:
                evaluator = local.evaluator = factory(True)
            evaluator.timeout_s = allowance
            multipliers = sample_multipliers(request, context.circuit, index, protected)
            begin = time.monotonic()
            try:
                circuit = perturb_circuit(context.circuit, multipliers)
                values = sample_values(context.circuit, multipliers)
            except ValueError as exc:   # pydantic ValidationError is a ValueError: IR limits exceeded
                reason = "perturbed value outside the circuit IR limits: " + str(exc).splitlines()[0][:160]
                return {"i": index, "status": "invalid", "valid": False, "feasible": None if not context.spec.requirements
                        else False, "multipliers": multipliers, "values": {}, "metrics": {}, "violations": {},
                        "error": reason, "seconds": round(time.monotonic() - begin, 4), "_checksums": None}
            result = evaluator.evaluate(circuit, context.spec)
            row = _record(index, multipliers, values, result, time.monotonic() - begin)
            row["_checksums"] = dict(result.model_checksums)
            return row

        with ThreadPoolExecutor(max_workers=request.workers) as pool:
            while state == "running":
                if len(samples) >= request.samples:
                    state = "complete"
                    break
                if store.cancel_path.exists():
                    store.cancel_path.unlink(missing_ok=True)
                    state = "cancelled"
                    break
                if chunk_elapsed() >= request.chunk_seconds:
                    break
                remaining = request.max_seconds - (elapsed_before + chunk_elapsed())
                if remaining < 0.05:
                    state = "truncated"
                    break
                allowance = min(request.sample_timeout_s, remaining)
                indices = list(range(len(samples), min(len(samples) + request.workers, request.samples)))
                block = list(pool.map(evaluate, indices, [allowance] * len(indices)))
                if any(row["status"] == "timeout" and allowance < request.sample_timeout_s - 1e-9
                       for row in block):
                    state = "truncated"   # cut short by the study cap: discard, never bias the yield
                    break
                if any(row["status"] == "unavailable" for row in block):
                    raise ValueError(block[0]["error"] or "ngspice is unavailable")
                expected = study["identity"].get("model_checksums") or {}
                if any(row.get("_checksums") is not None and row["_checksums"] != expected for row in block):
                    study["error"] = ("model files changed since this study started (checksums differ from the "
                                      "study identity); the block was discarded and samples are not pooled")
                    state = "identity_mismatch"
                    break
                for row in block:
                    row.pop("_checksums", None)
                    row["status"] = "failed" if row["status"] == "unavailable" else row["status"]
                store.append(block)
                samples.extend(block)
                new_total += len(block)
                if len(samples) >= FAIL_FAST_SAMPLES and all(
                        row["status"] != "ok" for row in samples[:FAIL_FAST_SAMPLES]):
                    state = "aborted"
        study["progress"]["elapsed_seconds"] = round(elapsed_before + chunk_elapsed(), 3)
        if new_total or study["progress"]["chunks"] == 0:
            study["progress"]["chunks"] += 1
        study["state"] = state
        _finish(study, store, context, request, samples, factory, new_samples=True)
        store.save(study)
        return study


def _run_nominal(study: dict[str, Any], context: StudyContext, request: MonteCarloRequest,
                 factory: EvaluatorFactory) -> bool:
    evaluator = factory(True)
    evaluator.timeout_s = request.sample_timeout_s
    result = evaluator.evaluate(context.circuit, context.spec)
    matches = result.status == "ok" and _metrics_match(result.metrics, context.saved_metrics)
    study["nominal"] = {"metrics": dict(result.metrics), "feasible": result.score.feasible,
                        "status": result.status, "error": "; ".join(result.errors) or None,
                        "matches_saved": matches}
    study["identity"]["model_checksums"] = dict(result.model_checksums)
    return matches


def _redraw_matches(samples: list[dict[str, Any]], request: MonteCarloRequest,
                    context: StudyContext) -> dict[str, Any]:
    protected = tuple(context.spec.protected_refs)
    equal = all(sample_multipliers(request, context.circuit, row["i"], protected) == row["multipliers"]
                for row in samples)
    return {"samples_redrawn": len(samples), "multipliers_equal": equal}


def _finish(study: dict[str, Any], store: MonteCarloStore, context: StudyContext,
            request: MonteCarloRequest, samples: list[dict[str, Any]], factory: EvaluatorFactory,
            *, new_samples: bool) -> None:
    """Recompute every statistic over the first ``min(target, saved)`` samples, never a larger set."""
    previous = study.get("worst_sample")
    prefix = samples[:request.samples]
    study["progress"]["completed"] = len(prefix)
    study.update(summarize(prefix, context.spec, study["nominal"]))
    study["repro_check"] = _redraw_matches(prefix, request, context)
    worst = study["worst_sample"]
    if worst is None:
        return
    protected = tuple(context.spec.protected_refs)
    circuit = perturb_circuit(context.circuit, sample_multipliers(request, context.circuit,
                                                                  worst["index"], protected))
    worst["circuit"] = circuit.model_dump(mode="json")
    if previous and previous.get("index") == worst["index"] and previous.get("reverification") in (
            "matched", "mismatch", "skipped: study time budget exhausted"):
        for key in ("artifact_dir", "reverified", "reverification"):
            worst[key] = previous.get(key)
        if worst["reverification"] != "skipped: study time budget exhausted" or study["state"] != "complete":
            return
    if study["state"] == "complete" and new_samples:
        _reverify_worst(study, store, context, request, factory, circuit)


def _reverify_worst(study: dict[str, Any], store: MonteCarloStore, context: StudyContext,
                    request: MonteCarloRequest, factory: EvaluatorFactory,
                    circuit: GeneralCircuitIR) -> None:
    """Full-fidelity rerun of the worst sample; its time is charged to the study budget."""
    worst = study["worst_sample"]
    remaining = request.max_seconds - float(study["progress"]["elapsed_seconds"])
    if remaining < 1.0:
        worst["reverified"], worst["reverification"] = False, "skipped: study time budget exhausted"
        return
    store._refuse_link(store.worst)
    store.worst.mkdir(exist_ok=True)
    store._refuse_link(store.worst / "circuit.json")
    (store.worst / "circuit.json").write_text(circuit.model_dump_json(indent=2))
    evaluator = factory(False)
    evaluator.timeout_s = min(request.sample_timeout_s, remaining)
    began = time.monotonic()
    result = evaluator.evaluate(circuit, context.spec)
    study["progress"]["elapsed_seconds"] = round(
        float(study["progress"]["elapsed_seconds"]) + time.monotonic() - began, 3)
    matched = result.status == "ok" and _metrics_match(result.metrics, worst["metrics"])
    worst["reverified"] = matched
    worst["reverification"] = "matched" if matched else "mismatch"
    if result.artifact_dir:
        if request.keep_worst_artifacts:
            worst["artifact_dir"] = result.artifact_dir
        else:
            shutil.rmtree(result.artifact_dir, ignore_errors=True)


def summary_of(study: dict[str, Any]) -> dict[str, Any]:
    """Small listing row for ``get_experiment`` and CLI tables."""
    return {"study_id": study["study_id"], "candidate_id": study["candidate_id"],
            "candidate_name": study.get("candidate_name"), "state": study["state"],
            "completed": study["progress"]["completed"], "target": study["progress"]["target"],
            "yield": study.get("yield"), "measured": study.get("measured", False),
            "seed": study["request"]["seed"],
            "scoring": study.get("scoring")}
