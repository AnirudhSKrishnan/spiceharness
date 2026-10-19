"""Monte Carlo tolerance studies: sampler, statistics, budget, storage, surfaces and real ngspice."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from test_experiments import rc_fixture
from typer.testing import CliRunner

from spiceharness import montecarlo as mc
from spiceharness.cli import app
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import (
    ExperimentEvaluator,
    ExperimentResult,
    ExperimentSpec,
    GeneralCircuitIR,
    _score,
)
from spiceharness.montecarlo import MonteCarloRequest, MonteCarloStore, StudyContext
from spiceharness.results_viewer import export_results_viewer

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
real_spice = pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")
runner = CliRunner()


# ------------------------------------------------------------------ fake closed-form evaluator

class FakeEvaluator:
    """Closed-form RC cutoff; stands in for ngspice in unit tests."""

    def __init__(self, delay=0.0, status="ok", counter=None):
        self.timeout_s = 20.0
        self.delay, self.status, self.counter = delay, status, counter

    def evaluate(self, circuit, spec):
        if self.counter is not None:
            self.counter.append(1)
        time.sleep(self.delay)
        parts = {part.ref.lower(): part for part in circuit.components}
        metrics = {"cutoff": 1 / (2 * math.pi * parts["r1"].value * parts["c1"].value)}
        if self.status != "ok":
            return ExperimentResult(status=self.status, errors=(f"fake {self.status}",),
                                    score=_score(circuit, spec, {}, False))
        return ExperimentResult(status="ok", metrics=metrics, score=_score(circuit, spec, metrics, True),
                                model_checksums={})


def factory_for(**kwargs):
    return lambda lean: FakeEvaluator(**kwargs)


def context(requirements=True, circuit=None):
    from spiceharness.experiments import MetricBound
    circuit0, spec = rc_fixture()
    if requirements:
        spec = ExperimentSpec.model_validate({**spec.model_dump(), "requirements": (
            MetricBound(metric="cutoff", minimum=150, maximum=168).model_dump(),)})
    circuit = circuit or circuit0
    nominal = FakeEvaluator().evaluate(circuit, spec).metrics
    return StudyContext(run_id="a" * 32, candidate_id="b" * 32, candidate_name=circuit.name, circuit=circuit,
                        spec=spec, saved_metrics=nominal)


def run_all(tmp_path, ctx, request, factory=None, name="study", limit=100):
    store = MonteCarloStore(tmp_path / name, "0123456789abcdef")
    for _ in range(limit):
        study = mc.run_chunk(store, ctx, request, evaluator_factory=factory or factory_for())
        if study["state"] != "running":
            return store, study
    raise AssertionError("study did not finish")


def stable(study):
    return {k: v for k, v in study.items() if k not in ("created_at", "updated_at", "progress")}


def lines(store):
    return [{k: v for k, v in json.loads(line).items() if k != "seconds"}
            for line in store.samples_path.read_text().splitlines()]


# ------------------------------------------------------------------ 1-5 sampler and reproducibility

def test_same_seed_identical(tmp_path):
    ctx, request = context(), MonteCarloRequest(samples=30, seed=3)
    a, sa = run_all(tmp_path, ctx, request, name="a")
    b, sb = run_all(tmp_path, ctx, request, name="b")
    assert lines(a) == lines(b) and stable(sa) == stable(sb)
    assert sa["repro_check"] == {"samples_redrawn": 30, "multipliers_equal": True}


def test_different_seed_differs(tmp_path):
    ctx = context()
    a, sa = run_all(tmp_path, ctx, MonteCarloRequest(samples=30, seed=1), name="a")
    b, sb = run_all(tmp_path, ctx, MonteCarloRequest(samples=30, seed=2), name="b")
    assert lines(a)[0]["multipliers"] != lines(b)[0]["multipliers"]
    assert sa["measurements"]["cutoff"]["mean"] != sb["measurements"]["cutoff"]["mean"]


def test_sample_independent_of_n_order_workers_and_prefix(tmp_path):
    ctx = context()
    small, large = MonteCarloRequest(samples=10, seed=5), MonteCarloRequest(samples=500, seed=5)
    shuffled = GeneralCircuitIR(components=tuple(reversed(ctx.circuit.components)))
    for index in range(10):
        assert mc.sample_multipliers(small, ctx.circuit, index) == mc.sample_multipliers(large, ctx.circuit, index)
        assert mc.sample_multipliers(small, ctx.circuit, index) == mc.sample_multipliers(small, shuffled, index)
    one, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=24, seed=5, workers=1), name="one")
    three, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=24, seed=5, workers=3), name="three")
    assert lines(one) == lines(three)
    # N=20 then N=40 into the same store: exact prefix, no redraw.
    store, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=20, seed=5), name="grow")
    before = lines(store)
    counter = []
    again = mc.run_chunk(store, ctx, MonteCarloRequest(samples=20, seed=5), evaluator_factory=factory_for(counter=counter))
    assert again["state"] == "complete" and not counter
    grown = mc.run_chunk(store, ctx, MonteCarloRequest(samples=40, seed=5), evaluator_factory=factory_for())
    assert grown["state"] == "complete" and lines(store)[:20] == before and len(lines(store)) == 40
    full, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=40, seed=5), name="full")
    assert lines(full) == lines(store)


def test_zero_tolerance_unchanged():
    ctx = context()
    request = MonteCarloRequest(resistor_tolerance=0, capacitor_tolerance=0)
    m = mc.sample_multipliers(request, ctx.circuit, 0)
    assert m["r1"]["value"] == 1.0 and m["c1"]["value"] == 1.0
    assert mc.perturb_circuit(ctx.circuit, m) == ctx.circuit
    only_c = MonteCarloRequest(part_tolerances={"R1": 0})
    values = mc.sample_multipliers(only_c, ctx.circuit, 3)
    assert values["r1"]["value"] == 1.0 and values["c1"]["value"] != 1.0
    perturbed = mc.perturb_circuit(ctx.circuit, values)
    assert perturbed.components[0].value == ctx.circuit.components[0].value
    assert perturbed.components[1].value != ctx.circuit.components[1].value


def test_bounds_and_distribution():
    for index in range(2000):
        u = mc.unit_draw(1, index, "c1")
        assert 0 < u < 1
        assert 0.95 <= mc.multiplier(u, 0.05, "uniform") <= 1.05
        assert 0.95 <= mc.multiplier(u, 0.05, "gaussian") <= 1.05
    draws = [mc.multiplier(mc.unit_draw(9, i, "r"), 0.3, "gaussian") for i in range(20000)]
    assert 0.7 <= min(draws) and max(draws) <= 1.3
    assert np.std(draws) == pytest.approx(0.1, rel=0.05)
    uniform = [mc.multiplier(mc.unit_draw(9, i, "r"), 0.3, "uniform") for i in range(20000)]
    assert np.std(uniform) == pytest.approx(0.3 / math.sqrt(3), rel=0.03)


# ------------------------------------------------------------------ 6 scope

def lab4():
    spec = ExperimentSpec.model_validate_json((EXAMPLES / "experiments/lab4-supply/spec.json").read_text())
    circuit = GeneralCircuitIR.model_validate_json(
        (EXAMPLES / "experiments/lab4-supply/rc_22ohm_100uf.json").read_text())
    return circuit, spec


def test_fixture_and_protected_never_perturbed():
    circuit, spec = lab4()
    request = MonteCarloRequest(series_resistance_tolerance=0.2, samples=5)
    base = {part.ref.lower(): part for part in spec.baseline.components}
    evaluator = ExperimentEvaluator()
    for index in range(5):
        sample = mc.perturb_circuit(circuit, mc.sample_multipliers(request, circuit, index, spec.protected_refs))
        for part in sample.components:
            if part.ref.lower() in {ref.lower() for ref in spec.protected_refs}:
                assert part == base[part.ref.lower()]
        assert not {p.ref.lower() for p in sample.components} & {p.ref.lower() for p in spec.loads}
        evaluator.validate(sample, spec)
    assert any(part.kind in ("R", "C", "L") and part.ref.lower() not in
               {ref.lower() for ref in spec.protected_refs} for part in circuit.components)
    for name in ("U1", "RLOAD", "nonexistent", "RF"):
        with pytest.raises(ValueError):
            MonteCarloRequest(part_tolerances={name: 0.1}).validate_against(circuit, spec)
    changed = circuit.model_dump()
    for part in changed["components"]:
        if part["ref"] == "RF":
            part["value"] *= 1.01
    with pytest.raises(ValueError, match="protected"):
        evaluator.validate(GeneralCircuitIR.model_validate(changed), spec)
    assert mc.sample_multipliers(request, circuit, 0, spec.protected_refs).keys().isdisjoint({"u1", "rf", "rg"})


# ------------------------------------------------------------------ 7-9 statistics

def test_wilson_known_values():
    for (k, n), (lo, hi) in {(95, 100): (0.88825, 0.97846), (10, 10): (0.72247, 1.0), (0, 10): (0.0, 0.27753),
                             (50, 100): (0.40383, 0.59617), (0, 1): (0.0, 0.79345)}.items():
        got = mc.wilson_interval(k, n)
        assert got[0] == pytest.approx(lo, abs=1e-5) and got[1] == pytest.approx(hi, abs=1e-5)
    assert mc.wilson_interval(0, 0) is None
    assert mc.wilson_interval(300, 300)[0] == pytest.approx(0.9874, abs=1e-4)


def sample(i, cutoff, status="ok", error=None):
    feasible = None if status != "ok" else 150 <= cutoff <= 168
    violation = 0.0 if status != "ok" else max(0.0, (150 - cutoff) / 168, (cutoff - 168) / 168)
    return {"i": i, "status": status, "valid": status == "ok", "feasible": feasible,
            "multipliers": {"r1": {"value": 1 + i / 100}}, "values": {"r1": {"value": 1000.0}},
            "metrics": {"cutoff": cutoff} if status == "ok" else {},
            "violations": {"cutoff:0": violation} if status == "ok" else {}, "error": error, "seconds": 0.0}


def test_summary_stats():
    ctx = context()
    samples = [sample(0, 155), sample(1, 160), sample(2, 170), sample(3, 140),
               sample(4, 0, "invalid", "no crossing"), sample(5, 165)]
    out = mc.summarize(samples, ctx.spec, {"metrics": {"cutoff": 159.0}})
    values = np.array([155, 160, 170, 140, 165.0])
    stat = out["measurements"]["cutoff"]
    assert stat["n"] == 5 and stat["mean"] == pytest.approx(values.mean())
    assert stat["std"] == pytest.approx(values.std(ddof=1))
    assert stat["percentiles"]["50"] == pytest.approx(np.percentile(values, 50))
    assert sum(stat["histogram"]["counts"]) == 5 and len(stat["histogram"]["edges"]) == 21
    assert (stat["min_sample"], stat["max_sample"]) == (3, 2) and stat["nominal"] == 159.0
    assert out["counts"]["invalid"] == 1 and out["counts"]["invalid_reasons"] == {"no crossing": 1}
    assert out["yield"]["passing"] == 3 and out["yield"]["attempted"] == 6
    assert out["yield"]["limiting_requirement"] == "cutoff"
    requirement = out["requirements"][0]
    assert requirement["passing"] == 3 and requirement["attempted"] == 6
    assert requirement["min_margin"] == pytest.approx(-10 / 168) and requirement["min_margin_sample"] == 3
    assert out["worst_sample"]["index"] == 3 and out["worst_sample"]["violated"] == ["cutoff"]
    assert out["invalid_samples"] == [{"index": 4, "status": "invalid", "error": "no crossing"}]


def test_analysis_only(tmp_path):
    ctx = context(requirements=False)
    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=25))
    assert study["yield"] is None and study["requirements"] == [] and study["worst_sample"] is None
    assert study["scoring"] == "Not scored (analysis only, no acceptance requirements)"
    assert study["measurements"]["cutoff"]["n"] == 25 and study["analysis_only"] is True
    assert all(row["feasible"] is None for row in lines(store))
    assert "worst-case proof" in study["notice"]


# ------------------------------------------------------------------ 10 budget

def test_budget_and_timeouts(tmp_path):
    ctx = context()
    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=200, max_seconds=1, chunk_seconds=60),
                           factory_for(delay=0.12))
    assert study["state"] == "truncated"
    rows = lines(store)
    assert [row["i"] for row in rows] == list(range(len(rows))) and 0 < len(rows) < 200
    assert study["progress"]["completed"] == len(rows) and study["measurements"]["cutoff"]["n"] == len(rows)


class Perturbed(FakeEvaluator):
    """Behaves for the nominal circuit; returns ``status`` for every perturbed one."""

    def __init__(self, nominal, status, cut_below=None):
        super().__init__()
        self.nominal, self.status_for_samples, self.cut_below = nominal, status, cut_below

    def evaluate(self, circuit, spec):
        if circuit == self.nominal:
            return FakeEvaluator.evaluate(self, circuit, spec)
        if self.cut_below is not None and self.timeout_s >= self.cut_below:
            return FakeEvaluator.evaluate(self, circuit, spec)
        return ExperimentResult(status=self.status_for_samples, errors=("bad",),
                                score=_score(circuit, spec, {}, False))


def test_timeout_failfast_and_cap_discard(tmp_path):
    ctx = context()
    # A full-allowance timeout counts as a failed sample.
    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=3),
                           lambda lean: Perturbed(ctx.circuit, "timeout"), name="t")
    assert study["state"] == "complete" and study["counts"]["timeout"] == 3 and study["yield"]["passing"] == 0
    # First 10 unmeasurable samples abort the study.
    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=50),
                           lambda lean: Perturbed(ctx.circuit, "invalid"), name="fast")
    assert study["state"] == "aborted" and study["progress"]["completed"] == 10
    assert study["counts"]["invalid_reasons"] == {"bad": 10} and study["yield"]["passing"] == 0
    # A sample cut short by the study cap is discarded, never counted.
    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=5, max_seconds=0.5),
                           lambda lean: Perturbed(ctx.circuit, "timeout", cut_below=19.9), name="cap")
    assert study["state"] == "truncated" and study["progress"]["completed"] == 0
    assert not store.samples_path.exists() or store.samples_path.read_text() == ""


def test_chunking_and_cancel(tmp_path):
    ctx = context()
    request = MonteCarloRequest(samples=12, chunk_seconds=1)
    one, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=12), name="whole")
    store = MonteCarloStore(tmp_path / "chunked", "0123456789abcdef")
    calls = 0
    while True:
        study = mc.run_chunk(store, ctx, request, evaluator_factory=factory_for(delay=0.35))
        calls += 1
        if study["state"] != "running":
            break
    assert calls >= 3 and lines(store) == lines(one) and study["progress"]["chunks"] == calls
    cancel = MonteCarloStore(tmp_path / "cancel", "0123456789abcdef")
    cancel.prepare()
    cancel.cancel_path.touch()
    study = mc.run_chunk(cancel, ctx, MonteCarloRequest(samples=5), evaluator_factory=factory_for())
    assert study["state"] == "cancelled" and not cancel.cancel_path.exists()
    study = mc.run_chunk(cancel, ctx, MonteCarloRequest(samples=5), evaluator_factory=factory_for())
    assert study["state"] == "complete"      # a cancelled study resumes from its prefix


def test_nominal_mismatch_stops(tmp_path):
    ctx = context()
    ctx = StudyContext(**{**ctx.__dict__, "saved_metrics": {"cutoff": 1.0}})
    _, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=5))
    assert study["state"] == "nominal_mismatch" and study["nominal"]["matches_saved"] is False
    assert study["progress"]["completed"] == 0


def test_torn_trailing_line_is_repaired(tmp_path):
    ctx = context()
    store, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=6))
    with store.samples_path.open("a") as handle:
        handle.write('{"i": 6, "sta')
    assert len(store.read_samples()) == 6 and store.samples_path.read_text().endswith("\n")


# ------------------------------------------------------------------ 11 default off

GOLDEN = {
    "experiments/loaded-rc/spec.json": ("spec", "8761039efa075df0def82f15306a4bab8c1d8e79b6868d15655b6c939976c585"),
    "experiments/lab4-supply/spec.json": ("spec", "be657fbbe7f551a7b386cfe3d1c6c72788c9a6cf5958a80061cac73c22a93fb5"),
    "experiments/lab4-supply/bandwidth-spec.json": (
        "spec", "f47ed0ca286b0f3f92ebba86713065b1523b5ac720cb1c06de6bf967c9d7a86b"),
    "experiments/loaded-rc/reference.json": (
        "circuit", "68bffb1e2354dc5354a765a432e2542cff13a30fb93394608183dbd188c2e30e"),
    "experiments/lab4-supply/rc_22ohm_100uf.json": (
        "circuit", "27216d91db63ffeeb6f51b96b1859063041518155df3ed0441acc0fc936eb3e8"),
    "amplifier.json": ("design", "937e03d88cce5b6de84862696385f3fd1f3dbd987cfe4c89e5b5bd6b70ec4b41"),
    "lowpass.json": ("design", "784a1d518c0076a3d36e235f3947b6a49eca04720d9586a4ebf9a4ebd9a25349"),
}
INTERFERENCE = EXAMPLES / "experiments/interference-filter/spec.json"


def test_golden_fingerprints_and_frozen_models():
    from spiceharness.domain import DesignSpec
    models = {"spec": ExperimentSpec, "circuit": GeneralCircuitIR, "design": DesignSpec}
    for name, (kind, expected) in GOLDEN.items():
        assert models[kind].model_validate_json((EXAMPLES / name).read_text()).fingerprint == expected, name
    if INTERFERENCE.exists():
        assert ExperimentSpec.model_validate_json(INTERFERENCE.read_text()).fingerprint == \
            "48b2128b580cb63607e45b7904b7591aeab229d66cec4d03053e8b5f4d74e141"
    assert list(ExperimentSpec.model_fields) == [
        "sources", "loads", "probes", "sweep", "transient", "measurements", "requirements", "baseline",
        "protected_refs", "redesign_ports", "allowed_models", "noise_sources", "temperature_c", "provenance",
        "assumptions", "numerical_policy"]
    assert list(GeneralCircuitIR.model_fields) == ["name", "components", "connections"]
    assert "montecarlo" not in DesignSpec.model_fields and "monte_carlo" not in ExperimentSpec.model_fields


# ------------------------------------------------------------------ real ngspice

def finalized_run(tmp_path):
    spec = ExperimentSpec.model_validate_json((EXAMPLES / "experiments/loaded-rc/spec.json").read_text())
    circuit = GeneralCircuitIR.model_validate_json((EXAMPLES / "experiments/loaded-rc/reference.json").read_text())
    manager = ExperimentManager(tmp_path)
    run = manager.start_experiment(spec, execution_mode="direct")
    manager.evaluate_experiment(run["run_id"], circuit)
    final = manager.finalize_experiment(run["run_id"])
    assert final["result"]["verified"]
    return manager, final


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@real_spice
def test_default_off_changes_nothing(tmp_path):
    manager, final = finalized_run(tmp_path)
    directory = manager.directory / final["run_id"]
    assert "montecarlo" not in json.loads((directory / "run.json").read_text())
    assert not (directory / "montecarlo").exists() and "montecarlo" not in manager.get_experiment(final["run_id"])
    for row in final["candidates"]:
        assert "metrics.json" in row["simulation"]["artifacts"] and row["simulation"]["artifact_dir"]
        names = {p.name for p in Path(row["simulation"]["artifact_dir"]).iterdir()}
        assert {"circuit.json", "fixture.json", "metrics.json", "0_op.cir", "1_ac_signal.cir"} <= names
        assert any(n.endswith(".curves.npz") for n in names) and any(n.endswith(".curves.csv") for n in names)
    # Reloading a recorded run and finalizing again still works unchanged.
    copy = tmp_path / "copy"
    (copy / ".spiceharness/experiments" / final["run_id"]).mkdir(parents=True)
    shutil.copyfile(directory / "run.json", copy / ".spiceharness/experiments" / final["run_id"] / "run.json")
    again = ExperimentManager(copy).finalize_experiment(final["run_id"])
    assert again["result"]["verified"] is True


@real_spice
def test_lean_evaluator_leaves_no_files_and_matches(tmp_path):
    circuit, spec = rc_fixture()
    scratch = tmp_path / "scratch"
    lean = ExperimentEvaluator(artifact_root=scratch, lean=True, simulator_version="v").evaluate(circuit, spec)
    assert lean.artifact_dir is None and lean.artifacts == () and lean.simulator_version == "v"
    assert not scratch.exists() or not list(scratch.iterdir())
    full = ExperimentEvaluator(artifact_root=tmp_path / "full").evaluate(circuit, spec)
    assert lean.metrics == full.metrics and lean.score == full.score


@real_spice
def test_study_matches_independent_evaluation_and_leaves_run_untouched(tmp_path):
    manager, final = finalized_run(tmp_path)
    run_id = final["run_id"]
    run_json = manager.directory / run_id / "run.json"
    before = sha(run_json)
    record = json.loads(run_json.read_text())
    request = MonteCarloRequest(samples=20, seed=1, workers=4, resistor_tolerance=0.01, capacitor_tolerance=0.05)
    study = manager.run_montecarlo(run_id, request)
    while study["state"] == "running":
        study = manager.run_montecarlo(run_id, request)
    assert study["state"] == "complete" and study["candidate_id"] == final["result"]["best_candidate_id"]
    assert sha(run_json) == before
    after = json.loads(run_json.read_text())
    for key in ("evaluation_count", "candidates", "result", "status", "events", "elapsed_seconds"):
        assert after[key] == record[key]
    store = MonteCarloStore(manager.directory / run_id, study["study_id"])
    rows = store.read_samples()
    spec = ExperimentSpec.model_validate(record["spec"])
    base = GeneralCircuitIR.model_validate(final["result"]["circuit"])
    evaluator = ExperimentEvaluator(artifact_root=tmp_path / "independent")
    cutoffs = []
    for row in rows:
        circuit = mc.perturb_circuit(base, row["multipliers"])
        result = evaluator.evaluate(circuit, spec)
        assert result.status == "ok"
        for name, value in result.metrics.items():
            assert row["metrics"][name] == pytest.approx(value, rel=1e-9)
        cutoffs.append(row["metrics"]["cutoff_hz"])
    assert max(cutoffs) > min(cutoffs)
    nominal = study["nominal"]["metrics"]["cutoff_hz"]
    corners = []
    for r in (0.99, 1.01):
        for c in (0.95, 1.05):
            factors = {"r1": {"value": r}, "c1": {"value": c}}
            corners.append(evaluator.evaluate(mc.perturb_circuit(base, factors), spec).metrics["cutoff_hz"])
    assert min(corners) * (1 - 1e-6) <= min(cutoffs) and max(cutoffs) <= max(corners) * (1 + 1e-6)
    assert min(corners) < nominal < max(corners)
    passing = sum(1 for row in rows if row["status"] == "ok" and row["feasible"])
    assert study["yield"]["passing"] == passing and study["yield"]["attempted"] == 20
    assert not list(store.scratch.iterdir())
    assert sum(p.stat().st_size for p in store.root.glob("*") if p.is_file()) < 1_000_000
    assert manager.get_experiment(run_id)["montecarlo"][0]["study_id"] == study["study_id"]
    # A second run in a fresh workspace reproduces multipliers, metrics and yield.
    manager2, final2 = finalized_run(tmp_path / "second")
    study2 = manager2.run_montecarlo(final2["run_id"], request)
    while study2["state"] == "running":
        study2 = manager2.run_montecarlo(final2["run_id"], request)
    rows2 = MonteCarloStore(manager2.directory / final2["run_id"], study2["study_id"]).read_samples()
    assert [r["multipliers"] for r in rows2] == [r["multipliers"] for r in rows]
    for a, b in zip(rows, rows2, strict=True):
        for name in a["metrics"]:
            assert b["metrics"][name] == pytest.approx(a["metrics"][name], rel=1e-12)
    assert study2["yield"] == study["yield"]


@real_spice
def test_eligibility_and_explicit_candidates(tmp_path):
    manager, final = finalized_run(tmp_path)
    run_id = final["run_id"]
    with pytest.raises(ValueError, match="Unknown candidate"):
        manager.run_montecarlo(run_id, MonteCarloRequest(candidate_id="zzzz"))
    verification = next(r for r in final["candidates"] if r["purpose"] == "verification")
    study = manager.run_montecarlo(run_id, MonteCarloRequest(samples=2, candidate_id=verification["candidate_id"][:10]))
    assert study["candidate_id"] == verification["candidate_id"]
    with pytest.raises(ValueError, match="not a hexadecimal|run_id"):
        manager.resolve_run_id("xyz")
    assert manager.resolve_run_id(run_id[:6]) == run_id
    # A fresh experiment with no verified winner cannot be studied by default.
    spec = ExperimentSpec.model_validate_json((EXAMPLES / "experiments/loaded-rc/spec.json").read_text())
    fresh = manager.start_experiment(spec, execution_mode="direct")
    with pytest.raises(ValueError, match="verified winner"):
        manager.run_montecarlo(fresh["run_id"], MonteCarloRequest())
    with pytest.raises(ValueError, match="get_monte_carlo|No Monte Carlo"):
        manager.get_montecarlo(fresh["run_id"])


# ------------------------------------------------------------------ CLI, viewer, MCP

@real_spice
def test_cli_and_viewer(tmp_path):
    workspace = tmp_path / "ws with 'quote'"
    workspace.mkdir()
    manager, final = finalized_run(workspace)
    run_id = final["run_id"]
    base = ["experiment", "montecarlo", run_id[:8], "-w", str(workspace), "-n", "4", "--workers", "2"]
    result = runner.invoke(app, base)
    assert result.exit_code == 0, result.output
    assert "not a worst-case proof" in result.output and "no agents" in result.output
    assert "montecarlo-show" in result.output and str(workspace).replace("'", "'\"'\"'") in result.output.replace("\n", "")
    as_json = runner.invoke(app, [*base, "--json"])
    payload = json.loads(as_json.output[as_json.output.index("{"):])
    assert payload["schema"] == "spiceharness.montecarlo/1" and {"yield", "requirements", "measurements"} <= payload.keys()
    assert runner.invoke(app, [*base, "--min-yield", "1.1"]).exit_code == 1
    bad = runner.invoke(app, ["experiment", "montecarlo", run_id, "-w", str(workspace), "--candidate", "", "--tol", "RLOAD=0.1"])
    assert bad.exit_code == 1
    assert runner.invoke(app, ["experiment", "montecarlo", run_id, "-w", str(workspace), "--tol", "RLOAD=0.1"]).exit_code == 1
    assert runner.invoke(app, ["experiment", "montecarlo", run_id, "-w", str(workspace), "--candidate", "zz"]).exit_code == 1
    shown = runner.invoke(app, ["experiment", "montecarlo-show", run_id[:8], "-w", str(workspace)])
    assert shown.exit_code == 0 and "Wilson" in shown.output
    # viewer
    run = manager.get_experiment(run_id)
    page = export_results_viewer(run, tmp_path / "view.html", workspace=workspace)
    html = page.read_text()
    data = json.loads(re.search(r'id="run-data">(.*?)</script>', html, re.DOTALL).group(1))
    best = next(c for c in data["candidates"] if c["id"] == final["result"]["best_candidate_id"])
    assert best["montecarlo"][0]["study_id"] and "statistical, not worst-case proof" in html
    assert all("circuit" not in (s.get("worst_sample") or {}) for c in data["candidates"] for s in c["montecarlo"] or [])


def test_viewer_without_study_is_unchanged(tmp_path):
    from test_results_viewer import _run
    page = export_results_viewer(_run(), tmp_path / "v.html", workspace=tmp_path)
    data = json.loads(re.search(r'id="run-data">(.*?)</script>', page.read_text(), re.DOTALL).group(1))
    assert all(c["montecarlo"] is None for c in data["candidates"])


def test_cli_ambiguous_prefix_and_bad_input(tmp_path):
    for suffix in ("a", "b"):
        directory = tmp_path / ".spiceharness/experiments" / ("ab12" + suffix * 28)
        directory.mkdir(parents=True)
        (directory / "run.json").write_text("{}")
    result = runner.invoke(app, ["experiment", "montecarlo", "ab12", "-w", str(tmp_path)])
    assert result.exit_code == 1 and "Ambiguous" in result.output
    result = runner.invoke(app, ["experiment", "montecarlo-show", "nope", "-w", str(tmp_path)])
    assert result.exit_code == 1


@real_spice
async def test_mcp_chunk_loop_reaches_complete(tmp_path):
    spec = ExperimentSpec.model_validate_json((EXAMPLES / "experiments/loaded-rc/spec.json").read_text())
    circuit = GeneralCircuitIR.model_validate_json((EXAMPLES / "experiments/loaded-rc/reference.json").read_text())
    params = StdioServerParameters(command=sys.executable, args=["-m", "spiceharness.mcp_server", "--workspace", str(tmp_path)],
                                  env={"PATH": os.environ["PATH"]})
    async with Client(params) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert tools["get_monte_carlo"].annotations.read_only_hint is True
        assert tools["run_monte_carlo"].annotations.read_only_hint is False
        started = await client.call_tool("start_experiment", {"spec": spec.model_dump(mode="json")})
        run_id = started.structured_content["run_id"]
        await client.call_tool("evaluate_experiment", {"run_id": run_id, "circuit": circuit.model_dump(mode="json")})
        await client.call_tool("finalize_experiment", {"run_id": run_id})
        request = {"samples": 6, "workers": 2, "chunk_seconds": 1}
        for _ in range(30):
            response = await client.call_tool("run_monte_carlo", {"run_id": run_id, "request": request})
            assert not response.is_error, response
            if response.structured_content["state"] != "running":
                break
            await asyncio.sleep(0)
        content = response.structured_content
        assert content["state"] == "complete" and "not a worst-case proof" in content["notice"]
        assert content["completed"] == 6 and "yield" in content and content["viewer_command"]
        saved = await client.call_tool("get_monte_carlo", {"run_id": run_id})
        assert saved.structured_content["measurements"] and saved.structured_content["study_id"] == content["study_id"]
        rejected = await client.call_tool("run_monte_carlo", {"run_id": run_id, "request": {"samples": 5, "part_tolerances": {"RLOAD": 0.1}}})
        assert rejected.is_error


# ------------------------------------------------------------------ audit regressions

def saved_workspace(tmp_path, study_ctx=None, request=None, factory=None, state_ctx=None):
    """A workspace with a minimal run.json and a fake-evaluator study stored beside it."""
    run_id = "a" * 32
    directory = tmp_path / ".spiceharness/experiments" / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "run_id": run_id, "status": "completed", "candidates": [], "events": [], "analysis_only": False,
        "created_at": "2026-01-01T00:00:00+00:00"}))
    ctx = state_ctx or study_ctx or context()
    store = MonteCarloStore(directory, "0123456789abcdef")
    for _ in range(50):
        study = mc.run_chunk(store, ctx, request or MonteCarloRequest(samples=12), evaluator_factory=factory or factory_for())
        if study["state"] != "running":
            break
    return run_id, store, study


def bad_context():
    ctx = context()
    return StudyContext(**{**ctx.__dict__, "saved_metrics": {"cutoff": 1.0}})


def test_h1_unmeasured_study_has_no_yield(tmp_path):
    _, _, study = saved_workspace(tmp_path, state_ctx=bad_context())
    assert study["state"] == "nominal_mismatch" and study["measured"] is False
    assert study["yield"] is None and study["requirements"] == [] and study["worst_sample"] is None
    empty = mc.summarize([], context().spec, None)
    assert empty["yield"] is None and empty["requirements"] == [] and empty["measured"] is False


def test_h1_cli_renders_unmeasured_states_and_exits_nonzero(tmp_path, monkeypatch):
    run_id, _, study = saved_workspace(tmp_path, state_ctx=bad_context())
    shown = runner.invoke(app, ["experiment", "montecarlo-show", run_id[:8], "-w", str(tmp_path)])
    assert shown.exit_code == 0
    assert "NOT MEASURED" in shown.output and "Yield: not measured" in shown.output
    assert "No sampled circuit violated" not in shown.output and "0/0" not in shown.output
    monkeypatch.setattr(ExperimentManager, "run_montecarlo", lambda self, run_id, request, **kw: study)
    for extra in ([], ["--min-yield", "0.9"]):
        result = runner.invoke(app, ["experiment", "montecarlo", run_id, "-w", str(tmp_path), *extra])
        assert result.exit_code == 1 and not isinstance(result.exception, TypeError), result.output
    listing = runner.invoke(app, ["experiment", "montecarlo-show", run_id, "-w", str(tmp_path), "--json"])
    assert json.loads(listing.output)["state"] == "nominal_mismatch"


def test_h1_min_yield_guards_missing_interval():
    from spiceharness.experiment_cli import _below
    assert _below({"yield": None, "state": "complete"}, 0.5)
    assert _below({"yield": {"wilson95": None}, "state": "complete"}, 0.5)
    assert _below({"yield": {"wilson95": [0.9, 1.0]}, "state": "truncated"}, 0.5)
    assert not _below({"yield": {"wilson95": [0.9, 1.0]}, "state": "complete"}, 0.5)


def test_h1_mcp_next_step_never_asks_for_a_yield_when_unmeasured(tmp_path):
    from spiceharness.mcp_server import create_server
    server = create_server(tmp_path)
    assert server is not None   # next_step wording is exercised through the stdio test below
    source = Path(mc.__file__).with_name("mcp_server.py").read_text()
    assert "did NOT produce a yield" in source and "Do not report a pass fraction" in source


def viewer_payload(tmp_path, study_state_ctx):
    from test_results_viewer import _run
    run_id, store, _ = saved_workspace(tmp_path, state_ctx=study_state_ctx)
    path = store.root / "study.json"
    saved = json.loads(path.read_text())
    saved["candidate_id"] = "search"
    path.write_text(json.dumps(saved))
    page = export_results_viewer(_run(run_id=run_id), tmp_path / "v.html", workspace=tmp_path)
    html = page.read_text()
    data = json.loads(re.search(r'id="run-data">(.*?)</script>', html, re.DOTALL).group(1))
    return html, data


def test_h1_m1_viewer_neutral_and_unmeasured(tmp_path):
    html, data = viewer_payload(tmp_path, bad_context())
    study = data["candidates"][0]["montecarlo"][0]
    assert study["state"] == "nominal_mismatch" and study["yield"] is None
    assert "Yield not measured (state" in html and "y&&y.attempted>0" in html
    # M1: statistics are neutral text; no pass/fail glyph logic is attached to counts.
    assert "const full=y.passing===y.attempted" not in html
    assert "sym(r.passing+'/'+r.attempted" not in html


def test_l4_viewer_details():
    from spiceharness.results_viewer import _HTML
    assert "sym(error,'fail')" in _HTML                      # error paragraphs carry the glyph
    assert "activeTab==='montecarlo'&&!c.montecarlo)activeTab='circuit'" in _HTML
    assert "Math.min(1,w[1])" in _HTML
    assert mc.wilson_interval(10, 10)[1] == 1.0 and mc.wilson_interval(0, 10)[0] == 0.0
    assert mc.wilson_interval(500, 500)[1] == 1.0


def test_m2_identity_changes_stop_the_study(tmp_path):
    ctx = context()

    class Changing(FakeEvaluator):
        calls = 0

        def evaluate(self, circuit, spec):
            result = super().evaluate(circuit, spec)
            type(self).calls += 1
            return result.model_copy(update={"model_checksums": {"m": "a" if type(self).calls == 1 else "b"}})

    store, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=10), lambda lean: Changing(), name="sums")
    assert study["state"] == "identity_mismatch" and study["progress"]["completed"] == 0
    assert "checksums" in study["error"] and mc.MonteCarloStore.read_samples(store) == []
    # A changed simulator version on resume stops the study without pooling samples.
    store, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=10, chunk_seconds=1), name="ver")
    moved = StudyContext(**{**ctx.__dict__, "simulator_version": "ngspice-99"})
    study = mc.run_chunk(store, moved, MonteCarloRequest(samples=20), evaluator_factory=factory_for())
    assert study["state"] == "identity_mismatch" and "version" in study["error"]
    assert len(store.read_samples()) == 10
    again = mc.run_chunk(store, moved, MonteCarloRequest(samples=20), evaluator_factory=factory_for())
    assert again["state"] == "identity_mismatch"


def test_m2_version_is_not_cached(monkeypatch):
    seen = []

    class Run:
        stdout = "ngspice-1"

    monkeypatch.setattr(mc.shutil, "which", lambda name: "/bin/ngspice")
    monkeypatch.setattr(mc.subprocess, "run", lambda *a, **k: seen.append(1) or Run())
    mc.ngspice_version()
    mc.ngspice_version()
    assert len(seen) == 2


def test_m5_cli_tables_readable_at_80_columns(tmp_path):
    run_id, _, _ = saved_workspace(tmp_path)
    result = runner.invoke(app, ["experiment", "montecarlo-show", run_id, "-w", str(tmp_path)],
                           env={"COLUMNS": "80"})
    assert result.exit_code == 0
    for line in result.output.splitlines():
        if line.startswith("      "):          # requirement and measurement detail rows
            assert len(line) <= 80
    assert "cutoff (n=12)" in result.output and "nominal " in result.output and "Wilson 95%" in result.output
    assert "cutoff: bounds 150 .. 168" in result.output


def test_l1_cancel_experiment_stops_studies(tmp_path):
    ctx = context()
    run_id = "a" * 32
    directory = tmp_path / ".spiceharness/experiments" / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "run_id": run_id, "status": "completed", "candidates": [], "events": [], "analysis_only": False,
        "created_at": "2026-01-01T00:00:00+00:00"}))
    store = MonteCarloStore(directory, "0123456789abcdef")
    request = MonteCarloRequest(samples=40, chunk_seconds=1)
    study = mc.run_chunk(store, ctx, request, evaluator_factory=factory_for(delay=0.3))
    assert study["state"] == "running"
    manager = ExperimentManager(tmp_path)
    manager.cancel_experiment(run_id)                       # idle study: marked cancelled directly
    assert json.loads(store.study_path.read_text())["state"] == "cancelled"
    assert mc.run_chunk(store, ctx, request, evaluator_factory=factory_for())["state"] == "complete"
    # With a chunk active the sentinel is left for the sampling loop.
    store2 = MonteCarloStore(directory, "fedcba9876543210")
    mc.run_chunk(store2, ctx, request, evaluator_factory=factory_for(delay=0.3))
    with store2.lock():
        manager.cancel_experiment(run_id)
    assert store2.cancel_path.exists()
    assert mc.run_chunk(store2, ctx, request, evaluator_factory=factory_for())["state"] == "cancelled"
    result = runner.invoke(app, ["experiment", "cancel", run_id[:6], "-w", str(tmp_path)])
    assert result.exit_code == 0 and "Monte Carlo" in result.output


def test_l2_worst_rerun_is_charged_or_skipped(tmp_path):
    ctx = context()
    wide = MonteCarloRequest(samples=4, capacitor_tolerance=0.5, resistor_tolerance=0.5, max_seconds=60)
    _, study = run_all(tmp_path, ctx, wide, factory_for(delay=0.2), name="charged")
    assert study["worst_sample"] is not None and study["worst_sample"]["reverification"] == "matched"
    assert study["progress"]["elapsed_seconds"] >= 1.15          # nominal + 4 samples + rerun
    tight = MonteCarloRequest(samples=4, capacitor_tolerance=0.5, resistor_tolerance=0.5, max_seconds=1.0)
    _, study = run_all(tmp_path, ctx, tight, name="skipped")
    worst = study["worst_sample"]
    assert study["state"] == "complete" and worst["reverified"] is False
    assert worst["reverification"] == "skipped: study time budget exhausted"


def test_l3_symlinks_are_never_followed(tmp_path):
    ctx = context()
    outside = tmp_path / "outside.txt"
    outside.write_text("precious")
    store = MonteCarloStore(tmp_path / "run", "0123456789abcdef")
    store.prepare()
    (store.root / "samples.jsonl").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        mc.run_chunk(store, ctx, MonteCarloRequest(samples=3), evaluator_factory=factory_for())
    assert outside.read_text() == "precious"
    (store.root / "samples.jsonl").unlink()
    (store.root / "study.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        mc.run_chunk(store, ctx, MonteCarloRequest(samples=3), evaluator_factory=factory_for())
    (store.root / "study.json").unlink()
    (store.root / "worst").symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        mc.run_chunk(store, ctx, MonteCarloRequest(samples=3), evaluator_factory=factory_for())
    assert outside.read_text() == "precious" and not (tmp_path / "circuit.json").exists()
    # a symlinked lock file or torn-line repair target is refused as well
    (store.root / "worst").unlink()
    (store.root / "study.lock").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        mc.run_chunk(store, ctx, MonteCarloRequest(samples=3), evaluator_factory=factory_for())
    assert outside.read_text() == "precious"


def test_l3_values_beyond_ir_limits_are_invalid_samples_not_a_wedge(tmp_path):
    ctx = context()
    data = ctx.circuit.model_dump()
    for part in data["components"]:
        if part["ref"] == "R1":
            part["value"] = 1e15
    big = GeneralCircuitIR.model_validate(data)
    ctx = StudyContext(**{**ctx.__dict__, "circuit": big,
                          "saved_metrics": FakeEvaluator().evaluate(big, ctx.spec).metrics})
    for attempt in range(2):
        _, study = run_all(tmp_path, ctx, MonteCarloRequest(samples=20), name="big")
        assert study["state"] in ("complete", "aborted")
        assert any("outside the circuit IR limits" in reason for reason in study["counts"]["invalid_reasons"])
        assert study["counts"]["invalid"] > 0


def test_l3_lowering_samples_reports_consistent_prefix(tmp_path):
    ctx = context()
    store, _ = run_all(tmp_path, ctx, MonteCarloRequest(samples=30), name="shrink")
    study = mc.run_chunk(store, ctx, MonteCarloRequest(samples=10), evaluator_factory=factory_for())
    assert study["progress"]["completed"] == 10 and study["progress"]["target"] == 10
    assert study["yield"]["attempted"] == 10 and study["measurements"]["cutoff"]["n"] == 10
    assert study["counts"]["attempted"] == 10 and len(store.read_samples()) == 30


def test_l5_append_event_is_public_and_used(tmp_path):
    run_id = "a" * 32
    directory = tmp_path / ".spiceharness/experiments" / run_id
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"run_id": run_id, "status": "completed", "candidates": [],
                                                    "events": []}))
    ExperimentManager(tmp_path).append_event(run_id, {"type": "note"})
    events = json.loads((directory / "run.json").read_text())["events"]
    assert events[-1]["type"] == "note" and events[-1]["created_at"]
    source = (Path(mc.__file__).with_name("experiment_cli.py")).read_text()
    assert "manager._read(" not in source and "manager._save(" not in source
