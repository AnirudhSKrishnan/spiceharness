"""No live providers: verify graph, budgets, role isolation and durable evidence."""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from spiceharness.domain import DesignSpec, SimulationResult
from spiceharness.providers import ProviderError, ProviderRouter, RoleSettings, RunLimits, Settings
from spiceharness.scoring import score_result
from spiceharness.templates import make_candidate
from spiceharness.workflow import (
    DesignManager,
    ExpertAdvice,
    NumericReview,
    RevisionDecision,
    ShortlistDecision,
)


@pytest.fixture
def spec() -> DesignSpec:
    return DesignSpec(topology="passive_lowpass", supply_positive_v=5, input_amplitude_v=.1,
                      input_frequency_hz=100, load_ohm=100_000, cutoff_hz=1000, max_candidates=12)


class MeasuredSimulator:
    """Deliberate test fixture, never used for demos or production results."""
    def __init__(self, fail_at: int | None = None):
        self.calls = []
        self.fail_at = fail_at
        self.after_call = None

    def evaluate(self, circuit, spec):
        self.calls.append(circuit)
        if self.after_call:
            self.after_call()
        if self.fail_at == len(self.calls):
            return SimulationResult(status="failed", errors=["test fixture verification failed"],
                                    spec_fingerprint=spec.fingerprint)
        return SimulationResult(status="ok", spec_fingerprint=spec.fingerprint,
            metrics={"gain": 1, "gain_at_input_frequency": 1, "phase_deg": 0,
                     "thd": 0, "gain_compression": 0, "output_offset_v": 0,
                     "supply_current_a": 0, "output_swing_v": .1, "transient_gain": 1,
                     "cutoff_hz": 1000})


def test_offline_graph_calls_no_provider_and_reserves_verification(tmp_path: Path, spec: DesignSpec):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline mode instantiated a provider")
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator, router_factory=forbidden,
                            settings=Settings(limits=RunLimits(max_candidates=4)))
    result = manager.run_design(spec, offline=True)
    assert result["status"] == "completed", result["error"]
    assert len(simulator.calls) == len(result["candidates"]) == 4
    assert result["result"]["verified"]
    assert result["result"]["agents_called"] is False
    assert "No agents were called" in result["result"]["report"]
    assert result["usage"] == []
    assert sum(row["verified"] for row in result["candidates"]) == 1
    assert {row["node"] for row in result["checkpoints"]} >= {
        "normalize", "retrieve", "shortlist", "analytical_seed", "simulate", "score", "revise", "verify", "report"}
    for path in result["result"]["artifacts"].values():
        assert Path(path).is_file()
    assert Path(result["result"]["artifacts"]["schematic_text"]).suffix == ".txt"
    assert (manager.directory / "graph-checkpoints.sqlite3").stat().st_size > 0


def test_fresh_failure_cannot_claim_verified_feasibility(tmp_path: Path, spec: DesignSpec):
    manager = DesignManager(tmp_path, simulator=MeasuredSimulator(fail_at=2),
                            settings=Settings(limits=RunLimits(max_candidates=2)))
    result = manager.run_design(spec, offline=True)
    assert result["result"]["candidate_feasible"] is True
    assert result["result"]["feasible"] is False
    assert result["result"]["verified"] is False
    assert not any(row["verified"] for row in result["candidates"])
    assert result["result"]["verification"]["simulation"]["status"] == "failed"


def test_cancel_and_resume_use_persisted_candidates(tmp_path: Path, spec: DesignSpec):
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator,
                            settings=Settings(limits=RunLimits(max_candidates=5)))
    run_id = manager._create(spec, offline=True)
    simulator.after_call = lambda: manager.cancel_run(run_id)
    manager._execute(run_id)
    cancelled = manager.get_run(run_id)
    assert cancelled["status"] == "cancelled"
    assert len(cancelled["candidates"]) == 1
    original_id = cancelled["candidates"][0]["candidate_id"]
    simulator.after_call = None
    resumed = manager.resume_run(run_id, background=False)
    assert resumed["status"] == "completed", resumed["error"]
    assert resumed["candidates"][0]["candidate_id"] == original_id
    assert len(resumed["candidates"]) <= 5
    assert resumed["result"]["verified"]


def test_cancelling_an_interrupted_run_takes_effect_without_a_worker(tmp_path: Path, spec: DesignSpec):
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator,
                            settings=Settings(limits=RunLimits(max_candidates=5)))
    run_id = manager._create(spec, offline=True)
    manager.store.update_run(run_id, status="interrupted", error="Worker exited before completion")
    cancelled = manager.cancel_run(run_id)
    # Previously the status stayed "interrupted" (only cancel_requested was set), so the
    # cancellation had no visible effect because no worker existed to observe the flag.
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_requested"]
    assert cancelled["error"] is None  # the stale "resume this run" message is cleared
    assert manager.cancel_run(run_id)["status"] == "cancelled"
    assert simulator.calls == []


class RecordedRouter:
    instances: ClassVar[list] = []

    def __init__(self, settings, *, on_usage=lambda _: None, **kwargs):
        self.on_usage = on_usage
        self.calls = kwargs.get("calls_used", 0)
        self.roles = []
        self.instances.append(self)

    def validate_access(self):
        self.calls += 1
        self.on_usage({"operation": "models.list", "success": True})

    def structured(self, role, prompt, schema):
        self.calls += 1
        self.roles.append(role)
        self.on_usage({"role": role, "operation": "structured", "success": True})
        if schema is ShortlistDecision:
            return schema(families=["passive_lowpass"], preferred_variant_indices=[], rationale="test choice")
        if schema is NumericReview:
            assert "numeric_score" in prompt
            return schema(assessment="consistent", concerns=[], rationale="test evidence review")
        if schema is RevisionDecision:
            assert "scorer_review" in prompt
            return schema(preferred_remaining_indices=[], rationale="keep analytical priority")
        if schema is ExpertAdvice:
            return schema(action="continue", preferred_remaining_indices=[], diagnosis="No improvement in three trials")
        raise AssertionError(schema)


def test_live_graph_routes_roles_and_escalates_once(tmp_path: Path, spec: DesignSpec):
    RecordedRouter.instances.clear()
    manager = DesignManager(tmp_path, simulator=MeasuredSimulator(), router_factory=RecordedRouter,
                            settings=Settings(limits=RunLimits(max_candidates=7)))
    result = manager.run_design(spec)
    assert result["status"] == "completed", result["error"]
    router = RecordedRouter.instances[-1]
    assert router.roles.count("expert") == 1
    assert router.roles.count("scorer") == 6
    assert router.roles[0] == "designer"
    assert result["result"]["agents_called"]
    assert result["usage"]


def test_provider_failure_preserves_scored_candidate_report(tmp_path: Path, spec: DesignSpec):
    class BrokenScorer(RecordedRouter):
        def structured(self, role, prompt, schema):
            if role == "scorer":
                raise ProviderError("test scorer unavailable")
            return super().structured(role, prompt, schema)
    manager = DesignManager(tmp_path, simulator=MeasuredSimulator(), router_factory=BrokenScorer)
    result = manager.run_design(spec)
    assert result["status"] == "failed"
    assert result["candidates"][0]["score"]["feasible"]
    assert result["result"]["best_candidate_id"]
    assert result["result"]["verified"] is False
    assert "test scorer unavailable" in Path(result["result"]["artifacts"]["report"]).read_text()


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


def test_openai_responses_strict_retry_and_token_ledger():
    calls = []
    bodies = iter(['{"unexpected":true}', '{"answer":"measured"}'])
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="completed", output_text=next(bodies),
                               usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 5, "output_tokens": 3}))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    usage = []
    router = ProviderRouter(Settings(), clients={"openai": client}, on_usage=usage.append)
    assert router.structured("scorer", "Only actual evidence.", Output).answer == "measured"
    assert len(calls) == 2
    assert calls[0]["text"]["format"]["strict"]
    assert calls[0]["model"] == "gpt-6.1-sol"
    assert calls[0]["reasoning"] == {"effort": "high"}
    assert usage[-1]["tokens"]["input_tokens"] == 5
    assert [row["success"] for row in usage] == [False, True]


def test_anthropic_native_structured_outputs_and_mixed_roles():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn", usage=None,
            content=[SimpleNamespace(type="text", text='{"answer":"Claude review"}')])
    settings = Settings(expert=RoleSettings(provider="anthropic", model="explicit-claude-model"))
    router = ProviderRouter(settings, clients={"anthropic": SimpleNamespace(messages=SimpleNamespace(create=create))})
    assert router.structured("expert", "Review measured results.", Output).answer == "Claude review"
    assert calls[0]["model"] == "explicit-claude-model"
    assert calls[0]["output_config"]["format"]["type"] == "json_schema"
    assert "responses" not in calls[0]


def test_access_validation_fails_closed_without_generic_fallback():
    client = SimpleNamespace(models=SimpleNamespace(list=lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(id="different-model")], has_more=False)))
    router = ProviderRouter(Settings(), clients={"openai": client})
    with pytest.raises(ProviderError, match="no fallback"):
        router.validate_access()


def test_missing_credentials_fail_before_simulation(tmp_path: Path, spec: DesignSpec, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator)
    result = manager.run_design(spec)
    assert result["status"] == "failed"
    assert "OPENAI_API_KEY is required" in result["error"]
    assert simulator.calls == []


def test_config_roles_and_env_without_secrets(tmp_path: Path, monkeypatch):
    config = tmp_path / "roles.toml"
    config.write_text('[roles.designer]\nprovider="anthropic"\nmodel="explicit-claude"\n'
                      '[roles.expert]\nmodel="gpt-6-astra"\n[limits]\nmax_seconds=240\nmax_candidates=32\n')
    monkeypatch.setenv("SPICEHARNESS_SCORER_MODEL", "explicit-scorer")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret-value-that-must-not-persist")
    settings = Settings.load(config)
    assert settings.designer.provider == "anthropic"
    assert settings.scorer.model == "explicit-scorer"
    assert settings.limits.max_candidates == 32
    assert "sk-test" not in json.dumps(settings.model_dump())


@pytest.mark.parametrize("simulation_saved", [False, True])
def test_resume_recovers_incomplete_candidate_without_double_counting(tmp_path: Path, spec: DesignSpec, simulation_saved: bool):
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator,
                            settings=Settings(limits=RunLimits(max_candidates=2)))
    run_id = manager._create(spec, offline=True)
    candidate = make_candidate(spec)
    candidate_id = manager.store.add_candidate(run_id, candidate)
    if simulation_saved:
        manager.store.save_evaluation(candidate_id, simulator.evaluate(candidate, spec))
    manager.store.update_run(run_id, status="interrupted")
    result = manager.resume_run(run_id, background=False)
    assert result["status"] == "completed", result["error"]
    assert len(result["candidates"]) == 2
    assert result["candidates"][0]["candidate_id"] == candidate_id
    assert result["candidates"][0]["score"]["feasible"]
    assert len(simulator.calls) == 2  # saved simulation is reused; one new fresh check
    assert result["result"]["verified"]


@pytest.mark.skipif(shutil.which("ngspice") is None or os.name != "posix", reason="requires POSIX and ngspice")
def test_hard_killed_worker_is_reaped_and_can_resume(tmp_path: Path, spec: DesignSpec):
    manager = DesignManager(tmp_path, settings=Settings(limits=RunLimits(max_candidates=20)))
    run = manager.start_design(spec, offline=True)
    pid = run["worker_pid"]
    assert pid
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        record = manager.store.get_run(run["run_id"], include_details=False)
        if record["status"] == "running":
            break
        time.sleep(.01)
    else:
        pytest.fail("worker never started")
    os.kill(pid, signal.SIGKILL)
    manager._workers[run["run_id"]].wait(timeout=5)
    interrupted = manager.get_run(run["run_id"])
    assert interrupted["status"] == "interrupted"
    assert interrupted["elapsed_seconds"] > 0
    resumed = manager.resume_run(run["run_id"], background=False)
    assert resumed["status"] == "completed", resumed["error"]
    assert resumed["result"]["verified"]


def test_expert_retry_cannot_exceed_expensive_call_budget():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(status="completed", output_text='{"bad":1}', usage=None)
    router = ProviderRouter(Settings(), clients={"openai": SimpleNamespace(responses=SimpleNamespace(create=create))})
    with pytest.raises(ProviderError, match="Expert API call limit"):
        router.structured("expert", "Review actual evidence", Output)
    assert len(calls) == 1


@pytest.mark.parametrize("saved_stage", ["inserted", "simulated", "scored"])
def test_resume_full_budget_recovers_reserved_verification_row(tmp_path: Path, spec: DesignSpec, saved_stage: str):
    simulator = MeasuredSimulator()
    manager = DesignManager(tmp_path, simulator=simulator,
                            settings=Settings(limits=RunLimits(max_candidates=2)))
    run_id = manager._create(spec, offline=True)
    circuit = make_candidate(spec)
    search_id = manager.store.add_candidate(run_id, circuit)
    measured = simulator.evaluate(circuit, spec)
    manager.store.save_evaluation(search_id, measured, score_result(spec, circuit, measured))
    verification_id = manager.store.add_candidate(run_id, circuit, purpose="verification")
    if saved_stage != "inserted":
        fresh = simulator.evaluate(circuit, spec)
        numeric = score_result(spec, circuit, fresh) if saved_stage == "scored" else None
        manager.store.save_evaluation(verification_id, fresh, numeric)
    manager.store.update_run(run_id, status="interrupted")
    result = manager.resume_run(run_id, background=False)
    assert result["status"] == "completed", result["error"]
    assert len(result["candidates"]) == 2
    assert len(simulator.calls) == 2
    assert result["result"]["best_candidate_id"] == search_id
    assert result["result"]["verification_candidate_id"] == verification_id
    assert result["result"]["verified"]
    assert result["candidates"][-1]["purpose"] == "verification"
    assert result["candidates"][-1]["verified"]


def test_stale_running_read_cannot_replace_concurrent_completion(tmp_path: Path, spec: DesignSpec):
    manager = DesignManager(tmp_path)
    run_id = manager._create(spec, offline=True)
    manager.store.claim_run(run_id, os.getpid())
    class FinishingChild:
        def poll(self):
            # Simulate completion between get_run's SELECT and child.poll().
            manager.store.update_run(run_id, status="completed", worker_pid=None)
            return 0
    manager._workers[run_id] = FinishingChild()
    result = manager.get_run(run_id)
    assert result["status"] == "completed"
    assert result["error"] is None
