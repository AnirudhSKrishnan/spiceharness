"""Client-led sessions keep electrical evidence and final acceptance on the server."""

from __future__ import annotations

import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from spiceharness.client_sessions import ClientDesignManager
from spiceharness.domain import DesignSpec, SimulationResult
from spiceharness.providers import ProviderRouter, Settings
from spiceharness.simulator import NgSpiceSimulator

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")


@pytest.fixture
def spec() -> DesignSpec:
    return DesignSpec(
        topology="passive_lowpass",
        gain=1,
        gain_tolerance=0.1,
        supply_positive_v=5,
        input_amplitude_v=0.1,
        input_frequency_hz=100,
        load_ohm=100000,
        cutoff_hz=1000,
        cutoff_tolerance=0.2,
    )


def test_no_keys_parameter_refinement_and_reconnect(tmp_path, spec, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Client-led mode must never call a provider")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(ProviderRouter, "validate_access", forbidden)
    monkeypatch.setattr(ProviderRouter, "structured", forbidden)
    manager = ClientDesignManager(tmp_path)
    run = manager.start_design(spec)
    run_id = run["run_id"]
    assert run["status"] == "awaiting_client" and not run["usage"]
    circuit = manager.propose_candidates(run_id)[0]
    data = circuit.model_dump(mode="json")
    data["components"][0]["value"] = 1100
    data["metadata"] = {"fake_score": 999999}
    candidate = manager.evaluate_candidate(run_id, data)
    assert candidate["score"]["valid"] and candidate["score"]["feasible"]
    assert candidate["circuit"]["components"][0]["value"] == 1100
    assert "fake_score" not in candidate["circuit"]["metadata"]
    # A new server/client object sees the same frozen run and evidence.
    reconnected = ClientDesignManager(tmp_path)
    final = reconnected.finalize_design(run_id)
    assert final["result"]["verified"] and final["result"]["feasible"]
    assert len(final["candidates"]) == 2 and not final["usage"]
    assert final["result"]["execution_mode"] == "client"
    for path in final["result"]["artifacts"].values():
        assert Path(path).is_file()
    assert final["result"]["verification_candidate_id"] != candidate["candidate_id"]


@pytest.mark.parametrize("mutation", ["spec", "wiring", "nonstandard", "metrics"])
def test_reject_client_evidence_or_changed_fixture(tmp_path, spec, mutation):
    manager = ClientDesignManager(tmp_path)
    run_id = manager.start_design(spec)["run_id"]
    data = manager.propose_candidates(run_id)[0].model_dump(mode="json")
    if mutation == "spec":
        data["spec"]["load_ohm"] = 1e9
    elif mutation == "wiring":
        data["components"][0]["nodes"][1] = "0"
    elif mutation == "nonstandard":
        data["components"][0]["value"] = 1234
    else:
        data["simulation"] = {"gain": 1}
    with pytest.raises(ValueError):
        manager.evaluate_candidate(run_id, data)
    assert not manager.get_run(run_id)["candidates"]


def test_trial_cap_reserves_fresh_verification_and_recovers_pending_row(tmp_path, spec):
    settings = Settings.model_validate({"limits": {"max_candidates": 2}})
    manager = ClientDesignManager(tmp_path, settings)
    run_id = manager.start_design(spec)["run_id"]
    proposals = manager.propose_candidates(run_id)
    manager.evaluate_candidate(run_id, proposals[0])
    changed = proposals[0].model_dump(mode="json")
    changed["components"][0]["value"] = 1100
    with pytest.raises(ValueError, match="reserved verification"):
        manager.evaluate_candidate(run_id, changed)
    # Simulate termination just after allocating the final reserved row.
    pending = manager.store.add_candidate(run_id, proposals[0], purpose="verification")
    manager.store.update_run(run_id, execution_started_at=time.time() - 0.1)
    final = ClientDesignManager(tmp_path).finalize_design(run_id)
    assert final["result"]["verified"]
    assert final["result"]["verification_candidate_id"] == pending
    assert len(final["candidates"]) == 2


def test_search_reserves_time_for_verification(tmp_path, spec):
    manager = ClientDesignManager(tmp_path, Settings.model_validate({"limits": {"max_seconds": 10}}))
    run_id = manager.start_design(spec)["run_id"]
    circuit = manager.propose_candidates(run_id)[0]
    manager.evaluate_candidate(run_id, circuit)
    manager.store.update_run(run_id, elapsed_seconds=8.1)
    with pytest.raises(ValueError, match="reserved verification time"):
        manager.evaluate_candidate(run_id, circuit)
    final = manager.finalize_design(run_id)
    assert final["result"]["verified"]
    assert final["elapsed_seconds"] >= 8.1


def test_failed_fresh_verification_never_claims_feasible(tmp_path, spec, monkeypatch):
    manager = ClientDesignManager(tmp_path)
    run_id = manager.start_design(spec)["run_id"]
    manager.evaluate_candidate(run_id, manager.propose_candidates(run_id)[0])
    monkeypatch.setattr(
        NgSpiceSimulator,
        "evaluate",
        lambda self, c, s: SimulationResult(
            status="failed", errors=["fresh verification failed"], spec_fingerprint=s.fingerprint
        ),
    )
    final = manager.finalize_design(run_id)
    assert not final["result"]["verified"] and not final["result"]["feasible"]
    assert final["result"]["candidate_feasible"]


@pytest.mark.parametrize("during_verification", [False, True])
def test_poll_and_cancel_remain_available_during_spice(tmp_path, spec, monkeypatch, during_verification):
    manager = ClientDesignManager(tmp_path)
    run_id = manager.start_design(spec)["run_id"]
    circuit = manager.propose_candidates(run_id)[0]
    if during_verification:
        manager.evaluate_candidate(run_id, circuit)
    entered, release = threading.Event(), threading.Event()
    original = NgSpiceSimulator.evaluate

    def blocked(self, c, s):
        entered.set()
        assert release.wait(5)
        return original(self, c, s)

    monkeypatch.setattr(NgSpiceSimulator, "evaluate", blocked)
    other = ClientDesignManager(tmp_path)
    with ThreadPoolExecutor(max_workers=1) as pool:
        work = (
            pool.submit(manager.finalize_design, run_id)
            if during_verification
            else (pool.submit(manager.evaluate_candidate, run_id, circuit))
        )
        try:
            assert entered.wait(3)
            assert other.get_run(run_id)["execution_started_at"] is not None
            cancelled = other.cancel_run(run_id)
            assert cancelled["status"] == "cancelled" and cancelled["cancel_requested"]
        finally:
            release.set()
        work.result(timeout=5)
    final = other.get_run(run_id)
    assert final["status"] == "cancelled"
    assert not final["result"]["verified"] and not final["result"]["feasible"]


def test_foreign_candidates_and_path_identifiers_are_rejected(tmp_path, spec):
    manager = ClientDesignManager(tmp_path)
    first = manager.start_design(spec)["run_id"]
    second = manager.start_design(spec)["run_id"]
    circuit = manager.propose_candidates(first)[0]
    candidate = manager.evaluate_candidate(first, circuit)
    manager.evaluate_candidate(second, circuit)
    with pytest.raises(ValueError, match="belonging"):
        manager.finalize_design(second, candidate["candidate_id"])
    with pytest.raises(ValueError, match="hexadecimal"):
        manager.get_run("../../outside")
