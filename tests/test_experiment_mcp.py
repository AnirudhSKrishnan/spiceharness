"""Real no-key experiment protocol and bounded session lifecycle checks."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from typer.testing import CliRunner

from spiceharness.cli import app
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import (
    ExperimentSource,
    ExperimentSpec,
    GeneralCircuitIR,
    GeneralComponent,
    Measurement,
    MetricBound,
    VoltageProbe,
)


def fixture(*, analysis_only=False):
    circuit = GeneralCircuitIR(components=(
        GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1000),
        GeneralComponent(ref="R2", kind="R", nodes=("out", "0"), value=1000),
    ))
    spec = ExperimentSpec(
        sources=(ExperimentSource(ref="input", nodes=("in", "0"), dc=2),),
        loads=(GeneralComponent(ref="load", kind="R", nodes=("out", "0"), value=1e9),),
        probes=(VoltageProbe(name="output", positive="out"),),
        measurements=(Measurement(name="voltage", operator="op_voltage", probe="output"),),
        requirements=() if analysis_only else (MetricBound(metric="voltage", minimum=0.9, maximum=1.1),),
        baseline=circuit, protected_refs=("R1",), redesign_ports=("out", "0"),
        assumptions=("Nominal DC fixture; no tolerance guarantee",),
    )
    return circuit, spec


real_spice = pytest.mark.skipif(not shutil.which("ngspice"), reason="real ngspice required")


@real_spice
@pytest.mark.parametrize("analysis_only", [False, True])
async def test_experiment_stdio_full_workflow_without_api_keys(tmp_path, monkeypatch, analysis_only):
    workspace = tmp_path / "workspace with 'quotes' and $cash"
    workspace.mkdir()
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SPICEHARNESS_CONFIG"):
        monkeypatch.delenv(name, raising=False)
    circuit, spec = fixture(analysis_only=analysis_only)
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "spiceharness.mcp_server", "--workspace", str(workspace)],
        env={"PATH": os.environ["PATH"], "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""},
    )
    async with Client(params) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        for name in ("start_experiment", "evaluate_experiment", "finalize_experiment", "cancel_experiment"):
            assert tools[name].annotations.read_only_hint is False
            assert tools[name].annotations.open_world_hint is False
            assert tools[name].input_schema["additionalProperties"] is False
        assert tools["get_experiment"].annotations.read_only_hint is True
        guide = await client.read_resource("spiceharness://experiments")
        assert "not every SPICE device" in guide.contents[0].text
        prompt = await client.get_prompt("investigate_circuit", {"question": "Explain this divider"})
        assert prompt.messages
        start = await client.call_tool("start_experiment", {"spec": spec.model_dump(mode="json")})
        assert not start.is_error, start
        run = start.structured_content
        run_id = run["run_id"]
        assert run["status"] == "awaiting_client" and not run["usage"]
        dashboard_args = shlex.split(run["dashboard_command"])
        assert dashboard_args == [str(Path(sys.executable).absolute()), "-m", "spiceharness",
                                  "dashboard", run_id, "--workspace", str(workspace)]
        viewer_args = shlex.split(run["viewer_command"])
        assert viewer_args == [str(Path(sys.executable).absolute()), "-m", "spiceharness",
                               "view", run_id, "--workspace", str(workspace)]
        assert shlex.split(run["export_command"]) == [
            str(Path(sys.executable).absolute()), "-m", "spiceharness", "export", run_id,
            "--workspace", str(workspace)]
        assert not (workspace / ".spiceharness" / "exports").exists()
        assert not list(workspace.rglob("results.html"))
        payload = {"run_id": run_id, "circuit": circuit.model_dump(mode="json")}
        for supplied in ({"score": {"feasible": True}}, {"spec": {}}, {"measurements": {"voltage": 1}}):
            assert (await client.call_tool("evaluate_experiment", {**payload, **supplied})).is_error
        protected = json.loads(circuit.model_dump_json())
        protected["components"][0]["value"] = 2000
        assert (await client.call_tool("evaluate_experiment", {**payload, "circuit": protected})).is_error
        injected = json.loads(circuit.model_dump_json())
        injected["components"][1]["nodes"][0] = "out\n.control"
        assert (await client.call_tool("evaluate_experiment", {**payload, "circuit": injected})).is_error
        assert (await client.call_tool("get_experiment", {"run_id": "../../outside"})).is_error
        checked = await client.call_tool("get_experiment", {"run_id": run_id})
        assert checked.structured_content["evaluation_count"] == 0
        worse = json.loads(circuit.model_dump_json())
        worse["components"][1]["value"] = 2000
        first = await client.call_tool("evaluate_experiment", {**payload, "circuit": worse})
        assert not first.is_error, first
        first_id = first.structured_content["candidate_id"]
        if analysis_only:
            assert first.structured_content["score"]["rank_key"] is None
        else:
            assert first.structured_content["score"]["feasible"] is False
        passing = await client.call_tool("evaluate_experiment", payload)
        assert not passing.is_error, passing
        winner_id = passing.structured_content["candidate_id"]
        assert passing.structured_content["simulation"]["metrics"]["voltage"] == pytest.approx(1, rel=1e-5)
    async with Client(params) as client:
        if not analysis_only:
            rejected = await client.call_tool("finalize_experiment", {"run_id": run_id, "candidate_id": first_id})
            assert rejected.is_error
        args = {"run_id": run_id, **({"candidate_id": first_id} if analysis_only else {})}
        finished = await client.call_tool("finalize_experiment", args)
        assert not finished.is_error, finished
        run = finished.structured_content
        result = run["result"]
        assert run["status"] == "completed" and result["verified"]
        assert result["feasible"] is (None if analysis_only else True)
        assert result["best_candidate_id"] == (first_id if analysis_only else winner_id)
        assert result["verification_candidate_id"] not in (first_id, winner_id)
        assert len(run["candidates"]) == 3 and not run["usage"]
        assert not run["leaderboard"] if analysis_only else run["leaderboard"][0]["candidate_id"] == winner_id
        for artifact in result["artifacts"].values():
            assert Path(artifact).is_file()
        final_again = await client.call_tool("finalize_experiment", {"run_id": run_id})
        assert final_again.structured_content["evaluation_count"] == 3
        other = await client.call_tool("start_experiment", {"spec": spec.model_dump(mode="json")})
        other_id = other.structured_content["run_id"]
        cancelled = await client.call_tool("cancel_experiment", {"run_id": other_id})
        assert not cancelled.is_error and cancelled.structured_content["status"] == "cancelled"
        assert (await client.call_tool("evaluate_experiment", {**payload, "run_id": other_id})).is_error
    dashboard = await asyncio.to_thread(subprocess.run, dashboard_args, cwd=tmp_path.parent,
                                        capture_output=True, text=True, timeout=10, check=False)
    assert dashboard.returncode == 0, dashboard.stderr
    assert run_id in dashboard.stdout and "completed" in dashboard.stdout


@real_spice
@pytest.mark.parametrize("stage", ["evaluation", "final_export"])
def test_concurrent_cancellation_wins_without_waiting_for_spice(tmp_path, monkeypatch, stage):
    from spiceharness.experiments import ExperimentEvaluator

    circuit, spec = fixture()
    manager = ExperimentManager(tmp_path)
    run_id = manager.start_experiment(spec)["run_id"]
    reached, release = threading.Event(), threading.Event()
    if stage == "evaluation":
        original = ExperimentEvaluator.evaluate

        def delayed(self, *args, **kwargs):
            reached.set()
            assert release.wait(5)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(ExperimentEvaluator, "evaluate", delayed)
        operation = lambda: manager.evaluate_experiment(run_id, circuit)
    else:
        manager.evaluate_experiment(run_id, circuit)
        original = manager._export

        def delayed(record):
            reached.set()
            assert release.wait(5)
            return original(record)

        monkeypatch.setattr(manager, "_export", delayed)
        operation = lambda: manager.finalize_experiment(run_id)
    with ThreadPoolExecutor() as executor:
        future = executor.submit(operation)
        try:
            assert reached.wait(5)
            started = time.monotonic()
            cancelled = manager.cancel_experiment(run_id)
            assert time.monotonic() - started < 1
            assert cancelled["status"] in {"cancelling", "cancelled"}
        finally:
            release.set()
        future.result(timeout=5)
    run = manager.get_experiment(run_id)
    assert run["status"] == "cancelled"
    if run.get("result"):
        assert run["result"]["verified"] is False
        assert run["result"]["feasible"] is True  # retain measured electrical truth
        evidence = json.loads(Path(run["result"]["artifacts"]["evidence"]).read_text())
        assert evidence["status"] == "cancelled" and not evidence["result"]["verified"]


@real_spice
@pytest.mark.parametrize("blocker", ["count", "time"])
def test_interrupted_pending_recovery_and_reserved_budget(tmp_path, blocker):
    circuit, spec = fixture()
    manager = ExperimentManager(tmp_path)
    run_id = manager.start_experiment(spec)["run_id"]
    saved = manager.evaluate_experiment(run_id, circuit)
    with manager._locked(run_id) as record:
        record["candidates"].append({**saved, "candidate_id": "f" * 32, "simulation": None,
                                     "score": None, "status": "pending", "verified": False})
        record["operation_started_at"] = time.time() - 2
        record["status"] = "evaluating"
        if blocker == "count":
            record["evaluation_count"] = 19
        else:
            record["elapsed_seconds"] = 161
        manager._save(record)
    pending = manager.get_experiment(run_id)
    assert pending["status"] == "interrupted"
    assert pending["candidates"][-1]["status"] == "interrupted"
    assert len(pending["leaderboard"]) == 1
    with pytest.raises(ValueError, match="budget exhausted"):
        manager.evaluate_experiment(run_id, circuit)
    final = manager.finalize_experiment(run_id)
    assert final["evaluation_count"] == (20 if blocker == "count" else 2)
    assert final["result"]["verified"]
    assert final["candidates"][1]["status"] == "interrupted"


@real_spice
def test_fresh_validity_and_feasibility_are_independent(tmp_path):
    circuit, spec = fixture()
    spec = spec.model_copy(update={"requirements": (MetricBound(metric="voltage", maximum=0.1),)})
    manager = ExperimentManager(tmp_path)
    run_id = manager.start_experiment(spec)["run_id"]
    manager.evaluate_experiment(run_id, circuit)
    run = manager.finalize_experiment(run_id)
    assert run["status"] == "completed"
    assert run["result"]["verified"] is True and run["result"]["feasible"] is False


@real_spice
def test_cli_direct_comparison_and_latest_dashboard_across_workflows(tmp_path):
    from spiceharness.domain import DesignSpec
    from spiceharness.storage import RunStore

    circuit, spec = fixture()
    spec_path = tmp_path / "spec.json"
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    spec_path.write_text(spec.model_dump_json())
    (candidates / "spec.json").write_text(spec.model_dump_json())
    (candidates / "alternate-spec.json").write_text(spec.model_dump_json())
    (candidates / "divider.json").write_text(circuit.model_dump_json())
    store = RunStore(tmp_path / ".spiceharness" / "runs.sqlite3")
    legacy_id = store.create_run(DesignSpec(topology="buffer", supply_positive_v=5, input_amplitude_v=0.1,
                                          input_frequency_hz=1000, load_ohm=10000))
    store.update_run(legacy_id, status="completed")
    runner = CliRunner()
    comparison = runner.invoke(app, ["experiment", "compare", "--spec", str(spec_path),
                                     "--candidates", str(candidates), "--json", "-w", str(tmp_path)])
    assert comparison.exit_code == 0, comparison.output
    run = json.loads(comparison.output)
    assert run["offline"] and run["execution_mode"] == "direct" and not run["usage"]
    assert run["result"]["verified"] and run["result"]["feasible"]
    dashboard = runner.invoke(app, ["dashboard", "-w", str(tmp_path)])
    assert dashboard.exit_code == 0 and run["run_id"] in dashboard.output
    assert "no agents" in dashboard.output
    # A later legacy run must win the same latest-run routing decision.
    newer_id = store.create_run(DesignSpec(topology="buffer", supply_positive_v=5, input_amplitude_v=0.1,
                                         input_frequency_hz=1000, load_ohm=10000))
    store.update_run(newer_id, status="completed")
    latest = runner.invoke(app, ["dashboard", "-w", str(tmp_path)])
    assert latest.exit_code == 0 and newer_id in latest.output
    explicit = runner.invoke(app, ["dashboard", run["run_id"], "-w", str(tmp_path)])
    assert explicit.exit_code == 0 and run["run_id"] in explicit.output


@real_spice
def test_session_charges_actual_evaluation_time_not_only_timeout_allowance(tmp_path, monkeypatch):
    from types import SimpleNamespace

    circuit, spec = fixture()
    manager = ExperimentManager(tmp_path)
    run_id = manager.start_experiment(spec)["run_id"]
    ticks = iter((100.0, 127.0))
    monkeypatch.setattr("spiceharness.experiment_sessions.time",
                        SimpleNamespace(time=time.time, monotonic=lambda: next(ticks)))
    manager.evaluate_experiment(run_id, circuit)
    assert manager.get_experiment(run_id)["elapsed_seconds"] == 27.0


@real_spice
async def test_stdio_cancel_is_available_during_another_clients_simulation(tmp_path):
    from spiceharness.experiments import TransientAnalysis

    circuit, spec = fixture()
    spec = spec.model_copy(update={
        "transient": TransientAnalysis(start_s=0.01, stop_s=0.02, step_s=1e-7),
        "measurements": (Measurement(name="voltage", operator="transient_mean", probe="output"),),
    })
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "spiceharness.mcp_server", "--workspace", str(tmp_path)],
        env={"PATH": os.environ["PATH"], "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""},
    )
    async with Client(params) as evaluator, Client(params) as observer:
        started = await evaluator.call_tool("start_experiment", {"spec": spec.model_dump(mode="json")})
        assert not started.is_error, started
        run_id = started.structured_content["run_id"]
        pending = asyncio.create_task(evaluator.call_tool(
            "evaluate_experiment", {"run_id": run_id, "circuit": circuit.model_dump(mode="json")}))
        path = tmp_path / ".spiceharness" / "experiments" / run_id / "run.json"
        deadline = time.monotonic() + 5
        while json.loads(path.read_text())["status"] != "evaluating":
            assert not pending.done(), "expected to observe the active persisted operation"
            assert time.monotonic() < deadline
            await asyncio.sleep(0.005)
        cancelled = await observer.call_tool("cancel_experiment", {"run_id": run_id})
        assert not cancelled.is_error, cancelled
        assert cancelled.structured_content["cancel_requested"]
        measured = await asyncio.wait_for(pending, timeout=25)
        assert not measured.is_error, measured
        saved = await observer.call_tool("get_experiment", {"run_id": run_id})
        assert not saved.is_error and saved.structured_content["status"] == "cancelled"
        assert saved.structured_content["result"] is None


async def test_domain_rejections_explain_the_reason_to_the_client(tmp_path):
    """Expected rejections must not collapse into an opaque, crash-logged tool failure."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "spiceharness.mcp_server", "--workspace", str(tmp_path)],
        env={"PATH": os.environ["PATH"], "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""})
    unknown = "f" * 32
    async with Client(params) as client:
        for tool, arguments, expected in (
            ("get_run", {"run_id": unknown}, f"Unknown run: {unknown}"),
            ("finalize_design", {"run_id": unknown}, f"Unknown run: {unknown}"),
            ("get_experiment", {"run_id": unknown}, f"Unknown experiment: {unknown}"),
            ("get_experiment", {"run_id": "../../x"}, "32-character hexadecimal"),
            ("export_run", {"run_id": "../../x"}, "32-character hexadecimal"),
            ("search_knowledge", {"query": "x", "limit": 0}, "limit must be between 1 and 50"),
        ):
            result = await client.call_tool(tool, arguments)
            text = result.content[0].text
            assert result.is_error and expected in text, (tool, text)
            assert not text.endswith(f"Error executing tool {tool}") and "'" not in text.split(": ", 1)[1][:1]
