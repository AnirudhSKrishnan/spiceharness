"""CLI and real SDK protocol checks over the common persisted run engine."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from typer.testing import CliRunner

from spiceharness.cli import TERMINAL_STATUSES, app
from spiceharness.domain import DesignSpec
from spiceharness.simulator import MODEL_DIR
from spiceharness.templates import make_candidate

runner = CliRunner()


@pytest.fixture
def spec() -> DesignSpec:
    return DesignSpec(topology="passive_lowpass", supply_positive_v=5, input_amplitude_v=0.1,
                      input_frequency_hz=100, load_ohm=100_000, cutoff_hz=1000, max_candidates=2)


def test_help_and_doctor_redact_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "sk-test-secret-do-not-print"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["doctor", "--json"])
    assert secret not in result.output
    data = json.loads(result.output)
    assert data["credentials"]["OPENAI_API_KEY"] is True
    assert set(data["models"]) == set(json.loads((MODEL_DIR / "catalog.json").read_text()))
    assert isinstance(data["ngspice"]["available"], bool)


def test_invalid_spec_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.json"
    path.write_text('{"topology": ".control"}')
    result = runner.invoke(app, ["design", "--spec", str(path), "--offline", "-w", str(tmp_path)])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_dashboard_reports_empty_workspace(tmp_path: Path) -> None:
    result = runner.invoke(app, ["dashboard", "-w", str(tmp_path)])
    assert result.exit_code == 1, result.output
    assert "No saved runs" in result.output
    assert "--workspace" in result.output


@pytest.mark.parametrize("latest", [True, False])
def test_dashboard_selects_saved_run_and_renders_completed(
    tmp_path: Path, spec: DesignSpec, latest: bool,
) -> None:
    from spiceharness.storage import RunStore

    store = RunStore(tmp_path / ".spiceharness" / "runs.sqlite3")
    older = store.create_run(spec, run_id="1" * 32)
    newer = store.create_run(spec, options={"execution_mode": "client"}, run_id="2" * 32)
    for run_id in (older, newer):
        store.update_run(run_id, status="completed")
    selected = newer if latest else older
    result = runner.invoke(app, ["dashboard", *([] if latest else [older]), "-w", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert selected in result.output
    assert "completed" in result.output
    assert "Circuit schematic" in result.output and "Candidate leaderboard" in result.output
    assert len(store.list_runs()) == 2


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")
def test_cli_design_and_saved_inspection(tmp_path: Path, spec: DesignSpec) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(spec.model_dump_json())
    result = runner.invoke(app, ["design", "--spec", str(spec_path), "--offline", "--json", "-w", str(tmp_path)])
    assert result.exit_code == 0, result.output
    run = json.loads(result.output)
    assert run["status"] == "completed"
    assert run["offline"] is True
    assert run["leaderboard"]
    assert not run["usage"]
    assert all(candidate["simulation"]["status"] == "ok" for candidate in run["leaderboard"])
    shown = runner.invoke(app, ["show", run["run_id"], "--json", "-w", str(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["run_id"] == run["run_id"]
    dashboard = runner.invoke(app, ["show", run["run_id"], "-w", str(tmp_path)])
    assert "schematic" in dashboard.output
    assert "cutoff_hz" in dashboard.output
    assert "leaderboard" in dashboard.output


async def _wait_for_run(client: Client, run_id: str) -> dict:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = await client.call_tool("get_run", {"run_id": run_id})
        assert not response.is_error, response
        run = response.structured_content
        if run["status"] in TERMINAL_STATUSES:
            return run
        await asyncio.sleep(.1)
    pytest.fail("MCP run did not finish within 30 seconds")


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")
async def test_sdk_stdio_lifecycle_and_structured_tools(tmp_path: Path, spec: DesignSpec) -> None:
    """Handshake/list/call/poll/cancel with a real stdio server, including reconnect."""
    params = StdioServerParameters(command=sys.executable,
                                  args=["-m", "spiceharness.mcp_server", "--workspace", str(tmp_path)],
                                  env={"PATH": os.environ.get("PATH", "")})
    async with Client(params) as client:
        tools = await client.list_tools()
        names = {tool.name for tool in tools.tools}
        assert names == {"start_design", "get_run", "cancel_run", "validate_circuit", "simulate_circuit",
                         "score_candidate", "search_knowledge", "render_circuit", "propose_candidates",
                         "evaluate_candidate", "finalize_design", "start_experiment", "evaluate_experiment",
                         "get_experiment", "finalize_experiment", "cancel_experiment", "export_run"}
        circuit = make_candidate(spec).model_dump(mode="json")
        validated = await client.call_tool("validate_circuit", {"circuit": circuit})
        assert not validated.is_error, validated
        assert validated.structured_content["valid"] is True
        unknown_spec = DesignSpec.model_validate({**spec.model_dump(), "topology": "buffer", "cutoff_hz": None,
                                                 "opamp_model": "NOT_IN_CATALOG"})
        unknown_circuit = make_candidate(unknown_spec).model_dump(mode="json")
        unknown = await client.call_tool("validate_circuit", {"circuit": unknown_circuit})
        assert not unknown.is_error, unknown
        assert unknown.structured_content["valid"] is False
        assert any("trusted catalog" in error for error in unknown.structured_content["errors"])
        rendered = await client.call_tool("render_circuit", {"circuit": circuit})
        assert not rendered.is_error, rendered
        assert len(rendered.structured_content["bom"]) == len(circuit["components"])
        assert rendered.structured_content["schematic"]
        injected = json.loads(json.dumps(circuit))
        injected["components"][0]["nodes"][0] = "in\n.control\nquit"
        rejected = await client.call_tool("simulate_circuit", {"circuit": injected})
        assert rejected.is_error
        started = await client.call_tool("start_design", {"spec": spec.model_dump(mode="json"), "offline": True})
        assert not started.is_error, started
        run_id = started.structured_content["run_id"]
    # Closing the transport cannot discard an already-launched design worker.
    async with Client(params) as client:
        run = await _wait_for_run(client, run_id)
        assert run["status"] == "completed", run.get("error")
        assert run["offline"] is True
        assert run["leaderboard"]
        candidate = run["leaderboard"][0]
        score = await client.call_tool("score_candidate", {"candidate_id": candidate["candidate_id"]})
        assert not score.is_error, score
        assert score.structured_content["valid"] is True
        assert "cutoff_hz" in score.structured_content["metrics"]
        searched = await client.call_tool("search_knowledge", {"query": "feedback"})
        assert not searched.is_error
        assert isinstance(searched.structured_content["matches"], list)
        next_run = await client.call_tool("start_design", {"spec": spec.model_dump(mode="json"), "offline": True})
        next_id = next_run.structured_content["run_id"]
        cancelled = await client.call_tool("cancel_run", {"run_id": next_id})
        assert not cancelled.is_error
        assert cancelled.structured_content["cancel_requested"] is True
        stopped = await _wait_for_run(client, next_id)
        assert stopped["status"] == "cancelled"


def test_request_reports_missing_fields_without_starting_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class IncompleteManager:
        def normalize_request(self, request: str) -> dict:
            assert request == "build a buffer"
            return {"spec": None, "partial_spec": {"topology": "buffer"},
                    "missing_fields": ["load_ohm"], "notes": []}

        def start_design(self, *args, **kwargs):
            pytest.fail("Incomplete requirements must never start a run")

    monkeypatch.setattr("spiceharness.cli._manager", lambda *args: IncompleteManager())
    result = runner.invoke(app, ["design", "--request", "build a buffer", "-w", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert json.loads(result.output)["missing_fields"] == ["load_ohm"]
    rejected = runner.invoke(app, ["design", "--request", "build a buffer", "--offline", "-w", str(tmp_path)])
    assert rejected.exit_code == 1
    assert "requires provider access" in rejected.output


def test_compact_live_dashboard_fits_80_by_24(spec: DesignSpec) -> None:
    import io

    from rich.console import Console

    from spiceharness.cli import run_dashboard

    active_spec = DesignSpec.model_validate({**spec.model_dump(), "topology": "active_lowpass_2",
                                           "supply_positive_v": 12, "supply_negative_v": -12})
    circuit = make_candidate(active_spec).model_dump(mode="json")
    circuit["components"].extend({"ref": f"RLOAD{index}", "kind": "R", "nodes": ["out", "0"],
                                  "value": 100_000} for index in range(12))
    metrics = {"gain": 1, "cutoff_hz": 1000, "thd": .001, "supply_current_a": .002,
               "output_offset_v": .001, "output_swing_v": .1}
    candidates = [{"candidate_id": f"candidate{index}", "circuit": circuit, "verified": True,
                   "score": {"points": 1010, "max_violation": 0, "feasible": True},
                   "simulation": {"status": "ok", "metrics": metrics}} for index in range(6)]
    run = {"run_id": "a" * 32, "status": "running", "offline": True,
           "spec": active_spec.model_dump(), "leaderboard": candidates,
           "events": [{"type": "candidate_evaluated"}]}
    output = Console(file=io.StringIO(), record=True, width=80, height=24, color_system=None)
    output.print(run_dashboard(run, compact=True, height=output.height))
    snapshot = output.export_text()
    lines = snapshot.splitlines()
    assert len(lines) <= 24, snapshot
    assert "Circuit schematic" in lines[1]
    assert "U1A" in "\n".join(lines[:12]), snapshot
    assert "▷" in snapshot and "more devices" in snapshot
    assert "Spec → measured" in snapshot
    assert "Candidate leaderboard" in snapshot and "Progress: candidate_evaluated" in snapshot
    assert "Fresh check" in snapshot and "passed" in snapshot
    assert "Exact connection table" not in snapshot
    assert '"supply_positive_v"' not in snapshot
    detailed = Console(file=io.StringIO(), record=True, width=80)
    detailed.print(run_dashboard(run))
    assert "Exact connection table" in detailed.export_text()

    # A completed initial simulation is not itself a fresh verification pass.
    run["leaderboard"] = [{**candidates[0], "verified": False,
                           "simulation": {"status": "failed", "metrics": {}}}]
    for compact in (True, False):
        failed_output = Console(file=io.StringIO(), record=True, width=80, height=24)
        failed_output.print(run_dashboard(run, compact=compact, height=24))
        failed_snapshot = failed_output.export_text()
        assert "Fresh check" in failed_snapshot
        assert "passed" not in failed_snapshot
        assert "unverified" not in failed_snapshot
