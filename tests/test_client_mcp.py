"""No-key protocol workflow used by Codex and Claude Code MCP clients."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from typer.testing import CliRunner

from spiceharness.cli import app

pytestmark = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not installed")


async def test_default_client_session_without_api_keys(tmp_path, monkeypatch):
    tmp_path = tmp_path / "workspace with 'quotes' and $cash"
    tmp_path.mkdir()
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "SPICEHARNESS_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    spec = json.loads((Path(__file__).parents[1] / "examples/lowpass.json").read_text())
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "spiceharness.mcp_server", "--workspace", str(tmp_path)],
        env={"PATH": os.environ["PATH"], "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""},
    )
    async with Client(params) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert tools["search_knowledge"].annotations.read_only_hint is True
        assert tools["evaluate_candidate"].annotations.read_only_hint is False
        assert tools["start_design"].annotations.open_world_hint is True
        guide = await client.read_resource("spiceharness://guide")
        assert "no OpenAI or Anthropic API keys" in guide.contents[0].text
        prompt = await client.get_prompt("design_circuit", {"request": "a low-pass filter"})
        assert prompt.messages
        start = await client.call_tool("start_design", {"spec": spec})
        assert not start.is_error, start
        run_id = start.structured_content["run_id"]
        dashboard_command = start.structured_content["dashboard_command"]
        dashboard_args = shlex.split(dashboard_command)
        assert Path(dashboard_args[0]).is_absolute()
        assert dashboard_args == [str(Path(sys.executable).absolute()), "-m", "spiceharness",
                                  "dashboard", run_id, "--workspace", str(tmp_path)]
        viewer_args = shlex.split(start.structured_content["viewer_command"])
        assert viewer_args == [str(Path(sys.executable).absolute()), "-m", "spiceharness",
                               "view", run_id, "--workspace", str(tmp_path)]
        assert shlex.split(start.structured_content["export_command"]) == [
            str(Path(sys.executable).absolute()), "-m", "spiceharness", "export", run_id,
            "--workspace", str(tmp_path)]
        assert not (tmp_path / ".spiceharness" / "exports").exists()
        assert not list(tmp_path.rglob("results.html"))
        polled = await client.call_tool("get_run", {"run_id": run_id})
        assert not polled.is_error
        assert polled.structured_content["dashboard_command"] == dashboard_command
        assert start.structured_content["status"] == "awaiting_client"
        assert not start.structured_content["usage"]
        proposal = await client.call_tool("propose_candidates", {"run_id": run_id})
        assert not proposal.is_error, proposal
        circuit = proposal.structured_content["candidates"][0]
        injected = await client.call_tool(
            "evaluate_candidate", {"run_id": run_id, "circuit": circuit, "measurements": {"gain": 1}}
        )
        assert injected.is_error
        evaluated = await client.call_tool("evaluate_candidate", {"run_id": run_id, "circuit": circuit})
        assert not evaluated.is_error, evaluated
        assert evaluated.structured_content["score"]["feasible"]
        candidate_id = evaluated.structured_content["candidate_id"]
    # Reopening either coding client can continue the same persisted session.
    async with Client(params) as client:
        scored = await client.call_tool("score_candidate", {"candidate_id": candidate_id})
        assert not scored.is_error and scored.structured_content["feasible"]
        final = await client.call_tool("finalize_design", {"run_id": run_id})
        assert not final.is_error, final
        run = final.structured_content
        assert run["dashboard_command"] == dashboard_command
        assert run["result"]["verified"] and run["result"]["feasible"]
        assert run["result"]["execution_mode"] == "client" and not run["usage"]
        for path in run["result"]["artifacts"].values():
            assert Path(path).is_file()
        next_run = await client.call_tool("start_design", {"spec": spec})
        stopped = await client.call_tool("cancel_run", {"run_id": next_run.structured_content["run_id"]})
        assert not stopped.is_error and stopped.structured_content["status"] == "cancelled"
        assert stopped.structured_content["dashboard_command"] == next_run.structured_content["dashboard_command"]
    # Copyable metadata works outside the repository without uv project discovery.
    dashboard = await asyncio.to_thread(subprocess.run, dashboard_args, cwd=tmp_path.parent,
                                        capture_output=True, text=True, timeout=10, check=False)
    assert dashboard.returncode == 0, dashboard.stderr
    assert run_id in dashboard.stdout and "completed" in dashboard.stdout
    runner = CliRunner()
    shown = runner.invoke(app, ["show", run_id, "--json", "-w", str(tmp_path)])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output)["result"]["verified"]
    resumed = runner.invoke(app, ["resume", run_id, "--json", "-w", str(tmp_path)])
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.output)["result"]["verified"]
