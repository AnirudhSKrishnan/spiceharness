"""The `spice` shortcut: workspace resolution, short run listing, prefixes and the installer."""
from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest
from test_result_commands import saved_experiment
from typer.testing import CliRunner

from spiceharness.cli import app
from spiceharness.domain import DesignSpec
from spiceharness.storage import RunStore
from spiceharness.workspace import command_hint, resolve_workspace, shortcut_installed

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
SPEC = DesignSpec(topology="buffer", supply_positive_v=5, input_amplitude_v=.1,
                  input_frequency_hz=100, load_ohm=10000)


def design_run(workspace: Path, run_id: str) -> str:
    store = RunStore(workspace / ".spiceharness" / "runs.sqlite3")
    store.create_run(SPEC, run_id=run_id)
    store.update_run(run_id, status="completed")
    return run_id


def test_workspace_resolution_order(tmp_path, monkeypatch):
    cwd, env_ws, default_ws, flag_ws = (tmp_path / name for name in ("cwd", "env", "default", "flag"))
    for directory in (cwd, env_ws, default_ws, flag_ws):
        directory.mkdir()
    monkeypatch.chdir(cwd)
    # 4 (last resort): the current directory when nothing else is configured
    assert resolve_workspace() == cwd.resolve()
    # 3: the install-time default beats a bare cwd
    monkeypatch.setenv("SPICEHARNESS_DEFAULT_WORKSPACE", str(default_ws))
    assert resolve_workspace() == default_ws.resolve()
    # 2: cwd containing .spiceharness beats the install default
    (cwd / ".spiceharness").mkdir()
    assert resolve_workspace() == cwd.resolve()
    # 1: SPICEHARNESS_WORKSPACE beats cwd
    monkeypatch.setenv("SPICEHARNESS_WORKSPACE", str(env_ws))
    assert resolve_workspace() == env_ws.resolve()
    # 0: an explicit -w beats everything
    assert resolve_workspace(flag_ws) == flag_ws.resolve()
    monkeypatch.setenv("SPICEHARNESS_WORKSPACE", "  ")  # blank is ignored
    assert resolve_workspace() == cwd.resolve()


def test_cli_uses_resolved_workspace_without_flag(tmp_path, monkeypatch):
    default_ws, elsewhere = tmp_path / "default", tmp_path / "elsewhere"
    elsewhere.mkdir()
    design_run(default_ws, "a" * 32)
    monkeypatch.chdir(elsewhere)
    monkeypatch.setenv("SPICEHARNESS_DEFAULT_WORKSPACE", str(default_ws))
    listed = runner.invoke(app, ["runs", "--json"])
    assert listed.exit_code == 0, listed.output
    assert json.loads(listed.output)["workspace"] == str(default_ws.resolve())
    # an explicit flag overrides the environment
    other = runner.invoke(app, ["runs", "--json", "-w", str(elsewhere)])
    assert json.loads(other.output)["runs"] == []
    # SPICEHARNESS_WORKSPACE overrides the install default
    monkeypatch.setenv("SPICEHARNESS_WORKSPACE", str(elsewhere))
    assert json.loads(runner.invoke(app, ["runs", "--json"]).output)["runs"] == []


def test_runs_lists_both_kinds_read_only(tmp_path):
    design = design_run(tmp_path, "1" * 32)
    experiment = saved_experiment(tmp_path, analysis_only=False)
    before = {path for path in tmp_path.rglob("*") if path.is_file()}
    result = runner.invoke(app, ["runs", "--json", "-w", str(tmp_path)])
    assert result.exit_code == 0, result.output
    rows = {row["run_id"]: row for row in json.loads(result.output)["runs"]}
    assert rows[design]["kind"] == "design" and rows[design]["candidates"] == 0
    assert rows[experiment["run_id"]]["kind"] == "experiment"
    assert rows[experiment["run_id"]]["candidates"] == 2 and rows[experiment["run_id"]]["status"] == "completed"
    limited = json.loads(runner.invoke(app, ["runs", "--json", "--limit", "1", "-w", str(tmp_path)]).output)
    assert len(limited["runs"]) == 1
    table = runner.invoke(app, ["runs", "-w", str(tmp_path)])
    assert table.exit_code == 0 and experiment["run_id"][:8] in table.output and "Winner" in table.output
    created = {path.name for path in tmp_path.rglob("*") if path.is_file() and path not in before}
    assert created <= {"runs.sqlite3-wal", "runs.sqlite3-shm"}, created
    assert not (tmp_path / ".spiceharness" / "knowledge.sqlite3").exists()


def test_runs_empty_workspace_creates_nothing(tmp_path):
    result = runner.invoke(app, ["runs", "-w", str(tmp_path / "missing")])
    assert result.exit_code == 0 and "No saved runs" in result.output
    assert not (tmp_path / "missing").exists()


def test_unique_prefix_ambiguity_and_unknown(tmp_path):
    one, two = "abcd" + "1" * 28, "abcd" + "2" * 28
    design_run(tmp_path, one)
    experiment = saved_experiment(tmp_path)
    args = ["-w", str(tmp_path)]
    assert runner.invoke(app, ["show", "abcd", *args]).exit_code == 0  # unique so far
    design_run(tmp_path, two)
    for command in ("show", "dashboard", "view", "candidates"):
        ambiguous = runner.invoke(app, [command, "abcd", *args] + (["--no-open"] if command == "view" else []))
        assert ambiguous.exit_code == 1 and "Ambiguous" in ambiguous.output, command
    assert runner.invoke(app, ["show", "abcd1", *args]).exit_code == 0
    exact = runner.invoke(app, ["show", experiment["run_id"][:10], "--json", *args])
    assert exact.exit_code == 0 and json.loads(exact.output)["run_id"] == experiment["run_id"]
    for bad, message in (("ffff", "No saved run"), ("abc", "at least 4"), ("../x", "hexadecimal")):
        result = runner.invoke(app, ["dashboard", bad, *args])
        assert result.exit_code == 1 and message in " ".join(result.output.split()), (bad, result.output)


def test_candidates_and_candidate_for_both_kinds(tmp_path, monkeypatch):
    experiment = saved_experiment(tmp_path, analysis_only=False)
    design = design_run(tmp_path, "9" * 32)
    store = RunStore(tmp_path / ".spiceharness" / "runs.sqlite3")
    from spiceharness.templates import make_candidate

    circuit = make_candidate(SPEC)
    candidate_id = store.add_candidate(design, circuit)
    store.save_evaluation(candidate_id, {"status": "ok", "metrics": {"gain": 1.0}, "errors": []},
                          {"valid": True, "feasible": True, "points": 12.5, "max_violation": 0, "rank_key": [0, 1]})
    args = ["-w", str(tmp_path)]
    listed = runner.invoke(app, ["candidates", experiment["run_id"][:8], *args])
    assert listed.exit_code == 0 and "saved RC candidate" in listed.output
    assert "candidate " + experiment["run_id"][:8] in listed.output
    inspect = runner.invoke(app, ["candidate", experiment["run_id"][:8], "a" * 31 + "b", *args])
    assert inspect.exit_code == 0 and "PASS" in inspect.output
    assert runner.invoke(app, ["candidate", experiment["run_id"][:8], "a", *args]).exit_code == 1
    board = runner.invoke(app, ["candidates", "9999", *args])
    assert board.exit_code == 0 and "Candidate leaderboard" in board.output
    assert candidate_id[:8] in board.output and "12.5" in board.output
    one = runner.invoke(app, ["candidate", "9999", candidate_id[:6], *args])
    assert one.exit_code == 0 and "gain" in one.output and candidate_id in one.output
    missing = runner.invoke(app, ["candidate", "9999", "deadbeef", *args])
    assert missing.exit_code == 1 and "Unknown candidate" in missing.output


def test_command_hints_prefer_shortcut_and_omit_default_workspace(tmp_path, monkeypatch):
    other = tmp_path / "other ws"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPICEHARNESS_DEFAULT_WORKSPACE", str(tmp_path))
    assert command_hint("view abc", tmp_path) == "uv run spiceharness view abc"
    assert shlex.split(command_hint("view abc", other)) == ["uv", "run", "spiceharness", "view", "abc", "-w", str(other)]
    monkeypatch.setenv("SPICEHARNESS_SHORTCUT", "1")
    assert command_hint("view abc", tmp_path) == "spice view abc"
    assert shlex.split(command_hint("view abc", other)) == ["spice", "view", "abc", "-w", str(other)]


def test_shortcut_detection_requires_spiceharness_wrapper(tmp_path, monkeypatch):
    monkeypatch.delenv("SPICEHARNESS_SHORTCUT")
    monkeypatch.setenv("HOME", str(tmp_path))
    foreign = tmp_path / "bin" / "spice"
    foreign.parent.mkdir()
    foreign.write_text("#!/bin/sh\necho some other spice\n")
    foreign.chmod(0o755)
    monkeypatch.setenv("PATH", str(foreign.parent))
    assert shortcut_installed() is False
    foreign.write_text("#!/bin/sh\n# spiceharness-shortcut\n")
    assert shortcut_installed() is True


def test_console_script_aliases_share_one_app():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts["spice"] == scripts["spiceharness"] == "spiceharness.cli:app"


def run_installer(*args: str, home: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run(["sh", str(ROOT / "scripts" / "install_shortcuts.sh"), *args], env=env,
                          capture_output=True, text=True, check=False, timeout=60)


@pytest.mark.skipif(not (ROOT / ".venv" / "bin" / "python").exists(), reason="project environment missing")
def test_installer_is_idempotent_safe_and_wrapper_runs(tmp_path):
    bin_dir, workspace = tmp_path / "bin dir", tmp_path / "work 'space' $x"
    workspace.mkdir()
    design = design_run(workspace, "7" * 32)
    created = run_installer("--bin-dir", str(bin_dir), "--workspace", str(workspace), home=tmp_path)
    assert created.returncode == 0, created.stderr
    target = bin_dir / "spice"
    assert "created" in created.stdout and target.stat().st_mode & stat.S_IXUSR
    again = run_installer("--bin-dir", str(bin_dir), "--workspace", str(workspace), home=tmp_path)
    assert again.returncode == 0 and "already up to date" in again.stdout
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPICEHARNESS_")}
    ran = subprocess.run([str(target), "runs", "--json"], env=env, cwd="/", capture_output=True, text=True,
                         check=False, timeout=60)
    assert ran.returncode == 0, ran.stderr
    assert json.loads(ran.stdout)["runs"][0]["run_id"] == design
    # An explicit SPICEHARNESS_WORKSPACE wins over the baked-in default.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    env["SPICEHARNESS_WORKSPACE"] = str(elsewhere)
    ran = subprocess.run([str(target), "runs", "--json"], env=env, cwd="/", capture_output=True, text=True,
                         check=False, timeout=60)
    assert json.loads(ran.stdout)["runs"] == []
    # Foreign files and symlinks are never overwritten or removed.
    target.write_text("#!/bin/sh\necho mine\n")
    refused = run_installer("--bin-dir", str(bin_dir), home=tmp_path)
    assert refused.returncode == 1 and "Refusing" in refused.stderr and "mine" in target.read_text()
    assert run_installer("--bin-dir", str(bin_dir), "--uninstall", home=tmp_path).returncode == 1
    target.unlink()
    assert run_installer("--bin-dir", str(bin_dir), "--workspace", str(workspace), home=tmp_path).returncode == 0
    assert run_installer("--bin-dir", str(bin_dir), "--uninstall", home=tmp_path).returncode == 0
    assert not target.exists()
