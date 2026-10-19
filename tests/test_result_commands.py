"""Result discovery must expose persisted evidence without simulating or calling agents."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from spiceharness.cli import app
from spiceharness.domain import DesignSpec
from spiceharness.experiment_cli import experiment_dashboard
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import ExperimentSpec, GeneralCircuitIR, GeneralComponent
from spiceharness.storage import RunStore

runner = CliRunner()


def saved_experiment(workspace: Path, *, analysis_only: bool = True) -> dict:
    from spiceharness.experiments import ExperimentSource, Measurement, MetricBound, VoltageProbe

    circuit = GeneralCircuitIR(name="saved RC candidate", components=(
        GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1000),
    ), connections=(("out", "tap"),))
    spec = ExperimentSpec(sources=(ExperimentSource(ref="source", nodes=("in", "0"), dc=1),),
                          probes=(VoltageProbe(name="output", positive="out"),),
                          measurements=(Measurement(name="voltage", operator="op_voltage", probe="output"),),
                          requirements=() if analysis_only else (
                              MetricBound(metric="voltage", minimum=.9, maximum=1.1),),
                          baseline=circuit)
    manager = ExperimentManager(workspace)
    run = manager.start_experiment(spec)
    artifact = manager.directory / run["run_id"] / "evaluations" / "artifacts"
    artifact.mkdir(parents=True)
    (artifact / "0_op.cir").write_text("* preserved saved deck\n.end\n")
    for index, candidate_id in enumerate(("a" * 32, "a" * 31 + "b")):
        score = {"valid": True, "feasible": None if analysis_only else True,
                 "package_count": 1, "violations": {} if analysis_only else {"voltage:0": 0},
                 "rank_key": None if analysis_only else [0, 0, 1]}
        run["candidates"].append({"candidate_id": candidate_id, "run_id": run["run_id"],
                                  "ordinal": index + 1, "circuit": circuit.model_dump(mode="json"),
                                  "purpose": "search", "status": "completed", "verified": False,
                                  "score": score, "simulation": {"status": "ok", "score": score,
                                      "metrics": {"voltage": 1}, "artifact_dir": str(artifact), "errors": []}})
    run["status"] = "completed"
    manager._save(run)
    return manager.get_experiment(run["run_id"])


def test_view_selects_latest_both_workflows_and_defaults_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace with 'quotes' and $cash"
    workspace.mkdir()
    store = RunStore(workspace / ".spiceharness" / "runs.sqlite3")
    spec = DesignSpec(topology="buffer", supply_positive_v=5, input_amplitude_v=.1,
                      input_frequency_hz=100, load_ohm=10000)
    old_id = store.create_run(spec)
    store.update_run(old_id, status="completed")
    run = saved_experiment(workspace)
    monkeypatch.chdir(workspace)
    monkeypatch.setattr("spiceharness.cli.webbrowser.open", lambda uri: pytest.fail("--no-open launched browser"))
    result = runner.invoke(app, ["view", "--no-open"])
    assert result.exit_code == 0, result.output
    path = workspace / ".spiceharness" / "experiments" / run["run_id"] / "report" / "results.html"
    assert path.is_file() and str(path) in result.output and path.as_uri() in result.output
    assert "2 candidate records" in result.output
    newer_id = store.create_run(spec)
    store.update_run(newer_id, status="completed")
    latest = runner.invoke(app, ["view", "--no-open"])
    assert latest.exit_code == 0 and newer_id in latest.output
    explicit = runner.invoke(app, ["view", run["run_id"], "--no-open"])
    assert explicit.exit_code == 0 and run["run_id"] in explicit.output


@pytest.mark.parametrize("run_id", ["../../outside", "not-a-run", "f" * 32])
def test_view_rejects_invalid_or_unknown_run(tmp_path, run_id):
    result = runner.invoke(app, ["view", run_id, "--no-open", "-w", str(tmp_path)])
    assert result.exit_code == 1 and "Error:" in result.output
    assert not list(tmp_path.rglob("results.html"))


def test_view_reports_browser_failure_but_retains_artifact(tmp_path, monkeypatch):
    run = saved_experiment(tmp_path)
    monkeypatch.setattr("spiceharness.cli.webbrowser.open", lambda uri: False)
    result = runner.invoke(app, ["view", run["run_id"], "-w", str(tmp_path)])
    assert result.exit_code == 1 and "Browser launch failed" in result.output
    assert "file://" in result.output and list(tmp_path.rglob("results.html"))



def test_view_opens_exported_file_uri_by_default(tmp_path, monkeypatch):
    run = saved_experiment(tmp_path)
    opened = []
    monkeypatch.setattr("spiceharness.cli.webbrowser.open", lambda uri: opened.append(uri) or True)
    result = runner.invoke(app, ["view", run["run_id"], "-w", str(tmp_path)])
    assert result.exit_code == 0, result.output
    path = tmp_path / ".spiceharness" / "experiments" / run["run_id"] / "report" / "results.html"
    assert opened == [path.as_uri()]


def test_candidate_inspection_and_prefix_errors(tmp_path):
    run = saved_experiment(tmp_path, analysis_only=False)
    args = ["-w", str(tmp_path)]
    listed = runner.invoke(app, ["experiment", "candidates", run["run_id"], *args])
    assert listed.exit_code == 0 and "saved RC candidate" in listed.output
    inspected = runner.invoke(app, ["experiment", "candidate", run["run_id"], "a" * 32, *args])
    assert inspected.exit_code == 0, inspected.output
    assert "wire 1" in inspected.output and "out · tap" in inspected.output
    assert "PASS" in inspected.output and "voltage:0" in inspected.output
    assert "Generated netlist:" in inspected.output and "0_op.cir" in inspected.output
    for reference, expected in (("a", "Ambiguous"), ("bad", "Unknown")):
        bad = runner.invoke(app, ["experiment", "candidate", run["run_id"], reference, *args])
        assert bad.exit_code == 1 and expected in bad.output


def test_analysis_only_explains_absent_ranking_and_compact_fits(tmp_path):
    run = saved_experiment(tmp_path)
    output = Console(file=io.StringIO(), record=True, width=80, height=24, color_system=None)
    output.print(experiment_dashboard(run, compact=True, height=24))
    rendered = output.export_text()
    assert len(rendered.splitlines()) <= 24
    assert "no acceptance requirements" in rendered and "saved RC candidate" in rendered
    assert "Candidate leaderboard" in rendered and "R1" in rendered and "Progress:" in rendered
    listing = runner.invoke(app, ["experiment", "candidates", run["run_id"], "-w", str(tmp_path)])
    assert "Not scored (analysis only, no acceptance requirements)" in listing.output


def test_completed_dashboard_never_uses_clipped_live(tmp_path, monkeypatch):
    import spiceharness.experiment_cli as module

    run = saved_experiment(tmp_path)
    output = Console(file=io.StringIO(), record=True, width=80, height=24, force_terminal=True)
    monkeypatch.setattr(module, "console", output)
    monkeypatch.setattr(module, "Live", lambda *a, **k: pytest.fail("Completed results used Live"))
    module.watch_experiment(ExperimentManager(tmp_path), run["run_id"])
    rendered = output.export_text()
    assert len(rendered.splitlines()) > 24
    assert rendered.index("Candidate evidence") < rendered.index("Frozen operating conditions")
    assert "wire 1" in rendered and "voltage" in rendered
    assert "no acceptance requirements" in rendered
    # Result viewing never changes frozen inputs or candidate evidence.
    stored = json.loads((ExperimentManager(tmp_path).directory / run["run_id"] / "run.json").read_text())
    assert stored["candidates"] == run["candidates"]


def _hint(output: str, label: str) -> list[str]:
    import shlex

    line = next(line for line in output.splitlines() if line.startswith(label))
    words = shlex.split(line.split(": ", 1)[1])
    assert words[:3] == ["uv", "run", "spiceharness"], line
    return words[3:]


def test_printed_follow_up_commands_carry_quoted_workspace(tmp_path, monkeypatch):
    workspace = tmp_path / "work space 'quoted' $cash"
    workspace.mkdir()
    run = saved_experiment(workspace)
    monkeypatch.chdir(tmp_path)
    shown = runner.invoke(app, ["experiment", "show", run["run_id"], "-w", str(workspace)])
    assert shown.exit_code == 0, shown.output
    args = _hint(shown.output, "All candidate circuits and scores")
    # The printed hint is one unwrapped line and works when pasted as-is.
    assert args == ["view", run["run_id"], "-w", str(workspace.resolve())]
    followed = runner.invoke(app, [*args, "--no-open"])
    assert followed.exit_code == 0, followed.output
    listing = runner.invoke(app, ["experiment", "candidates", run["run_id"], "-w", str(workspace)])
    inspect = _hint(listing.output, "Inspect a circuit")
    assert inspect[:3] == ["experiment", "candidate", run["run_id"]] and inspect[3] == "CANDIDATE_ID"
    pasted = runner.invoke(app, [*inspect[:3], "a" * 32, *inspect[4:]])
    assert pasted.exit_code == 0, pasted.output
    # In the default workspace no -w is needed.
    monkeypatch.chdir(workspace)
    here = runner.invoke(app, ["experiment", "show", run["run_id"]])
    assert _hint(here.output, "All candidate circuits and scores") == ["view", run["run_id"]]


def test_viewer_refresh_command_names_workspace(tmp_path):
    from spiceharness.results_viewer import _viewer_data

    run = saved_experiment(tmp_path)
    data = _viewer_data(run, tmp_path / "a b")
    import shlex

    assert shlex.split(data["refresh_command"]) == [
        "uv", "run", "spiceharness", "view", run["run_id"], "-w", str(tmp_path / "a b")]
