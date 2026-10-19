"""Portable exports preserve evaluator evidence and enforce local artifact trust."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from spiceharness.cli import app
from spiceharness.domain import Circuit, Component, DesignSpec
from spiceharness.experiment_sessions import ExperimentManager
from spiceharness.experiments import (
    ExperimentEvaluator,
    ExperimentSource,
    ExperimentSpec,
    GeneralCircuitIR,
    GeneralComponent,
    Measurement,
    MetricBound,
    VoltageProbe,
)
from spiceharness.simulator import MODEL_DIR, NgSpiceSimulator, tolerance_cases
from spiceharness.spice_bundle import export_spice_bundle


def _run(workspace: Path, *, measured=True, active=False, analysis_only=False) -> dict:
    parts = [GeneralComponent(ref="R1", kind="R", nodes=("in", "out"), value=1000),
             GeneralComponent(ref="R2", kind="R", nodes=("out", "0"), value=1000)]
    if active:
        parts.append(GeneralComponent(ref="D1", kind="DIODE", nodes=("out", "0"), model="1N4148"))
    circuit = GeneralCircuitIR(name="Saved RC circuit", components=tuple(parts))
    spec = ExperimentSpec(sources=(ExperimentSource(ref="input", nodes=("in", "0"), dc=1),),
                          probes=(VoltageProbe(name="output", positive="out"),),
                          measurements=(Measurement(name="dc", operator="op_voltage", probe="output"),),
                          requirements=() if analysis_only else (MetricBound(metric="dc", minimum=.4),),
                          allowed_models=("1N4148",) if active else (), baseline=circuit)
    root = workspace / ".spiceharness" / "experiments" / ("a" * 32) / "evaluations" / "saved"
    simulation = None
    score = None
    if measured:
        root.mkdir(parents=True)
        model_files = {}
        checksums = {}
        if active:
            model = MODEL_DIR / "1N4148.lib"
            if not model.exists():
                pytest.skip("Original pinned diode model is not available locally")
            shutil.copyfile(model, root / "model_0.lib")
            model_files = {"1N4148": "model_0.lib"}
            checksums = {"1N4148": hashlib.sha256(model.read_bytes()).hexdigest()}
        (root / "0_op.cir").write_text(ExperimentEvaluator().generate_netlist(circuit, spec, "op", model_files=model_files))
        (root / "0_op.log").write_text("Original saved log\n")
        (root / "0_op.raw").write_text("Original saved raw\n")
        (root / ".env").write_text("API_KEY=must-not-export\n")
        (root / ".spiceinit").write_text("shell must-not-execute\n")
        score = {"valid": True, "feasible": None if analysis_only else True,
                 "rank_key": None if analysis_only else [0, 0, 0, len(parts)], "package_count": len(parts),
                 "violations": {} if analysis_only else {"dc:0": 0}}
        simulation = {"status": "ok", "metrics": {"dc": .5}, "artifact_dir": str(root), "model_checksums": checksums}
    row = {"candidate_id": "candidate", "circuit": circuit.model_dump(mode="json"), "purpose": "search",
           "verified": False, "score": score, "simulation": simulation, "status": "pending" if not measured else "completed"}
    fresh = {**row, "candidate_id": "fresh", "purpose": "verification", "verified": bool(measured)}
    return {"run_id": "a" * 32, "spec": spec.model_dump(mode="json"), "analysis_only": analysis_only,
            "created_at": "2026-10-15T00:00:00Z", "status": "completed", "operation_started_at": None,
            "candidates": [row, fresh], "leaderboard": [row, fresh],
            "result": {"best_candidate_id": "candidate", "verification_candidate_id": "fresh", "verified": bool(measured)}}


def test_bundle_preserves_decks_models_scores_and_fresh_selection(tmp_path, monkeypatch):
    run = _run(tmp_path, active=True)
    before = copy.deepcopy(run)
    monkeypatch.setattr("subprocess.run", lambda *a, **kw: pytest.fail("Export must not invoke SPICE or agents"))
    manifest_path = export_spice_bundle(run, tmp_path / "portable with 'quotes' $cash", workspace=tmp_path)
    manifest = json.loads(manifest_path.read_text())
    bundle = manifest_path.parent
    assert run == before
    assert len(manifest["candidates"]) == 2
    assert manifest["open_schematic"] == "fresh/candidate.kicad_sch"
    assert [row["rank"] for row in manifest["candidates"]] == [1, None]
    assert "fresh verification of the best candidate" in (bundle / "README.md").read_text()
    for row in manifest["candidates"]:
        target = bundle / row["path"]
        original = Path(run["candidates"][0]["simulation"]["artifact_dir"])
        assert (target / "0_op.cir").read_bytes() == (original / "0_op.cir").read_bytes()
        assert '.include "model_0.lib"' in (target / "0_op.cir").read_text()
        assert (target / "model_0.lib").read_bytes() == (original / "model_0.lib").read_bytes()
        assert (target / ".spiceinit").read_text() == "set ngbehavior=ltpsa\nset filetype=ascii\n"
        assert not (target / ".env").exists()
        assert json.loads((target / "score.json").read_text()) == run["candidates"][0]["score"]
        assert json.loads((target / "metrics.json").read_text()) == {"dc": .5}
        assert row["native_runnable"] and row["saved_decks_runnable"]
        text = (target / "candidate.kicad_sch").read_text()
        assert text.startswith("(kicad_sch") and '"Sim.Library" "model_0.lib"' in text
        assert json.loads((target / "candidate.kicad_pro").read_text()) == {}
        assert 'HOME="$PWD"' in (target / "README.md").read_text()
    assert all(hashlib.sha256((bundle / name).read_bytes()).hexdigest() == checksum
               for name, checksum in manifest["files"].items())


def test_real_ngspice_can_run_relocated_bundle(tmp_path):
    executable = shutil.which("ngspice")
    if not executable:
        pytest.skip("ngspice required for real portable model validation")
    import os
    import subprocess

    manifest_path = export_spice_bundle(_run(tmp_path, active=True), tmp_path / "export", workspace=tmp_path)
    relocated = tmp_path / "another location with spaces"
    shutil.move(manifest_path.parent, relocated)
    candidate = relocated / "candidate"
    process = subprocess.run([executable, "-b", "-r", "user.raw", "0_op.cir"], cwd=candidate,
                             env=dict(os.environ, HOME=str(candidate), SPICE_ASCIIRAWFILE="1"),
                             capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stdout + process.stderr
    assert (candidate / "user.raw").is_file()


def test_pending_analysis_only_and_safe_ids_do_not_invent_results(tmp_path):
    run = _run(tmp_path, measured=False, analysis_only=True)
    run["candidates"][0]["candidate_id"] = "../../unsafe;$(echo key)"
    run["candidates"][0]["circuit"]["name"] = "Circuit name with $(touch stolen) and `shell`"
    path = export_spice_bundle(run, tmp_path / "bundle", workspace=tmp_path)
    manifest = json.loads(path.read_text())
    assert manifest["analysis_only"]
    assert all(row["rank"] is None and row["feasible"] is None for row in manifest["candidates"])
    assert all("/" not in row["path"] and ";" not in row["path"] for row in manifest["candidates"])
    assert not any(path.parent.rglob("*.cir"))
    assert "no feasibility ranking or winner" in (path.parent / "README.md").read_text()
    assert "No readable saved" in next(path.parent.rglob("OUTPUTS.md")).read_text()
    assert not (tmp_path / "stolen").exists()


@pytest.mark.parametrize("attack", ["outside", "traversal", "directory_link", "file_link", "model_hash", "deck_control",
                                    "uppercase_deck", "unknown_ir"])
def test_rejects_untrusted_paths_models_decks_and_ir(tmp_path, attack):
    run = _run(tmp_path, active=attack == "model_hash")
    simulation = run["candidates"][0]["simulation"]
    artifacts = Path(simulation["artifact_dir"])
    if attack == "outside":
        simulation["artifact_dir"] = str(tmp_path / "outside")
    elif attack == "traversal":
        simulation["artifact_dir"] = str(artifacts / ".." / "saved")
    elif attack == "directory_link":
        link = artifacts.parent / "link"
        link.symlink_to(artifacts, target_is_directory=True)
        simulation["artifact_dir"] = str(link)
    elif attack == "file_link":
        raw = artifacts / "0_op.raw"
        raw.unlink()
        raw.symlink_to(artifacts / ".env")
    elif attack == "model_hash":
        (artifacts / "model_0.lib").write_text("* not the pinned model\n")
    elif attack == "deck_control":
        with (artifacts / "0_op.cir").open("a") as handle:
            handle.write(".control\nshell cat ~/.env\n.endc\n")
    elif attack == "uppercase_deck":
        (artifacts / "INJECTED.CIR").write_text(".control\nshell cat ~/.env\n.endc\n")
    else:
        run["candidates"][0]["circuit"]["spice"] = ".control\nshell exfiltrate\n"
    with pytest.raises(ValueError):
        export_spice_bundle(run, tmp_path / "bundle", workspace=tmp_path)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".spiceharness-export-*"))


def test_reexport_keeps_user_edits_and_versions_destination(tmp_path):
    run = _run(tmp_path)
    first = export_spice_bundle(run, tmp_path / "bundle", workspace=tmp_path)
    schematic = first.parent / "candidate" / "candidate.kicad_sch"
    schematic.write_text("User's edited schematic\n")
    second = export_spice_bundle(run, tmp_path / "bundle", workspace=tmp_path)
    assert second.parent.name == "bundle-2"
    assert schematic.read_text() == "User's edited schematic\n"
    assert (second.parent / "candidate" / "candidate.kicad_sch").read_text().startswith("(kicad_sch")


def test_missing_original_model_is_explicit_not_idealized(tmp_path):
    run = _run(tmp_path, active=True)
    (Path(run["candidates"][0]["simulation"]["artifact_dir"]) / "model_0.lib").unlink()
    path = export_spice_bundle(run, tmp_path / "bundle", workspace=tmp_path)
    manifest = json.loads(path.read_text())
    assert not any(row["native_runnable"] or row["saved_decks_runnable"] for row in manifest["candidates"])
    assert all(row["missing"] for row in manifest["candidates"])
    assert not any(path.parent.rglob("*.cir"))
    assert not any(path.parent.rglob("*.lib"))


def test_legacy_bundle_preserves_nominal_and_original_tolerance_decks(tmp_path):
    spec = DesignSpec(topology="passive_lowpass", supply_positive_v=5, input_amplitude_v=.1,
                      input_frequency_hz=100, cutoff_hz=1000, load_ohm=10000, corner_policy="tolerance")
    circuit = Circuit(name="Legacy RC", topology="passive_lowpass", spec=spec, components=(
        Component(ref="R1", kind="R", nodes=("in", "out"), value=1000),
        Component(ref="C1", kind="C", nodes=("out", "0"), value=1e-7)))
    artifacts = tmp_path / ".spiceharness" / "runs" / ("b" * 32) / "simulations" / "nominal"
    artifacts.mkdir(parents=True)
    simulator = NgSpiceSimulator()
    (artifacts / "op.cir").write_text(simulator.generate_netlist(circuit, "op", spec))
    corner_dir = artifacts / "corner"
    corner_dir.mkdir()
    corner_spec = spec.model_copy(update={"corner_policy": "nominal"})
    corner = circuit.model_copy(update={"components": tolerance_cases(spec, circuit)["R1.low"], "spec": corner_spec})
    original_corner = simulator.generate_netlist(corner, "op", corner_spec)
    (corner_dir / "op.cir").write_text(original_corner)
    simulation = {"status": "ok", "metrics": {"gain": .9}, "artifact_dir": str(artifacts),
                  "analyses": {"corners": {"R1.low": {"status": "ok", "artifact_dir": str(corner_dir),
                                                    "metrics": {"gain": .91}}}}}
    score = {"valid": True, "feasible": True, "rank_key": [0, 0, 0, 2], "points": 500}
    row = {"candidate_id": "legacy", "purpose": "search", "circuit": circuit.model_dump(mode="json"),
           "simulation": simulation, "score": score}
    run = {"run_id": "b" * 32, "spec": spec.model_dump(mode="json"), "candidates": [row]}
    path = export_spice_bundle(run, tmp_path / "legacy-export", workspace=tmp_path)
    target = path.parent / "legacy"
    assert (target / "op.cir").read_bytes() == (artifacts / "op.cir").read_bytes()
    assert next((target / "corners").rglob("op.cir")).read_text() == original_corner
    assert json.loads((target / "score.json").read_text()) == score


def test_cli_export_latest_filter_and_quoted_paths(tmp_path):
    workspace = tmp_path / "workspace with 'quotes' and $cash"
    workspace.mkdir()
    run = _run(workspace)
    manager = ExperimentManager(workspace)
    manager._save(run)
    runner = CliRunner()
    result = runner.invoke(app, ["export", "-w", str(workspace), "--candidate", "fresh", "--output", str(tmp_path / "output")])
    assert result.exit_code == 0, result.output
    manifest = json.loads((tmp_path / "output" / "manifest.json").read_text())
    assert [row["candidate_id"] for row in manifest["candidates"]] == ["fresh"]
    assert "1 candidate records" in result.output and "Open in KiCad:" in result.output
    ambiguous = runner.invoke(app, ["export", "-w", str(workspace), "--candidate", ""])
    assert ambiguous.exit_code == 1 and "exactly one saved" in " ".join(ambiguous.output.split())
    assert "Ambiguous candidate prefix" in ambiguous.output
    unknown = runner.invoke(app, ["export", "-w", str(workspace), "--candidate", "no-such-candidate"])
    assert unknown.exit_code == 1 and "Unknown candidate" in unknown.output
    selected = runner.invoke(app, ["export", "-w", str(workspace), "--candidate", "candidate",
                                   "--output", str(tmp_path / "selected")])
    assert selected.exit_code == 0, selected.output
    selected_manifest = json.loads((tmp_path / "selected" / "manifest.json").read_text())
    assert selected_manifest["open_candidate_id"] == "candidate"
    assert "no fresh verification record supplied" in (tmp_path / "selected" / "README.md").read_text()


@pytest.mark.parametrize("name", ["experiments", "runs.sqlite3"])
def test_cli_rejects_symlinked_saved_records_before_loading(tmp_path, name):
    root = tmp_path / ".spiceharness"
    root.mkdir()
    outside = tmp_path / "unrelated"
    outside.mkdir()
    target = outside if name == "experiments" else outside / "private.sqlite3"
    (root / name).symlink_to(target, target_is_directory=name == "experiments")
    result = CliRunner().invoke(app, ["export", "-w", str(tmp_path)])
    assert result.exit_code == 1 and "symlinks" in result.output
    assert not list((root / "exports").glob("*"))
