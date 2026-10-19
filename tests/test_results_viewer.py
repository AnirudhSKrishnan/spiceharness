"""Snapshots preserve recorded electrical truth and remain offline and inert."""
import copy
import json
import re
from pathlib import Path

import pytest

from spiceharness.results_viewer import export_results_viewer


def _payload(path: Path) -> dict:
    html = path.read_text()
    encoded = re.search(r'<script type="application/json" id="run-data">(.*?)</script>', html, re.DOTALL)
    assert encoded is not None
    return json.loads(encoded.group(1))


def _candidate(name="RC alternative", candidate_id="search", **changes):
    value = {"candidate_id": candidate_id, "ordinal": 1, "purpose": "search", "verified": False,
             "circuit": {"name": name, "components": [
                 {"ref": "R1", "kind": "R", "nodes": ["in", "out"], "value": 1000},
                 {"ref": "C1", "kind": "C", "nodes": ["out", "0"], "value": 1e-6}]},
             "simulation": {"status": "ok", "metrics": {"gain": 0.5, "dc": 0.1}},
             "score": {"valid": True, "feasible": True, "package_count": 2,
                       "rank_key": [0, 0, 0, 2], "violations": {"gain:0": 0}}}
    value.update(changes)
    return value


def _run(**changes):
    value = {"run_id": "a" * 32, "kind": "experiment", "status": "completed",
             "execution_mode": "direct", "offline": True,
             "spec": {"requirements": [{"metric": "gain", "minimum": None, "maximum": 0.75}]},
             "candidates": [_candidate()], "result": {"best_candidate_id": "search", "verified": False}}
    value.update(changes)
    return value


def test_general_snapshot_keeps_all_candidates_and_recorded_rank(tmp_path):
    run = _run()
    # Saved leaderboard wins over any attempt to rank by presentation values or points.
    second = _candidate("Smaller but infeasible", "second", ordinal=2,
                        score={"valid": True, "feasible": False, "package_count": 1,
                               "rank_key": [1, 2, 2, 1], "violations": {"gain:0": 2}})
    fresh = _candidate("RC alternative", "fresh", purpose="verification", verified=True, ordinal=3)
    run["candidates"] += [second, fresh]
    run["leaderboard"] = [run["candidates"][0], second, fresh]
    run["result"].update(verification_candidate_id="fresh", verified=True, feasible=True)
    before = copy.deepcopy(run)
    path = export_results_viewer(run, tmp_path / "nested" / "viewer.html")
    data = _payload(path)
    assert run == before
    assert len(data["candidates"]) == 3
    winner, loser, verification = data["candidates"]
    assert [winner["rank"], loser["rank"], verification["rank"]] == [1, 2, None]
    assert data["default_id"] == "fresh"
    assert verification["purpose"] == "verification"
    assert winner["metrics"][0] == {"metric": "gain", "scope": "nominal", "value": 0.5,
                                     "bound": "≤ 0.75", "violation": 0, "state": "pass", "key": "gain:0"}
    assert winner["metrics"][1]["state"] == "analysis-only"
    assert "points" not in winner["score"]
    assert loser["worst"] == 2
    assert "<svg" in winner["svg"]
    assert data["refresh_command"] == "uv run spiceharness view " + "a" * 32


def test_analysis_only_and_missing_failed_scores_stay_unknown(tmp_path):
    analyzed = _candidate(score={"valid": True, "feasible": None, "package_count": 2,
                                 "rank_key": None, "violations": {}})
    pending = _candidate("Pending design", "pending", simulation=None, score=None, status="pending")
    failed = _candidate("Failed design", "failed", simulation={"status": "failed", "errors": ["ngspice failed"]},
                        score={"valid": False, "feasible": False, "rank_key": [2, 0, 0, 2],
                               "package_count": 2, "violations": {}})
    run = _run(spec={"requirements": []}, analysis_only=True, candidates=[analyzed, pending, failed], result=None)
    data = _payload(export_results_viewer(run, tmp_path / "viewer.html"))
    assert data["analysis_only"] is True
    assert data["candidates"][0]["rank"] is None
    assert data["candidates"][0]["score"]["feasible"] is None
    assert all(row["state"] == "analysis-only" for row in data["candidates"][0]["metrics"])
    assert data["candidates"][1]["score"] == {}
    assert data["candidates"][1]["status"] == "pending"
    assert data["candidates"][1]["worst"] is None
    assert data["candidates"][2]["errors"] == ["ngspice failed"]
    assert "Not scored: no acceptance requirements were frozen." in (tmp_path / "viewer.html").read_text()


def test_nullable_minimum_maximum_and_missing_requirement_metrics(tmp_path):
    run = _run(spec={"requirements": [{"metric": "gain", "minimum": 0.1, "maximum": None},
                                     {"metric": "missing", "minimum": 1, "maximum": 2}]})
    run["candidates"][0]["score"]["violations"] = {"gain:0": 0}
    rows = _payload(export_results_viewer(run, tmp_path / "v.html"))["candidates"][0]["metrics"]
    assert rows[0]["bound"] == "≥ 0.1"
    assert rows[1]["value"] is None
    assert rows[1]["state"] == "unknown"
    assert rows[1]["violation"] is None


def test_legacy_recorded_scores_corner_constraints_and_points(tmp_path):
    row = _candidate()
    row["score"] = {"valid": True, "feasible": False, "package_count": 2,
                    "rank_key": [1, 0.4, 0.4, 2, 0], "max_violation": 0.4,
                    "points": 410.25, "violations": {"nominal.gain": 0, "nominal.thd": 0.4,
                                                     "r_hi.gain": 0}}
    row["simulation"]["analyses"] = {"corners": {"r_hi": {"status": "ok", "metrics": {"gain": 1}}}}
    run = _run(kind="template", spec={"topology": "buffer", "gain": 1, "gain_tolerance": 0.05,
                                     "max_thd": 0.03, "max_output_offset_v": None}, candidates=[row])
    data = _payload(export_results_viewer(run, tmp_path / "legacy.html"))
    candidate = data["candidates"][0]
    assert candidate["score"]["points"] == 410.25
    assert candidate["score"]["rank_key"] == row["score"]["rank_key"]
    assert any(m["scope"] == "r_hi" and m["metric"] == "gain" and m["state"] == "pass"
               for m in candidate["metrics"])
    assert not any(m["metric"] == "output_offset_v" for m in candidate["metrics"])
    assert any(m["metric"] == "gain_compression" and m["state"] == "unknown" for m in candidate["metrics"])


def test_user_strings_cannot_close_script_or_inject_html(tmp_path):
    malicious = '</script><script>alert("x")</script><img src=x onerror=alert(1)>'
    run = _run(candidates=[_candidate(malicious)])
    path = export_results_viewer(run, tmp_path / "attack.html")
    html = path.read_text()
    assert malicious not in html
    assert _payload(path)["candidates"][0]["name"] == malicious
    assert "connect-src 'none'" in html
    assert '<script src=' not in html
    assert '<link ' not in html
    assert '.textContent=' in html
    assert 'drawing.innerHTML=c.svg' in html  # Only the trusted escaped renderer SVG uses HTML insertion.


def test_artifact_links_and_decks_stay_inside_workspace_artifact_tree(tmp_path):
    root = tmp_path / ".spiceharness" / "experiments" / "saved"
    root.mkdir(parents=True)
    (root / "tran.cir").write_text('* generated deck\nR1 in out 1000\n.end\n')
    (root / "op.log").write_text('ok')
    (root / "vendor.lib").write_text('private model')
    secret = tmp_path / ".env"
    secret.write_text('API_KEY=do-not-read')
    (root / "escaped.log").symlink_to(secret)
    row = _candidate()
    row["simulation"].update(artifact_dir=str(root), artifacts=["tran.cir", "op.log", "vendor.lib",
                                                              "escaped.log", "../../../.env"])
    path = export_results_viewer(_run(candidates=[row]), tmp_path / "v.html", workspace=tmp_path)
    artifacts = _payload(path)["candidates"][0]["artifacts"]
    assert [item["name"] for item in artifacts] == ["op.log", "tran.cir"]
    assert artifacts[1]["deck"].startswith('* generated deck')
    assert artifacts[1]["href"] == (root / "tran.cir").as_uri()
    assert 'do-not-read' not in path.read_text()
    outside = _candidate(simulation={"artifact_dir": str(tmp_path), "artifacts": [".env"]})
    data = _payload(export_results_viewer(_run(candidates=[outside]), tmp_path / "outside.html", workspace=tmp_path))
    assert data["candidates"][0]["artifacts"] == []
    data = _payload(export_results_viewer(_run(candidates=[row]), tmp_path / "disabled.html"))
    assert data["candidates"][0]["artifacts"] == []


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_historical_nonfinite_measurements_render_as_unknown(tmp_path, value):
    row = _candidate()
    row["simulation"]["metrics"]["gain"] = value
    data = _payload(export_results_viewer(_run(candidates=[row]), tmp_path / "v.html"))
    assert data["candidates"][0]["metrics"][0]["value"] is None


def test_kicad_export_latest_safe_files_and_missing_status(tmp_path):
    import os

    root = tmp_path / '.spiceharness' / 'exports'
    run = _run()

    def bundle(suffix, stamp, **changes):
        directory = root / (run['run_id'] + suffix)
        candidate = directory / 'search'
        candidate.mkdir(parents=True)
        for extension in ('kicad_sch', 'kicad_pro'):
            (candidate / ('candidate.' + extension)).write_text('native')
        entry = {'candidate_id': 'search', 'schematic': 'search/candidate.kicad_sch',
                 'status': 'ok', 'native_runnable': True, 'missing': []}
        entry.update(changes)
        manifest = directory / 'manifest.json'
        manifest.write_text(json.dumps({'run_id': run['run_id'], 'candidates': [entry]}))
        os.utime(manifest, (stamp, stamp))
        return directory

    bundle('', 100)
    latest = bundle('-2', 200, native_runnable=False, missing=['missing original model'])
    bundle('-wrong', 300)
    data = _payload(export_results_viewer(run, tmp_path / 'view.html', workspace=tmp_path))
    exported = data['candidates'][0]['kicad']
    assert len(exported['files']) == 2
    assert all(str(latest) in item['path'] for item in exported['files'])
    assert exported['native_runnable'] is False
    assert exported['missing'] == ['missing original model']
    assert data['export_command'].startswith('uv run spiceharness export ')


@pytest.mark.parametrize('unsafe', ['../outside.kicad_sch', '/tmp/outside.kicad_sch',
                                    'https://example.org/candidate.kicad_sch', 'linked.kicad_sch'])
def test_kicad_export_rejects_unsafe_file_links(tmp_path, unsafe):
    directory = tmp_path / '.spiceharness' / 'exports' / ('a' * 32)
    directory.mkdir(parents=True)
    outside = tmp_path / 'outside.kicad_sch'
    outside.write_text('native')
    (directory / 'linked.kicad_sch').symlink_to(outside)
    (directory / 'manifest.json').write_text(json.dumps({'run_id': 'a' * 32, 'candidates': [
        {'candidate_id': 'search', 'schematic': unsafe, 'native_runnable': True}]}))
    data = _payload(export_results_viewer(_run(), tmp_path / 'view.html', workspace=tmp_path))
    assert data['candidates'][0]['kicad']['files'] == []


def test_kicad_export_manifest_bounds_and_run_identity(tmp_path):
    directory = tmp_path / '.spiceharness' / 'exports' / ('a' * 32)
    directory.mkdir(parents=True)
    manifest = directory / 'manifest.json'
    for contents in (' ' * (2 * 1024 * 1024 + 1), '{', json.dumps({'run_id': 'b' * 32, 'candidates': []})):
        manifest.write_text(contents)
        data = _payload(export_results_viewer(_run(), tmp_path / 'view.html', workspace=tmp_path))
        assert data['candidates'][0]['kicad'] is None


def test_kicad_export_rejects_symlinked_manifest_and_directory(tmp_path):
    root = tmp_path / '.spiceharness' / 'exports'
    root.mkdir(parents=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    manifest = outside / 'manifest.json'
    manifest.write_text(json.dumps({'run_id': 'a' * 32, 'candidates': []}))
    link = root / ('a' * 32)
    link.symlink_to(outside, target_is_directory=True)
    from spiceharness.results_viewer import _kicad_exports
    assert _kicad_exports(_run(), tmp_path) == {}
    link.unlink()
    link.mkdir()
    (link / 'manifest.json').symlink_to(manifest)
    assert _kicad_exports(_run(), tmp_path) == {}
