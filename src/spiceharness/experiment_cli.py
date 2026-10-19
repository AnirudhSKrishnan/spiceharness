"""Terminal views and explicit no-agent benchmarks for general experiments."""
from __future__ import annotations

import json
import shlex
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .experiment_sessions import TERMINAL, ExperimentManager
from .workspace import command_hint, workspace_option

app = typer.Typer(help="Investigate supported circuit IR under frozen fixtures.", no_args_is_help=True)
Workspace = Annotated[Path, typer.Option("--workspace", "-w", callback=workspace_option)]
console = Console()


def _hint_line(label: str, command: str) -> Text:
    # Never hard-wrap a command: inserted newlines would break copy and paste.
    return Text(f"{label}: {command}", no_wrap=True, overflow="ignore")


def _number(value: Any) -> str:
    if value is None:
        return "—"
    return f"{value:.8g}" if isinstance(value, float) else str(value)


def candidate_table(run: dict[str, Any]) -> Table:
    """Show every persisted candidate, retaining the evaluator's saved ranks."""
    table = Table(title="Candidate evidence" if run["analysis_only"] else "Candidate leaderboard",
                  box=box.SIMPLE, expand=True, padding=(0, 0))
    table.add_column("Rank", max_width=4)
    table.add_column("Candidate / ID prefix", min_width=24, overflow="fold")
    for label in ("Purpose", "Valid", "Feasible", "Parts", "Max viol."):
        table.add_column(label, overflow="fold")
    ranks = {row["candidate_id"]: str(index + 1) for index, row in enumerate(run.get("leaderboard", []))}
    candidates = run.get("candidates") or []
    ranked = run.get("leaderboard") or []
    rows = ranked + [row for row in candidates if row["candidate_id"] not in ranks]
    for row in rows:
        score = row.get("score") or {}
        violations = score.get("violations") or {}
        prefix_length = next((length for length in range(8, 33)
                              if sum(other["candidate_id"].startswith(row["candidate_id"][:length])
                                     for other in candidates) == 1), 32)
        table.add_row(ranks.get(row["candidate_id"], "—"),
                      Text(f"{row['circuit'].get('name', 'candidate')}\n{row['candidate_id'][:prefix_length]}",
                           overflow="fold"), "verify" if row["purpose"] == "verification" else row["purpose"],
                      _number(score.get("valid")), _number(score.get("feasible")),
                      _number(score.get("package_count")),
                      _number(max(violations.values(), default=0)) if score and not run["analysis_only"] else "—")
    return table


def connectivity_table(circuit: dict[str, Any]) -> Table:
    table = Table(title="Circuit connectivity · validated IR", box=box.SIMPLE, expand=True)
    for label in ("Part", "Kind", "Connections", "Value / model", "Series R (Ω)"):
        table.add_column(label)
    for part in circuit.get("components", []):
        table.add_row(part["ref"], part["kind"], " · ".join(part["nodes"]),
                      str(part.get("model") or part.get("value") or "—"),
                      str(part.get("series_resistance_ohm", 0)))
    for index, (a, b) in enumerate(circuit.get("connections", []), 1):
        table.add_row(f"wire {index}", "ideal wire", f"{a} · {b}", "0 Ω", "—")
    return table


def measurement_table(run: dict[str, Any], simulation: dict[str, Any], score: dict[str, Any]) -> Table:
    table = Table(title="Probe measurements and per-check score (SI units)", box=box.SIMPLE,
                  expand=True, padding=(0, 0))
    table.add_column("Metric", min_width=22, overflow="fold")
    for label in ("Measured", "Minimum", "Maximum", "Violation", "Check"):
        table.add_column(label, overflow="fold")
    metrics = simulation.get("metrics") or {}
    bounds = run["spec"]["requirements"]
    for index, bound in enumerate(bounds):
        name = bound["metric"]
        violation = (score.get("violations") or {}).get(f"{name}:{index}")
        check = ("unmeasured" if name not in metrics else "PASS" if violation == 0 else
                 "FAIL" if violation is not None else "unscored")
        table.add_row(name, _number(metrics.get(name)), _number(bound.get("minimum")),
                      _number(bound.get("maximum")), _number(violation), check)
    bounded_metrics = {bound["metric"] for bound in bounds}
    for name, value in metrics.items():
        if name not in bounded_metrics:
            table.add_row(name, _number(value), "—", "—", "—", "analysis only")
    return table


def experiment_dashboard(run: dict[str, Any], *, compact: bool = False, height: int = 24,
                         workspace: Path | None = None) -> Group:
    result = run.get("result") or {}
    candidates = run.get("candidates") or []
    selected_id = result.get("best_candidate_id")
    selected = next((row for row in candidates if row["candidate_id"] == selected_id), None)
    selected = selected or next(iter(run.get("leaderboard") or []), None)
    selected = selected or next((row for row in reversed(candidates) if row.get("simulation")), {})
    simulation = result.get("simulation") or selected.get("simulation") or {}
    circuit = result.get("circuit") or selected.get("circuit") or run["spec"].get("baseline") or {}
    score = result.get("score") or selected.get("score") or {}
    mode = "deterministic benchmark / no agents" if run.get("offline") else "connected client / no model API calls"
    rule = ("Not scored (analysis only, no acceptance requirements)." if run["analysis_only"] else
            "Ranking: all required bounds dominate package count; lower saved rank_key wins.")
    if compact:
        def line(value: str) -> Text:
            return Text(value, no_wrap=True, overflow="ellipsis")

        rows = [line(f"SPICEHARNESS {run['run_id'][:12]} {run['status']}"), line(mode),
                line(f"Selected: {circuit.get('name', 'awaiting candidate')}"),
                line(f"{len(candidates)} candidate records · valid={score.get('valid')} "
                     f"feasible={score.get('feasible')} packages={score.get('package_count', '—')}"),
                line(rule), line("Candidate leaderboard · ID / name / feasible / packages")]
        for row in (run.get("leaderboard") or candidates)[:3]:
            scored = row.get("score") or {}
            rows.append(line(f"{row['candidate_id'][:8]} {row['circuit'].get('name', 'candidate')} "
                             f"{scored.get('feasible')} {scored.get('package_count', '—')}"))
        rows.append(line("Circuit connectivity · validated IR (excerpt)"))
        devices = [f"{part['ref']} {part['kind']}: {' · '.join(part['nodes'])} "
                   f"{part.get('model') or part.get('value') or ''}" for part in circuit.get("components", [])]
        devices += [f"wire: {a} · {b}" for a, b in circuit.get("connections", [])]
        rows.extend(line(device) for device in devices[:max(0, height - len(rows) - 3)])
        rows.append(line(f"Progress: {run.get('evaluation_count', len(candidates))} evaluations saved"))
        rows.append(_hint_line("Full results", command_hint(f"view {run['run_id']}", workspace)))
        return Group(*rows)
    details = {"sources": run["spec"]["sources"], "loads": run["spec"]["loads"],
               "probes": run["spec"]["probes"], "assumptions": run["spec"]["assumptions"]}
    return Group(Panel(Text(f"SPICEHARNESS EXPERIMENT {run['run_id']}  {run['status']}\n{mode}")),
                 candidate_table(run), Text(rule),
                 Text(f"Selected circuit: {circuit.get('name', 'candidate')} · {selected.get('candidate_id', 'baseline')}"),
                 connectivity_table(circuit), measurement_table(run, simulation, score),
                 Panel(Text(json.dumps(details, indent=2)), title="Frozen operating conditions"),
                 Panel(Text("\n".join(simulation.get("errors") or []) or
                            str(result.get("selection_policy", "Awaiting client evaluation"))),
                       title="Evidence / limitations"),
                 _hint_line("All candidate circuits and scores", command_hint(f"view {run['run_id']}", workspace)))


def watch_experiment(manager: ExperimentManager, run_id: str) -> None:
    run = manager.get_experiment(run_id)
    workspace = manager.workspace
    terminal = TERMINAL | {"interrupted"}
    if not console.is_terminal or run["status"] in terminal:
        console.print(experiment_dashboard(run, workspace=workspace), crop=False)
        return
    with Live(experiment_dashboard(run, compact=True, height=console.height, workspace=workspace), console=console,
              refresh_per_second=2, transient=True) as live:
        while run["status"] not in terminal:
            time.sleep(0.5)
            run = manager.get_experiment(run_id)
            live.update(experiment_dashboard(run, compact=True, height=console.height, workspace=workspace))
    console.print(experiment_dashboard(run, workspace=workspace), crop=False)


def _error(exc: Exception) -> None:
    console.print(Text(f"Error: {exc}", style="red"))
    raise typer.Exit(1) from exc


@app.command("show")
def show(run_id: str, workspace: Workspace = None,
         json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Inspect a saved experiment and its measurements."""
    try:
        run = ExperimentManager(workspace).get_experiment(run_id)
        if json_output:
            typer.echo(json.dumps(run, indent=2))
        else:
            console.print(experiment_dashboard(run, workspace=workspace), crop=False)
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


@app.command("candidates")
def candidates(run_id: str, workspace: Workspace = None) -> None:
    """List all saved candidate names, IDs, ranks and deterministic scores."""
    try:
        run = ExperimentManager(workspace).get_experiment(run_id)
        console.print(candidate_table(run))
        typer.echo("Not scored (analysis only, no acceptance requirements)." if run["analysis_only"] else
                   "Required electrical bounds dominate package count; lower saved rank_key wins.")
        typer.echo("Inspect a circuit: " + command_hint(
            f"experiment candidate {shlex.quote(run_id)} CANDIDATE_ID", workspace))
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


@app.command("candidate")
def candidate(run_id: str, candidate_id: str, workspace: Workspace = None) -> None:
    """Inspect a saved circuit, per-check score and artifacts by ID or unique ID prefix."""
    try:
        run = ExperimentManager(workspace).get_experiment(run_id)
        matches = [row for row in run["candidates"] if row["candidate_id"].startswith(candidate_id)]
        if not candidate_id or not matches:
            raise ValueError(f"Unknown candidate: {candidate_id}")
        if len(matches) != 1:
            raise ValueError(f"Ambiguous candidate prefix {candidate_id!r}; use a longer prefix or the full ID")
        row = matches[0]
        simulation, score = row.get("simulation") or {}, row.get("score") or {}
        typer.echo(f"{row['circuit'].get('name', 'candidate')} · {row['candidate_id']} · {row['purpose']}")
        console.print(connectivity_table(row["circuit"]))
        console.print(measurement_table(run, simulation, score))
        if run["analysis_only"]:
            typer.echo("Not scored (analysis only, no acceptance requirements).")
        typer.echo("Saved deterministic score: " + json.dumps(score))
        if simulation.get("errors"):
            typer.echo("Simulator errors: " + "; ".join(simulation["errors"]))
        directory = simulation.get("artifact_dir")
        if directory:
            root = Path(directory).expanduser()
            if not root.is_absolute():
                root = workspace.expanduser().resolve() / root
            typer.echo(f"Simulator artifacts: {root}")
            for path in sorted(root.glob("*.cir")):
                typer.echo(f"Generated netlist: {path}")
        for artifact in simulation.get("artifacts") or []:
            typer.echo(f"Artifact: {artifact}")
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


@app.command("dashboard")
def dashboard(run_id: str | None = None, workspace: Workspace = None) -> None:
    """Watch a saved experiment; defaults to the latest investigation."""
    try:
        manager = ExperimentManager(workspace)
        if run_id is None:
            runs = manager.list_experiments()
            if not runs:
                raise ValueError("No saved experiments; start_experiment through MCP or use experiment run")
            run_id = runs[0]["run_id"]
        watch_experiment(manager, run_id)
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


def _benchmark(spec_path: Path, candidate_paths: list[Path], workspace: Path,
               json_output: bool) -> None:
    from .experiments import ExperimentSpec, GeneralCircuitIR

    manager = ExperimentManager(workspace)
    spec = ExperimentSpec.model_validate_json(spec_path.read_text())
    circuits = [GeneralCircuitIR.model_validate_json(path.read_text()) for path in candidate_paths]
    if not circuits:
        raise ValueError("Provide at least one candidate circuit JSON file")
    if len(circuits) > 19:
        raise ValueError("At most 19 search candidates are allowed; one evaluation is reserved for verification")
    run = manager.start_experiment(spec, execution_mode="direct")
    for circuit in circuits:
        manager.evaluate_experiment(run["run_id"], circuit)
    run = manager.finalize_experiment(run["run_id"])
    if json_output:
        typer.echo(json.dumps(run, indent=2))
    else:
        console.print(experiment_dashboard(run, workspace=workspace), crop=False)
        typer.echo(f"Run ID: {run['run_id']} (deterministic benchmark / no agent calls)")
        typer.echo(f"Report: {run['result']['artifacts']['report']}")
    if not run["result"]["verified"] or run["result"]["feasible"] is False:
        raise typer.Exit(1)


@app.command("run")
def run(spec: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
        candidate: Annotated[Path, typer.Option("--candidate", exists=True, dir_okay=False)],
        workspace: Workspace = None,
        json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Measure a supplied circuit and freshly rerun it; no agents or model APIs."""
    try:
        _benchmark(spec, [candidate], workspace, json_output)
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


@app.command("compare")
def compare(spec: Annotated[Path, typer.Option("--spec", exists=True, dir_okay=False)],
            candidates: Annotated[Path, typer.Option("--candidates", exists=True, file_okay=False)],
            workspace: Workspace = None,
            json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Compare supplied circuit JSON files, then freshly rerun the numerical winner; no agents."""
    try:
        paths = []
        for path in sorted(candidates.glob("*.json")):
            if path.name == "spec.json" or path.name.endswith("-spec.json"):
                continue
            data = json.loads(path.read_text())
            if isinstance(data, dict) and ({"components", "connections"} & data.keys()):
                paths.append(path)
        _benchmark(spec, paths, workspace, json_output)
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


def _check_rows(run: dict[str, Any], simulation: dict[str, Any], score: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-requirement pass/fail from the evaluator's saved violations (never recomputed here)."""
    metrics = simulation.get("metrics") or {}
    rows = []
    for index, bound in enumerate(run["spec"]["requirements"]):
        name = bound["metric"]
        violation = (score.get("violations") or {}).get(f"{name}:{index}")
        rows.append({"metric": name, "value": metrics.get(name), "minimum": bound.get("minimum"),
                     "maximum": bound.get("maximum"), "violation": violation,
                     "passed": None if violation is None or name not in metrics else violation == 0})
    return rows


def _placement(original: dict[str, Any], edit_key: list[float] | None) -> dict[str, Any] | None:
    """Where the edit's saved rank_key would sit among the original run's saved search ranks."""
    if edit_key is None:
        return None
    board = original.get("leaderboard") or []
    keys = [tuple(row["score"]["rank_key"]) for row in board]
    better = sum(key < tuple(edit_key) for key in keys)
    tied = sum(key == tuple(edit_key) for key in keys)
    return {"position": better + 1, "of": len(keys) + 1, "better_than_edit": better, "ties": tied,
            "ahead_of": [{"name": row["circuit"].get("name"), "candidate_id": row["candidate_id"]}
                         for row, key in zip(board, keys, strict=True) if key < tuple(edit_key)]}


def _original_winner(original: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The original run's finalized winner (its fresh metrics) or, if unfinalized, its best saved search row."""
    result = original.get("result") or {}
    rows = {row["candidate_id"]: row for row in original["candidates"]}
    row = rows.get(result.get("best_candidate_id")) or next(iter(original.get("leaderboard") or []), None)
    metrics = (result.get("simulation") or (row or {}).get("simulation") or {}).get("metrics") or {}
    return row, metrics


@app.command("rescore-kicad")
def rescore_kicad(
    run_id: Annotated[str, typer.Argument(help="Saved experiment run ID or unique prefix (its frozen spec is reused).")],
    schematic: Annotated[Path, typer.Argument(metavar="PATH.kicad_sch", help="Edited KiCad schematic from `export`.")],
    workspace: Workspace = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    name: Annotated[str | None, typer.Option("--name", help="Candidate name; default adds _kicad_edit.")] = None,
) -> None:
    """Re-score an edited KiCad schematic as a NEW candidate in a NEW run under the SAME frozen spec.

    The edit never inherits the original score and the original run is not modified. No agents are called.
    """
    try:
        _rescore_kicad(run_id, schematic, workspace, json_output, name)
    except (ValueError, OSError, KeyError) as exc:
        _error(exc)


def _rescore_kicad(run_id: str, schematic: Path, workspace: Path, json_output: bool, name: str | None) -> None:
    from .cli import _expand_run_id
    from .kicad_import import import_kicad_bytes, read_schematic_bytes
    from .storage import utcnow

    workspace = Path(workspace).expanduser().resolve()
    run_id = _expand_run_id(workspace, run_id)
    manager = ExperimentManager(workspace)
    if not (manager.directory / run_id / "run.json").is_file():
        from .storage import RunStore

        database = workspace / ".spiceharness" / "runs.sqlite3"
        if database.is_file():
            with RunStore(database).connection() as db:
                if db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone():
                    raise ValueError(f"Run {run_id} is a template design run; rescore-kicad: only general "
                                     "experiment runs are supported")
        raise ValueError(f"Unknown experiment: {run_id}")
    original = manager.get_experiment(run_id)
    spec = original["spec"]
    path = Path(schematic).expanduser().resolve()
    data = read_schematic_bytes(path)
    imported = import_kicad_bytes(data, original_dir=path.parent, spec=spec, source=str(path))
    if not imported.ok:
        raise ValueError("The edited schematic cannot be scored; nothing was started:\n"
                         + "\n".join(f"  - {item}" for item in imported.problems()))
    title = imported.circuit["name"]
    imported.circuit["name"] = (name or f"{title}_kicad_{'edit' if imported.changes or not imported.original_found else 'reimport'}")[:200]
    started = manager.start_experiment(spec, execution_mode="direct")
    new_id = started["run_id"]
    if started["fixture_fingerprint"] != original["fixture_fingerprint"]:
        raise ValueError("internal error: the new run's frozen fixture differs from the original run")
    provenance = {"original_run_id": run_id, "schematic_path": str(path), "schematic_sha256": imported.sha256,
                  "changes": imported.changes, "warnings": imported.warnings, "imported_circuit": imported.circuit,
                  "original_circuit_json_found": imported.original_found, "imported_at": utcnow()}
    run_dir = manager.directory / new_id
    (run_dir / "kicad_import.json").write_text(json.dumps(provenance, indent=2))
    (run_dir / "imported.kicad_sch").write_bytes(data)
    record = manager._read(new_id)
    record["events"].append({"type": "kicad_rescore_import", "created_at": utcnow(),
                             "original_run_id": run_id, "schematic_path": str(path),
                             "schematic_sha256": imported.sha256})
    manager._save(record)
    row = manager.evaluate_experiment(new_id, imported.circuit)
    simulation = row.get("simulation") or {}
    if not (row.get("score") or {}).get("valid"):
        manager.cancel_experiment(new_id)
        raise ValueError(f"The edited circuit could not be measured (run {new_id}): "
                         + ("; ".join(simulation.get("errors") or []) or f"simulator status {simulation.get('status')}"))
    final = manager.finalize_experiment(new_id)
    result = final["result"]
    score, fresh = result["score"], result["simulation"]
    analysis_only = final["analysis_only"]
    checks = _check_rows(final, fresh, score)
    winner, winner_metrics = _original_winner(original)
    edit_key = (row.get("score") or {}).get("rank_key")
    placement = None if analysis_only else _placement(original, edit_key)
    metric_names = list(dict.fromkeys([*winner_metrics, *(fresh.get("metrics") or {})]))
    comparison = [{"metric": metric, "original_winner": winner_metrics.get(metric),
                   "edited": (fresh.get("metrics") or {}).get(metric),
                   "change": None if metric not in winner_metrics or metric not in (fresh.get("metrics") or {})
                   else fresh["metrics"][metric] - winner_metrics[metric]} for metric in metric_names]
    ok = bool(result["verified"]) and (analysis_only or result["feasible"] is True)
    report = {"new_run_id": new_id, "original_run_id": run_id, "analysis_only": analysis_only,
              "schematic": {"path": str(path), "sha256": imported.sha256, "copied_to": str(run_dir / "imported.kicad_sch")},
              "candidate": {"name": imported.circuit["name"], "candidate_id": row["candidate_id"]},
              "changes": imported.changes, "warnings": imported.warnings,
              "original_circuit_found": imported.original_found,
              "valid": score["valid"], "feasible": score["feasible"], "verified": result["verified"],
              "package_count": score["package_count"], "rank_key": edit_key, "fresh_rank_key": score["rank_key"],
              "checks": checks, "metrics": fresh.get("metrics"),
              "original_winner": None if winner is None else {
                  "name": winner["circuit"].get("name"), "candidate_id": winner["candidate_id"],
                  "rank_key": (winner.get("score") or {}).get("rank_key")},
              "comparison": comparison, "placement": placement,
              "note": "Not scored (analysis only, no acceptance requirements)." if analysis_only else None,
              "passed": ok}
    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        _print_rescore(final, report, workspace)
    if not ok:
        raise typer.Exit(1)


def _print_rescore(final: dict[str, Any], report: dict[str, Any], workspace: Path) -> None:
    result = final["result"]
    typer.echo(f"Imported {report['schematic']['path']}")
    typer.echo(f"  sha256 {report['schematic']['sha256']}")
    typer.echo(f"Original run {report['original_run_id']} was not modified; its frozen spec was copied.")
    if report["changes"]:
        typer.echo("Changes vs the exported circuit.json:")
        for change in report["changes"]:
            typer.echo(f"  - {change['text']}")
    else:
        typer.echo("No differences from the exported circuit.json." if report["original_circuit_found"] else
                   "No circuit.json next to the schematic, so no change list is available.")
    for warning in report["warnings"]:
        typer.echo(f"Warning: {warning}")
    typer.echo(f"New run: {report['new_run_id']}  candidate {report['candidate']['name']} "
               f"({report['candidate']['candidate_id']})")
    typer.echo(f"valid={report['valid']}  feasible={report['feasible']}  verified={report['verified']}  "
               f"packages={report['package_count']}  rank_key={report['rank_key']}")
    if report["analysis_only"]:
        typer.echo("Not scored (analysis only, no acceptance requirements).")
    checks = Table(title="Frozen requirements (fresh verification, SI units)" if report["checks"] else
                   "Fresh measurements (SI units)", box=box.SIMPLE, expand=True, padding=(0, 0))
    for label in (("Metric", "Measured", "Minimum", "Maximum", "Violation", "Check") if report["checks"]
                  else ("Metric", "Measured")):
        checks.add_column(label, overflow="fold")
    if report["checks"]:
        for row in report["checks"]:
            checks.add_row(row["metric"], _number(row["value"]), _number(row["minimum"]), _number(row["maximum"]),
                           _number(row["violation"]),
                           "unmeasured" if row["passed"] is None else "PASS" if row["passed"] else "FAIL")
    else:
        for metric, value in (report["metrics"] or {}).items():
            checks.add_row(metric, _number(value))
    console.print(checks)
    winner = report["original_winner"]
    if winner:
        typer.echo(f"Comparison with the original run's winner {winner['name']} ({winner['candidate_id'][:8]}):")
        if not report["analysis_only"]:
            typer.echo(f"  rank_key original {winner['rank_key']}  edited {report['rank_key']} "
                       "(lower is better; required bounds dominate package count)")
        bounded = {row["metric"] for row in report["checks"]}
        shown = [row for row in report["comparison"] if not bounded or row["metric"] in bounded]
        table = Table(box=box.SIMPLE, expand=True, padding=(0, 0))
        for label in ("Metric", "Original winner", "Edited", "Change"):
            table.add_column(label, overflow="fold")
        for row in shown:
            table.add_row(row["metric"], _number(row["original_winner"]), _number(row["edited"]),
                          _number(row["change"]))
        console.print(table)
        if len(shown) < len(report["comparison"]):
            typer.echo(f"({len(report['comparison']) - len(shown)} further analysis-only metrics are in --json "
                       "and the run report.)")
    else:
        typer.echo("The original run has no finalized winner or ranked candidate to compare against.")
    placement = report["placement"]
    if placement:
        tie = f" (ties with {placement['ties']})" if placement["ties"] else ""
        typer.echo(f"On the original leaderboard this edit would place {placement['position']} of "
                   f"{placement['of']}{tie}; {placement['better_than_edit']} original candidate(s) rank better.")
    if report["passed"]:
        typer.echo("RESULT: the edited circuit passes under the frozen specification (fresh verification confirmed).")
    elif report["analysis_only"]:
        typer.echo("RESULT: measured, but fresh verification did not succeed.")
    else:
        typer.echo("RESULT: the edited circuit does NOT meet every frozen requirement "
                   f"(feasible={report['feasible']}, verified={report['verified']}).")
    new_id = report["new_run_id"]
    typer.echo(f"Report: {result['artifacts']['report']}")
    for label, sub in (("Inspect", f"experiment show {new_id}"), ("Browser view", f"view {new_id}"),
                       ("Export the edit", f"export {new_id}")):
        typer.echo(f"{label}: {command_hint(sub, workspace)}")
