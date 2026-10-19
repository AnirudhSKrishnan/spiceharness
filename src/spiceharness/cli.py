"""Terminal interface to the same persisted runs exposed by MCP."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import webbrowser
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .experiment_cli import app as experiment_app
from .workspace import command_hint, workspace_option

app = typer.Typer(help="Design analog circuits and verify them with ngspice.", no_args_is_help=True)
knowledge_app = typer.Typer(help="Import and search local circuit knowledge.", no_args_is_help=True)
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(experiment_app, name="experiment")
console = Console()
TERMINAL_STATUSES = {"completed", "complete", "failed", "cancelled", "canceled", "budget_exhausted", "interrupted"}
Workspace = Annotated[Path, typer.Option(
    "--workspace", "-w", callback=workspace_option,
    help="Project directory containing .spiceharness. Default: $SPICEHARNESS_WORKSPACE, else the current "
         "directory if it has .spiceharness/, else the installed default workspace, else the current directory.")]
Config = Annotated[Path | None, typer.Option("--config", help="Role and provider configuration TOML.")]


def _manager(workspace: Path, config: Path | None = None):
    from .workflow import DesignManager

    if config is None:
        return DesignManager(workspace=workspace)
    from .providers import Settings

    return DesignManager(workspace=workspace, settings=Settings.load(config))


def _saved_manager(workspace: Path, run_id: str):
    manager = _manager(workspace)
    record = manager.storage.get_run(run_id, include_details=False)
    if record["options"].get("execution_mode") == "client":
        from .client_sessions import ClientDesignManager

        return ClientDesignManager(workspace=workspace, settings=manager.settings)
    return manager


def _read_spec(path: Path):
    from .domain import DesignSpec

    return DesignSpec.model_validate_json(path.read_text())


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def _handle_error(error: Exception) -> None:
    # Validation errors contain the user input, so keep secrets out of provider diagnostics.
    from .providers import safe_error

    # KeyError's str() is its repr ("'Unknown run: X'"); show the plain message instead.
    detail = error.args[0] if isinstance(error, KeyError) and error.args else error
    console.print(Text(f"Error: {safe_error(detail)}", style="red"))
    raise typer.Exit(1) from error


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _fresh_check(candidate: dict[str, Any]) -> str:
    simulation = candidate.get("simulation") or {}
    return "passed" if candidate.get("verified") and simulation.get("status") == "ok" else "—"


def _leaderboard_table(candidates: list[dict[str, Any]], limit: int | None = 10, *,
                       names: bool = False) -> Table:
    leaderboard = Table(title="Candidate leaderboard", box=box.SIMPLE, expand=True)
    for name in ["Candidate", *(["Name"] if names else []), "Topology", "Points", "Max violation ↓",
                 "Fresh check", "Feasible"]:
        leaderboard.add_column(name, overflow="fold" if name == "Name" else "ellipsis")
    for candidate in candidates[:limit]:
        score = candidate.get("score") or {}
        candidate_circuit = candidate.get("circuit") or {}
        leaderboard.add_row(str(candidate.get("candidate_id", ""))[:8],
                            *([str(candidate_circuit.get("name", ""))] if names else []),
                            str(candidate_circuit.get("topology", candidate.get("family", ""))),
                            _fmt(score.get("points")),
                            _fmt(score.get("max_violation", score.get("total"))),
                            _fresh_check(candidate),
                            _fmt(score.get("passed", score.get("feasible"))))
    return leaderboard


def run_dashboard(run: dict[str, Any], *, compact: bool = False, height: int | None = None) -> Group:
    """Render specs, measured results and schematic from the persisted circuit IR."""
    status = run.get("status", "unknown")
    style = "green" if status in {"completed", "complete"} else "red" if status == "failed" else "cyan"
    header = Text.assemble(("SPICEHARNESS  ", "bold cyan"), (str(run.get("run_id", "")), "bold"),
                           (f"  {status}", style),
                           ("  deterministic / no agents" if run.get("offline") else
                            "  connected client" if run.get("options", {}).get("execution_mode") == "client"
                            else "  agent design", "dim"))
    result = run.get("result") or {}
    candidates = run.get("leaderboard") or run.get("candidates") or []
    best_id = result.get("best_candidate_id")
    best = next((c for c in candidates if c.get("candidate_id") == best_id), candidates[0] if candidates else {})
    circuit = result.get("circuit") or result.get("best_circuit") or best.get("circuit")
    simulation = result.get("simulation") or result.get("best_simulation") or best.get("simulation") or {}
    measured = simulation.get("metrics") or simulation.get("measurements") or {}
    spec = run.get("spec") or {}
    specs = Table(title="Specification and measured performance (SI units)", box=box.SIMPLE, expand=True)
    specs.add_column("Metric / constraint")
    specs.add_column("Requested")
    specs.add_column("Measured")
    score = best.get("score") or result.get("score") or {}
    comparisons = [
        ("gain", "gain", "", f" ±{spec.get('gain_tolerance', .05):.0%}"),
        ("cutoff_hz", "cutoff_hz", "", f" ±{spec.get('cutoff_tolerance', .1):.0%}"),
        ("max_thd", "thd", "≤ ", ""),
        ("max_supply_current_a", "supply_current_a", "≤ ", ""),
        ("max_output_offset_v", "output_offset_v", "≤ ", ""),
        ("min_output_swing_v", "output_swing_v", "≥ ", ""),
        ("max_parts", "package_count", "≤ ", ""),
    ]
    for requested, metric, prefix, suffix in comparisons:
        target = spec.get(requested)
        measured_value = measured.get(metric, score.get(metric))
        if target is not None or measured_value is not None:
            specs.add_row(metric, prefix + _fmt(target) + suffix if target is not None else "—",
                          _fmt(measured_value))
    conditions = {key: spec[key] for key in ("supply_positive_v", "supply_negative_v", "input_amplitude_v",
                  "input_frequency_hz", "load_ohm", "source_resistance_ohm", "temperature_c", "corner_policy")
                  if key in spec}
    leaderboard = _leaderboard_table(candidates)
    schematic = "Waiting for the first circuit candidate…"
    if circuit:
        from .domain import Circuit
        from .render import render_schematic

        try:
            schematic = render_schematic(Circuit.model_validate(circuit))
        except (ValidationError, ValueError, TypeError) as exc:
            schematic = f"Schematic unavailable: {exc}"
    events = run.get("events") or []
    messages = [str(e.get("message", e.get("type", e.get("event", e.get("kind", ""))))) for e in events[-5:]]
    usage = run.get("usage") or []
    if usage:
        messages.append(f"Provider requests: {len(usage)}")
    if run.get("error"):
        from .providers import safe_error
        messages.append(safe_error(run["error"]))
    progress = Text("\n".join(messages) or "Run initialized. Results are saved after every candidate.")
    if compact:
        return _compact_dashboard(run, circuit, measured, candidates, messages, height or console.height)
    return Group(Panel(header), leaderboard, specs, Panel(Text(schematic), title="Circuit schematic · shared IR"),
                 Panel(Text(_json(conditions)), title="Operating conditions"), Panel(progress, title="Progress"))


def _compact_dashboard(run: dict[str, Any], circuit: dict[str, Any] | None, measured: dict[str, Any],
                       candidates: list[dict[str, Any]], messages: list[str], height: int) -> Group:
    """Allocate physical rows so the circuit remains visible in a small live terminal."""
    from .domain import Circuit
    from .render import format_value

    def line(value: str, style: str = "") -> Text:
        return Text(value, style=style, no_wrap=True, overflow="ellipsis")

    spec = run.get("spec") or {}
    mode = ("offline" if run.get("offline") else
            "client" if run.get("options", {}).get("execution_mode") == "client" else "agents")
    header = line(f"SPICEHARNESS {str(run.get('run_id', ''))[:12]}  {run.get('status', 'unknown')}  {mode}", "bold cyan")
    comparisons = [("gain", "gain", ""), ("cutoff_hz", "cutoff_hz", ""), ("max_thd", "thd", "≤"),
                   ("max_supply_current_a", "supply_current_a", "≤"),
                   ("max_output_offset_v", "output_offset_v", "≤"),
                   ("min_output_swing_v", "output_swing_v", "≥"), ("max_parts", "package_count", "≤")]
    score = (candidates[0].get("score") or {}) if candidates else {}
    metrics = [f"{metric}: {prefix}{_fmt(spec.get(target))} → {_fmt(measured.get(metric, score.get(metric)))}"
               for target, metric, prefix in comparisons if spec.get(target) is not None or metric in measured]
    metric_lines = [" | ".join(metrics[i:i + 2]) for i in range(0, len(metrics), 2)]
    metric_limit = max(0, (height - 10) // 3)
    metric_lines = metric_lines[:metric_limit]
    if metric_lines:
        metric_lines[0] = "Spec → measured: " + metric_lines[0]
    top_count = min(3 if height >= 16 else 1 if height >= 12 else 0, len(candidates))
    leaderboard_lines = [f"Candidate leaderboard · {len(candidates)} evaluated"]
    if top_count:
        leaderboard_lines.append("ID        Points     Violation    Fresh check  Feasible")
        for candidate in candidates[:top_count]:
            scored = candidate.get("score") or {}
            leaderboard_lines.append(
                f"{str(candidate.get('candidate_id', ''))[:8]:8}  {_fmt(scored.get('points')):9}  "
                f"{_fmt(scored.get('max_violation')):11}  "
                f"{_fresh_check(candidate):11}  {_fmt(scored.get('feasible'))}"
            )
    # Header and progress each consume one row; reserve the rest before the circuit.
    available = max(3, height - 2 - len(metric_lines) - len(leaderboard_lines))
    schematic_height = min(max(3, height // 2), available)
    devices = ["Waiting for the first circuit candidate…"]
    if circuit:
        try:
            parsed = Circuit.model_validate(circuit)
            devices = [f"{parsed.topology} · {parsed.input_node} → {parsed.output_node}"]
            for part in parsed.components:
                value = part.model or format_value(part.value, part.kind)
                if part.kind == "OPAMP":
                    plus, minus, positive, negative, output = part.nodes
                    glyph = f"(+) {plus}  (−) {minus}  ▷ {output}  rails {positive}/{negative}"
                elif part.kind == "BJT":
                    collector, base, emitter = part.nodes
                    glyph = f"B {base} ─[Q]─ C {collector}  E {emitter}"
                else:
                    symbol = {"R": "[R]", "C": "┤├", "L": "∿∿", "DIODE": "▷│"}[part.kind]
                    glyph = f"{part.nodes[0]} ─{symbol}─ {part.nodes[1]}"
                devices.append(f"{part.ref} {value}: {glyph}")
        except (ValidationError, ValueError, TypeError) as exc:
            devices = [f"Schematic unavailable: {exc}"]
    inner_rows = schematic_height - 2
    if len(devices) > inner_rows:
        hidden = len(devices) - inner_rows + 1
        devices = devices[:inner_rows - 1] + [f"… {hidden} more devices; show this run for the full circuit"]
    schematic = Panel(line("\n".join(devices)), title="Circuit schematic · shared IR", height=schematic_height,
                      padding=(0, 1))
    progress = messages[-1] if messages else "Initializing; every candidate is saved."
    return Group(header, schematic, *(line(value) for value in metric_lines),
                 *(line(value, "bold" if index == 0 else "") for index, value in enumerate(leaderboard_lines)),
                 line("Progress: " + progress, "dim"))


def _watch(manager: Any, run_id: str, json_output: bool = False) -> dict[str, Any]:
    run = manager.get_run(run_id)
    if json_output or not console.is_terminal:
        while run.get("status") not in TERMINAL_STATUSES:
            time.sleep(0.2)
            run = manager.get_run(run_id)
        if json_output:
            typer.echo(_json(run))
        else:
            console.print(run_dashboard(run))
        return run
    if run.get("status") in TERMINAL_STATUSES:
        console.print(run_dashboard(run))
        return run
    with Live(run_dashboard(run, compact=True, height=console.height), console=console,
              refresh_per_second=4, transient=True) as live:
        while run.get("status") not in TERMINAL_STATUSES:
            time.sleep(0.25)
            run = manager.get_run(run_id)
            live.update(run_dashboard(run, compact=True, height=console.height))
    console.print(run_dashboard(run))
    return run


@app.command()
def design(
    spec: Annotated[Path | None, typer.Option("--spec", exists=True, dir_okay=False, help="DesignSpec JSON file.")] = None,
    request: Annotated[str | None, typer.Option("--request", help="Natural-language requirements; requires provider access.")] = None,
    workspace: Workspace = None,
    config: Config = None,
    offline: Annotated[bool, typer.Option("--offline", help="Explicit deterministic benchmark; no provider calls.")] = False,
    detach: Annotated[bool, typer.Option("--detach", help="Return the persistent background run ID immediately.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print machine-readable run JSON.")] = False,
) -> None:
    """Start a design run; watch its candidates, schematic and measurements."""
    try:
        if (spec is None) == (request is None):
            raise ValueError("Provide exactly one of --spec or --request")
        if request is not None and offline:
            raise ValueError("--request requires provider access; use --spec JSON with --offline")
        if spec is not None:
            parsed_spec = _read_spec(spec)
            manager = _manager(workspace, config)
        else:
            manager = _manager(workspace, config)
            normalized = manager.normalize_request(request)
            if normalized.get("missing_fields"):
                typer.echo(_json(normalized))
                raise typer.Exit(2)
            from .domain import DesignSpec
            parsed_spec = DesignSpec.model_validate(normalized["spec"])
        run = manager.start_design(parsed_spec, offline=offline)
        if detach:
            typer.echo(_json(run) if json_output else run["run_id"])
            return
        run = _watch(manager, run["run_id"], json_output)
        if run.get("status") in {"failed", "interrupted"}:
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        _handle_error(exc)


@app.command()
def show(run_id: Annotated[str | None, typer.Argument(help="Saved run ID or unique prefix; default: latest.")] = None,
         workspace: Workspace = None,
         json_output: Annotated[bool, typer.Option("--json")] = False,
         watch: Annotated[bool, typer.Option("--watch", help="Refresh until the run stops.")] = False) -> None:
    """Inspect a saved run (design or experiment), including its verification and leaderboard."""
    try:
        manager, selected_id, experiment = _select_saved_run(workspace, run_id)
        if experiment:
            from .experiment_cli import experiment_dashboard, watch_experiment

            if watch and not json_output:
                watch_experiment(manager, selected_id)
                return
            run = manager.get_experiment(selected_id)
            if json_output:
                typer.echo(_json(run))
            else:
                console.print(experiment_dashboard(run, workspace=workspace), crop=False)
        elif watch:
            _watch(manager, selected_id, json_output)
        else:
            run = manager.get_run(selected_id)
            if json_output:
                typer.echo(_json(run))
            else:
                console.print(run_dashboard(run))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


def _expand_run_id(workspace: Path, run_id: str) -> str:
    """Complete a unique hexadecimal prefix against both run kinds; ambiguity is an error."""
    if re.fullmatch(r"[0-9a-f]{32}", run_id):
        return run_id
    if re.fullmatch(r"[0-9a-f]{4,32}", run_id) is None:
        raise ValueError("run_id must be a 32-character hexadecimal run identifier or a unique prefix "
                         "of at least 4 characters")
    root = workspace / ".spiceharness"
    matches: set[str] = set()
    if (root / "experiments").is_dir():
        matches |= {path.parent.name for path in (root / "experiments").glob(f"{run_id}*/run.json")}
    if (root / "runs.sqlite3").is_file():
        from .storage import RunStore

        with RunStore(root / "runs.sqlite3").connection() as db:
            matches |= {row[0] for row in db.execute("SELECT run_id FROM runs WHERE run_id LIKE ?",
                                                     (run_id + "%",))}
    if len(matches) > 1:
        listed = ", ".join(sorted(item[:12] for item in matches)[:6])
        raise ValueError(f"Ambiguous run prefix {run_id!r} matches {len(matches)} runs ({listed}); "
                         "use more characters")
    if not matches:
        raise ValueError(f"No saved run in {workspace} matches prefix {run_id!r}")
    return next(iter(matches))


def _select_saved_run(workspace: Path, run_id: str | None) -> tuple[Any, str, bool]:
    """Use one routing rule for terminal dashboards and HTML result snapshots."""
    from .experiment_sessions import ExperimentManager

    if run_id is not None and re.fullmatch(r"[0-9a-f]{4,32}", run_id) is None:
        raise ValueError("run_id must be a 32-character hexadecimal run identifier or a unique prefix "
                         "of at least 4 characters")
    workspace = workspace.expanduser().resolve()
    root = workspace / ".spiceharness"

    def regular_path(path: Path) -> None:
        for ancestor in (path, *path.parents):
            if ancestor == workspace:
                break
            if ancestor.is_symlink():
                raise ValueError("Saved run paths cannot contain symlinks")

    regular_path(root)
    for name in ("runs.sqlite3", "knowledge.sqlite3", "runs.sqlite3-wal", "runs.sqlite3-shm",
                 "runs.sqlite3-journal", "knowledge.sqlite3-wal", "knowledge.sqlite3-shm"):
        regular_path(root / name)
    regular_path(root / "experiments")
    regular_path(root / "runs")
    if run_id is not None:
        run_id = _expand_run_id(workspace, run_id)
    records = [root / "experiments" / run_id] if run_id else list((root / "experiments").glob("*"))
    for directory in records:
        regular_path(directory)
        for name in ("run.json", "session.lock", "cancel.requested"):
            regular_path(directory / name)
    experiments = ExperimentManager(workspace)
    if run_id is None:
        runs = _manager(workspace).storage.list_runs(limit=1)
        investigations = experiments.list_experiments()
        if investigations and (not runs or investigations[0]["created_at"] > runs[0]["created_at"]):
            return experiments, investigations[0]["run_id"], True
        if not runs:
            raise ValueError(
                f"No saved runs in {workspace.expanduser().resolve()}. "
                "Start a design through MCP or the design command, or select another --workspace."
            )
        run_id = runs[0]["run_id"]
    if (experiments.directory / run_id / "run.json").is_file():
        return experiments, run_id, True
    return _saved_manager(workspace, run_id), run_id, False


def _load_saved_run(workspace: Path, run_id: str | None) -> tuple[dict[str, Any], str, bool]:
    """Read the selected legacy/general snapshot through the shared safe router."""
    manager, selected_id, experiment = _select_saved_run(workspace, run_id)
    root = workspace.expanduser().resolve() / ".spiceharness"
    directory = root / ("experiments" if experiment else "runs") / selected_id
    for path in (directory, directory / "report", directory / "report" / "results.html"):
        if path.is_symlink():
            raise ValueError("Saved result paths cannot contain symlinks")
    run = manager.get_experiment(selected_id) if experiment else manager.get_run(selected_id)
    return run, selected_id, experiment


@app.command()
def dashboard(
    run_id: Annotated[str | None, typer.Argument(help="Saved run ID; defaults to the latest run.")] = None,
    workspace: Workspace = None,
) -> None:
    """Watch a saved run in the terminal, or inspect its final results."""
    try:
        from .experiment_cli import watch_experiment

        manager, selected_id, experiment = _select_saved_run(workspace, run_id)
        if experiment:
            watch_experiment(manager, selected_id)
        else:
            _watch(manager, selected_id)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def view(
    run_id: Annotated[str | None, typer.Argument(help="Saved run ID; defaults to the latest run.")] = None,
    workspace: Workspace = None,
    no_open: Annotated[bool, typer.Option("--no-open", help="Export HTML without launching a browser.")] = False,
) -> None:
    """Open all saved candidate circuits, scores and evidence in a local HTML viewer."""
    try:
        from .results_viewer import export_results_viewer

        workspace = workspace.expanduser().resolve()
        run, selected_id, experiment = _load_saved_run(workspace, run_id)
        directory = workspace / ".spiceharness" / ("experiments" if experiment else "runs") / selected_id
        path = export_results_viewer(run, directory / "report" / "results.html", workspace=workspace).resolve()
        typer.echo(f"Run: {selected_id} · {len(run.get('candidates') or [])} candidate records")
        typer.echo(f"Results: {path}")
        typer.echo(f"URL: {path.as_uri()}")
        if not no_open:
            try:
                opened = webbrowser.open(path.as_uri())
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                typer.echo("Browser launch failed. Open the URL above, or use --no-open to export only.", err=True)
                raise typer.Exit(1)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def export(
    run_id: Annotated[str | None, typer.Argument(help="Saved run ID; defaults to the latest run.")] = None,
    workspace: Workspace = None,
    output: Annotated[Path | None, typer.Option("--output", help="Portable bundle directory.")] = None,
    candidate: Annotated[str | None, typer.Option("--candidate", help="One saved candidate ID or unique prefix.")] = None,
) -> None:
    """Export all saved candidates as editable KiCad schematics and original ngspice evidence."""
    try:
        from .spice_bundle import export_spice_bundle

        workspace = workspace.expanduser().resolve()
        run, selected_id, _ = _load_saved_run(workspace, run_id)
        if candidate is not None:
            matches = [row for row in run.get("candidates", []) if row.get("candidate_id", "").startswith(candidate)]
            exact = [row for row in matches if row["candidate_id"] == candidate]
            matches = exact or matches
            if len(matches) != 1:
                reason = "Ambiguous candidate prefix" if len(matches) > 1 else "Unknown candidate"
                raise ValueError(f"{reason} {candidate!r}: candidate must match exactly one saved candidate "
                                 "ID or unique prefix")
            run = {**run, "candidates": matches}
        path = export_spice_bundle(run, output or workspace / ".spiceharness" / "exports" / selected_id,
                                   workspace=workspace)
        manifest = json.loads(path.read_text())
        typer.echo(f"Run: {selected_id} · {len(manifest['candidates'])} candidate records")
        typer.echo(f"Portable circuits: {path.parent}")
        typer.echo(f"Manifest: {path}")
        if manifest["open_schematic"]:
            typer.echo(f"Open in KiCad: {path.parent / manifest['open_schematic']}")
        for row in manifest["candidates"]:
            if row["missing"]:
                typer.echo(f"Missing evidence for {row['candidate_id']}: {'; '.join(row['missing'])}")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


def _local_time(value: str | None) -> str:
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value)).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(value or "—")


def _yes_no(value: Any) -> str:
    return "—" if value is None else "yes" if value else "no"


def _run_rows(workspace: Path, limit: int) -> list[dict[str, Any]]:
    """Summarise saved design and experiment runs from stored data only (no simulation)."""
    from .experiment_sessions import ExperimentManager

    root = workspace / ".spiceharness"
    rows: list[dict[str, Any]] = []
    if (root / "runs.sqlite3").is_file():
        from .storage import RunStore

        store = RunStore(root / "runs.sqlite3")  # not DesignManager: it would create other state files
        for run in store.list_runs(limit=limit):
            result = run.get("result") or {}
            candidates = [row for row in store.list_candidates(run["run_id"]) if row.get("purpose") != "verification"]
            best = result.get("best") or {}
            rows.append({"run_id": run["run_id"], "kind": "design", "status": run["status"],
                         "created_at": run["created_at"], "candidates": len(candidates),
                         "winner": (result.get("circuit") or best.get("circuit") or {}).get("name"),
                         "verified": result.get("verified"), "feasible": result.get("feasible")})
    if (root / "experiments").is_dir():
        for run in ExperimentManager(workspace).list_experiments()[:limit]:
            result = run.get("result") or {}
            analysis = bool(run.get("analysis_only"))
            candidates = [row for row in run.get("candidates", []) if row.get("purpose") != "verification"]
            rows.append({"run_id": run["run_id"], "kind": "experiment", "status": run["status"],
                         "created_at": run["created_at"], "candidates": len(candidates),
                         "winner": None if analysis else (result.get("circuit") or {}).get("name"),
                         "verified": result.get("verified"),
                         "feasible": None if analysis else result.get("feasible"),
                         "analysis_only": analysis})
    rows.sort(key=lambda row: row["created_at"], reverse=True)
    return rows[:limit]


@app.command()
def runs(
    workspace: Workspace = None,
    limit: Annotated[int, typer.Option("--limit", "-n", min=1, max=500, help="Most recent runs to list.")] = 10,
    json_output: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """List recent saved design and experiment runs (read-only; no simulation, no agents)."""
    try:
        workspace = workspace.expanduser().resolve()
        rows = _run_rows(workspace, limit)
        if json_output:
            typer.echo(_json({"workspace": str(workspace), "runs": rows}))
            return
        if not rows:
            typer.echo(f"No saved runs in {workspace}.")
            return
        table = Table(title=f"Saved runs in {workspace}", box=box.SIMPLE, pad_edge=False)
        for name in ("Run", "Kind", "Status", "Created", "Cand.", "Winner", "Verified", "Feasible"):
            table.add_column(name, no_wrap=name != "Winner", overflow="fold" if name == "Winner" else "ellipsis")
        for row in rows:
            table.add_row(row["run_id"][:8], row["kind"], row["status"], _local_time(row["created_at"]),
                          str(row["candidates"]), row["winner"] or "—", _yes_no(row["verified"]),
                          "analysis" if row.get("analysis_only") else _yes_no(row["feasible"]))
        # Piped output has no terminal width; keep rows on single lines.
        Console(width=max(console.width, 130)).print(table) if not console.is_terminal else console.print(table)
        first = rows[0]["run_id"][:8]
        typer.echo("Next: " + " · ".join(command_hint(f"{name} {first}", workspace)
                                        for name in ("dashboard", "candidates", "view")))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def candidates(
    run_id: Annotated[str | None, typer.Argument(help="Saved run ID or unique prefix; default: latest.")] = None,
    workspace: Workspace = None,
) -> None:
    """Show the candidate leaderboard of a saved design or experiment run."""
    try:
        from .experiment_cli import candidate_table

        manager, selected_id, experiment = _select_saved_run(workspace, run_id)
        if experiment:
            run = manager.get_experiment(selected_id)
            console.print(candidate_table(run))
            typer.echo("Not scored (analysis only, no acceptance requirements)." if run["analysis_only"] else
                       "Required electrical bounds dominate package count; lower saved rank_key wins.")
        else:
            run = manager.get_run(selected_id)
            ranked = run.get("leaderboard") or []
            ranked_ids = {row["candidate_id"] for row in ranked}
            rows = ranked + [row for row in run.get("candidates", []) if row["candidate_id"] not in ranked_ids]
            console.print(_leaderboard_table(rows, None, names=True))
        typer.echo(f"Run {selected_id}")
        typer.echo("Inspect a circuit: " + command_hint(f"candidate {selected_id[:8]} CANDIDATE_ID", workspace))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def candidate(
    run_id: Annotated[str, typer.Argument(help="Saved run ID or unique prefix.")],
    candidate_id: Annotated[str, typer.Argument(help="Candidate ID or unique prefix.")],
    workspace: Workspace = None,
) -> None:
    """Inspect one saved circuit with its score, measurements and artifacts."""
    try:
        manager, selected_id, experiment = _select_saved_run(workspace, run_id)
        if experiment:
            from .experiment_cli import candidate as experiment_candidate

            experiment_candidate(selected_id, candidate_id, workspace)
            return
        run = manager.get_run(selected_id)
        matches = [row for row in run["candidates"] if candidate_id and row["candidate_id"].startswith(candidate_id)]
        if not matches:
            raise ValueError(f"Unknown candidate: {candidate_id}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous candidate prefix {candidate_id!r}; use a longer prefix or the full ID")
        row = matches[0]
        circuit, simulation, score = row.get("circuit") or {}, row.get("simulation") or {}, row.get("score") or {}
        typer.echo(f"{circuit.get('name', 'candidate')} · {row['candidate_id']} · {row.get('purpose', 'search')}"
                   f" · fresh check {_fresh_check(row)}")
        try:
            from .domain import Circuit
            from .render import render_schematic

            console.print(Panel(Text(render_schematic(Circuit.model_validate(circuit))),
                                title="Circuit schematic · shared IR"))
        except (ValidationError, ValueError, TypeError) as exc:
            typer.echo(f"Schematic unavailable: {exc}")
        metrics = Table(title="Measured performance (SI units)", box=box.SIMPLE)
        metrics.add_column("Metric")
        metrics.add_column("Measured")
        for name, value in (simulation.get("metrics") or {}).items():
            metrics.add_row(name, _fmt(value))
        console.print(metrics)
        typer.echo("Saved deterministic score: " + json.dumps(
            {key: score[key] for key in ("valid", "feasible", "points", "max_violation", "package_count",
                                         "violations") if key in score}, default=str))
        if simulation.get("errors"):
            typer.echo("Simulator errors: " + "; ".join(map(str, simulation["errors"])))
        directory = simulation.get("artifact_dir")
        if directory:
            root = Path(directory).expanduser()
            if not root.is_absolute():
                root = workspace.expanduser().resolve() / root
            typer.echo(f"Simulator artifacts: {root}")
            for path in sorted(root.glob("*.cir")):
                typer.echo(f"Generated netlist: {path}")
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def resume(run_id: str, workspace: Workspace = None, config: Config = None,
           detach: Annotated[bool, typer.Option("--detach")] = False,
           json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Resume an interrupted run using its persisted checkpoints."""
    try:
        from .client_sessions import ClientDesignManager

        saved = _saved_manager(workspace, run_id)
        if isinstance(saved, ClientDesignManager):
            run = saved.get_run(run_id)
            if json_output:
                typer.echo(_json(run))
            else:
                console.print(run_dashboard(run))
                typer.echo("Continue this client-driven run from Codex or Claude Code using its run_id.")
            return
        manager = _manager(workspace, config)
        run = manager.resume_run(run_id)
        if detach:
            typer.echo(_json(run) if json_output else run["run_id"])
        else:
            run = _watch(manager, run_id, json_output)
            if run.get("status") in {"failed", "interrupted"}:
                raise typer.Exit(1)
    except typer.Exit:
        raise
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@app.command()
def cancel(run_id: str, workspace: Workspace = None) -> None:
    """Request cancellation; running SPICE processes have a bounded timeout."""
    try:
        typer.echo(_json(_saved_manager(workspace, run_id).cancel_run(run_id)))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        _handle_error(exc)


@knowledge_app.command("import")
def knowledge_import(path: Annotated[Path, typer.Argument(exists=True)], workspace: Workspace = None) -> None:
    """Import local Markdown, text, JSON or PDF knowledge."""
    try:
        typer.echo(_json(_manager(workspace).import_knowledge(path)))
    except (ValueError, OSError, RuntimeError) as exc:
        _handle_error(exc)


@knowledge_app.command("search")
def knowledge_search(query: str, workspace: Workspace = None,
                     limit: Annotated[int, typer.Option(min=1, max=50)] = 5) -> None:
    """Search local source documents with provenance."""
    try:
        typer.echo(_json(_manager(workspace).search_knowledge(query, limit=limit)))
    except (ValueError, OSError, RuntimeError) as exc:
        _handle_error(exc)


@app.command()
def doctor(workspace: Workspace = None, config: Config = None,
           simulate_models: Annotated[bool, typer.Option("--simulate-models", help="Run bounded ngspice smoke tests for every catalog device.")] = False,
           json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Check ngspice, bundled models and credential presence without revealing keys."""
    import hashlib
    from importlib.resources import files

    spice_path = shutil.which("ngspice")
    spice_version = None
    if spice_path:
        try:
            check = subprocess.run([spice_path, "--version"], capture_output=True, text=True, timeout=5,
                                   check=False)
            spice_version = next((line.strip() for line in check.stdout.splitlines() if "ngspice-" in line), None)
        except (OSError, subprocess.TimeoutExpired):
            pass
    model_root = files("spiceharness").joinpath("data", "models")
    catalog = json.loads(model_root.joinpath("catalog.json").read_text())
    models = {}
    for name, record in catalog.items():
        path = model_root.joinpath(record["path"])
        models[name] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    credentials = {name: bool(os.environ.get(name)) for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY")}
    data: dict[str, Any] = {"version": __version__, "workspace": str(workspace.resolve()),
                            "ngspice": {"available": bool(spice_path), "version": spice_version},
                            "models": models, "credentials": credentials}
    if config:
        try:
            from .providers import Settings
            settings = Settings.load(config)
            data["configuration"] = {"valid": True}
            # Report only environment variable names and booleans, never settings reprs.
            dumped = settings.model_dump() if hasattr(settings, "model_dump") else {}
            def find_env_names(value: Any) -> None:
                if isinstance(value, dict):
                    for key, item in value.items():
                        if key in {"api_key_env", "credential_env"} and isinstance(item, str):
                            credentials[item] = bool(os.environ.get(item))
                        else:
                            find_env_names(item)
                elif isinstance(value, list):
                    for item in value:
                        find_env_names(item)
            find_env_names(dumped)
        except (ValueError, OSError) as exc:
            data["configuration"] = {"valid": False, "error": type(exc).__name__}
    if simulate_models:
        from .simulator import smoke_models
        data["model_simulations"] = {
            name: {"status": result.status, "errors": result.errors,
                   "artifact_dir": result.artifact_dir, "model_checksums": result.model_checksums}
            for name, result in smoke_models().items()
        }
    if json_output:
        typer.echo(_json(data))
    else:
        table = Table(title=f"SpiceHarness {__version__} · environment", box=box.SIMPLE)
        table.add_column("Dependency")
        table.add_column("Status")
        table.add_row("ngspice", spice_version or ("found" if spice_path else "missing"))
        for name, present in models.items():
            table.add_row(name, "available" if present else "missing")
        for name, present in credentials.items():
            table.add_row(name, "set" if present else "not set")
        if "configuration" in data:
            table.add_row("configuration", "valid" if data["configuration"]["valid"] else "invalid")
        for name, check in data.get("model_simulations", {}).items():
            table.add_row(f"{name} ngspice smoke", check["status"])
        console.print(table)
    smoke_failed = any(item["status"] != "ok" for item in data.get("model_simulations", {}).values())
    if not spice_path or not all(models.values()) or data.get("configuration", {}).get("valid") is False or smoke_failed:
        raise typer.Exit(1)


@app.command()
def serve(workspace: Workspace = None, config: Config = None) -> None:
    """Serve MCP over stdio. Standard output carries protocol messages only."""
    from .mcp_server import create_server

    create_server(workspace=workspace, config=config).run(transport="stdio")
