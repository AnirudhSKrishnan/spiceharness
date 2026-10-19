"""Portable native schematics and unchanged, validated saved simulator evidence.

Export is offline: it neither downloads models nor runs SPICE nor changes scores.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .domain import Circuit, DesignSpec
from .experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR
from .simulator import MODEL_DIR, NgSpiceSimulator, tolerance_cases

_INIT = "set ngbehavior=ltpsa\nset filetype=ascii\n"
_SAFE_FILE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,150}")
_SUFFIXES = {".cir", ".raw", ".log", ".csv", ".npz", ".png"}
_MAX_FILE_BYTES = 256 * 1024 * 1024


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _no_links(path: Path) -> None:
    for ancestor in (path, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Symlink paths are not allowed in an export: {ancestor}")


def _artifact_directory(value: Any, workspace: Path) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or ".." in Path(value).parts:
        raise ValueError("Artifact directory must be a local path without traversal")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else workspace / path
    _no_links(path)
    if not path.resolve().is_relative_to(workspace / ".spiceharness"):
        raise ValueError("Saved artifact directory is outside the workspace .spiceharness tree")
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError("Saved artifact directory is not a regular directory")
    return path


def _safe_id(value: str) -> str:
    # IDs, never names or purposes, determine filenames. Retain normal UUIDs and
    # suffix transformed IDs with a hash so two hostile strings cannot collide.
    clean = re.sub(r"[^A-Za-z0-9_-]", "_", value)[:80].strip("_") or "candidate"
    return clean if clean == value else clean + "-" + hashlib.sha256(value.encode()).hexdigest()[:12]


def _models(directory: Path | None, circuit: Any, spec: Any, simulation: dict,
            output: Path) -> tuple[dict, list[str]]:
    """Copy only original files matching both saved checksums and trusted catalog."""
    from .kicad_export import model_pin_names

    catalog = json.loads((MODEL_DIR / "catalog.json").read_text())
    names = sorted({part.model for part in (*circuit.components, *getattr(spec, "loads", ())) if part.model})
    checksums = simulation.get("model_checksums") or {}
    bindings, missing = {}, []
    for index, name in enumerate(names):
        entry = catalog.get(name)
        if entry is None:
            missing.append(f"{name}: unavailable in the trusted model catalog; no substitute model was created")
            continue
        filename = f"model_{index}.lib"
        source = directory / filename if directory else None
        if source is not None:
            _no_links(source)
        if source is None or not source.exists():
            missing.append(f"{name}: original saved {filename} is missing; fetch models explicitly and re-evaluate")
            continue
        if not source.is_file():
            raise ValueError(f"Saved model {filename} is not a regular file")
        if source.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"Saved model {filename} exceeds the export size limit")
        expected = entry["sha256"]
        if checksums.get(name) != expected or _digest(source) != expected:
            raise ValueError(f"Saved model checksum/provenance mismatch for {name}")
        shutil.copyfile(source, output / filename)
        binding = {**entry, "path": filename, "library": filename}
        binding["model_pins"] = model_pin_names(source.read_text(errors="strict"), entry["spice_name"])
        bindings[name] = binding
    return bindings, missing


def _validate_deck(path: Path, circuit: Any, spec: Any, bindings: dict, output: Path) -> None:
    """Compare saved executable input to the deterministic IR compiler, byte for byte."""
    catalog = {name: {**entry, "path": str(output / entry["library"])} for name, entry in bindings.items()}
    model_files = {name: entry["library"] for name, entry in bindings.items()}
    if isinstance(circuit, GeneralCircuitIR):
        match = re.fullmatch(r"\d+_(op|ac|tran)(?:_([A-Za-z0-9_]{1,48}))?\.cir", path.name)
        if match is None:
            raise ValueError(f"Unrecognized saved experiment deck name: {path.name}")
        evaluator = ExperimentEvaluator(model_catalog=catalog)
        generated = evaluator.generate_netlist(circuit, spec, match[1], match[2], model_files)
    else:
        if path.name not in {"op.cir", "ac.cir", "tran.cir"}:
            raise ValueError(f"Unrecognized saved template deck name: {path.name}")
        generated = NgSpiceSimulator(model_catalog=catalog).generate_netlist(circuit, path.stem, spec, model_files)
    if path.read_bytes() != generated.encode():
        raise ValueError(f"Saved deck {path.name} differs from the validated circuit and frozen fixture")


def _copy_evidence(directory: Path | None, output: Path, circuit: Any, spec: Any,
                   simulation: dict) -> tuple[dict, list[str], list[str]]:
    bindings, missing = _models(directory, circuit, spec, simulation, output)
    copied = [entry["library"] for entry in bindings.values()]
    if directory is None:
        return bindings, copied, missing
    used_models = {part.model for part in (*circuit.components, *getattr(spec, "loads", ())) if part.model}
    for source in sorted(directory.iterdir()):
        allowed = source.suffix.lower() in _SUFFIXES or source.name.endswith(".console.txt")
        if not allowed:
            continue
        if not _SAFE_FILE.fullmatch(source.name):
            raise ValueError("Saved evidence filenames must be simple local filenames")
        _no_links(source)
        if not source.is_file():
            raise ValueError(f"Saved evidence is not a regular file: {source.name}")
        if source.stat().st_size > _MAX_FILE_BYTES:
            raise ValueError(f"Saved evidence exceeds the export size limit: {source.name}")
        if source.suffix.lower() == ".cir":
            if used_models - bindings.keys():
                missing.append(f"{source.name}: omitted because its original pinned models are unavailable")
                continue
            _validate_deck(source, circuit, spec, bindings, output)
        shutil.copyfile(source, output / source.name)
        copied.append(source.name)
    if any(name.endswith(".cir") for name in copied):
        (output / ".spiceinit").write_text(_INIT)
        copied.append(".spiceinit")
    return bindings, copied, missing


def _candidate_readme(row: dict, files: list[str], missing: list[str]) -> str:
    decks = [name for name in files if name.endswith(".cir") and "/" not in name]
    lines = ["# Saved candidate circuit", "", "Open `candidate.kicad_sch` in KiCad Schematic Editor.",
             "Keep `candidate.kicad_pro` alongside it so relative model libraries resolve when opened from another folder.",
             "The editable drawing includes the frozen fixture and circuit connections. Active devices use only",
             "copied original pinned vendor models. Missing active models make simulation unavailable; they are never idealized.",
             "The recorded score and measurements are in `score.json`, `metrics.json`, and `evidence.json`.",
             "Editing or simulating this copy does not revise those saved results.",
             "The saved score applies only to the original circuit. After a KiCad edit, submit validated Circuit IR",
             "as a new candidate, re-evaluate it under the frozen specification, and run fresh verification before claiming a pass.", "",
             f"Saved simulation status: {row.get('status', 'pending')}. Fresh verification: {bool(row.get('verified'))}.", ""]
    if decks:
        lines += ["## Run the original ngspice decks", "", "In a terminal, change into this candidate folder first.",
                  "Run each desired saved deck, for example:", "", "```sh",
                  f'HOME="$PWD" SPICE_ASCIIRAWFILE=1 ngspice -b -r user-output.raw -o user-output.log {decks[0]}',
                  "```", "", "`HOME` is scoped to this command so ngspice reads this folder's `.spiceinit`",
                  "(PSpice compatibility and ASCII raw output) without changing your personal configuration.",
                  "Keep every relative `model_N.lib` beside the unchanged `.cir` files.",
                  "Use another raw/log output filename for each deck to retain your new simulations.",
                  "The existing `.raw`, `.log`, `.csv`, and `.npz` files are original evaluator evidence.",
                  "General-experiment CSV columns contain real/imaginary pairs in SI units; NPZ stores `names` and `data`.",
                  "ASCII `.raw` files can be loaded by ngspice (`load filename.raw`) or a compatible waveform viewer.",
                  "Analysis commands are already in the saved decks. KiCad Simulator may use its own analysis settings.", ""]
    else:
        lines += ["No runnable validated saved deck is available for this record. No simulation or curve was invented.", ""]
    if missing:
        lines += ["## Missing evidence", "", *[f"- {note}" for note in missing], ""]
    return "\n".join(lines)


def _rank_map(run: dict, analysis_only: bool) -> dict[str, int]:
    if analysis_only:
        return {}
    rows = run.get("leaderboard")
    if rows is None:
        rows = sorted((row for row in run.get("candidates", []) if (row.get("score") or {}).get("rank_key")
                       is not None), key=lambda row: tuple(row["score"]["rank_key"]))
    eligible = [row for row in rows if row.get("purpose", "search") == "search"
                and (row.get("score") or {}).get("rank_key") is not None]
    return {row["candidate_id"]: rank for rank, row in enumerate(eligible, 1)}


def export_spice_bundle(run: dict, destination: Path, *, workspace: Path) -> Path:
    """Export every supplied saved candidate; return the portable manifest path.

    Existing destinations are retained. A new sibling ``NAME-2``, ``NAME-3``, ...
    is published atomically, so reruns never overwrite edits to exported circuits.
    Caller may filter ``run['candidates']`` without changing the saved record.
    """
    from .kicad_export import render_kicad_schematic

    workspace = Path(workspace).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if ".." in destination.parts:
        raise ValueError("Export destination cannot contain traversal")
    _no_links(destination)
    candidates = run.get("candidates") or []
    if not candidates:
        raise ValueError("This run has no saved candidate circuits to export")
    general = "sources" in (run.get("spec") or {})
    spec = (ExperimentSpec if general else DesignSpec).model_validate(run.get("spec"))
    analysis_only = general and not spec.requirements
    rank = _rank_map(run, analysis_only)
    result = run.get("result") or {}
    manifest = {"format": "spiceharness-portable-v1", "run_id": run.get("run_id"),
                "analysis_only": analysis_only, "best_candidate_id": result.get("best_candidate_id"),
                "verification_candidate_id": result.get("verification_candidate_id"),
                "verified": result.get("verified", False), "selection_policy": result.get("selection_policy"),
                "candidates": []}
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".spiceharness-export-", dir=destination.parent))
    try:
        _json(stage / "spec.json", run["spec"])
        seen = set()
        for candidate in candidates:
            candidate_id = candidate.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
                raise ValueError("Each exported candidate must have a unique nonempty saved ID")
            seen.add(candidate_id)
            circuit = (GeneralCircuitIR if general else Circuit).model_validate(candidate.get("circuit"))
            folder = _safe_id(candidate_id)
            output = stage / folder
            output.mkdir()
            simulation = candidate.get("simulation") or {}
            directory = _artifact_directory(simulation.get("artifact_dir"), workspace)
            bindings, files, missing = _copy_evidence(directory, output, circuit, spec, simulation)
            if simulation.get("artifact_dir") and directory is None:
                missing.append("The saved artifact directory is missing from this workspace")
            # Preserve legacy tolerance screens as separate folders, using their
            # deterministic perturbed circuits rather than treating them as nominal.
            if not general:
                cases = tolerance_cases(spec, circuit)
                corners = (simulation.get("analyses") or {}).get("corners") or {}
                for label, corner in corners.items():
                    if label not in cases:
                        raise ValueError("Unknown saved tolerance case")
                    corner_dir = _artifact_directory(corner.get("artifact_dir"), workspace)
                    child = output / "corners" / _safe_id(label)
                    child.mkdir(parents=True)
                    corner_spec = spec.model_copy(update={"corner_policy": "nominal"})
                    corner_circuit = circuit.model_copy(update={"components": cases[label], "spec": corner_spec})
                    _, corner_files, corner_missing = _copy_evidence(
                        corner_dir, child, corner_circuit, corner_spec,
                        {**corner, "model_checksums": simulation.get("model_checksums", {})})
                    files += [str(Path("corners") / _safe_id(label) / name) for name in corner_files]
                    missing += [f"{label}: {message}" for message in corner_missing]
            (output / "candidate.kicad_sch").write_text(render_kicad_schematic(
                circuit.model_dump(mode="json"), spec=spec.model_dump(mode="json"), model_bindings=bindings))
            # KiCad needs a project context to resolve model libraries relative
            # to this folder when launched from another working directory.
            _json(output / "candidate.kicad_pro", {})
            _json(output / "circuit.json", candidate["circuit"])
            _json(output / "score.json", candidate.get("score"))
            _json(output / "metrics.json", simulation.get("metrics") or {})
            _json(output / "models.json", bindings)
            from .saved_outputs import render_saved_outputs

            plots = render_saved_outputs(simulation, workspace=workspace,
                                         spec=spec.model_dump(mode="json"), circuit=circuit.model_dump(mode="json"))
            for index, plot in enumerate(plots, 1):
                plot_path = f"saved-output-{index}.svg"
                (output / plot_path).write_text(plot["svg"])
                files.append(plot_path)
            _json(output / "outputs.json", [{key: value for key, value in plot.items() if key != "svg"}
                                           | {"path": f"saved-output-{index}.svg"}
                                           for index, plot in enumerate(plots, 1)])
            if not plots:
                (output / "OUTPUTS.md").write_text(
                    "No readable saved AC/transient curves are available for this record.\n"
                    "No new simulation or waveform was created. Inspect the original raw/log files, if present.\n")
            _json(output / "evidence.json", {"candidate_id": candidate_id, "purpose": candidate.get("purpose", "search"),
                                           "verified": candidate.get("verified", False), "simulation": simulation,
                                           "files": files, "missing": missing})
            row = {"candidate_id": candidate_id, "name": circuit.name,
                   "purpose": candidate.get("purpose", "search"), "rank": rank.get(candidate_id),
                   "feasible": (candidate.get("score") or {}).get("feasible"),
                   "verified": candidate.get("verified", False), "path": folder,
                   "status": simulation.get("status", candidate.get("status", "pending")),
                   "schematic": f"{folder}/candidate.kicad_sch", "missing": missing,
                   "native_runnable": not ({part.model for part in circuit.components if part.model} - bindings.keys()),
                   "saved_decks_runnable": any(name.endswith(".cir") and "/" not in name for name in files)}
            (output / "README.md").write_text(_candidate_readme(row, files, missing))
            manifest["candidates"].append(row)
        paths = {row["candidate_id"]: row["path"] for row in manifest["candidates"]}
        # Study figures are existing run-level evidence, never regenerated from
        # labels or agent claims. Their only allowed location is this run's report.
        run_id = run.get("run_id", "")
        if isinstance(run_id, str) and re.fullmatch(r"[0-9a-f]{32}", run_id):
            kind = "experiments" if general else "runs"
            report = workspace / ".spiceharness" / kind / run_id / "report"
            _no_links(report)
            if report.is_dir():
                for figure in sorted(report.glob("*.png")):
                    _no_links(figure)
                    if not figure.is_file() or not _SAFE_FILE.fullmatch(figure.name):
                        raise ValueError("Study figures must be regular files with simple filenames")
                    if figure.stat().st_size > _MAX_FILE_BYTES:
                        raise ValueError("Study figure exceeds the export size limit")
                    (stage / "study-figures").mkdir(exist_ok=True)
                    shutil.copyfile(figure, stage / "study-figures" / figure.name)
        default = next((candidate_id for candidate_id in (result.get("verification_candidate_id"),
                                                          result.get("best_candidate_id"))
                        if candidate_id in paths), None)
        manifest["open_candidate_id"] = default
        manifest["open_schematic"] = f"{paths[default]}/candidate.kicad_sch" if default in paths else None
        lines = ["# Portable saved circuits", "", f"Run: {run.get('run_id')}", "",
                 "Every included candidate has its own editable native KiCad schematic and saved evidence.",
                 "All files are local. This export made no agent calls, downloaded no models, and ran no simulations.",
                 "Scores, measurements, ranking and fresh-verification flags are copied from the saved evaluator results.", ""]
        if manifest["open_schematic"]:
            label = "fresh verification of the selected analysis candidate" if analysis_only else "fresh verification of the best candidate"
            if default != result.get("verification_candidate_id"):
                label = "saved selected candidate (no fresh verification record supplied)"
            lines += [f"Open `{manifest['open_schematic']}` in KiCad first: {label}.", ""]
        else:
            lines += ["No finalized candidate is included. Open any candidate schematic listed in `manifest.json`.", ""]
        if analysis_only:
            lines += ["Analysis only: no acceptance requirements were frozen, so there is no feasibility ranking or winner.", ""]
        lines += ["See `manifest.json` for every name, ID, purpose, saved rank, feasibility, verification and folder.",
                  "See each candidate's `README.md` for ngspice commands, compatibility settings and missing evidence.",
                  "No unavailable real model is replaced by an ideal model. Active KiCad devices without an original",
                  "pinned model cannot run; native_runnable is false. Pending/failed records retain their saved status.",
                  "Model source URLs and checksums are recorded in each `models.json`.",
                  "Retain all model files when moving this local folder. Vendor models are copied for local use;",
                  "this export does not grant permission to redistribute them. Vendor model terms still apply.",
                  "Sensitivity screens are not exhaustive worst-case analysis.", "",
                  "Re-exporting creates a new sibling folder with a numeric suffix; existing files and edits are retained.", ""]
        (stage / "README.md").write_text("\n".join(lines))
        # File checksums cover portable contents without a self-referential hash.
        manifest["files"] = {str(path.relative_to(stage)): _digest(path)
                             for path in sorted(stage.rglob("*")) if path.is_file()}
        _json(stage / "manifest.json", manifest)
        target, number = destination, 1
        while True:
            try:
                if target.exists():
                    raise FileExistsError
                os.rename(stage, target)
                break
            except FileExistsError:
                number += 1
                target = destination.with_name(f"{destination.name}-{number}")
                _no_links(target)
        return target / "manifest.json"
    finally:
        if stage.exists():
            shutil.rmtree(stage)
