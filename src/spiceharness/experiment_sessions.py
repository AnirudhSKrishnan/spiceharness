"""Bounded, durable circuit investigations driven by a connected client.

The saved specification owns fixtures and measurements. Clients submit circuit IR,
never measurements, ranking, or netlists. No model API calls occur in this module.
"""

from __future__ import annotations

import fcntl
import json
import re
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from .storage import encoded, utcnow

TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}
MAX_EVALUATIONS = 20
MAX_OPERATION_SECONDS = 20.0
MAX_SESSION_SECONDS = 180.0


class ExperimentManager:
    """Persist frozen investigations and independently verify the numerical winner."""

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).expanduser().resolve()
        self.directory = self.workspace / ".spiceharness" / "experiments"

    def _path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise ValueError("run_id must be a 32-character hexadecimal run identifier")
        return self.directory / run_id

    def _read(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id) / "run.json"
        if not path.is_file():
            raise ValueError(f"Unknown experiment: {run_id}")
        record = json.loads(path.read_text())
        record["cancel_requested"] = (path.parent / "cancel.requested").exists()
        return record

    def _save(self, record: dict[str, Any]) -> None:
        path = self._path(record["run_id"]) / "run.json"
        with (path.parent / "state.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            if (path.parent / "cancel.requested").exists():
                record["cancel_requested"] = True
                record["status"] = "cancelled"
                if record.get("result"):
                    record["result"]["verified"] = False
                for row in record["candidates"]:
                    if row["purpose"] == "verification":
                        row["verified"] = False
            record["updated_at"] = utcnow()
            temporary = path.with_name(f"run.{uuid.uuid4().hex}.tmp")
            temporary.write_text(encoded(record))
            temporary.replace(path)

    @contextmanager
    def _locked(self, run_id: str) -> Iterator[dict[str, Any]]:
        self._read(run_id)
        with (self._path(run_id) / "session.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("An experiment operation is active; retry when it finishes") from exc
            try:
                record = self._read(run_id)
                if self._recover(record):
                    self._save(record)
                yield record
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _recover(self, record: dict[str, Any]) -> bool:
        """Called after the operation lock proves no evaluator is running."""
        if record.get("operation_started_at") is None:
            return False
        # After a process crash the exact CPU duration is unknowable. Recover
        # at most one operation allowance; successful work charges actual time.
        record["elapsed_seconds"] += min(
            MAX_OPERATION_SECONDS, max(0, time.time() - record["operation_started_at"])
        )
        record["operation_started_at"] = None
        record["status"] = "cancelled" if record["cancel_requested"] else "interrupted"
        if record.get("result"):
            record["result"]["verified"] = False
        for row in record["candidates"]:
            if row["purpose"] == "verification":
                row["verified"] = False
            if row.get("simulation") is None and row.get("status", "pending") == "pending":
                row["status"] = "interrupted"
                row["error"] = "Evaluator stopped before measured results were committed"
                row["verified"] = False
        record["events"].append({"type": "operation_interrupted", "created_at": utcnow()})
        return True

    def _active(self, record: dict[str, Any]) -> None:
        if record["cancel_requested"] or (self._path(record["run_id"]) / "cancel.requested").exists():
            record["status"] = "cancelled"
            self._save(record)
            raise ValueError("Experiment is cancelled")
        if record["status"] in TERMINAL:
            raise ValueError(f"Experiment is {record['status']}; start another frozen investigation")

    def get_experiment(self, run_id: str) -> dict[str, Any]:
        record = self._read(run_id)
        lock_path = self._path(run_id) / "session.lock"
        if record.get("operation_started_at") is not None and lock_path.exists():
            with lock_path.open("r") as handle:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    # Read-only snapshots identify stale work; a later mutation
                    # persists recovery under the operation lock.
                    record = self._read(run_id)
                    self._recover(record)
        if record["cancel_requested"] and record["status"] not in TERMINAL:
            record["status"] = "cancelling" if record.get("operation_started_at") else "cancelled"
        if record["cancel_requested"]:
            if record.get("result"):
                record["result"]["verified"] = False
            for row in record["candidates"]:
                if row["purpose"] == "verification":
                    row["verified"] = False
        record["leaderboard"] = sorted(
            (row for row in record["candidates"] if (row.get("score") or {}).get("rank_key") is not None
             and row["purpose"] == "search"),
            key=lambda row: tuple(row["score"]["rank_key"]),
        )
        if (self._path(run_id) / "montecarlo").is_dir():
            # Read-only summary of optional sidecar studies; absent unless a study exists.
            record["montecarlo"] = self.list_montecarlo(run_id)
        return record

    def resolve_run_id(self, run_id: str) -> str:
        """Complete a unique hexadecimal prefix (at least 4 characters) of a saved experiment."""
        if re.fullmatch(r"[0-9a-f]{32}", run_id or ""):
            return run_id
        if re.fullmatch(r"[0-9a-f]{4,31}", run_id or "") is None:
            raise ValueError("run_id must be a 32-character hexadecimal run identifier or a unique prefix "
                             "of at least 4 characters")
        matches = sorted(path.parent.name for path in self.directory.glob(f"{run_id}*/run.json")) \
            if self.directory.is_dir() else []
        if len(matches) > 1:
            raise ValueError(f"Ambiguous run prefix {run_id!r} matches {len(matches)} experiments "
                             f"({', '.join(item[:12] for item in matches[:6])}); use more characters")
        if not matches:
            raise ValueError(f"No saved experiment in {self.workspace} matches prefix {run_id!r}")
        return matches[0]

    def _study_directories(self, run_id: str) -> list[Path]:
        root = self._path(run_id) / "montecarlo"
        if not root.is_dir() or root.is_symlink():
            return []
        return sorted(path for path in root.iterdir()
                      if path.is_dir() and not path.is_symlink()
                      and re.fullmatch(r"[0-9a-f]{16}", path.name) and (path / "study.json").is_file())

    def list_montecarlo(self, run_id: str) -> list[dict[str, Any]]:
        from .montecarlo import summary_of

        studies = []
        for directory in self._study_directories(run_id):
            try:
                studies.append(summary_of(json.loads((directory / "study.json").read_text())))
            except (OSError, ValueError, KeyError):
                continue
        return studies

    def get_montecarlo(self, run_id: str, study_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """The full saved study, or summaries when several exist and none was selected."""
        self._read(run_id)
        directories = self._study_directories(run_id)
        if study_id is not None:
            if re.fullmatch(r"[0-9a-f]{4,16}", study_id) is None:
                raise ValueError("study_id must be a hexadecimal study id or a unique prefix")
            directories = [path for path in directories if path.name.startswith(study_id)]
            if len(directories) > 1:
                raise ValueError(f"Ambiguous study prefix {study_id!r}; use more characters")
            if not directories:
                raise ValueError(f"No Monte Carlo study matches {study_id!r} for this experiment")
        elif not directories:
            raise ValueError("No Monte Carlo study exists for this experiment; call run_montecarlo first")
        elif len(directories) > 1:
            return self.list_montecarlo(run_id)
        return json.loads((directories[0] / "study.json").read_text())

    def _montecarlo_row(self, record: dict[str, Any], candidate_id: str | None) -> dict[str, Any]:
        if candidate_id is None:
            result = record.get("result") or {}
            if record["status"] != "completed" or not result.get("verified"):
                raise ValueError("Default Monte Carlo target is the verified winner; finalize the experiment "
                                 "first or name a saved candidate_id")
            candidate_id = result["best_candidate_id"]
        matches = [row for row in record["candidates"] if row["candidate_id"].startswith(candidate_id)]
        if not candidate_id or not matches:
            raise ValueError(f"Unknown candidate: {candidate_id}")
        exact = [row for row in matches if row["candidate_id"] == candidate_id]
        if len(exact) != 1 and len(matches) != 1:
            raise ValueError(f"Ambiguous candidate prefix {candidate_id!r}; use a longer prefix or the full ID")
        row = exact[0] if exact else matches[0]
        if not (row.get("score") or {}).get("valid") or not (row.get("simulation") or {}).get("metrics"):
            raise ValueError("Monte Carlo needs a saved candidate with a valid measured evaluation")
        return row

    def run_montecarlo(self, run_id: str, request: Any, *, evaluator_factory: Any = None) -> dict[str, Any]:
        """Run one bounded chunk of a tolerance study; never writes run.json or the leaderboard.

        Repeat with the same request until ``state != "running"``. Studies are sidecars under the run
        directory and use their own budget, separate from the 20-evaluation / 180 s session limits.
        """
        from .experiments import ExperimentSpec, GeneralCircuitIR
        from .montecarlo import MonteCarloRequest, MonteCarloStore, StudyContext, ngspice_version, run_chunk

        request = MonteCarloRequest.model_validate(
            request.model_dump() if hasattr(request, "model_dump") else request)
        record = self._read(run_id)
        if record.get("operation_started_at") is not None:
            raise ValueError("Experiment has an unfinished operation; call get_experiment or finalize first")
        if record["status"] in ("cancelled", "cancelling") or record["cancel_requested"]:
            raise ValueError("Experiment is cancelled")
        with (self._path(run_id) / "session.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("An experiment operation is active; retry when it finishes") from exc
            try:
                record = self._read(run_id)
                row = self._montecarlo_row(record, request.candidate_id)
                spec = ExperimentSpec.model_validate(record["spec"])
                circuit = GeneralCircuitIR.model_validate(row["circuit"])
                request = MonteCarloRequest.model_validate({**request.model_dump(), "candidate_id": row["candidate_id"]})
                request.validate_against(circuit, spec)
                study_id = request.study_id(
                    run_id=run_id, candidate_id=row["candidate_id"], fixture_fingerprint=spec.fingerprint,
                    circuit_fingerprint=circuit.fingerprint)
                context = StudyContext(
                    run_id=run_id, candidate_id=row["candidate_id"],
                    candidate_name=circuit.name, circuit=circuit, spec=spec,
                    saved_metrics=dict(row["simulation"]["metrics"]),
                    simulator_version=None if evaluator_factory else ngspice_version())
                return run_chunk(MonteCarloStore(self._path(run_id), study_id), context, request,
                                 evaluator_factory=evaluator_factory)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def list_experiments(self) -> list[dict[str, Any]]:
        if not self.directory.exists():
            return []
        records = [self.get_experiment(path.parent.name) for path in self.directory.glob("*/run.json")]
        return sorted(records, key=lambda record: record["created_at"], reverse=True)

    def cancel_experiment(self, run_id: str) -> dict[str, Any]:
        self._read(run_id)
        with (self._path(run_id) / "state.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            record = self._read(run_id)
            if record["status"] not in TERMINAL:
                (self._path(run_id) / "cancel.requested").touch()
        self._cancel_studies(run_id)
        return self.get_experiment(run_id)

    def _cancel_studies(self, run_id: str) -> None:
        """Stop running Monte Carlo studies of this run: at the next sample boundary, or at once if idle.

        A cancelled study keeps its committed prefix and resumes on the next run_montecarlo call.
        """
        from .montecarlo import MonteCarloStore

        for directory in self._study_directories(run_id):
            store = MonteCarloStore(self._path(run_id), directory.name)
            try:
                study = json.loads((directory / "study.json").read_text())
                if study.get("state") != "running":
                    continue
                try:
                    with store.lock():       # no chunk is active: mark it directly
                        study = store.load() or study
                        if study.get("state") == "running":
                            study["state"] = "cancelled"
                            store.save(study)
                except ValueError:           # a chunk is active: it checks the sentinel between samples
                    if not store.cancel_path.is_symlink():
                        store.cancel_path.touch()
            except (OSError, ValueError, KeyError):
                continue

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Record an auditable event on a saved experiment."""
        record = self._read(run_id)
        record["events"].append({"created_at": utcnow(), **event})
        self._save(record)

    def start_experiment(self, spec: Any, *,
                         execution_mode: Literal["client", "direct"] = "client") -> dict[str, Any]:
        from .experiments import ExperimentSpec

        if execution_mode not in {"client", "direct"}:
            raise ValueError("General experiments support only client or direct execution")
        spec = ExperimentSpec.model_validate(spec.model_dump() if hasattr(spec, "model_dump") else spec)
        run_id = uuid.uuid4().hex
        self._path(run_id).mkdir(parents=True)
        record = {
            "run_id": run_id, "kind": "experiment", "status": "awaiting_client",
            "execution_mode": execution_mode, "offline": execution_mode == "direct",
            "spec": spec.model_dump(mode="json"),
            "fixture_fingerprint": spec.fingerprint, "created_at": utcnow(), "updated_at": utcnow(),
            "candidates": [], "events": [{"type": "experiment_created", "created_at": utcnow()}],
            "usage": [], "elapsed_seconds": 0.0, "evaluation_count": 0,
            "operation_started_at": None, "cancel_requested": False, "result": None,
            "limits": {"evaluations": MAX_EVALUATIONS, "operation_seconds": MAX_OPERATION_SECONDS,
                       "session_seconds": MAX_SESSION_SECONDS},
            "analysis_only": not bool(spec.requirements),
        }
        self._save(record)
        return self.get_experiment(run_id)

    def _evaluate(self, record: dict[str, Any], circuit: Any, *, verification: bool) -> dict[str, Any]:
        from .experiments import ExperimentEvaluator, ExperimentSpec, GeneralCircuitIR

        self._active(record)
        circuit = GeneralCircuitIR.model_validate(circuit.model_dump() if hasattr(circuit, "model_dump") else circuit)
        spec = ExperimentSpec.model_validate(record["spec"])
        remaining = MAX_SESSION_SECONDS - record["elapsed_seconds"]
        reserve = 0 if verification else MAX_OPERATION_SECONDS
        count_limit = MAX_EVALUATIONS if verification else MAX_EVALUATIONS - 1
        if record["evaluation_count"] >= count_limit or remaining <= reserve:
            raise ValueError("Experiment search budget exhausted; finalize saved evidence or start a new run")
        candidate_id = uuid.uuid4().hex
        root = self._path(record["run_id"]) / ("verification" if verification else "evaluations") / candidate_id
        evaluator = ExperimentEvaluator(artifact_root=root, timeout_s=min(MAX_OPERATION_SECONDS, remaining - reserve))
        # Reject fixture/protected-section/model violations before consuming simulation budget.
        evaluator.validate(circuit, spec)
        row = {"candidate_id": candidate_id, "run_id": record["run_id"], "ordinal": len(record["candidates"]) + 1,
               "circuit": circuit.model_dump(mode="json"), "simulation": None, "score": None,
               "purpose": "verification" if verification else "search", "verified": False,
               "status": "pending"}
        record["candidates"].append(row)
        record["evaluation_count"] += 1
        record["status"] = "verifying" if verification else "evaluating"
        record["operation_started_at"] = time.time()
        self._save(record)
        started = time.monotonic()
        try:
            measured = evaluator.evaluate(circuit, spec)
            row["status"] = "completed"
            row["simulation"] = measured.model_dump(mode="json")
            row["score"] = measured.score.model_dump(mode="json")
            row["verified"] = verification and measured.score.valid
            record["events"].append({"type": "candidate_evaluated", "candidate_id": candidate_id,
                                     "status": measured.status, "created_at": utcnow()})
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"Evaluator failed before producing measurements: {type(exc).__name__}"
            raise
        finally:
            record["elapsed_seconds"] += time.monotonic() - started
            record["operation_started_at"] = None
            record["status"] = "cancelled" if (self._path(record["run_id"]) / "cancel.requested").exists() else "awaiting_client"
            self._save(record)
        return row

    def evaluate_experiment(self, run_id: str, circuit: Any) -> dict[str, Any]:
        with self._locked(run_id) as record:
            return self._evaluate(record, circuit, verification=False)

    def finalize_experiment(self, run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        with self._locked(run_id) as record:
            if record["status"] == "completed":
                result = record.get("result") or {}
                if candidate_id is not None and candidate_id not in {
                        result.get("best_candidate_id"), result.get("verification_candidate_id")}:
                    known = any(row["candidate_id"] == candidate_id for row in record["candidates"])
                    raise ValueError(
                        "Experiment is already finalized for candidate "
                        f"{result.get('best_candidate_id')}; "
                        + ("it cannot be re-finalized for a different candidate"
                           if known else "candidate_id is not a saved evaluation of this run"))
                return self.get_experiment(run_id)
            self._active(record)
            candidates = [row for row in record["candidates"]
                          if row["purpose"] == "search" and (row.get("score") or {}).get("valid")]
            if not candidates:
                raise ValueError("No valid measured candidate to verify; evaluate another eligible circuit")
            if record["analysis_only"]:
                selected = next((row for row in candidates if row["candidate_id"] == candidate_id), None)
                if candidate_id is not None and selected is None:
                    raise ValueError("Analysis candidate must be a valid saved search evaluation")
                selected = selected or candidates[0]
                policy = "analysis only: fresh remeasurement of selected candidate; no feasibility ranking"
            else:
                selected = min(candidates, key=lambda row: tuple(row["score"]["rank_key"]))
                if candidate_id is not None and candidate_id != selected["candidate_id"]:
                    raise ValueError("Fresh verification must use the deterministic numerical winner")
                policy = "best found under frozen requirements and bounded search"
            fresh = self._evaluate(record, selected["circuit"], verification=True)
            if record["status"] == "cancelled":
                return self.get_experiment(run_id)
            record["result"] = {"best_candidate_id": selected["candidate_id"],
                                "verification_candidate_id": fresh["candidate_id"],
                                "circuit": selected["circuit"], "simulation": fresh["simulation"],
                                "score": fresh["score"], "verified": fresh["verified"],
                                "feasible": fresh["score"]["feasible"], "selection_policy": policy}
            record["status"] = "finalizing"
            record["operation_started_at"] = time.time()
            export_started = time.monotonic()
            self._save(record)
            if record["status"] == "cancelled":
                return self.get_experiment(run_id)
            record["result"]["artifacts"] = self._export(record)
            record["elapsed_seconds"] += time.monotonic() - export_started
            record["status"] = "completed" if fresh["verified"] else "failed"
            record["operation_started_at"] = None
            self._save(record)
            # Cancellation during export must win the final save. The report
            # reflects the committed state, including a cancelled verification.
            self._export(record)
        return self.get_experiment(run_id)

    def _export(self, record: dict[str, Any]) -> dict[str, str]:
        directory = self._path(record["run_id"]) / "report"
        directory.mkdir(exist_ok=True)
        result = record["result"]
        circuit = directory / "circuit.json"
        fixture = directory / "fixture.json"
        report = directory / "report.md"
        evidence = directory / "evidence.json"
        circuit.write_text(json.dumps(result["circuit"], indent=2))
        fixture.write_text(json.dumps(record["spec"], indent=2))
        evidence.write_text(encoded(record))
        lines = ["# Circuit investigation", "", f"Run: {record['run_id']}", "",
                 f"Selection: {result['selection_policy']}", "",
                 f"Fresh analysis verified: {result['verified']}; feasibility: {result['feasible']}", "",
                 "Results are nominal sampled analyses under the saved fixture, not exhaustive worst-case proof.",
                 "", "## Fresh measurements", "", "| Metric | Value (SI units) |", "| --- | ---: |"]
        lines.extend(f"| {name} | {value:.10g} |" for name, value in result["simulation"]["metrics"].items())
        lines += ["", "## Assumptions", ""] + [f"- {value}" for value in record["spec"]["assumptions"]]
        lines += ["", "## Source provenance", ""] + [
            f"- {source['source']}, page {source.get('page') or 'unspecified'}: {source.get('note', '')}"
            for source in record["spec"]["provenance"]]
        lines += ["", "## Candidate evidence", "", "| Candidate | Purpose | Status | Packages | Feasible |",
                  "| --- | --- | --- | ---: | --- |"]
        for row in record["candidates"]:
            score = row.get("score") or {}
            lines.append(f"| {row['candidate_id']} | {row['purpose']} | "
                         f"{(row.get('simulation') or {}).get('status', row.get('status', 'interrupted'))} | "
                         f"{score.get('package_count', 'unknown')} | {score.get('feasible')} |")
        lines += ["", "Raw simulator decks, logs and sampled curves remain in each evaluation's artifact_dir.", ""]
        report.write_text("\n".join(lines))
        return {"circuit": str(circuit), "fixture": str(fixture), "report": str(report), "evidence": str(evidence)}
