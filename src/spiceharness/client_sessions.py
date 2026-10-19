"""Durable sessions driven by the connected coding agent, without provider calls.

Only the evaluator owns fixtures, measurements, ranking and fresh verification.
The client selects from analytically generated circuits; it never submits evidence.
"""

from __future__ import annotations

import fcntl
import json
import math
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .domain import Circuit, DesignSpec, SimulationResult, normalize_spec
from .knowledge import KnowledgeStore
from .providers import Settings, safe_error
from .render import export_bom, render_schematic
from .scoring import score_result
from .simulator import NgSpiceSimulator
from .storage import RunStore, encoded, utcnow
from .templates import E12, E24, generate_candidates, standard_value

_TERMINAL = {"completed", "failed", "cancelled", "budget_exhausted"}


def _physical(circuit: Circuit) -> str:
    return encoded(
        {
            key: circuit.model_dump(mode="json")[key]
            for key in ("topology", "components", "input_node", "output_node")
        }
    )


class ClientDesignManager:
    """Bounded, reconnectable host-agent orchestration using real SPICE only.

    ``elapsed_seconds`` counts server operation time, excluding client thinking
    and disconnected time. Each SPICE evaluation has a maximum 20 second budget;
    sessions permit at most 19 search trials and one fresh verification trial.
    """

    def __init__(self, workspace: str | Path, settings: Settings | dict[str, Any] | None = None):
        self.workspace = Path(workspace).expanduser().resolve()
        self.directory = self.workspace / ".spiceharness"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.settings = (
            settings
            if isinstance(settings, Settings)
            else (Settings.model_validate(settings) if settings is not None else Settings.load())
        )
        self.store = RunStore(self.directory / "runs.sqlite3")
        self.storage = self.store
        self.knowledge = KnowledgeStore(self.directory / "knowledge.sqlite3")

    def _path(self, run_id: str) -> Path:
        if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise ValueError("run_id must be a 32-character hexadecimal run identifier")
        return self.directory / "runs" / run_id

    @contextmanager
    def _locked(self, run_id: str) -> Iterator[dict[str, Any]]:
        directory = self._path(run_id)
        # Check ownership before creating any directory from a caller's identifier.
        self._record(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "session.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError(
                    "A server operation is already active for this run; retry after it finishes"
                ) from exc
            try:
                record = self._record(run_id)
                if record.get("execution_started_at") is not None:
                    # A previous host process died while holding the operation
                    # lock. Charge only its bounded operation, never idle time.
                    charged = min(20.0, max(0.0, time.time() - record["execution_started_at"]))
                    self.store.update_run(
                        run_id, execution_started_at=None, elapsed_seconds=record["elapsed_seconds"] + charged
                    )
                    spec = normalize_spec(record["spec"])
                    for row in record["candidates"]:
                        if row["score"] is None and row["simulation"] is not None:
                            circuit = Circuit.model_validate(row["circuit"])
                            result = SimulationResult.model_validate(row["simulation"])
                            numeric = score_result(spec, circuit, result)
                            self.store.save_evaluation(row["candidate_id"], result, numeric)
                            if row["purpose"] == "verification":
                                self.store.set_candidate_verified(
                                    row["candidate_id"], numeric.valid and numeric.feasible
                                )
                    self.store.event(run_id, "client_operation_interrupted", {"charged_seconds": charged})
                    record = self._record(run_id)
                yield record
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _record(self, run_id: str) -> dict[str, Any]:
        self._path(run_id)
        record = self.store.get_run(run_id)
        if record["options"].get("execution_mode") != "client":
            raise ValueError("This operation requires a client-driven run")
        return record

    @staticmethod
    def _limits(record: dict[str, Any]) -> tuple[int, float]:
        settings = Settings.model_validate(record["options"]["settings"])
        return min(20, settings.limits.max_candidates), min(180.0, settings.limits.max_seconds)

    @staticmethod
    def _active(record: dict[str, Any]) -> None:
        if record["status"] in _TERMINAL or record["cancel_requested"]:
            raise ValueError(f"Run is {record['status']}; create a new run to continue")

    @contextmanager
    def _operation(self, record: dict[str, Any]) -> Iterator[None]:
        run_id = record["run_id"]
        started = time.monotonic()
        self.store.update_run(run_id, execution_started_at=time.time())
        try:
            yield
        finally:
            self.store.update_run(
                run_id,
                execution_started_at=None,
                elapsed_seconds=record["elapsed_seconds"] + time.monotonic() - started,
            )

    def start_design(self, spec: DesignSpec | dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        frozen = normalize_spec(spec)
        run_id = self.store.create_run(
            frozen,
            options={
                "execution_mode": "client",
                "offline": False,
                "settings": self.settings.model_dump(mode="json"),
                "workspace": str(self.workspace),
                "time_accounting": "server_operations_only",
                "spec_fingerprint": frozen.fingerprint,
            },
        )
        with self._locked(run_id):
            provenance = self.knowledge.search(frozen.topology.replace("_", " "), limit=5)
            self.store.checkpoint(
                run_id, "client_start", {"spec_fingerprint": frozen.fingerprint, "provenance": provenance}
            )
            self.store.event(
                run_id,
                "client_session_started",
                {
                    "provider_calls": 0,
                    "time_accounting": "Server operations only; client thinking time is excluded",
                },
            )
            self.store.update_run(
                run_id, status="awaiting_client", elapsed_seconds=time.monotonic() - started
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        # Polling and cancellation must remain available while SPICE owns the
        # mutation lock. Interrupted operations recover on the next mutation.
        return self._record(run_id)

    @staticmethod
    def _verification_reserve(seconds: float) -> float:
        return min(20.0, seconds * 0.2)

    def propose_candidates(self, run_id: str, limit: int = 6) -> list[Circuit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("Candidate limit must be between 1 and 20")
        with self._locked(run_id) as record:
            self._active(record)
            count, seconds = self._limits(record)
            if record["elapsed_seconds"] >= seconds:
                self._report(record, status="budget_exhausted", reason="time_budget")
                return []
            with self._operation(record):
                known = {_physical(Circuit.model_validate(row["circuit"])) for row in record["candidates"]}
                remaining = max(0, count - 1 - len(record["candidates"]))
                if seconds - record["elapsed_seconds"] <= self._verification_reserve(seconds):
                    return []
                return [
                    c
                    for c in generate_candidates(normalize_spec(record["spec"]))
                    if _physical(c) not in known
                ][: min(limit, remaining)]

    def _canonical(self, record: dict[str, Any], circuit: Circuit | dict[str, Any]) -> Circuit:
        # Revalidate even a model instance, since model_copy/model_construct do
        # not validate updates. None of the client's commentary enters evidence.
        candidate = Circuit.model_validate(
            circuit.model_dump(mode="json") if isinstance(circuit, Circuit) else circuit
        )
        spec = normalize_spec(record["spec"])
        if candidate.spec is not None and candidate.spec.fingerprint != spec.fingerprint:
            raise ValueError("Candidate specification differs from the frozen specification")

        def skeleton(item: Circuit) -> dict[str, Any]:
            data = item.model_dump(mode="json")
            return {
                "topology": item.topology,
                "input_node": item.input_node,
                "output_node": item.output_node,
                "components": [
                    {key: value for key, value in part.items() if key != "value"}
                    for part in data["components"]
                ],
            }

        matching = next(
            (item for item in generate_candidates(spec) if skeleton(item) == skeleton(candidate)), None
        )
        if matching is None:
            raise ValueError(
                "Candidate wiring, devices and packages must match an eligible analytical template"
            )
        for part, original in zip(candidate.components, matching.components, strict=True):
            if part.kind in {"R", "C"}:
                series = E24 if part.kind == "R" else E12
                if not math.isclose(part.value, standard_value(part.value, series), rel_tol=1e-12):
                    raise ValueError("Resistors must use E24 and capacitors E12 standard values")
            elif part.value != original.value:
                raise ValueError("Only resistor and capacitor values may be adjusted")
        return matching.model_copy(update={"components": candidate.components})

    def _evaluate(
        self, record: dict[str, Any], circuit: Circuit, *, purpose: str, candidate_id: str | None = None
    ) -> dict[str, Any]:
        run_id = record["run_id"]
        _, seconds = self._limits(record)
        remaining = seconds - record["elapsed_seconds"]
        if purpose == "search":
            remaining -= self._verification_reserve(seconds)
        candidate_id = candidate_id or self.store.add_candidate(run_id, circuit, purpose=purpose)
        try:
            simulator = NgSpiceSimulator(
                timeout_s=min(20.0, remaining), artifact_root=self._path(run_id) / "simulations"
            )
            measured = simulator.evaluate(circuit, normalize_spec(record["spec"]))
        except (ValueError, OSError, RuntimeError, TypeError) as exc:
            measured = SimulationResult(
                status="failed",
                errors=[safe_error(exc)],
                spec_fingerprint=normalize_spec(record["spec"]).fingerprint,
            )
        numeric = score_result(normalize_spec(record["spec"]), circuit, measured)
        self.store.save_evaluation(candidate_id, measured, numeric)
        if purpose == "verification":
            self.store.set_candidate_verified(
                candidate_id,
                numeric.valid and numeric.feasible and not self._record(run_id)["cancel_requested"],
            )
        self.store.event(
            run_id,
            "client_evaluation",
            {"candidate_id": candidate_id, "purpose": purpose, "simulation_status": measured.status},
        )
        return self.store.get_candidate(candidate_id)

    def evaluate_candidate(self, run_id: str, circuit: Circuit | dict[str, Any]) -> dict[str, Any]:
        with self._locked(run_id) as record:
            self._active(record)
            canonical = self._canonical(record, circuit)
            count, seconds = self._limits(record)
            if record["elapsed_seconds"] >= seconds:
                self._report(record, status="budget_exhausted", reason="time_budget")
                raise ValueError("Run server-operation time budget is exhausted")
            if seconds - record["elapsed_seconds"] <= self._verification_reserve(seconds):
                raise ValueError(
                    "Search time budget exhausted; finalize to use the reserved verification time"
                )
            if any(row["purpose"] == "verification" for row in record["candidates"]):
                raise ValueError("Fresh verification has already started; finalize this run")
            pending = next(
                (
                    row
                    for row in record["candidates"]
                    if row["purpose"] == "search"
                    and row["score"] is None
                    and _physical(Circuit.model_validate(row["circuit"])) == _physical(canonical)
                ),
                None,
            )
            if len(record["candidates"]) >= count - 1 and pending is None:
                raise ValueError(
                    "Search candidate budget exhausted; finalize to use the reserved verification"
                )
            if pending is None and any(
                _physical(Circuit.model_validate(row["circuit"])) == _physical(canonical)
                for row in record["candidates"]
            ):
                raise ValueError("This physical candidate was already evaluated; inspect its saved evidence")
            with self._operation(record):
                result = self._evaluate(
                    record,
                    canonical,
                    purpose="search",
                    candidate_id=pending["candidate_id"] if pending else None,
                )
            if self._record(run_id)["cancel_requested"]:
                self._report(self._record(run_id), status="cancelled", reason="cancelled")
            if self._record(run_id)["elapsed_seconds"] >= seconds:
                self._report(self._record(run_id), status="budget_exhausted", reason="time_budget")
            return result

    def finalize_design(self, run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        with self._locked(run_id) as record:
            if record["status"] in _TERMINAL:
                return record
            self._active(record)
            rows = [row for row in record["candidates"] if row["purpose"] == "search" and row["score"]]
            if not rows:
                raise ValueError("Evaluate at least one candidate before finalizing")
            best = min(rows, key=lambda row: tuple(row["score"]["rank_key"]))
            selected = (
                best
                if candidate_id is None
                else next((row for row in rows if row["candidate_id"] == candidate_id), None)
            )
            if selected is None:
                raise ValueError("Selected candidate must be a saved search candidate belonging to this run")
            # Passing checks always dominates package count, including explicit
            # client selections. Numerical ranking cannot be overridden.
            if tuple(selected["score"]["rank_key"]) != tuple(best["score"]["rank_key"]):
                raise ValueError("Selected candidate must have the best deterministic rank in this run")
            count, seconds = self._limits(record)
            previous = next((row for row in record["candidates"] if row["purpose"] == "verification"), None)
            if previous and previous["score"] is not None:
                self._report(
                    record,
                    status="completed",
                    reason="client_finalized",
                    selected=selected,
                    verification=previous,
                )
            elif record["elapsed_seconds"] >= seconds or (
                len(record["candidates"]) >= count and previous is None
            ):
                self._report(
                    record,
                    status="budget_exhausted",
                    reason="time_budget" if record["elapsed_seconds"] >= seconds else "candidate_budget",
                    selected=selected,
                )
            else:
                with self._operation(record):
                    verification = self._evaluate(
                        record,
                        Circuit.model_validate(selected["circuit"]),
                        purpose="verification",
                        candidate_id=previous["candidate_id"] if previous else None,
                    )
                    self._report(
                        self._record(run_id),
                        status="completed",
                        reason="client_finalized",
                        selected=selected,
                        verification=verification,
                    )
        return self.get_run(run_id)

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        self._record(run_id)
        # An atomic conditional write prevents a stale cancellation from
        # replacing completion, and does not wait for the SPICE operation lock.
        with self.store.connection() as db:
            changed = db.execute(
                "UPDATE runs SET cancel_requested=1,status='cancelled',updated_at=? "
                "WHERE run_id=? AND status NOT IN ('completed','failed','cancelled','budget_exhausted')",
                (utcnow(), run_id),
            ).rowcount
        if changed:
            self.store.event(run_id, "cancel_requested")
            try:
                with self._locked(run_id) as record:
                    self._report(record, status="cancelled", reason="cancelled")
            except ValueError as exc:
                if "already active" not in str(exc):
                    raise
        return self.get_run(run_id)

    def _report(
        self,
        record: dict[str, Any],
        *,
        status: str,
        reason: str,
        selected: dict[str, Any] | None = None,
        verification: dict[str, Any] | None = None,
    ) -> None:
        # Serialize the final cancellation check and status write. A request
        # committed before this transaction always wins over verification.
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            cancelled = db.execute(
                "SELECT cancel_requested FROM runs WHERE run_id=?", (record["run_id"],)
            ).fetchone()[0]
            if cancelled:
                status, reason = "cancelled", "cancelled"
            self._write_report(
                record, status=status, reason=reason, selected=selected, verification=verification, db=db
            )

    def _write_report(
        self,
        record: dict[str, Any],
        *,
        status: str,
        reason: str,
        selected: dict[str, Any] | None,
        verification: dict[str, Any] | None,
        db: Any,
    ) -> None:
        run_id = record["run_id"]
        rows = [row for row in record["candidates"] if row["purpose"] == "search" and row["score"]]
        best = selected or (min(rows, key=lambda row: tuple(row["score"]["rank_key"])) if rows else None)
        verified = bool(
            verification
            and verification["verified"]
            and verification["score"]
            and verification["score"]["valid"]
            and verification["score"]["feasible"]
            and status == "completed"
        )
        evidence = verification or best
        checkpoint = self.store.latest_checkpoint(run_id)
        provenance = checkpoint["state"].get("provenance", []) if checkpoint else []
        directory = self._path(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        if best:
            circuit = Circuit.model_validate(best["circuit"])
            files = {
                "circuit_json": ("circuit.json", encoded(best["circuit"])),
                "bom_json": ("bom.json", encoded(export_bom(circuit))),
                "schematic_text": ("schematic.txt", render_schematic(circuit)),
            }
            for key, (filename, content) in files.items():
                (directory / filename).write_text(content, encoding="utf-8")
                artifacts[key] = str(directory / filename)
        report = "\n".join(
            [
                "# Client-driven circuit design run",
                "",
                f"Run: {run_id}",
                "The connected coding agent drives this session. The harness makes no provider API calls.",
                "This is not deterministic offline search; the host client selects candidates.",
                f"Stop reason: {reason}",
                f"Evaluations: {len(record['candidates'])}/{self._limits(record)[0]}",
                "Time budget counts server operations only; client thinking time is excluded.",
                f"Fresh verification passed: {verified}",
                "Verified feasible design established."
                if verified
                else "No verified feasible design was established.",
                "",
                (
                    "This is the best evaluated candidate in the supported analytical template families; "
                    "the search does not prove a global minimum part count."
                ),
                "Tolerance corners are sensitivity screening, not exhaustive worst-case analysis.",
                "",
                "## Reference provenance",
                *[
                    f"- {note['title']}: {note['source']} (SHA-256 {note['checksum']}; page {note['page']})"
                    for note in provenance
                ],
                "",
                "## Measured evidence",
                "```json",
                json.dumps(evidence["simulation"] if evidence else None, indent=2),
                "```",
                "",
            ]
        )
        (directory / "report.md").write_text(report, encoding="utf-8")
        artifacts["report"] = str(directory / "report.md")
        result = {
            "best_candidate_id": best["candidate_id"] if best else None,
            "verification_candidate_id": verification["candidate_id"] if verification else None,
            "best": best,
            "circuit": best["circuit"] if best else None,
            "simulation": evidence["simulation"] if evidence else None,
            "score": evidence["score"] if evidence else None,
            "verification": verification,
            "verified": verified,
            "feasible": verified,
            "candidate_feasible": bool(best and best["score"]["feasible"]),
            "execution_mode": "client",
            "offline": False,
            "agents_called": False,
            "stop_reason": reason,
            "report": report,
            "artifacts": artifacts,
            "reviews": [],
            "provenance": provenance,
        }
        db.execute(
            "UPDATE runs SET status=?,result=?,updated_at=? WHERE run_id=?",
            (status, encoded(result), utcnow(), run_id),
        )
