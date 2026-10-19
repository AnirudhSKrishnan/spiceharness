"""Small transactional SQLite ledger for inspectable, resumable design runs."""
from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


def encoded(value: Any) -> str:
    return json.dumps(value, default=plain, ensure_ascii=False, allow_nan=False)


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    spec TEXT NOT NULL, options TEXT NOT NULL,
                    result TEXT, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    worker_pid INTEGER, elapsed_seconds REAL NOT NULL DEFAULT 0,
                    execution_started_at REAL
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    ordinal INTEGER NOT NULL, circuit TEXT NOT NULL, simulation TEXT, score TEXT,
                    verified INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                    purpose TEXT NOT NULL DEFAULT 'search',
                    UNIQUE(run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    type TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    data TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(run_id),
                    node TEXT NOT NULL, state TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS candidates_run ON candidates(run_id, ordinal);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id, id);
                CREATE INDEX IF NOT EXISTS checkpoint_run ON checkpoints(run_id, id);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(runs)")}
            if "execution_started_at" not in columns:
                db.execute("ALTER TABLE runs ADD COLUMN execution_started_at REAL")
            candidate_columns = {row["name"] for row in db.execute("PRAGMA table_info(candidates)")}
            if "purpose" not in candidate_columns:
                db.execute("ALTER TABLE candidates ADD COLUMN purpose TEXT NOT NULL DEFAULT 'search'")
                db.execute("UPDATE candidates SET purpose='verification' WHERE verified=1")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def create_run(self, spec: Any, *, options: dict[str, Any] | None = None,
                   run_id: str | None = None) -> str:
        run_id = run_id or uuid.uuid4().hex
        now = utcnow()
        with self.connection() as db:
            db.execute("INSERT INTO runs(run_id,status,spec,options,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                       (run_id, "queued", encoded(plain(spec)), encoded(options or {}), now, now))
        self.event(run_id, "created", {"offline": (options or {}).get("offline", False)})
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        allowed = {"status", "spec", "options", "result", "error", "cancel_requested",
                   "worker_pid", "elapsed_seconds", "execution_started_at"}
        if not fields or fields.keys() - allowed:
            raise ValueError("Unsupported run update fields")
        fields["updated_at"] = utcnow()
        for key in ("spec", "options", "result"):
            if key in fields and fields[key] is not None:
                fields[key] = encoded(plain(fields[key]))
        with self.connection() as db:
            cursor = db.execute(f"UPDATE runs SET {','.join(f'{key}=?' for key in fields)} WHERE run_id=?",
                                (*fields.values(), run_id))
            if cursor.rowcount == 0:
                raise KeyError(run_id)

    def claim_run(self, run_id: str, pid: int) -> bool:
        with self.connection() as db:
            import time
            cursor = db.execute("UPDATE runs SET status='running',worker_pid=?,updated_at=?,execution_started_at=? "
                                "WHERE run_id=? AND status='queued'", (pid, utcnow(), time.time(), run_id))
            return cursor.rowcount == 1

    def mark_interrupted(self, run_id: str, *, elapsed_seconds: float, error: str) -> bool:
        """Do not replace completion committed after the caller's stale read."""
        with self.connection() as db:
            cursor = db.execute("UPDATE runs SET status='interrupted',worker_pid=NULL,"
                                "elapsed_seconds=?,execution_started_at=NULL,error=?,updated_at=? "
                                "WHERE run_id=? AND status IN ('queued','running')",
                                (elapsed_seconds, error, utcnow(), run_id))
            return cursor.rowcount == 1

    def cancel_interrupted(self, run_id: str) -> bool:
        """Finish a cancellation for a run that has no live worker to observe it."""
        with self.connection() as db:
            cursor = db.execute("UPDATE runs SET status='cancelled',cancel_requested=1,worker_pid=NULL,"
                                "execution_started_at=NULL,error=NULL,updated_at=? "
                                "WHERE run_id=? AND status='interrupted'", (utcnow(), run_id))
            return cursor.rowcount == 1

    def get_run(self, run_id: str, *, include_details: bool = True) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown run: {run_id}")
            result = dict(row)
            for key in ("spec", "options", "result"):
                result[key] = json.loads(result[key]) if result[key] is not None else None
            result["cancel_requested"] = bool(result["cancel_requested"])
            result["offline"] = result["options"].get("offline", False)
            result["limits"] = result["options"].get("settings", {}).get("limits", {})
            if include_details:
                result["events"] = self._entries(db, "events", run_id)
                result["usage"] = self._entries(db, "usage", run_id)
                checkpoints = db.execute("SELECT id,node,created_at FROM checkpoints WHERE run_id=? ORDER BY id",
                                         (run_id,)).fetchall()
                result["checkpoints"] = [dict(item) for item in checkpoints]
        if include_details:
            result["candidates"] = self.list_candidates(run_id)
            result["leaderboard"] = sorted(
                [item for item in result["candidates"] if item["score"] is not None],
                key=lambda item: tuple(item["score"].get("rank_key", [float("inf")])))
        return result

    @staticmethod
    def _entries(db: sqlite3.Connection, table: str, run_id: str) -> list[dict[str, Any]]:
        rows = db.execute(f"SELECT * FROM {table} WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def list_runs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT run_id FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_run(row["run_id"], include_details=False) for row in rows]

    def event(self, run_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO events(run_id,type,data,created_at) VALUES(?,?,?,?)",
                       (run_id, event_type, encoded(data or {}), utcnow()))

    def record_usage(self, run_id: str, data: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO usage(run_id,data,created_at) VALUES(?,?,?)",
                       (run_id, encoded(data), utcnow()))

    def checkpoint(self, run_id: str, node: str, state: dict[str, Any]) -> None:
        with self.connection() as db:
            db.execute("INSERT INTO checkpoints(run_id,node,state,created_at) VALUES(?,?,?,?)",
                       (run_id, node, encoded(state), utcnow()))

    def latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as db:
            row = db.execute("SELECT node,state,created_at FROM checkpoints WHERE run_id=? ORDER BY id DESC LIMIT 1",
                             (run_id,)).fetchone()
        return {**dict(row), "state": json.loads(row["state"])} if row else None

    def add_candidate(self, run_id: str, circuit: Any, *, ordinal: int | None = None,
                      verified: bool = False, purpose: str = "search") -> str:
        if purpose not in {"search", "verification"}:
            raise ValueError("Candidate purpose must be search or verification")
        candidate_id = uuid.uuid4().hex
        with self.connection() as db:
            if ordinal is None:
                ordinal = db.execute("SELECT COALESCE(MAX(ordinal),0)+1 FROM candidates WHERE run_id=?",
                                     (run_id,)).fetchone()[0]
            db.execute("INSERT INTO candidates(candidate_id,run_id,ordinal,circuit,verified,created_at,purpose) VALUES(?,?,?,?,?,?,?)",
                       (candidate_id, run_id, ordinal, encoded(plain(circuit)), int(verified), utcnow(), purpose))
        return candidate_id

    def save_evaluation(self, candidate_id: str, simulation: Any, score: Any | None = None) -> None:
        with self.connection() as db:
            db.execute("UPDATE candidates SET simulation=?,score=? WHERE candidate_id=?",
                       (encoded(plain(simulation)), encoded(plain(score)) if score is not None else None,
                        candidate_id))

    def set_candidate_verified(self, candidate_id: str, passed: bool) -> None:
        with self.connection() as db:
            db.execute("UPDATE candidates SET verified=? WHERE candidate_id=?", (int(passed), candidate_id))

    @staticmethod
    def _candidate(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("circuit", "simulation", "score"):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        result["verified"] = bool(result["verified"])
        return result

    def get_candidate(self, candidate_id: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM candidates WHERE candidate_id=?", (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown candidate: {candidate_id}")
        return self._candidate(row)

    def list_candidates(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM candidates WHERE run_id=? ORDER BY ordinal", (run_id,)).fetchall()
        return [self._candidate(row) for row in rows]

    def request_cancel(self, run_id: str) -> dict[str, Any]:
        self.update_run(run_id, cancel_requested=True)
        self.event(run_id, "cancel_requested")
        return self.get_run(run_id)
