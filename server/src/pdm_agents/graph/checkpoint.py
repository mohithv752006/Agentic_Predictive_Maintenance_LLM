"""Persistence: LangGraph SQLite checkpointer (keyed by run id) and a run/feedback store."""

from __future__ import annotations

import inspect
import json
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from pdm_agents import schemas
from pdm_agents.config import AppConfig
from pdm_agents.schemas import EscalationItem, FeedbackRecord, PredictResponse


def _schema_types() -> list[tuple[str, str]]:
    """Every Pydantic contract, allow-listed for checkpoint deserialization."""
    return [(schemas.__name__, name) for name, obj in vars(schemas).items()
            if inspect.isclass(obj) and issubclass(obj, BaseModel)
            and obj.__module__ == schemas.__name__]


def make_checkpointer(cfg: AppConfig, path: Path | None = None) -> SqliteSaver:
    path = path or cfg.paths.runtime / "checkpoints.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_schema_types())
    return SqliteSaver(conn, serde=serde)


class RunStore:
    """Finished runs and human feedback (for the escalation queue and later analysis)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, created_at TEXT, serial_number TEXT, model TEXT,
                    as_of TEXT, action TEXT, escalated INTEGER, response TEXT);
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, run_id TEXT, created_at TEXT, record TEXT);
                CREATE INDEX IF NOT EXISTS runs_serial ON runs(serial_number);
                """
            )

    @classmethod
    def for_config(cls, cfg: AppConfig) -> RunStore:
        return cls(cfg.paths.runtime / "runs.sqlite")

    def save(self, resp: PredictResponse) -> None:
        ctx = resp.context_summary
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
                (resp.run_id, resp.trace[0].ts.isoformat() if resp.trace else "",
                 resp.request.serial_number, ctx.get("model", ""), str(ctx.get("as_of_date", "")),
                 resp.decision.action if resp.decision else None,
                 int(bool(resp.decision and resp.decision.escalated)), resp.model_dump_json()),
            )
            self._conn.commit()

    def get(self, run_id: str) -> PredictResponse | None:
        with self._lock:
            row = self._conn.execute("SELECT response FROM runs WHERE run_id=?",
                                     (run_id,)).fetchone()
        return PredictResponse.model_validate_json(row[0]) if row else None

    def latest_by_serial(self) -> dict[str, dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT serial_number, action, response FROM runs r WHERE created_at = "
                "(SELECT max(created_at) FROM runs WHERE serial_number = r.serial_number)"
            ).fetchall()
        return {s: {"action": a, "response": r} for s, a, r in rows}

    def add_feedback(self, record: FeedbackRecord) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO feedback VALUES (?,?,?,?)",
                               (record.id, record.run_id, record.created_at.isoformat(),
                                record.model_dump_json()))
            self._conn.commit()

    def escalations(self, limit: int = 200) -> list[EscalationItem]:
        with self._lock:
            runs = self._conn.execute(
                "SELECT response FROM runs WHERE escalated=1 ORDER BY created_at DESC LIMIT ?",
                (limit,)).fetchall()
            fb = self._conn.execute("SELECT run_id, record FROM feedback "
                                    "ORDER BY created_at").fetchall()
        feedback = {rid: FeedbackRecord.model_validate_json(rec) for rid, rec in fb}
        items = []
        for (raw,) in runs:
            r = PredictResponse.model_validate(json.loads(raw))
            items.append(EscalationItem(
                run_id=r.run_id, serial_number=r.request.serial_number,
                model=r.context_summary.get("model", ""),
                as_of_date=r.context_summary["as_of_date"],
                action=r.decision.action if r.decision else "",
                reasons=r.decision.reasons if r.decision else [],
                probabilities=r.fused.probabilities if r.fused else {},
                created_at=r.trace[0].ts if r.trace else None,
                feedback=feedback.get(r.run_id)))
        return items
