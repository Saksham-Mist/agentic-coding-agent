"""Postgres run store — step 3.6.

Every trace event is written as a `steps` row together with the
`runs` row update (status, step and token counters), in one SQL
statement: atomic on its own, and one round trip instead of four
(Neon is ~290ms away from the dev machine). Postgres is the source of
truth for resuming a crashed run (the design notes §5, §7).
"""
from __future__ import annotations

import json
import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

_LLM_EVENTS = {"plan", "plan_error", "worker_code", "worker_error"}


class RunStore:
    def __init__(self, url: str | None = None):
        url = url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL not set")
        # autocommit: each statement below is its own atomic transaction
        self._conn = psycopg.connect(url, autocommit=True)

    def close(self) -> None:
        self._conn.close()

    def create_run(self, run_id: str, task_id: str, source: str = "cli") -> None:
        """Mark a run as running, creating its row if needed. An existing
        row (e.g. queued by POST /runs) keeps its own source."""
        self._conn.execute(
            """
            INSERT INTO runs (run_id, task_id, status, source) VALUES (%s, %s, 'running', %s)
            ON CONFLICT (run_id) DO UPDATE SET status = 'running', updated_at = now()
            """,
            (run_id, task_id, source),
        )

    def mark_running(self, run_id: str) -> None:
        self._conn.execute(
            "UPDATE runs SET status = 'running', updated_at = now() WHERE run_id = %s", (run_id,)
        )

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        kind = event["kind"]
        payload = json.loads(json.dumps(event, default=str))
        usage = event.get("usage") or {}
        is_llm = kind in _LLM_EVENTS
        is_finish = kind == "finish"
        self._conn.execute(
            """
            WITH ins AS (
                INSERT INTO steps (run_id, step_number, kind, payload, subtask, code,
                                   execution_result, verifier_decision)
                VALUES (%(run_id)s, %(seq)s, %(kind)s, %(payload)s, %(subtask)s, %(code)s,
                        %(exec_result)s, %(decision)s)
                RETURNING run_id
            )
            UPDATE runs SET
                steps_taken = steps_taken + %(step_inc)s,
                tokens_used = tokens_used + %(token_inc)s,
                status = COALESCE(%(status)s, status),
                result = COALESCE(%(result)s, result),
                updated_at = now()
            WHERE run_id = (SELECT run_id FROM ins)
            """,
            {
                "run_id": run_id, "seq": event["seq"], "kind": kind, "payload": Jsonb(payload),
                "subtask": str(event["subtask"]) if "subtask" in event else None,
                "code": event.get("code"),
                "exec_result": Jsonb(payload["result"]) if kind == "execute" and "result" in payload else None,
                "decision": event.get("decision") if kind == "verify" else None,
                "step_inc": 1 if is_llm else 0,
                "token_inc": int(usage.get("total_tokens", 0)) if is_llm else 0,
                "status": event["status"] if is_finish else None,
                "result": Jsonb(payload) if is_finish else None,
            },
        )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT run_id, task_id, status, steps_taken, tokens_used, result "
                "FROM runs WHERE run_id = %s",
                (run_id,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        keys = ("run_id", "task_id", "status", "steps_taken", "tokens_used", "result")
        out = dict(zip(keys, row))
        out["run_id"] = str(out["run_id"])
        return out

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT payload FROM steps WHERE run_id = %s AND payload IS NOT NULL "
                "ORDER BY step_number",
                (run_id,),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def create_queued(self, run_id: str, task_id: str, source: str = "api") -> None:
        """POST /runs: the run exists (status queued) before any work starts."""
        self._conn.execute(
            "INSERT INTO runs (run_id, task_id, status, source) VALUES (%s, %s, 'queued', %s)",
            (run_id, task_id, source),
        )

    def claim_queued(self, run_id: str) -> bool:
        """Step 4.5: atomically move a run from queued to running. Only
        one caller can win, so a queued run is never started twice, even
        by two server processes."""
        row = self._conn.execute(
            "UPDATE runs SET status = 'running', updated_at = now() "
            "WHERE run_id = %s AND status = 'queued' RETURNING run_id",
            (run_id,),
        ).fetchone()
        return row is not None

    def mark_error(self, run_id: str, reason: str) -> None:
        """Last-resort status for a run whose worker crashed outright."""
        self._conn.execute(
            "UPDATE runs SET status = 'error', result = %s, updated_at = now() WHERE run_id = %s",
            (Jsonb({"status": "error", "reason": reason}), run_id),
        )

    def list_unfinished(self) -> list[dict[str, Any]]:
        """Runs a restarted server must pick up again, oldest first."""
        rows = self._conn.execute(
            "SELECT run_id, task_id, status FROM runs WHERE status IN ('queued', 'running') "
            "ORDER BY created_at"
        ).fetchall()
        return [{"run_id": str(r[0]), "task_id": r[1], "status": r[2]} for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregates for GET /stats, per source and overall (excluding test runs)."""
        rows = self._conn.execute(
            """
            SELECT source,
                   count(*) FILTER (WHERE status IN ('passed', 'incomplete', 'error')) AS finished,
                   count(*) FILTER (WHERE status = 'passed') AS passed,
                   count(*) FILTER (WHERE status IN ('queued', 'running')) AS in_progress,
                   avg(steps_taken) FILTER (WHERE status = 'passed') AS avg_steps_passed,
                   avg(tokens_used) FILTER (WHERE status IN ('passed', 'incomplete', 'error')) AS avg_tokens,
                   avg(extract(epoch FROM updated_at - created_at))
                       FILTER (WHERE status IN ('passed', 'incomplete', 'error')) AS avg_seconds
            FROM runs GROUP BY source ORDER BY source
            """
        ).fetchall()
        overall = self._conn.execute(
            """
            SELECT count(*) FILTER (WHERE status IN ('passed', 'incomplete', 'error')),
                   count(*) FILTER (WHERE status = 'passed'),
                   count(*) FILTER (WHERE status IN ('queued', 'running')),
                   avg(steps_taken) FILTER (WHERE status = 'passed'),
                   avg(tokens_used) FILTER (WHERE status IN ('passed', 'incomplete', 'error')),
                   avg(extract(epoch FROM updated_at - created_at))
                       FILTER (WHERE status IN ('passed', 'incomplete', 'error'))
            FROM runs WHERE source <> 'test'
            """
        ).fetchone()

        def shape(finished, passed, in_progress, avg_steps, avg_tokens, avg_seconds):
            return {
                "finished_runs": finished,
                "passed": passed,
                "success_rate": round(passed / finished, 3) if finished else None,
                "in_progress": in_progress,
                "avg_steps_passed": round(float(avg_steps), 2) if avg_steps is not None else None,
                "avg_tokens_per_run": round(float(avg_tokens)) if avg_tokens is not None else None,
                "avg_wall_clock_seconds": round(float(avg_seconds), 1) if avg_seconds is not None else None,
            }

        return {
            "overall_excluding_test": shape(*overall),
            "by_source": {r[0]: shape(*r[1:]) for r in rows},
        }

    def consume_rate_limit(self, ip: str, limit: int) -> int | None:
        """Step 4.3: count one run against `ip` for today (UTC).

        Returns the new count, or None if `ip` already used `limit` runs
        today. One statement, so concurrent requests can't both take
        the last slot."""
        row = self._conn.execute(
            """
            INSERT INTO rate_limits (ip, day, count)
            VALUES (%(ip)s, (now() AT TIME ZONE 'utc')::date, 1)
            ON CONFLICT (ip, day) DO UPDATE SET count = rate_limits.count + 1
                WHERE rate_limits.count < %(limit)s
            RETURNING count
            """,
            {"ip": ip, "limit": limit},
        ).fetchone()
        return row[0] if row else None

    def delete_rate_limit(self, ip: str) -> None:
        """For tests."""
        self._conn.execute("DELETE FROM rate_limits WHERE ip = %s", (ip,))

    def delete_run(self, run_id: str) -> None:
        """For tests/fixtures; steps cascade."""
        self._conn.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
