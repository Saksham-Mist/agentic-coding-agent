"""Postgres run store — step 3.6.

Every trace event is written as a `steps` row together with the
`runs` row update (status, step and token counters), in one SQL
statement: atomic on its own, and one round trip instead of four
(Neon is ~290ms away from the dev machine). Postgres is the source of
truth for resuming a crashed run (the design notes §5, §7).

The connection is long-lived, and Neon closes idle connections (its
free tier suspends the database after a few minutes of inactivity).
Every statement goes through _execute(), which reconnects when the
connection is closed or broken and retries the statement once. That is
safe for the common case, a connection dropped while idle, where the
failed statement never reached the server. (If a connection died in
the middle of a statement that had already committed, the retry could
apply it twice; that window is tiny and accepted for v1.)
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

_LLM_EVENTS = {"plan", "plan_error", "worker_code", "worker_error"}

logger = logging.getLogger("agentic_coding_agent.store")


class RunStore:
    def __init__(self, url: str | None = None):
        url = url or os.environ.get("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL not set")
        self._url = url
        self._connect()

    def _connect(self) -> None:
        # autocommit: each statement below is its own atomic transaction
        self._conn = psycopg.connect(self._url, autocommit=True)

    def _usable(self) -> bool:
        return not (self._conn.closed or self._conn.broken)

    def _execute(self, query, params=None):
        """conn.execute with reconnect-and-retry-once on a dropped connection."""
        if not self._usable():
            self._connect()
        try:
            return self._conn.execute(query, params)
        except psycopg.OperationalError:
            if self._usable():
                raise  # a real query error, not a dead connection
            logger.warning("database connection was dropped; reconnecting and retrying once")
            self._connect()
            return self._conn.execute(query, params)

    def close(self) -> None:
        self._conn.close()

    def create_run(self, run_id: str, task_id: str, source: str = "cli") -> None:
        """Mark a run as running, creating its row if needed. An existing
        row (e.g. queued by POST /runs) keeps its own source."""
        self._execute(
            """
            INSERT INTO runs (run_id, task_id, status, source) VALUES (%s, %s, 'running', %s)
            ON CONFLICT (run_id) DO UPDATE SET status = 'running', updated_at = now()
            """,
            (run_id, task_id, source),
        )

    def mark_running(self, run_id: str) -> None:
        self._execute(
            "UPDATE runs SET status = 'running', updated_at = now() WHERE run_id = %s", (run_id,)
        )

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        kind = event["kind"]
        payload = json.loads(json.dumps(event, default=str))
        usage = event.get("usage") or {}
        is_llm = kind in _LLM_EVENTS
        is_finish = kind == "finish"
        self._execute(
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
        row = self._execute(
            "SELECT run_id, task_id, status, steps_taken, tokens_used, result "
            "FROM runs WHERE run_id = %s",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("run_id", "task_id", "status", "steps_taken", "tokens_used", "result")
        out = dict(zip(keys, row))
        out["run_id"] = str(out["run_id"])
        return out

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self._execute(
            "SELECT payload FROM steps WHERE run_id = %s AND payload IS NOT NULL "
            "ORDER BY step_number",
            (run_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def create_queued(self, run_id: str, task_id: str, source: str = "api") -> None:
        """POST /runs: the run exists (status queued) before any work starts."""
        self._execute(
            "INSERT INTO runs (run_id, task_id, status, source) VALUES (%s, %s, 'queued', %s)",
            (run_id, task_id, source),
        )

    def claim_queued(self, run_id: str) -> bool:
        """Step 4.5: atomically move a run from queued to running. Only
        one caller can win, so a queued run is never started twice, even
        by two server processes."""
        row = self._execute(
            "UPDATE runs SET status = 'running', updated_at = now() "
            "WHERE run_id = %s AND status = 'queued' RETURNING run_id",
            (run_id,),
        ).fetchone()
        return row is not None

    def mark_error(self, run_id: str, reason: str) -> None:
        """Last-resort status for a run whose worker crashed outright."""
        self._execute(
            "UPDATE runs SET status = 'error', result = %s, updated_at = now() WHERE run_id = %s",
            (Jsonb({"status": "error", "reason": reason}), run_id),
        )

    def list_unfinished(self) -> list[dict[str, Any]]:
        """Runs a restarted server must pick up again, oldest first."""
        rows = self._execute(
            "SELECT run_id, task_id, status FROM runs WHERE status IN ('queued', 'running') "
            "ORDER BY created_at"
        ).fetchall()
        return [{"run_id": str(r[0]), "task_id": r[1], "status": r[2]} for r in rows]

    def stats(self) -> dict[str, Any]:
        """Aggregates for GET /stats.

        `public` is the headline: runs submitted by public visitors through
        POST /runs (source "api") only. Every source, including the owner's
        cli/eval runs and tests, is still reported separately in
        `by_source`. (An earlier "overall_excluding_test" figure mixed the
        owner's eval runs into the headline, so it said 106/106 before a
        single visitor had tried the demo.)"""
        rows = self._execute(
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

        by_source = {r[0]: shape(*r[1:]) for r in rows}
        return {
            "public": by_source.get("api", shape(0, 0, 0, None, None, None)),
            "by_source": by_source,
        }

    def consume_rate_limit(self, ip: str, limit: int) -> int | None:
        """Step 4.3: count one run against `ip` for today (UTC).

        Returns the new count, or None if `ip` already used `limit` runs
        today. One statement, so concurrent requests can't both take
        the last slot."""
        row = self._execute(
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
        self._execute("DELETE FROM rate_limits WHERE ip = %s", (ip,))

    # --- step 5.2: eval batches ---

    def create_eval_batch(self, label: str | None, command: str, git_commit: str | None,
                          model: str | None, repeats: int, task_ids: list[str]) -> str:
        row = self._execute(
            """
            INSERT INTO eval_batches (label, command, git_commit, model, repeats, task_ids)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING batch_id
            """,
            (label, command, git_commit, model, repeats, Jsonb(task_ids)),
        ).fetchone()
        return str(row[0])

    def add_eval_run(self, batch_id: str, task_id: str, repeat: int,
                     run_id: str | None, error: str | None = None) -> None:
        self._execute(
            "INSERT INTO eval_runs (batch_id, task_id, repeat, run_id, error) VALUES (%s, %s, %s, %s, %s)",
            (batch_id, task_id, repeat, run_id, error),
        )

    def finish_eval_batch(self, batch_id: str) -> None:
        self._execute("UPDATE eval_batches SET finished_at = now() WHERE batch_id = %s", (batch_id,))

    def eval_batch(self, batch_id: str) -> dict[str, Any] | None:
        row = self._execute(
            "SELECT batch_id, label, command, git_commit, model, repeats, task_ids, started_at, finished_at "
            "FROM eval_batches WHERE batch_id = %s",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        keys = ("batch_id", "label", "command", "git_commit", "model", "repeats", "task_ids",
                "started_at", "finished_at")
        return dict(zip(keys, row))

    def eval_rows(self, batch_id: str) -> list[dict[str, Any]]:
        """One row per (task, repeat) with the run's outcome. Wall clock is
        first to last step event, so time spent queued isn't counted."""
        rows = self._execute(
            """
            SELECT e.task_id, e.repeat, e.run_id, e.error, r.status, r.steps_taken, r.tokens_used,
                   extract(epoch FROM (SELECT max(created_at) - min(created_at)
                                       FROM steps s WHERE s.run_id = e.run_id)) AS seconds
            FROM eval_runs e LEFT JOIN runs r ON r.run_id = e.run_id
            WHERE e.batch_id = %s ORDER BY e.task_id, e.repeat
            """,
            (batch_id,),
        ).fetchall()
        keys = ("task_id", "repeat", "run_id", "error", "status", "steps", "tokens", "seconds")
        out = []
        for r in rows:
            d = dict(zip(keys, r))
            d["run_id"] = str(d["run_id"]) if d["run_id"] else None
            d["seconds"] = float(d["seconds"]) if d["seconds"] is not None else None
            out.append(d)
        return out

    def delete_eval_batch(self, batch_id: str) -> None:
        """For tests; eval_runs cascade (runs are deleted separately)."""
        self._execute("DELETE FROM eval_batches WHERE batch_id = %s", (batch_id,))

    def delete_run(self, run_id: str) -> None:
        """For tests/fixtures; steps cascade."""
        self._execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
