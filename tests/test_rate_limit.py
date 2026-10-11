"""Step 4.3 done-when check: the 6th request from one IP in a day gets
429 + Retry-After, and rejected requests create no run and cost no
tokens. Runs against the real Postgres in DATABASE_URL; each test uses
its own documentation-range IP (203.0.113.0/24) and cleans up.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from agentic_coding_agent.api.app import seconds_until_utc_midnight

needs_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)


class CountingLLM:
    calls = 0

    def generate(self, prompt, system_instruction=None):
        CountingLLM.calls += 1
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 50}
        if "planning step" in (system_instruction or ""):
            return {"text": '{"approach": "a", "subtasks": ["s1"]}', "usage": usage}
        return {"text": "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```", "usage": usage}


def passing(code, language, cases, **kw):
    return {"status": "pass", "stdout": "", "stderr": "",
            "test_results": [{"input": c["input"], "expected": c["expected_output"],
                              "actual": c["expected_output"], "passed": True} for c in cases]}


def test_retry_after_counts_down_to_utc_midnight():
    assert seconds_until_utc_midnight(datetime(2026, 10, 9, 23, 59, 0, tzinfo=timezone.utc)) == 60
    assert seconds_until_utc_midnight(datetime(2026, 10, 9, 0, 0, 0, tzinfo=timezone.utc)) == 86400


@pytest.fixture
def make_client(tmp_path):
    """Builds API clients bound to a given caller IP; cleans up runs and
    rate-limit rows afterwards."""
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    opened, ips = [], []

    def build(ip, limit=5):
        ips.append(ip)
        store.delete_rate_limit(ip)
        app = create_app(llm_factory=CountingLLM, executor=passing, log_dir=tmp_path,
                         run_source="test", rate_limit=limit)
        c = TestClient(app, client=(ip, 50000))
        c.__enter__()
        opened.append(c)
        return c

    yield build
    for c in opened:
        c.__exit__(None, None, None)
    rows = store._conn.execute(
        "SELECT run_id FROM runs WHERE source = 'test' AND created_at > now() - interval '1 hour'"
    ).fetchall()
    for (run_id,) in rows:
        store.delete_run(str(run_id))
    for ip in ips:
        store.delete_rate_limit(ip)
    store.close()


def _test_run_count(store_conn):
    return store_conn.execute("SELECT count(*) FROM runs WHERE source = 'test'").fetchone()[0]


@needs_db
def test_sixth_request_gets_429_with_retry_after(make_client):
    from agentic_coding_agent.db.store import RunStore

    c = make_client("203.0.113.10", limit=5)
    for i in range(5):
        r = c.post("/runs", json={"task_id": "two_sum"})
        assert r.status_code == 202, (i, r.text)

    probe = RunStore()
    before = _test_run_count(probe._conn)

    r = c.post("/runs", json={"task_id": "two_sum"})
    assert r.status_code == 429
    retry_after = int(r.headers["Retry-After"])
    assert 0 < retry_after <= 86400
    detail = r.json()["detail"]
    assert detail["reason"] == "rate_limited"
    assert detail["limit"] == 5
    assert detail["retry_after_seconds"] == retry_after

    # the rejected request created no run (no-LLM-call is checked in
    # test_rate_limited_request_makes_no_llm_call, after earlier runs finish)
    assert _test_run_count(probe._conn) == before
    probe.close()


@needs_db
def test_limit_is_per_ip(make_client):
    a = make_client("203.0.113.20", limit=1)
    b = make_client("203.0.113.21", limit=1)
    assert a.post("/runs", json={"task_id": "two_sum"}).status_code == 202
    assert a.post("/runs", json={"task_id": "two_sum"}).status_code == 429
    assert b.post("/runs", json={"task_id": "two_sum"}).status_code == 202


@needs_db
def test_invalid_task_id_does_not_use_a_slot(make_client):
    c = make_client("203.0.113.30", limit=1)
    assert c.post("/runs", json={"task_id": "no_such_task"}).status_code == 422
    from agentic_coding_agent.agent.catalog import load_catalog, validate_entry
    not_ready = next(t for t, e in load_catalog().items() if validate_entry(e))
    assert c.post("/runs", json={"task_id": not_ready}).status_code == 422
    assert c.post("/runs", json={"task_id": "two_sum"}).status_code == 202


@needs_db
def test_rate_limited_request_makes_no_llm_call(make_client):
    c = make_client("203.0.113.40", limit=1)
    c.post("/runs", json={"task_id": "two_sum"})
    # wait for the allowed run to finish so its LLM calls are all counted
    import time
    from agentic_coding_agent.db.store import RunStore
    probe = RunStore()
    for _ in range(60):
        busy = probe._conn.execute(
            "SELECT count(*) FROM runs WHERE source='test' AND status IN ('queued','running')"
        ).fetchone()[0]
        if not busy:
            break
        time.sleep(0.5)
    probe.close()

    calls = CountingLLM.calls
    for _ in range(3):
        assert c.post("/runs", json={"task_id": "two_sum"}).status_code == 429
    time.sleep(1)
    assert CountingLLM.calls == calls


@needs_db
def test_concurrent_requests_cannot_exceed_limit():
    """The check-and-count is one SQL statement, so racing requests can't
    both take the last slot. Hits the store directly with 10 threads."""
    from agentic_coding_agent.db.store import RunStore

    ip = "203.0.113.50"
    setup = RunStore()
    setup.delete_rate_limit(ip)

    def attempt(_):
        s = RunStore()
        try:
            return s.consume_rate_limit(ip, 5)
        finally:
            s.close()

    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(attempt, range(10)))
    allowed = [r for r in results if r is not None]
    assert sorted(allowed) == [1, 2, 3, 4, 5]
    assert results.count(None) == 5
    setup.delete_rate_limit(ip)
    setup.close()
