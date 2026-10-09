"""Step 4.1: POST /runs, GET /runs/{id}, GET /stats.

In-process API with a fake LLM and fake executor (no quota, no
Docker), against the real Postgres in DATABASE_URL. Runs are tagged
source='test' and deleted afterwards. The real curl round trip against
a running uvicorn is checked separately.
"""
import json
import os
import time
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)

PLAN = '{"approach": "a", "subtasks": ["s1"]}'
CODE = "```python\ndef two_sum(nums, target):\n    print(nums)\n    return [0, 1]\n```"


class FakeLLM:
    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 50}
        text = PLAN if "planning step" in (system_instruction or "") else CODE
        return {"text": text, "usage": usage}


def passing(code, language, cases, **kw):
    # Mimic the real harness: per-case results plus captured stdout
    # that contains test inputs (the code above prints nums).
    return {
        "status": "pass",
        "stdout": "\n".join(json.dumps(c["input"]["nums"]) for c in cases),
        "stderr": "",
        "test_results": [{"input": c["input"], "expected": c["expected_output"],
                          "actual": c["expected_output"], "passed": True} for c in cases],
    }


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path, run_source="test")
    cleanup = RunStore()
    created = []
    with TestClient(app) as c:
        c.created = created
        yield c
    for run_id in created:
        cleanup.delete_run(run_id)
    cleanup.close()


def _wait(client, run_id, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.5)
    raise AssertionError(f"run {run_id} did not finish: {body['status']}")


def test_post_returns_202_queued_immediately(client):
    r = client.post("/runs", json={"task_id": "two_sum"})
    assert r.status_code == 202
    body = r.json()
    client.created.append(body["run_id"])
    assert body["status"] == "queued"


def test_round_trip_runs_to_passed(client):
    run_id = client.post("/runs", json={"task_id": "two_sum"}).json()["run_id"]
    client.created.append(run_id)
    body = _wait(client, run_id)
    assert body["status"] == "passed"
    assert body["steps"] == 2
    assert body["tokens"] == 100
    kinds = [e["kind"] for e in body["trace"]]
    assert kinds == ["plan", "worker_code", "execute", "verify", "finish"]


def test_public_trace_never_contains_hidden_cases(client):
    from agentic_coding_agent.agent.catalog import get_task

    run_id = client.post("/runs", json={"task_id": "two_sum"}).json()["run_id"]
    client.created.append(run_id)
    body = _wait(client, run_id)
    text = json.dumps(body)

    task = get_task("two_sum")
    hidden = [c for c in task["test_cases"] if not c["visible"]]
    visible = [c for c in task["test_cases"] if c["visible"]]
    assert hidden
    for c in hidden:
        assert json.dumps(c["input"]["nums"]) not in text, "hidden input leaked"
    for c in visible:
        assert json.dumps(c["input"]["nums"]) in text  # visible results are still shown

    execute = next(e for e in body["trace"] if e["kind"] == "execute")
    assert "stdout" not in execute["result"]
    assert "stderr" not in execute["result"]
    assert len(execute["result"]["visible_test_results"]) == len(visible)


def test_stats_counts_test_runs_separately(client):
    run_id = client.post("/runs", json={"task_id": "two_sum"}).json()["run_id"]
    client.created.append(run_id)
    _wait(client, run_id)
    stats = client.get("/stats").json()
    assert stats["by_source"]["test"]["passed"] >= 1
    assert stats["by_source"]["test"]["success_rate"] is not None
    assert "overall_excluding_test" in stats


def test_unknown_and_malformed_run_ids_are_404(client):
    assert client.get("/runs/00000000-0000-0000-0000-00000000ffff").status_code == 404
    assert client.get("/runs/not-a-uuid").status_code == 404


def test_queued_run_is_recovered_on_startup(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    run_id = "00000000-0000-0000-0000-00000000c001"
    store.create_queued(run_id, "two_sum", source="test")
    try:
        # A run left queued (e.g. the server died before starting it)
        # must be picked up when a new server starts.
        app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path, run_source="test")
        with TestClient(app) as c:
            body = _wait(c, run_id)
        assert body["status"] == "passed"
    finally:
        store.delete_run(run_id)
        store.close()
