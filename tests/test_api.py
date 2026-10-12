"""Step 4.1: POST /runs, GET /runs/{id}, GET /stats.

In-process API with a fake LLM and fake executor (no quota, no
Docker), against the real Postgres in DATABASE_URL. Runs are tagged
source='test' and deleted afterwards. The real curl round trip against
a running uvicorn is checked separately.
"""
import json
import os
import uuid
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

    app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path, run_source="test", rate_limit=10_000)
    cleanup = RunStore()
    created = []
    with TestClient(app) as c:
        c.created = created
        yield c
    for run_id in created:
        cleanup.delete_run(run_id)
    cleanup.delete_rate_limit("testclient")
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
    # Match whole input objects (nums + target): matching only the nums
    # list gave a false alarm once the catalog was rebuilt, because a
    # hidden case's nums [1, 2] equals a visible case's expected output.
    for c in hidden:
        assert json.dumps(c["input"]) not in text, f"hidden input leaked: {c['input']}"
    for c in visible:
        assert json.dumps(c["input"]) in text  # visible results are still shown

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
    assert "public" in stats


def test_root_serves_the_demo_page(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert "<title>" in r.text and "/tasks" in r.text


def test_tasks_lists_public_tasks_without_hidden_cases(client):
    from agentic_coding_agent.agent.catalog import load_catalog

    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = {t["task_id"]: t for t in r.json()}
    assert "two_sum" in tasks
    entry = load_catalog()["two_sum"]
    visible = [c for c in entry["test_cases"] if c.get("visible")]
    t = tasks["two_sum"]
    assert len(t["examples"]) == len(visible)
    assert t["hidden_cases"] == len(entry["test_cases"]) - len(visible)
    shown = json.dumps(t)
    for c in entry["test_cases"]:
        if not c.get("visible"):
            assert json.dumps(c["input"]) not in shown   # hidden inputs never leave the server


def test_unknown_and_malformed_run_ids_are_404(client):
    assert client.get("/runs/00000000-0000-0000-0000-00000000ffff").status_code == 404
    assert client.get("/runs/not-a-uuid").status_code == 404


def test_queued_run_is_recovered_on_startup(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    run_id = str(uuid.uuid4())
    store.create_queued(run_id, "two_sum", source="test")
    try:
        # A run left queued (e.g. the server died before starting it)
        # must be picked up when a new server starts.
        app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path, run_source="test", rate_limit=10_000)
        with TestClient(app) as c:
            body = _wait(c, run_id)
        assert body["status"] == "passed"
    finally:
        store.delete_run(run_id)
        store.close()


# --- step 4.2: catalog-only validation ---

_count_conn = None


def _test_rows():
    # One reused connection: a fresh Neon connection costs ~3s from the
    # dev machine (TLS + auth over a ~290ms link).
    global _count_conn
    import psycopg

    if _count_conn is None or _count_conn.closed:
        _count_conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    return _count_conn.execute("SELECT count(*) FROM runs WHERE source = 'test'").fetchone()[0]


class CountingLLM(FakeLLM):
    calls = 0

    def generate(self, prompt, system_instruction=None):
        CountingLLM.calls += 1
        return super().generate(prompt, system_instruction)


@pytest.fixture(scope="module")
def _shared_counting_client(tmp_path_factory):
    # One API instance for all validation tests: starting one per test
    # meant new Neon connections each time (~8s per test).
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    app = create_app(llm_factory=CountingLLM, executor=passing,
                     log_dir=tmp_path_factory.mktemp("logs"), run_source="test", rate_limit=10_000)
    cleanup = RunStore()
    with TestClient(app) as c:
        c.created = []
        yield c
    for run_id in c.created:
        cleanup.delete_run(run_id)
    cleanup.delete_rate_limit("testclient")
    cleanup.close()


@pytest.fixture
def counting_client(_shared_counting_client):
    CountingLLM.calls = 0
    return _shared_counting_client


STUB_ID = "zz_unfinished_stub"


@pytest.fixture
def with_stub(monkeypatch):
    """Adds an unfinished entry to the catalog the API sees. Picking one
    from the real catalog broke once all 18 entries were valid."""
    import agentic_coding_agent.api.app as app_module
    from agentic_coding_agent.agent.catalog import load_catalog

    real = load_catalog()
    stub = {"task_id": STUB_ID, "entry_point": "f", "comparator": "exact", "prompt": "",
            "function_signature": "", "constraints": "", "test_cases": []}
    monkeypatch.setattr(app_module, "load_catalog", lambda path=None: {**real, STUB_ID: stub})
    return STUB_ID


@pytest.mark.parametrize("payload, reason", [
    ({"task_id": "no_such_task"}, "unknown_task_id"),
    ({"task_id": None}, "task_not_ready"),  # filled in with the with_stub fixture
])
def test_rejected_task_ids_create_no_run(counting_client, with_stub, payload, reason):
    not_ready = with_stub
    if payload["task_id"] is None:
        payload = {"task_id": not_ready}
    before = _test_rows()
    r = counting_client.post("/runs", json=payload)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == reason
    assert "two_sum" in detail["available_task_ids"]
    assert not_ready not in detail["available_task_ids"]
    from agentic_coding_agent.agent.catalog import load_catalog, validate_entry
    assert all(not validate_entry(load_catalog()[t]) for t in detail["available_task_ids"])
    time.sleep(1)  # give the worker a chance to (wrongly) pick something up
    assert _test_rows() == before
    assert CountingLLM.calls == 0


@pytest.mark.parametrize("payload", [
    {"task_id": "../../etc/passwd"},
    {"task_id": "Two_Sum"},
    {"task_id": "x" * 65},
    {"task_id": ""},
    {"task_id": 123},
    {"task": "two_sum"},
    {"task_id": "two_sum", "code": "import os"},  # extra fields are ignored, never executed
])
def test_malformed_requests_are_rejected_or_ignored(counting_client, payload):
    before = _test_rows()
    r = counting_client.post("/runs", json=payload)
    if "code" in payload:
        # Only task_id is read; the run uses the catalog's own code-free spec.
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        counting_client.created.append(run_id)
        _wait(counting_client, run_id)  # don't delete under the worker's feet
        return
    assert r.status_code == 422
    assert _test_rows() == before
    assert CountingLLM.calls == 0


def test_interrupted_running_run_is_resumed_on_startup(tmp_path):
    """Startup recovery is the one place a 'running' run gets resumed."""
    from fastapi.testclient import TestClient

    from agentic_coding_agent.agent.graph import run_task
    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    class Crash(BaseException):
        pass

    def crashing(*a, **kw):
        raise Crash()

    store = RunStore()
    run_id = str(uuid.uuid4())
    try:
        with pytest.raises(Crash):  # the "previous server" dies mid-run
            run_task("two_sum", FakeLLM(), executor=crashing, run_id=run_id, store=store, source="test")
        assert store.get_run(run_id)["status"] == "running"

        app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path,
                         run_source="test", rate_limit=10_000)
        with TestClient(app) as c:
            body = _wait(c, run_id)
        assert body["status"] == "passed"
        kinds = [e["kind"] for e in body["trace"]]
        assert kinds.count("plan") == 1 and "resume" in kinds
    finally:
        store.delete_run(run_id)
        store.close()


# --- stress set: owner-only twisted tasks ---

@pytest.fixture
def with_stress_task(monkeypatch):
    """A fully valid owner-only stress entry (a copy of two_sum tagged
    set=stress), patched into the catalog the API sees."""
    import agentic_coding_agent.api.app as app_module
    from agentic_coding_agent.agent.catalog import load_catalog

    real = load_catalog()
    stress = {**real["two_sum"], "task_id": "zz_stress_copy", "set": "stress"}
    monkeypatch.setattr(app_module, "load_catalog", lambda path=None: {**real, "zz_stress_copy": stress})
    return "zz_stress_copy"


def test_public_caller_cannot_see_or_run_stress_tasks(counting_client, with_stress_task):
    r = counting_client.post("/runs", json={"task_id": with_stress_task})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "unknown_task_id"          # looks unknown, doesn't reveal the set
    assert with_stress_task not in detail["available_task_ids"]
    assert CountingLLM.calls == 0


def test_tasks_list_hides_stress_tasks(counting_client, with_stress_task):
    ids = [t["task_id"] for t in counting_client.get("/tasks").json()]
    assert with_stress_task not in ids and "two_sum" in ids


def test_owner_can_run_stress_tasks(tmp_path, with_stress_task):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path, run_source="test",
                     rate_limit=10_000, owner_token="owner-secret-for-test")
    store = RunStore()
    with TestClient(app) as c:
        r = c.post("/runs", json={"task_id": with_stress_task},
                   headers={"Authorization": "Bearer owner-secret-for-test"})
        assert r.status_code == 202
        run_id = r.json()["run_id"]
        _wait(c, run_id)
    store.delete_run(run_id)
    store.close()


def test_public_stats_count_only_visitor_runs():
    """GET /stats' headline counts public (source=api) runs only; owner,
    eval and test runs appear only in by_source. Checked by inserting known
    runs and comparing before/after deltas (other rows may exist)."""
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    ids = {src: [str(uuid.uuid4()) for _ in range(n)] for src, n in (("api", 2), ("cli", 1), ("eval", 1))}
    before = store.stats()
    try:
        for src, run_ids in ids.items():
            for run_id in run_ids:
                store.create_queued(run_id, "two_sum", source=src)
                store._conn.execute("UPDATE runs SET status = 'passed', steps_taken = 2, tokens_used = 100 "
                                    "WHERE run_id = %s", (run_id,))
        after = store.stats()
        assert after["public"]["finished_runs"] - before["public"]["finished_runs"] == 2
        assert after["public"]["passed"] - before["public"]["passed"] == 2
        assert after["public"] == after["by_source"]["api"]

        def finished(stats, src):
            return stats["by_source"].get(src, {}).get("finished_runs", 0)
        assert finished(after, "cli") - finished(before, "cli") == 1
        assert finished(after, "eval") - finished(before, "eval") == 1
    finally:
        for run_ids in ids.values():
            for run_id in run_ids:
                store.delete_run(run_id)
        store.close()


def test_public_stats_are_zero_without_visitor_runs():
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    try:
        stats = store.stats()
        if "api" not in stats["by_source"]:
            assert stats["public"]["finished_runs"] == 0
            assert stats["public"]["success_rate"] is None
        else:
            assert stats["public"] == stats["by_source"]["api"]
    finally:
        store.close()
