"""Step 4.5 done-when check: simultaneous POSTs cause no double
execution and no crash; runs execute one at a time (v1 is sequential).
Fake LLM and executor, real Postgres.
"""
import os
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)


class FakeLLM:
    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 50}
        if "planning step" in (system_instruction or ""):
            return {"text": '{"approach": "a", "subtasks": ["s1"]}', "usage": usage}
        return {"text": "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```", "usage": usage}


class RecordingExecutor:
    """Records each execution's time window; sleeps so that any parallel
    execution would overlap."""

    def __init__(self):
        self.windows = []
        self.lock = threading.Lock()

    def __call__(self, code, language, cases, **kw):
        start = time.monotonic()
        time.sleep(0.3)
        end = time.monotonic()
        with self.lock:
            self.windows.append((start, end))
        return {"status": "pass", "stdout": "", "stderr": "",
                "test_results": [{"input": c["input"], "expected": c["expected_output"],
                                  "actual": c["expected_output"], "passed": True} for c in cases]}

    def max_overlap(self):
        events = sorted([(s, 1) for s, _ in self.windows] + [(e, -1) for _, e in self.windows])
        current = peak = 0
        for _, delta in events:
            current += delta
            peak = max(peak, current)
        return peak


def _kinds(store, run_id):
    return [e["kind"] for e in store.load_events(run_id)]


def _wait_all(store, run_ids, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        statuses = [store.get_run(r)["status"] for r in run_ids]
        if all(s not in ("queued", "running") for s in statuses):
            return statuses
        time.sleep(0.5)
    raise AssertionError(f"runs did not finish: {statuses}")


@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    executor = RecordingExecutor()
    app = create_app(llm_factory=FakeLLM, executor=executor, log_dir=tmp_path,
                     run_source="test", rate_limit=10_000)
    store = RunStore()
    created = []
    with TestClient(app) as client:
        yield {"client": client, "app": app, "store": store, "executor": executor, "created": created}
        _wait_all(store, created)  # never delete rows under the worker
    for run_id in created:
        store.delete_run(run_id)
    store.delete_rate_limit("testclient")
    store.close()


def test_simultaneous_posts_each_run_once_and_never_overlap(api):
    client = api["client"]
    barrier = threading.Barrier(5)

    def post(_):
        barrier.wait()  # release all five requests at the same moment
        return client.post("/runs", json={"task_id": "two_sum"})

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(post, range(5)))

    assert [r.status_code for r in responses] == [202] * 5
    run_ids = [r.json()["run_id"] for r in responses]
    api["created"].extend(run_ids)
    assert len(set(run_ids)) == 5

    assert _wait_all(api["store"], run_ids) == ["passed"] * 5
    for run_id in run_ids:
        kinds = _kinds(api["store"], run_id)
        assert kinds.count("plan") == 1, (run_id, kinds)
        assert kinds.count("finish") == 1, (run_id, kinds)
    assert len(api["executor"].windows) == 5   # one execution per run, no double execution
    assert api["executor"].max_overlap() == 1  # strictly one at a time


def test_duplicate_queue_entries_execute_once(api):
    r = api["client"].post("/runs", json={"task_id": "two_sum"})
    run_id = r.json()["run_id"]
    api["created"].append(run_id)
    worker = api["app"].state.worker
    worker.submit(run_id)  # the same run lands in the queue three times
    worker.submit(run_id)

    _wait_all(api["store"], [run_id])
    time.sleep(1.5)  # let the duplicate entries be processed (and skipped)
    kinds = _kinds(api["store"], run_id)
    assert kinds.count("plan") == 1
    assert kinds.count("finish") == 1
    assert len(api["executor"].windows) == 1


def test_two_workers_cannot_both_start_one_queued_run(tmp_path):
    """Simulates two server processes recovering the same queued run:
    only the one that wins the claim executes it."""
    from agentic_coding_agent.api.app import RunWorker
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    run_id = str(uuid.uuid4())
    store.create_queued(run_id, "two_sum", source="test")
    executor = RecordingExecutor()
    workers = [RunWorker(FakeLLM, executor, RunStore, tmp_path) for _ in range(2)]
    try:
        for w in workers:
            w.start()
        for w in workers:
            w.submit(run_id)
        assert _wait_all(store, [run_id]) == ["passed"]
        time.sleep(1.5)
        kinds = _kinds(store, run_id)
        assert kinds.count("plan") == 1, kinds
        assert len(executor.windows) == 1
    finally:
        for w in workers:
            w.stop()
        store.delete_run(run_id)
        store.close()


def test_claim_is_exclusive():
    from agentic_coding_agent.db.store import RunStore

    run_id = str(uuid.uuid4())
    setup = RunStore()
    setup.create_queued(run_id, "two_sum", source="test")

    def claim(_):
        s = RunStore()
        try:
            return s.claim_queued(run_id)
        finally:
            s.close()

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(claim, range(8)))
        assert results.count(True) == 1
        assert setup.get_run(run_id)["status"] == "running"
    finally:
        setup.delete_run(run_id)
        setup.close()


def test_stop_finishes_current_run_and_leaves_the_rest_queued(tmp_path):
    """stop() must not keep draining the queue after shutdown: the run in
    progress finishes, the others stay queued for the next startup."""
    from agentic_coding_agent.api.app import RunWorker
    from agentic_coding_agent.db.store import RunStore

    store = RunStore()
    run_ids = [str(uuid.uuid4()) for _ in range(3)]
    for r in run_ids:
        store.create_queued(r, "two_sum", source="test")
    executor = RecordingExecutor()
    worker = RunWorker(FakeLLM, executor, RunStore, tmp_path)
    try:
        worker.start()
        for r in run_ids:
            worker.submit(r)
        # wait until the first run is underway, then stop
        deadline = time.monotonic() + 60
        while store.get_run(run_ids[0])["status"] == "queued" and time.monotonic() < deadline:
            time.sleep(0.1)
        worker.stop()

        statuses = [store.get_run(r)["status"] for r in run_ids]
        assert statuses[0] == "passed"          # the run in progress was allowed to finish
        assert statuses[1:] == ["queued", "queued"]  # nothing new started after stop
        assert len(executor.windows) == 1
    finally:
        for r in run_ids:
            store.delete_run(r)
        store.close()
