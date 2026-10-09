"""Step 3.7 done-when check: the plain-text log matches the Postgres rows.

Rendering/writer tests need nothing; the match tests run against the
real Postgres in DATABASE_URL and the local-only task catalog.
"""
import io
import os
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from agentic_coding_agent.agent.trace_log import (
    TraceLogWriter,
    fanout,
    render_event,
    render_log,
)

TS = "2026-10-09T10:00:00+00:00"


def test_render_each_kind():
    assert "PLAN  hash map" in render_event(
        {"seq": 1, "ts": TS, "kind": "plan", "approach": "hash map",
         "subtasks": ["a", "b"], "usage": {"total_tokens": 10}})
    write = render_event({"seq": 2, "ts": TS, "kind": "worker_code", "attempt": 2, "is_retry": True,
                          "diagnosis": "never looked up the complement",
                          "code": "def f():\n    return 1", "usage": {"total_tokens": 5}})
    assert "WRITE attempt 2 (retry)  tokens=5" in write
    assert "diagnosis: never looked up the complement" in write
    assert "    def f():\n        return 1" in write
    assert "RUN   attempt 1  status=fail  passed 3/7" in render_event(
        {"seq": 3, "ts": TS, "kind": "execute", "attempt": 1, "status": "fail", "passed": 3, "total": 7})
    assert "CHECK attempt 1  decision=retry  reason=fail" in render_event(
        {"seq": 4, "ts": TS, "kind": "verify", "attempt": 1, "decision": "retry", "reason": "fail"})
    assert "RESUME at write" in render_event(
        {"seq": 5, "ts": TS, "kind": "resume", "resume_at": "write", "attempt": 1, "steps": 1, "tokens": 9})
    assert "DONE  status=passed" in render_event(
        {"seq": 6, "ts": TS, "kind": "finish", "status": "passed", "reason": "all_tests_passed",
         "steps": 2, "tokens": 15})


def test_log_is_plain_text_not_json():
    line = render_event({"seq": 1, "ts": TS, "kind": "verify", "attempt": 1,
                         "decision": "pass", "reason": "all_tests_passed"})
    assert "{" not in line


def test_writer_writes_file_and_stream_and_appends(tmp_path):
    out = io.StringIO()
    w = TraceLogWriter("r1", log_dir=tmp_path, stream=out)
    e1 = {"seq": 1, "ts": TS, "kind": "verify", "attempt": 1, "decision": "retry", "reason": "fail"}
    e2 = {"seq": 2, "ts": TS, "kind": "finish", "status": "passed", "reason": "x", "steps": 1, "tokens": 1}
    w(e1)
    TraceLogWriter("r1", log_dir=tmp_path, stream=None)(e2)  # e.g. a resumed process
    text = (tmp_path / "r1.log").read_text(encoding="utf-8")
    assert text == render_log([e1, e2])
    assert out.getvalue() == render_log([e1])


def test_fanout_calls_store_before_log():
    order = []
    sink = fanout(lambda e: order.append("store"), None, lambda e: order.append("log"))
    sink({"seq": 1, "kind": "x"})
    assert order == ["store", "log"]
    assert fanout(None, None) is None


# --- log vs Postgres (done-when) ---

needs_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)

PLAN = '{"approach": "a", "subtasks": ["s1"]}'
FIRST = "```python\ndef two_sum(nums, target):\n    return [0, 0]\n```"
RETRY = "Diagnosis: always returned [0, 0].\n```python\ndef two_sum(nums, target):\n    return [0, 1]\n```"


class Crash(BaseException):
    pass


class FakeLLM:
    def __init__(self):
        self.writes = 0

    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 100}
        if "planning step" in (system_instruction or ""):
            return {"text": PLAN, "usage": usage}
        self.writes += 1
        return {"text": FIRST if self.writes == 1 else RETRY, "usage": usage}


def _res(cases, ok):
    return [{"input": c["input"], "expected": c["expected_output"],
             "actual": c["expected_output"] if ok else None, "passed": ok} for c in cases]


def fail_then_pass():
    calls = {"n": 0}

    def ex(code, language, cases, **kw):
        calls["n"] += 1
        ok = calls["n"] > 1
        return {"status": "pass" if ok else "fail", "stdout": "", "stderr": "",
                "test_results": _res(cases, ok)}
    return ex


@pytest.fixture
def store():
    from agentic_coding_agent.db.store import RunStore

    s = RunStore()
    ids = []
    s.track = ids.append
    yield s
    for r in ids:
        s.delete_run(r)
    s.close()


@needs_db
def test_log_matches_postgres_rows(store, tmp_path):
    from agentic_coding_agent.agent.graph import run_task

    run_id = str(uuid.uuid4())
    store.track(run_id)
    log = TraceLogWriter(run_id, log_dir=tmp_path, stream=None)
    result = run_task("two_sum", FakeLLM(), executor=fail_then_pass(), run_id=run_id,
                      store=store, log=log, source="test")
    assert result["status"] == "passed"

    file_text = log.path.read_text(encoding="utf-8")
    assert file_text == render_log(store.load_events(run_id))
    assert "CHECK attempt 1  decision=retry" in file_text
    assert "diagnosis: always returned [0, 0]." in file_text


@needs_db
def test_log_matches_postgres_rows_across_crash_and_resume(store, tmp_path):
    from agentic_coding_agent.agent.graph import resume_run, run_task

    def crash(*a, **kw):
        raise Crash()

    run_id = str(uuid.uuid4())
    store.track(run_id)
    with pytest.raises(Crash):
        run_task("two_sum", FakeLLM(), executor=crash, run_id=run_id, store=store,
                 log=TraceLogWriter(run_id, log_dir=tmp_path, stream=None), source="test")

    llm = FakeLLM()
    llm.writes = 1  # next write is the passing one
    resume_run(run_id, llm, store, executor=fail_then_pass(),
               log=TraceLogWriter(run_id, log_dir=tmp_path, stream=None))

    file_text = (tmp_path / f"{run_id}.log").read_text(encoding="utf-8")
    assert file_text == render_log(store.load_events(run_id))
    assert "RESUME at execute" in file_text
