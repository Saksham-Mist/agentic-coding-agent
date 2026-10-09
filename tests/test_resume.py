"""Step 3.6 done-when check: a run killed mid-way resumes from its
last persisted step and finishes without re-planning.

Runs against the real Postgres in DATABASE_URL (each test creates and
deletes its own run rows). A "crash" is a BaseException raised from
inside the run, which nothing in the loop catches, same as the process
dying at that point. The real process-kill check is done from the CLI.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from agentic_coding_agent.agent.graph import Limits, resume_run, run_task

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)

PLAN = '{"approach": "a", "subtasks": ["s1"]}'
CODE = "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```"


class Crash(BaseException):
    """Stands in for the process being killed."""


class FakeLLM:
    def __init__(self, crash_on_plan=False):
        self.plan_calls = 0
        self.write_calls = 0
        self.crash_on_plan = crash_on_plan

    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 100}
        if "planning step" in (system_instruction or ""):
            self.plan_calls += 1
            if self.crash_on_plan:
                raise Crash()
            return {"text": PLAN, "usage": usage}
        self.write_calls += 1
        return {"text": CODE, "usage": usage}


def _results(cases, ok):
    return [{"input": c["input"], "expected": c["expected_output"],
             "actual": c["expected_output"] if ok else None, "passed": ok} for c in cases]


def passing(code, language, cases, **kw):
    return {"status": "pass", "stdout": "", "stderr": "", "test_results": _results(cases, True)}


def failing(code, language, cases, **kw):
    return {"status": "fail", "stdout": "", "stderr": "", "test_results": _results(cases, False)}


def crashing(code, language, cases, **kw):
    raise Crash()


class ScriptedExecutor:
    """Runs the given executors in order, one per call."""

    def __init__(self, *steps):
        self.steps = list(steps)

    def __call__(self, *a, **kw):
        return self.steps.pop(0)(*a, **kw)


@pytest.fixture
def store():
    from agentic_coding_agent.db.store import RunStore

    s = RunStore()
    created = []
    s.track = created.append
    yield s
    for run_id in created:
        s.delete_run(run_id)
    s.close()


def _start(store, llm, executor, run_id):
    store.track(run_id)
    with pytest.raises(Crash):
        run_task("two_sum", llm, executor=executor, run_id=run_id, store=store)


def test_crash_during_execute_resumes_without_replanning(store):
    run_id = "00000000-0000-0000-0000-00000000a001"
    _start(store, FakeLLM(), crashing, run_id)

    assert [e["kind"] for e in store.load_events(run_id)] == ["plan", "worker_code"]
    assert store.get_run(run_id)["status"] == "running"

    llm = FakeLLM()
    result = resume_run(run_id, llm, store, executor=passing)
    assert result["status"] == "passed"
    assert llm.plan_calls == 0       # no re-planning
    assert llm.write_calls == 0      # saved code is re-executed, not rewritten
    kinds = [e["kind"] for e in result["trace"]]
    assert kinds == ["plan", "worker_code", "resume", "execute", "verify", "finish"]
    assert result["trace"][2]["resume_at"] == "execute"


def test_crash_after_failed_attempt_resumes_with_feedback(store):
    run_id = "00000000-0000-0000-0000-00000000a002"
    # attempt 1 fails and is verified (retry), attempt 2 crashes mid-execute
    _start(store, FakeLLM(), ScriptedExecutor(failing, crashing), run_id)

    llm = FakeLLM()
    result = resume_run(run_id, llm, store, executor=passing)
    assert result["status"] == "passed"
    assert llm.plan_calls == 0
    resume_ev = next(e for e in result["trace"] if e["kind"] == "resume")
    assert resume_ev["resume_at"] == "execute"
    assert resume_ev["attempt"] == 2
    assert result["steps"] == 3       # plan + 2 writes, all from before the crash


def test_resume_after_verify_retry_uses_saved_feedback(store):
    from agentic_coding_agent.agent.graph import _state_from_events

    run_id = "00000000-0000-0000-0000-00000000a003"
    # attempt 1 fails + verify(retry); crash happens inside the attempt-2 worker call
    class CrashOnSecondWrite(FakeLLM):
        def generate(self, prompt, system_instruction=None):
            if "planning step" not in (system_instruction or "") and self.write_calls == 1:
                raise Crash()
            return super().generate(prompt, system_instruction)

    _start(store, CrashOnSecondWrite(), failing, run_id)
    events = store.load_events(run_id)
    assert events[-1]["kind"] == "verify" and events[-1]["decision"] == "retry"

    restored = _state_from_events(events)
    assert restored["resume_at"] == "write"
    assert restored["attempt"] == 2
    assert "Your previous attempt failed" in restored["feedback"]

    prompts = []

    class Recording(FakeLLM):
        def generate(self, prompt, system_instruction=None):
            prompts.append(prompt)
            return super().generate(prompt, system_instruction)

    llm = Recording()
    result = resume_run(run_id, llm, store, executor=passing)
    assert result["status"] == "passed"
    assert llm.plan_calls == 0
    assert "Your previous attempt failed" in prompts[0]


def test_crash_during_planning_resumes_at_plan(store):
    run_id = "00000000-0000-0000-0000-00000000a004"
    _start(store, FakeLLM(crash_on_plan=True), passing, run_id)
    assert store.load_events(run_id) == []

    llm = FakeLLM()
    result = resume_run(run_id, llm, store, executor=passing)
    assert result["status"] == "passed"
    assert llm.plan_calls == 1


def test_finished_run_resume_makes_no_calls(store):
    run_id = "00000000-0000-0000-0000-00000000a005"
    store.track(run_id)
    first = run_task("two_sum", FakeLLM(), executor=passing, run_id=run_id, store=store)
    assert first["status"] == "passed"

    llm = FakeLLM()
    again = resume_run(run_id, llm, store, executor=passing)
    assert again["status"] == "passed"
    assert llm.plan_calls == llm.write_calls == 0


def test_postgres_rows_match_trace(store):
    run_id = "00000000-0000-0000-0000-00000000a006"
    store.track(run_id)
    result = run_task("two_sum", FakeLLM(), executor=ScriptedExecutor(failing, passing),
                      run_id=run_id, store=store)

    rows = store.load_events(run_id)
    assert [(r["seq"], r["kind"]) for r in rows] == [(e["seq"], e["kind"]) for e in result["trace"]]
    run = store.get_run(run_id)
    assert run["status"] == "passed"
    assert run["steps_taken"] == result["steps"] == 3
    assert run["tokens_used"] == result["tokens"] == 300


def test_unknown_run_is_structured_error(store):
    result = resume_run("00000000-0000-0000-0000-00000000dead", FakeLLM(), store)
    assert result["status"] == "error"
    assert result["reason"] == "run_not_found"
