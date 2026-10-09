"""Step 3.5 done-when check: each limit, forced in a test, yields
`incomplete` with a trace; the happy path passes. Uses a fake LLM and
a fake executor, so no quota or Docker is needed. The live two_sum run
from the CLI is checked separately.
"""
from pathlib import Path

import pytest

from agentic_coding_agent.agent.graph import Limits, run_task

# task_catalog.json is local-only (gitignored); skip in a fresh clone.
pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="task_catalog.json not present (local-only file)",
)

PLAN = '{"approach": "a", "subtasks": ["s1", "s2", "s3"]}'
CODE = "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```"


class FakeLLM:
    def __init__(self, tokens_per_call=100, fail_with=None):
        self.calls = 0
        self.tokens = tokens_per_call
        self.fail_with = fail_with

    def generate(self, prompt, system_instruction=None):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        text = PLAN if "planning step" in (system_instruction or "") else CODE
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0,
                 "total_tokens": self.tokens}
        return {"text": text, "usage": usage}


def passing_executor(code, language, cases, **kw):
    return {"status": "pass", "stdout": "", "stderr": "",
            "test_results": [{"input": c["input"], "expected": c["expected_output"],
                              "actual": c["expected_output"], "passed": True} for c in cases]}


def failing_executor(code, language, cases, **kw):
    return {"status": "fail", "stdout": "", "stderr": "",
            "test_results": [{"input": c["input"], "expected": c["expected_output"],
                              "actual": None, "passed": False} for c in cases]}


def kinds(result):
    return [e["kind"] for e in result["trace"]]


def test_happy_path_passes_and_stops_early():
    llm = FakeLLM()
    result = run_task("two_sum", llm, executor=passing_executor)
    assert result["status"] == "passed"
    # plan + one write: the remaining 2 subtasks are skipped once tests pass
    assert result["steps"] == 2
    assert llm.calls == 2
    assert kinds(result) == ["plan", "worker_code", "execute", "verify", "finish"]
    from agentic_coding_agent.agent.catalog import get_task
    assert result["passed"] == result["total"] == len(get_task("two_sum")["test_cases"])


def test_retry_limit_yields_incomplete():
    llm = FakeLLM()
    result = run_task("two_sum", llm, limits=Limits(max_retries=3, max_steps=15, max_tokens=100_000),
                      executor=failing_executor)
    assert result["status"] == "incomplete"
    assert result["reason"] == "retries_exhausted:fail"
    # 1 plan + 4 attempts (first + 3 retries)
    assert result["steps"] == 5
    assert kinds(result).count("worker_code") == 4
    assert kinds(result)[-1] == "finish"


def test_step_limit_yields_incomplete():
    llm = FakeLLM()
    result = run_task("two_sum", llm, limits=Limits(max_retries=10, max_steps=3, max_tokens=100_000),
                      executor=failing_executor)
    assert result["status"] == "incomplete"
    assert result["reason"] == "step_limit"
    assert result["steps"] == 3
    assert llm.calls == 3  # never makes a call past the limit


def test_token_limit_yields_incomplete():
    llm = FakeLLM(tokens_per_call=60_000)
    result = run_task("two_sum", llm, limits=Limits(max_retries=10, max_steps=15, max_tokens=100_000),
                      executor=failing_executor)
    assert result["status"] == "incomplete"
    assert result["reason"] == "token_limit"
    # plan (60k) + write (120k) -> over budget before the third call
    assert llm.calls == 2
    assert result["tokens"] == 120_000


def test_default_limits_fit_under_langgraph_recursion_cap():
    # 15 steps with retries never exhausting: must end on step_limit,
    # not a LangGraph GraphRecursionError.
    llm = FakeLLM()
    result = run_task("two_sum", llm, limits=Limits(max_retries=100, max_steps=15, max_tokens=10**9),
                      executor=failing_executor)
    assert result["status"] == "incomplete"
    assert result["reason"] == "step_limit"
    assert result["steps"] == 15


def test_llm_exception_is_structured_error():
    llm = FakeLLM(fail_with=RuntimeError("quota exhausted"))
    result = run_task("two_sum", llm, executor=passing_executor)
    assert result["status"] == "error"
    assert "quota exhausted" in result["reason"]
    assert kinds(result) == ["finish"]


def test_unknown_task_is_structured_error():
    result = run_task("no_such_task", FakeLLM(), executor=passing_executor)
    assert result["status"] == "error"
    assert result["reason"] == "not_found"


def test_retry_feedback_reaches_worker():
    prompts = []

    class Recording(FakeLLM):
        def generate(self, prompt, system_instruction=None):
            prompts.append(prompt)
            return super().generate(prompt, system_instruction)

    run_task("two_sum", Recording(), limits=Limits(max_retries=1, max_steps=15, max_tokens=100_000),
             executor=failing_executor)
    # prompts: plan, attempt 1, attempt 2 (retry)
    assert "previous attempt failed" not in prompts[1]
    assert "Your previous attempt failed" in prompts[2]
    assert "hidden test(s) also failed" in prompts[2]
