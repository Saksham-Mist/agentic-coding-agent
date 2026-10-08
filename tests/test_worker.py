"""Step 3.3 done-when check: the retry prompt demonstrably contains the
failure. Also: hidden-case data never reaches the prompt.
"""
import os

import pytest
from dotenv import load_dotenv

from agentic_coding_agent.agent.trace import Trace
from agentic_coding_agent.agent.worker import format_feedback, write_code

load_dotenv()

USAGE = {"prompt_tokens": 10, "response_tokens": 5, "thoughts_tokens": 0, "total_tokens": 15}

SPEC = {
    "task_id": "two_sum",
    "prompt": "Return indices of the two numbers that add up to target.",
    "function_signature": "def two_sum(nums: list[int], target: int) -> list[int]",
    "constraints": "2 <= len(nums) <= 10^4",
    "test_cases": [{"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]}],
}
PLAN = {"approach": "hash map", "subtasks": ["write two_sum with a dict"]}

WRONG_CODE = "def two_sum(nums, target):\n    return [0, 0]"

EXEC_RESULT = {
    "status": "fail",
    "stdout": "",
    "stderr": "",
    "test_results": [
        {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected": [0, 1],
         "actual": [0, 0], "passed": False},
        {"input": {"nums": [-3, 4, 3, 90], "target": 0}, "expected": [0, 2],
         "actual": [0, 0], "passed": False},
    ],
}
VISIBLE = [True, False]


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.prompts = []

    def generate(self, prompt, system_instruction=None):
        self.prompts.append(prompt)
        return {"text": self.text, "usage": USAGE}


GOOD_REPLY = "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```"


def test_first_attempt_has_no_retry_context():
    llm = FakeLLM(GOOD_REPLY)
    result = write_code(SPEC, PLAN, 0, llm, Trace("r1"))
    assert result["status"] == "ok"
    assert result["code"].startswith("def two_sum")
    assert "previous attempt failed" not in llm.prompts[0]


def test_retry_prompt_contains_prior_code_and_visible_failure():
    llm = FakeLLM(GOOD_REPLY)
    feedback = format_feedback(WRONG_CODE, EXEC_RESULT, VISIBLE)
    trace = Trace("r2")
    write_code(SPEC, PLAN, 0, llm, trace, attempt=2, feedback=feedback)

    prompt = llm.prompts[0]
    assert WRONG_CODE in prompt
    assert '"nums": [2, 7, 11, 15]' in prompt
    assert "expected=[0, 1]" in prompt
    assert "actual=[0, 0]" in prompt
    assert trace.events[0]["is_retry"] is True


def test_hidden_failure_is_counted_but_not_leaked():
    feedback = format_feedback(WRONG_CODE, EXEC_RESULT, VISIBLE)
    assert "1 hidden test(s) also failed" in feedback
    assert "[-3, 4, 3, 90]" not in feedback
    assert "[0, 2]" not in feedback


def test_long_stderr_is_truncated():
    result = dict(EXEC_RESULT, status="error", stderr="x" * 10_000)
    feedback = format_feedback(WRONG_CODE, result, VISIBLE)
    assert feedback.count("x") < 2000


def test_no_code_in_reply_is_structured_error():
    trace = Trace("r3")
    result = write_code(SPEC, PLAN, 0, FakeLLM("   "), trace)
    assert result["status"] == "error"
    assert result["reason"] == "no_code"
    assert trace.events[0]["kind"] == "worker_error"


@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") and os.environ.get("RUN_LIVE_LLM") == "1"),
    reason="live LLM test; set RUN_LIVE_LLM=1",
)
def test_live_worker_retry_fixes_wrong_code():
    from agentic_coding_agent.agent.llm import GeminiClient

    trace = Trace("live")
    feedback = format_feedback(WRONG_CODE, EXEC_RESULT, VISIBLE)
    result = write_code(SPEC, PLAN, 0, GeminiClient(), trace, attempt=2, feedback=feedback)
    assert result["status"] == "ok"
    assert "def two_sum" in result["code"]
    assert "return [0, 0]" not in result["code"]
