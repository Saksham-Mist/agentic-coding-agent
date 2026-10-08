"""Step 3.2 done-when check: the plan appears in the trace first.

Fake-LLM tests are deterministic and cost no quota; the live test
makes one real Gemini call and skips without GEMINI_API_KEY.
"""
import os

import pytest
from dotenv import load_dotenv

from agentic_coding_agent.agent.planner import MAX_SUBTASKS, plan_task
from agentic_coding_agent.agent.trace import Trace

load_dotenv()

USAGE = {"prompt_tokens": 10, "response_tokens": 5, "thoughts_tokens": 0, "total_tokens": 15}

SPEC = {
    "task_id": "two_sum",
    "prompt": "Return indices of the two numbers that add up to target.",
    "function_signature": "def two_sum(nums: list[int], target: int) -> list[int]",
    "constraints": "2 <= len(nums) <= 10^4",
    "test_cases": [{"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]}],
}


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, prompt, system_instruction=None):
        self.calls.append(prompt)
        return {"text": self.text, "usage": USAGE}


def test_plan_is_first_trace_event():
    trace = Trace("run-1")
    llm = FakeLLM('{"approach": "hash map", "subtasks": ["write two_sum with a dict"]}')
    result = plan_task(SPEC, llm, trace)

    assert result["status"] == "ok"
    assert result["subtasks"] == ["write two_sum with a dict"]
    assert trace.events[0]["kind"] == "plan"
    assert trace.events[0]["seq"] == 1
    assert trace.events[0]["subtasks"] == ["write two_sum with a dict"]


def test_fenced_json_is_accepted():
    llm = FakeLLM('```json\n{"approach": "a", "subtasks": ["s1"]}\n```')
    result = plan_task(SPEC, llm, Trace("run-2"))
    assert result["status"] == "ok"


def test_subtasks_capped():
    many = ", ".join(f'"s{i}"' for i in range(10))
    llm = FakeLLM(f'{{"approach": "a", "subtasks": [{many}]}}')
    result = plan_task(SPEC, llm, Trace("run-3"))
    assert len(result["subtasks"]) == MAX_SUBTASKS


def test_unparseable_output_is_structured_error_not_exception():
    trace = Trace("run-4")
    result = plan_task(SPEC, FakeLLM("sure! here's my plan: just do it"), trace)
    assert result["status"] == "error"
    assert result["reason"] == "unparseable_plan"
    assert trace.events[0]["kind"] == "plan_error"


def test_prompt_contains_only_given_cases():
    llm = FakeLLM('{"approach": "a", "subtasks": ["s1"]}')
    plan_task(SPEC, llm, Trace("run-5"))
    assert "[2, 7, 11, 15]" in llm.calls[0]


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="GEMINI_API_KEY not set")
def test_live_planner_produces_valid_plan():
    from agentic_coding_agent.agent.llm import GeminiClient

    trace = Trace("run-live")
    result = plan_task(SPEC, GeminiClient(), trace)
    assert result["status"] == "ok", result
    assert 1 <= len(result["subtasks"]) <= MAX_SUBTASKS
    assert trace.events[0]["kind"] == "plan"
