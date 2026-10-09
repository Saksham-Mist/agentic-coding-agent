"""Hidden test data must never reach the model or the public API, even
when the agent's code prints inputs or raises exceptions whose
messages contain them (e.g. KeyError: <input value>).

Runs in the real Docker sandbox; skips without Docker or the image.
"""
import shutil
import subprocess

import pytest

from agentic_coding_agent.api.app import public_trace
from agentic_coding_agent.agent.worker import format_feedback
from agentic_coding_agent.sandbox.executor import execute_code

SECRET = 987654321  # appears only in the hidden case's input


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        out = subprocess.run(["docker", "images", "-q", "agentic-sandbox:latest"],
                             capture_output=True, text=True)
        return bool(out.stdout.strip())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _docker_ready(), reason="Docker or agentic-sandbox image unavailable")

# Prints its input to stdout and stderr, then raises with the input in
# the message -- the worst case for leaking.
LEAKY_CODE = """
import sys
def two_sum(nums, target):
    print("stdout sees", nums)
    print("stderr sees", nums, file=sys.stderr)
    raise KeyError(nums[0])
"""

CASES = [
    {"input": {"nums": [11, 22], "target": 33}, "expected_output": [0, 1]},        # visible
    {"input": {"nums": [SECRET, 1], "target": SECRET + 1}, "expected_output": [0, 1]},  # hidden
]
FLAGS = [True, False]


@pytest.fixture(scope="module")
def result():
    return execute_code(LEAKY_CODE, "python", CASES, entry_point="two_sum", comparator="unordered")


def test_errors_and_output_are_attached_to_their_own_case(result):
    visible, hidden = result["test_results"]
    assert "KeyError: 11" in visible["error"]
    assert "stdout sees [11, 22]" in visible["stdout"]
    assert f"KeyError: {SECRET}" in hidden["error"]  # recorded, but only on the hidden case
    assert not visible["passed"] and not hidden["passed"]


def test_shared_output_carries_no_case_data(result):
    assert str(SECRET) not in result["stdout"]
    assert str(SECRET) not in result["stderr"]
    assert "11, 22" not in result["stderr"]


def test_retry_feedback_shows_visible_error_but_not_hidden_data(result):
    feedback = format_feedback(LEAKY_CODE, result, FLAGS)
    assert str(SECRET) not in feedback
    assert "raised:" in feedback and "KeyError: 11" in feedback
    assert "1 hidden test(s) also failed" in feedback


def test_public_trace_has_no_hidden_data(result):
    task = {"test_cases": [dict(c, visible=v) for c, v in zip(CASES, FLAGS)]}
    events = [{"seq": 1, "kind": "execute", "status": result["status"], "passed": 0,
               "total": 2, "result": result}]
    assert str(SECRET) not in str(public_trace(events, task))


def test_load_time_error_is_still_reported_in_shared_stderr():
    r = execute_code("def two_sum(nums target): pass", "python", CASES,
                     entry_point="two_sum", comparator="unordered")
    assert r["status"] == "error"
    assert "SyntaxError" in r["stderr"]
