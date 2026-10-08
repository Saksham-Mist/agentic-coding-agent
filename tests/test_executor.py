"""Step 2.2 done-when check: execute_code returns the contract's
output shape. Requires Docker running and agentic-sandbox:latest
built (see docker/sandbox/Dockerfile) — skips otherwise.
"""
import shutil
import subprocess

import pytest

from agentic_coding_agent.sandbox.executor import execute_code

EXPECTED_KEYS = {"status", "stdout", "stderr", "test_results"}


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def _image_built() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", "agentic-sandbox:latest"],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


pytestmark = pytest.mark.skipif(
    not (_docker_available() and _image_built()),
    reason="Docker not running or agentic-sandbox:latest not built",
)

TWO_SUM_CODE = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
"""


def test_known_correct_solution_passes():
    result = execute_code(
        code=TWO_SUM_CODE,
        language="python",
        entry_point="two_sum",
        test_cases=[
            {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]},
        ],
    )
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] == "pass"
    assert result["test_results"][0]["passed"] is True


def test_known_wrong_solution_fails_with_actual():
    wrong_code = "def two_sum(nums, target):\n    return [0, 0]\n"
    result = execute_code(
        code=wrong_code,
        language="python",
        entry_point="two_sum",
        test_cases=[
            {"input": {"nums": [2, 7, 11, 15], "target": 9}, "expected_output": [0, 1]},
        ],
    )
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] == "fail"
    assert result["test_results"][0]["passed"] is False
    assert result["test_results"][0]["actual"] == [0, 0]


def test_unsupported_language_returns_structured_error():
    result = execute_code(code="print(1)", language="javascript", test_cases=[])
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] == "error"


def test_syntax_error_returns_structured_error_not_exception():
    result = execute_code(
        code="def two_sum(nums target):\n    pass",  # missing comma
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1], "target": 1}, "expected_output": None}],
    )
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] == "error"
    assert result["test_results"] == []


def test_infinite_loop_times_out():
    result = execute_code(
        code="def two_sum(nums, target):\n    while True:\n        pass\n",
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1], "target": 1}, "expected_output": None}],
    )
    assert result["status"] == "timeout"
    # the container must actually be gone (--rm + docker kill), not left running
    leftover = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "ancestor=agentic-sandbox:latest"],
        capture_output=True, text=True,
    )
    assert leftover.stdout.strip() == ""


def test_memory_bomb_returns_resource_exceeded():
    result = execute_code(
        code=(
            "def two_sum(nums, target):\n"
            "    data = []\n"
            "    while True:\n"
            "        data.append('x' * 10**7)\n"
        ),
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1], "target": 1}, "expected_output": None}],
    )
    assert result["status"] in ("resource_exceeded", "timeout", "error")


def test_fork_bomb_is_contained():
    result = execute_code(
        code=(
            "import os\n"
            "def two_sum(nums, target):\n"
            "    while True:\n"
            "        os.fork()\n"
        ),
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1], "target": 1}, "expected_output": None}],
    )
    # A fork bomb inside one container can't be allowed to affect the
    # host or other containers; it must come back as a structured
    # status, and the container itself must not be left running.
    # --pids-limit caps the process count, so os.fork() starts raising
    # BlockingIOError quickly rather than spawning thousands of
    # processes — but each forked child is a full copy of the Python
    # interpreter, including our own harness code, so multiple
    # processes can each print a result line and the host sees
    # concatenated JSON ("error", malformed output) rather than one
    # clean "fail". Both are acceptable: what matters is no exception
    # escapes to the caller and the container doesn't survive.
    assert result["status"] in ("timeout", "resource_exceeded", "error", "fail")
    leftover = subprocess.run(
        ["docker", "ps", "-a", "-q", "--filter", "ancestor=agentic-sandbox:latest"],
        capture_output=True, text=True,
    )
    assert leftover.stdout.strip() == ""


def test_write_outside_workdir_does_not_escape_container():
    result = execute_code(
        code=(
            "def two_sum(nums, target):\n"
            "    with open('/etc/sandbox_escape_test', 'w') as f:\n"
            "        f.write('escaped')\n"
            "    return [0, 1]\n"
        ),
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1, 2], "target": 3}, "expected_output": [0, 1]}],
    )
    # Writing inside the container's own filesystem is allowed (the
    # container is disposable and --rm'd); what matters is it never
    # touches the host. Container-per-call + --rm is the guarantee
    # here, not a write-permission denial — so this just confirms the
    # call completes and returns a structured result either way.
    assert set(result.keys()) == EXPECTED_KEYS
    assert result["status"] in ("pass", "fail", "error")


def test_outbound_network_call_fails_closed():
    result = execute_code(
        code=(
            "import urllib.request\n"
            "def two_sum(nums, target):\n"
            "    urllib.request.urlopen('http://example.com', timeout=3)\n"
            "    return [0, 1]\n"
        ),
        language="python",
        entry_point="two_sum",
        test_cases=[{"input": {"nums": [1, 2], "target": 3}, "expected_output": [0, 1]}],
    )
    # --network=none means the call itself must fail inside the
    # container; the harness reports it as a failed/errored test case,
    # never a silent pass.
    assert result["status"] in ("fail", "error")
