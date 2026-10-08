"""Step 2.3 done-when check: known-correct two_sum, merge_two_sorted_lists
and the LRUCache example pass; a wrong version of each fails with
per-case `actual`. Requires Docker + agentic-sandbox:latest — skips
otherwise.
"""
import shutil
import subprocess

import pytest

from agentic_coding_agent.sandbox.executor import execute_code


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

# --- two_sum: plain values, "unordered" comparator (index order doesn't matter) ---

TWO_SUM_CORRECT = """
def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
"""

TWO_SUM_WRONG = "def two_sum(nums, target):\n    return [0, 0]\n"


def test_two_sum_correct_passes_with_unordered_comparator():
    result = execute_code(
        code=TWO_SUM_CORRECT, language="python", entry_point="two_sum",
        comparator="unordered",
        test_cases=[{"input": {"nums": [3, 2, 4], "target": 6}, "expected_output": [1, 2]}],
    )
    assert result["status"] == "pass"


def test_two_sum_wrong_fails_with_actual():
    result = execute_code(
        code=TWO_SUM_WRONG, language="python", entry_point="two_sum",
        comparator="unordered",
        test_cases=[{"input": {"nums": [3, 2, 4], "target": 6}, "expected_output": [1, 2]}],
    )
    assert result["status"] == "fail"
    assert result["test_results"][0]["actual"] == [0, 0]


# --- merge_two_sorted_lists: linked_list arg_types + return_type ---

MERGE_CORRECT = """
def merge_two_lists(list1, list2):
    dummy = ListNode()
    tail = dummy
    while list1 and list2:
        if list1.val <= list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail = tail.next
    tail.next = list1 or list2
    return dummy.next
"""

MERGE_WRONG = """
def merge_two_lists(list1, list2):
    return list1
"""


def test_merge_two_sorted_lists_correct_passes():
    result = execute_code(
        code=MERGE_CORRECT, language="python", entry_point="merge_two_lists",
        arg_types={"list1": "linked_list", "list2": "linked_list"},
        return_type="linked_list",
        test_cases=[{
            "input": {"list1": [1, 2, 4], "list2": [1, 3, 4]},
            "expected_output": [1, 1, 2, 3, 4, 4],
        }],
    )
    assert result["status"] == "pass"


def test_merge_two_sorted_lists_wrong_fails_with_actual():
    result = execute_code(
        code=MERGE_WRONG, language="python", entry_point="merge_two_lists",
        arg_types={"list1": "linked_list", "list2": "linked_list"},
        return_type="linked_list",
        test_cases=[{
            "input": {"list1": [1, 2, 4], "list2": [1, 3, 4]},
            "expected_output": [1, 1, 2, 3, 4, 4],
        }],
    )
    assert result["status"] == "fail"
    assert result["test_results"][0]["actual"] == [1, 2, 4]


# --- lru_cache: ops_sequence class mode ---

LRU_CORRECT = """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key):
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
"""

LRU_WRONG = """
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        return self.cache.get(key, -1)

    def put(self, key, value):
        self.cache[key] = value
"""

LRU_CASE = {
    "input": {
        "operations": ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"],
        "arguments": [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]],
    },
    "expected_output": [None, None, None, 1, None, -1, None, -1, 3, 4],
}


def test_lru_cache_correct_passes():
    result = execute_code(
        code=LRU_CORRECT, language="python", entry_point="LRUCache",
        comparator="ops_sequence",
        test_cases=[LRU_CASE],
    )
    assert result["status"] == "pass"


def test_lru_cache_wrong_fails_with_actual():
    # LRU_WRONG never evicts, so it never returns -1 for the evicted key.
    result = execute_code(
        code=LRU_WRONG, language="python", entry_point="LRUCache",
        comparator="ops_sequence",
        test_cases=[LRU_CASE],
    )
    assert result["status"] == "fail"
    assert result["test_results"][0]["actual"] != LRU_CASE["expected_output"]
