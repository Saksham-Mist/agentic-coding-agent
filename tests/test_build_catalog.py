"""Step 5.1 pipeline: expected outputs come from the reference solution
through the sandbox harness, cross-checked against a brute force.

Uses toy tasks in a temp dir, never the private reference/ files.
"""
import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_catalog", ROOT / "scripts" / "build_catalog.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def _entry(task_id, entry_point, comparator="exact", **extra):
    return {"task_id": task_id, "entry_point": entry_point, "comparator": comparator,
            "prompt": "toy task", "function_signature": f"def {entry_point}(...)",
            "constraints": "small", "test_cases": [], **extra}


def _setup(tmp_path, entry, reference, cases):
    (tmp_path / "reference" / "cases").mkdir(parents=True)
    (tmp_path / "reference" / f"{entry['task_id']}.py").write_text(textwrap.dedent(reference))
    (tmp_path / "reference" / "cases" / f"{entry['task_id']}.py").write_text(textwrap.dedent(cases))
    cat = tmp_path / "task_catalog.json"
    cat.write_text(json.dumps([entry]))
    return cat, tmp_path / "reference"


SORT_CASES = """
SEED = 7
VISIBLE = [{"nums": [3, 1, 2]}, {"nums": [5, 4]}]
EDGE = [{"nums": []}, {"nums": [1]}]
MAX = [{"nums": list(range(2000, 0, -1))}]
def generate(rng):
    return [{"nums": [rng.randint(-50, 50) for _ in range(rng.randint(0, 20))]} for _ in range(5)]
def brute(nums):
    return sorted(nums)
"""


def test_builds_cases_from_reference(tmp_path):
    cat, ref = _setup(tmp_path, _entry("toy_sort", "sort_it"),
                      "def sort_it(nums):\n    return sorted(nums)\n", SORT_CASES)
    report = bc.build_task("toy_sort", cat, ref)
    built = json.loads(cat.read_text())[0]

    assert report["cases"] == len(built["test_cases"]) >= 8
    assert report["by_kind"]["visible"] == 2 and report["by_kind"]["max"] == 1
    assert report["brute_checked"] == report["cases"] - 1  # brute skips the max case
    by_input = {json.dumps(c["input"]): c for c in built["test_cases"]}
    assert by_input[json.dumps({"nums": [3, 1, 2]})]["expected_output"] == [1, 2, 3]
    assert by_input[json.dumps({"nums": [3, 1, 2]})]["visible"] is True
    assert sum(1 for c in built["test_cases"] if c.get("max_size")) == 1
    assert sum(1 for c in built["test_cases"] if not c["visible"]) >= 6
    assert bc.validate_entry(built) == []


def test_build_is_reproducible(tmp_path):
    cat, ref = _setup(tmp_path, _entry("toy_sort", "sort_it"),
                      "def sort_it(nums):\n    return sorted(nums)\n", SORT_CASES)
    bc.build_task("toy_sort", cat, ref)
    first = cat.read_text()
    bc.build_task("toy_sort", cat, ref)
    assert cat.read_text() == first


def test_in_place_mutation_does_not_change_stored_inputs(tmp_path):
    # This reference sorts its input in place before returning it.
    cat, ref = _setup(tmp_path, _entry("toy_sort", "sort_it"),
                      "def sort_it(nums):\n    nums.sort()\n    return nums\n", SORT_CASES)
    bc.build_task("toy_sort", cat, ref)
    first = json.loads(cat.read_text())[0]["test_cases"][0]
    assert first["input"] == {"nums": [3, 1, 2]}
    assert first["expected_output"] == [1, 2, 3]


def test_wrong_reference_is_caught_by_brute(tmp_path):
    wrong = "def sort_it(nums):\n    return sorted(nums)[:3]\n"  # truncates longer lists
    cat, ref = _setup(tmp_path, _entry("toy_sort", "sort_it"), wrong, SORT_CASES)
    with pytest.raises(bc.BuildError, match="reference and brute disagree"):
        bc.build_task("toy_sort", cat, ref)
    assert json.loads(cat.read_text())[0]["test_cases"] == []  # nothing written


def test_reference_that_raises_fails_the_build(tmp_path):
    cat, ref = _setup(tmp_path, _entry("toy_sort", "sort_it"),
                      "def sort_it(nums):\n    return [nums[0]]\n", SORT_CASES)
    with pytest.raises(bc.BuildError, match="reference raised"):
        bc.build_task("toy_sort", cat, ref)


def test_empty_prompt_is_rejected(tmp_path):
    entry = _entry("toy_sort", "sort_it")
    entry["prompt"] = ""
    cat, ref = _setup(tmp_path, entry, "def sort_it(nums):\n    return sorted(nums)\n", SORT_CASES)
    with pytest.raises(bc.BuildError, match="'prompt' is empty"):
        bc.build_task("toy_sort", cat, ref)


def test_linked_list_conversion(tmp_path):
    entry = _entry("toy_rev", "reverse", arg_types={"head": "linked_list"}, return_type="linked_list")
    reference = """
    def reverse(head):
        prev = None
        while head:
            head.next, prev, head = prev, head, head.next
        return prev
    """
    cases = """
    VISIBLE = [{"head": [1, 2, 3]}]
    EDGE = [{"head": []}, {"head": [7]}]
    MAX = [{"head": list(range(500))}]
    def generate(rng):
        return [{"head": [rng.randint(0, 9) for _ in range(rng.randint(1, 8))]} for _ in range(6)]
    """
    cat, ref = _setup(tmp_path, entry, reference, cases)
    bc.build_task("toy_rev", cat, ref)
    built = json.loads(cat.read_text())[0]["test_cases"]
    assert built[0] == {"input": {"head": [1, 2, 3]}, "expected_output": [3, 2, 1], "visible": True}
    assert built[1]["expected_output"] == []


def test_class_mode(tmp_path):
    entry = _entry("toy_counter", "Counter", comparator="ops_sequence")
    entry["function_signature"] = "class Counter:"
    reference = """
    class Counter:
        def __init__(self, start):
            self.n = start
        def inc(self):
            self.n += 1
        def get(self):
            return self.n
    """
    cases = """
    VISIBLE = [{"operations": ["Counter", "inc", "get"], "arguments": [[5], [], []]}]
    EDGE = [{"operations": ["Counter", "get"], "arguments": [[0], []]}]
    MAX = [{"operations": ["Counter"] + ["inc"] * 1000 + ["get"], "arguments": [[0]] + [[]] * 1001}]
    def generate(rng):
        out = []
        for _ in range(6):
            n = rng.randint(1, 10)
            out.append({"operations": ["Counter"] + ["inc"] * n + ["get"], "arguments": [[n]] + [[]] * (n + 1)})
        return out
    """
    brute = """
    class _Brute:
        def __init__(self, start):
            self.calls = [start]
        def inc(self):
            self.calls.append(1)
        def get(self):
            return sum(self.calls)
    brute = _Brute
    """
    cat, ref = _setup(tmp_path, entry, reference, textwrap.dedent(cases) + textwrap.dedent(brute))
    report = bc.build_task("toy_counter", cat, ref)
    built = json.loads(cat.read_text())[0]["test_cases"]
    assert built[0]["expected_output"] == [None, None, 6]
    assert built[-1]["expected_output"][-1] == 1000
    assert report["brute_checked"] == report["cases"] - 1  # class-mode brute check runs and agrees


def test_class_mode_brute_catches_wrong_reference(tmp_path):
    entry = _entry("toy_counter", "Counter", comparator="ops_sequence")
    entry["function_signature"] = "class Counter:"
    wrong = """
    class Counter:
        def __init__(self, start):
            self.n = start
        def inc(self):
            self.n += 2
        def get(self):
            return self.n
    """
    cases = """
    VISIBLE = [{"operations": ["Counter", "inc", "get"], "arguments": [[5], [], []]}]
    EDGE = [{"operations": ["Counter", "get"], "arguments": [[0], []]}]
    MAX = [{"operations": ["Counter", "get"], "arguments": [[1], []]}]
    def generate(rng):
        return [{"operations": ["Counter"] + ["inc"] * n + ["get"], "arguments": [[0]] + [[]] * (n + 1)} for n in range(1, 7)]
    class _Brute:
        def __init__(self, start):
            self.n = start
        def inc(self):
            self.n += 1
        def get(self):
            return self.n
    brute = _Brute
    """
    cat, ref = _setup(tmp_path, entry, wrong, cases)
    with pytest.raises(bc.BuildError, match="reference and brute disagree"):
        bc.build_task("toy_counter", cat, ref)


def test_check_reports_invalid_entries(tmp_path, capsys):
    cat = tmp_path / "task_catalog.json"
    cat.write_text(json.dumps([_entry("toy", "f")]))
    assert bc.check(cat) == 1
    assert "only 0 test cases" in capsys.readouterr().out


def test_unreviewed_draft_builds_but_fails_check(tmp_path, capsys):
    entry = _entry("toy_sort", "sort_it", reviewed=False)
    cat, ref = _setup(tmp_path, entry, "def sort_it(nums):\n    return sorted(nums)\n", SORT_CASES)
    bc.build_task("toy_sort", cat, ref)  # building isn't blocked by review status
    assert bc.check(cat) == 1
    assert "unreviewed draft" in capsys.readouterr().out

    built = json.loads(cat.read_text())
    built[0]["reviewed"] = True
    cat.write_text(json.dumps(built))
    assert bc.check(cat) == 0
