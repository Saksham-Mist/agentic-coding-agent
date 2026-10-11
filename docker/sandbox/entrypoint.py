"""Sandbox entrypoint — step 2.3: full harness (function + class mode).

Reads a JSON payload from stdin:
  { code, entry_point, test_cases,
    comparator?: "exact"|"unordered"|"unordered_nested"|"ops_sequence",
    arg_types?: {argname: "linked_list"|"tree"|"linked_list_list"},
    return_type?: "linked_list"|"tree" }

Prints the the design notes output shape to stdout. See
the design notes "Harness modes" for what each comparator/mode means.

Each test case's printed output and exception are captured separately
and attached to that case's result (stdout/stderr/error), so hidden
cases' data never mixes into the shared output the agent sees.
"""
import contextlib
import copy
import io
import json
import sys
import traceback

# Python's default limit (1000) would fail correct recursive solutions on
# inputs within the stated constraints, which measures an interpreter
# setting, not the algorithm (LeetCode raises it too). 200,000 covers the
# deepest legitimate case: recursive DFS on an all-land 300x300 grid goes
# ~90,000 deep (measured: passes with this limit, fails with 10,000).
# Python 3.11 keeps pure-Python frames on the heap, so this doesn't crash;
# runaway recursion is still bounded by --memory and the 10s timeout.
# Set at import so it also applies when scripts/build_catalog.py loads this.
sys.setrecursionlimit(200_000)


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def _array_to_linked_list(arr):
    head = None
    tail = None
    for v in arr:
        node = ListNode(v)
        if head is None:
            head = node
        else:
            tail.next = node
        tail = node
    return head


def _linked_list_to_array(node):
    out = []
    while node is not None:
        out.append(node.val)
        node = node.next
    return out


def _array_to_tree(arr):
    """LeetCode-style level-order array (null = missing child) -> TreeNode."""
    if not arr or arr[0] is None:
        return None
    it = iter(arr)
    root = TreeNode(next(it))
    queue = [root]
    while queue:
        node = queue.pop(0)
        try:
            left_val = next(it)
        except StopIteration:
            break
        if left_val is not None:
            node.left = TreeNode(left_val)
            queue.append(node.left)
        try:
            right_val = next(it)
        except StopIteration:
            break
        if right_val is not None:
            node.right = TreeNode(right_val)
            queue.append(node.right)
    return root


def _tree_to_array(root):
    """TreeNode -> level-order array with None for missing children,
    trailing Nones trimmed (matches LeetCode's own output convention)."""
    if root is None:
        return []
    out = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node is None:
            out.append(None)
            continue
        out.append(node.val)
        queue.append(node.left)
        queue.append(node.right)
    while out and out[-1] is None:
        out.pop()
    return out


def _convert_input(test_input: dict, arg_types: dict) -> dict:
    converted = dict(test_input)
    for arg_name, type_name in arg_types.items():
        if arg_name not in converted:
            continue
        value = converted[arg_name]
        if type_name == "linked_list":
            converted[arg_name] = _array_to_linked_list(value)
        elif type_name == "tree":
            converted[arg_name] = _array_to_tree(value)
        elif type_name == "linked_list_list":
            converted[arg_name] = [_array_to_linked_list(v) for v in value]
    return converted


def _convert_output(value, return_type: str | None):
    if return_type == "linked_list":
        return _linked_list_to_array(value)
    if return_type == "tree":
        return _tree_to_array(value)
    return value


def _compare(actual, expected, comparator: str) -> bool:
    if comparator == "exact":
        return actual == expected
    if comparator == "unordered":
        try:
            return sorted(actual) == sorted(expected)
        except TypeError:
            return actual == expected
    if comparator == "unordered_nested":
        def normalize(groups):
            return sorted(tuple(sorted(g)) for g in groups)
        try:
            return normalize(actual) == normalize(expected)
        except TypeError:
            return actual == expected
    # Unknown comparator: fail closed rather than silently claim a pass.
    return False


MAX_CASE_OUTPUT_CHARS = 2000


def _case_error() -> str:
    """The current exception's traceback, trimmed from the front so the
    exception line itself (at the end) is always kept."""
    tb = traceback.format_exc()
    return tb[-MAX_CASE_OUTPUT_CHARS:]


def _run_case(fn_call):
    """Runs one test case with its own stdout/stderr capture. Returns
    (value, stdout, stderr, error). Everything a case prints or raises
    stays attached to that case's result, never to the shared output,
    so the caller can show visible cases' details without exposing
    hidden ones (their inputs can appear in printed text and in
    exception messages, e.g. KeyError: <input value>)."""
    out, err = io.StringIO(), io.StringIO()
    value, error = None, None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            value = fn_call()
    except Exception:
        error = _case_error()
    return value, out.getvalue()[-MAX_CASE_OUTPUT_CHARS:], err.getvalue()[-MAX_CASE_OUTPUT_CHARS:], error


def _result(raw_input, expected, actual, passed, stdout, stderr, error):
    r = {"input": raw_input, "expected": expected, "actual": actual, "passed": passed}
    if stdout:
        r["stdout"] = stdout
    if stderr:
        r["stderr"] = stderr
    if error:
        r["error"] = error
    return r


def _run_function_mode(namespace, entry_point, test_cases, comparator, arg_types, return_type):
    fn = namespace[entry_point]
    test_results = []
    for case in test_cases:
        raw_input = case["input"]
        expected = case["expected_output"]

        def call(raw_input=raw_input):
            # deepcopy: a solution may sort or modify its input in place,
            # which must not change the input reported back (or stored).
            converted = _convert_input(copy.deepcopy(raw_input), arg_types or {})
            return _convert_output(fn(**converted), return_type)

        actual, out, err, error = _run_case(call)
        passed = error is None and _compare(actual, expected, comparator)
        test_results.append(_result(raw_input, expected, actual, passed, out, err, error))
    return test_results


def _run_class_mode(namespace, entry_point, test_cases):
    cls = namespace[entry_point]
    test_results = []
    for case in test_cases:
        raw_input = case["input"]
        expected = case["expected_output"]

        def call(raw_input=raw_input):
            operations = raw_input["operations"]
            arguments = copy.deepcopy(raw_input["arguments"])
            obj = cls(*arguments[0])
            recorded = [None]
            for op, args in zip(operations[1:], arguments[1:]):
                recorded.append(getattr(obj, op)(*args))
            return recorded

        actual, out, err, error = _run_case(call)
        passed = error is None and actual == expected
        test_results.append(_result(raw_input, expected, actual, passed, out, err, error))
    return test_results


def main() -> None:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({
            "status": "error", "stdout": "", "stderr": f"invalid payload: {e}",
            "test_results": [],
        }))
        return

    code = payload.get("code", "")
    entry_point = payload.get("entry_point", "")
    test_cases = payload.get("test_cases", [])
    comparator = payload.get("comparator", "exact")
    arg_types = payload.get("arg_types")
    return_type = payload.get("return_type")

    namespace: dict = {"ListNode": ListNode, "TreeNode": TreeNode}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, namespace)
            if entry_point not in namespace:
                raise NameError(f"entry_point {entry_point!r} not defined by submitted code")
    except Exception:
        print(json.dumps({
            "status": "error",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue() + traceback.format_exc(),
            "test_results": [],
        }))
        return

    # From here on the shared stdout/stderr only hold load-time output
    # (from exec above); per-case output lives on each test result.
    if comparator == "ops_sequence":
        test_results = _run_class_mode(namespace, entry_point, test_cases)
    else:
        test_results = _run_function_mode(
            namespace, entry_point, test_cases, comparator, arg_types, return_type,
        )

    overall_status = "pass" if all(r["passed"] for r in test_results) else "fail"
    print(json.dumps({
        "status": overall_status,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "test_results": test_results,
    }))


if __name__ == "__main__":
    main()
