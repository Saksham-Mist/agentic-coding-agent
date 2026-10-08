"""Sandbox entrypoint — step 2.3: full harness (function + class mode).

Reads a JSON payload from stdin:
  { code, entry_point, test_cases,
    comparator?: "exact"|"unordered"|"unordered_nested"|"ops_sequence",
    arg_types?: {argname: "linked_list"|"tree"|"linked_list_list"},
    return_type?: "linked_list"|"tree" }

Prints the the design notes output shape to stdout. See
the design notes "Harness modes" for what each comparator/mode means.
"""
import contextlib
import io
import json
import sys
import traceback


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


def _run_function_mode(namespace, entry_point, test_cases, comparator, arg_types, return_type,
                        stdout_buf, stderr_buf):
    fn = namespace[entry_point]
    test_results = []
    for case in test_cases:
        raw_input = case["input"]
        expected = case["expected_output"]
        try:
            converted_input = _convert_input(raw_input, arg_types or {})
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                raw_actual = fn(**converted_input)
            actual = _convert_output(raw_actual, return_type)
            test_results.append({
                "input": raw_input, "expected": expected, "actual": actual,
                "passed": _compare(actual, expected, comparator),
            })
        except Exception:
            test_results.append({
                "input": raw_input, "expected": expected, "actual": None,
                "passed": False,
            })
            stderr_buf.write(traceback.format_exc())
    return test_results


def _run_class_mode(namespace, entry_point, test_cases, stdout_buf, stderr_buf):
    cls = namespace[entry_point]
    test_results = []
    for case in test_cases:
        raw_input = case["input"]
        expected = case["expected_output"]
        operations = raw_input["operations"]
        arguments = raw_input["arguments"]
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                obj = cls(*arguments[0])
                recorded = [None]
                for op, args in zip(operations[1:], arguments[1:]):
                    method = getattr(obj, op)
                    recorded.append(method(*args))
            test_results.append({
                "input": raw_input, "expected": expected, "actual": recorded,
                "passed": recorded == expected,
            })
        except Exception:
            test_results.append({
                "input": raw_input, "expected": expected, "actual": None,
                "passed": False,
            })
            stderr_buf.write(traceback.format_exc())
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

    if comparator == "ops_sequence":
        test_results = _run_class_mode(namespace, entry_point, test_cases, stdout_buf, stderr_buf)
    else:
        test_results = _run_function_mode(
            namespace, entry_point, test_cases, comparator, arg_types, return_type,
            stdout_buf, stderr_buf,
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
