"""Sandbox entrypoint — step 2.2: basic function-mode harness.

Reads a JSON payload from stdin: {code, entry_point, test_cases}.
Executes `code` in an isolated namespace, calls entry_point(**input)
for each test case, and prints the the design notes output shape to
stdout.

This step covers plain-value arguments and exact comparison only.
Step 2.3 extends this with arg_types conversion (linked_list/tree),
non-exact comparators (unordered/unordered_nested), and class mode
(ops_sequence, for lru_cache) — this file's I/O contract (stdin JSON
in, one JSON line out) does not change when that happens.
"""
import contextlib
import io
import json
import sys
import traceback


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

    namespace: dict = {}
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            exec(code, namespace)
            fn = namespace[entry_point]
    except Exception:
        print(json.dumps({
            "status": "error",
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue() + traceback.format_exc(),
            "test_results": [],
        }))
        return

    test_results = []
    for case in test_cases:
        test_input = case["input"]
        expected = case["expected_output"]
        try:
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                actual = fn(**test_input)
            test_results.append({
                "input": test_input, "expected": expected, "actual": actual,
                "passed": actual == expected,
            })
        except Exception:
            test_results.append({
                "input": test_input, "expected": expected, "actual": None,
                "passed": False,
            })
            stderr_buf.write(traceback.format_exc())

    overall_status = "pass" if all(r["passed"] for r in test_results) else "fail"
    print(json.dumps({
        "status": overall_status,
        "stdout": stdout_buf.getvalue(),
        "stderr": stderr_buf.getvalue(),
        "test_results": test_results,
    }))


if __name__ == "__main__":
    main()
