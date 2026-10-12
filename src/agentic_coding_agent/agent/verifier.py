"""Verifier — step 3.4. Deterministic, no LLM call.

Decides pass / retry / give_up from execute_code's output. A model
grading its own output is the failure mode guardrails exist to
prevent, so this is plain code.

"Pass" is recomputed from test_results, never taken from the status
string alone: the result must say "pass", contain exactly the
expected number of test results, and every one must have passed.
Anything else is not a pass (never report a false "pass").
"""
from __future__ import annotations

from typing import Any

# Max 3 retries per subtask -> 1 first attempt + 3 retries.
MAX_RETRIES = 3

_KNOWN_STATUSES = {"pass", "fail", "error", "timeout", "resource_exceeded"}


def verify(
    exec_result: dict[str, Any],
    expected_count: int,
    attempt: int,
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    """Returns {decision: "pass"|"retry"|"give_up", reason, passed, total}.

    `attempt` is 1-based: attempt 1 is the first try, so retries used
    so far = attempt - 1."""
    status = exec_result.get("status")
    results = exec_result.get("test_results") or []
    passed = sum(1 for r in results if r.get("passed") is True)
    summary = {"passed": passed, "total": expected_count}

    if status == "pass" and len(results) == expected_count and passed == expected_count and expected_count > 0:
        return {"decision": "pass", "reason": "all_tests_passed", **summary}

    if status == "pass":
        # Harness said pass but the results don't back it up.
        reason = "pass_status_not_backed_by_results"
    elif status in _KNOWN_STATUSES:
        reason = status
    else:
        reason = "unknown_status"

    retries_used = attempt - 1
    if retries_used < max_retries:
        return {"decision": "retry", "reason": reason, **summary}
    return {"decision": "give_up", "reason": f"retries_exhausted:{reason}", **summary}
