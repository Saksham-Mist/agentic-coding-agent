"""Step 3.4 done-when check: unit tests cover pass / fail / error /
timeout, plus the false-pass guards and retry exhaustion."""
import pytest

from agentic_coding_agent.agent.verifier import MAX_RETRIES, verify


def _results(*passed_flags):
    return [{"input": {}, "expected": 1, "actual": 1 if p else 0, "passed": p} for p in passed_flags]


def test_pass():
    out = verify({"status": "pass", "test_results": _results(True, True)}, expected_count=2, attempt=1)
    assert out == {"decision": "pass", "reason": "all_tests_passed", "passed": 2, "total": 2}


def test_fail_retries():
    out = verify({"status": "fail", "test_results": _results(True, False)}, expected_count=2, attempt=1)
    assert out["decision"] == "retry"
    assert out["reason"] == "fail"
    assert out["passed"] == 1


def test_error_retries():
    out = verify({"status": "error", "test_results": []}, expected_count=2, attempt=1)
    assert out["decision"] == "retry"
    assert out["reason"] == "error"


def test_timeout_retries():
    out = verify({"status": "timeout", "test_results": []}, expected_count=2, attempt=2)
    assert out["decision"] == "retry"
    assert out["reason"] == "timeout"


def test_resource_exceeded_retries():
    out = verify({"status": "resource_exceeded", "test_results": []}, expected_count=2, attempt=1)
    assert out["decision"] == "retry"


@pytest.mark.parametrize("status", ["fail", "error", "timeout", "resource_exceeded"])
def test_gives_up_after_max_retries(status):
    # attempt MAX_RETRIES + 1 is the last allowed try (1 first + 3 retries)
    out = verify({"status": status, "test_results": []}, expected_count=2, attempt=MAX_RETRIES + 1)
    assert out["decision"] == "give_up"
    assert out["reason"] == f"retries_exhausted:{status}"


def test_last_retry_is_still_allowed():
    out = verify({"status": "fail", "test_results": _results(False)}, expected_count=1, attempt=MAX_RETRIES)
    assert out["decision"] == "retry"


def test_pass_status_with_a_failed_case_is_not_a_pass():
    out = verify({"status": "pass", "test_results": _results(True, False)}, expected_count=2, attempt=1)
    assert out["decision"] == "retry"
    assert out["reason"] == "pass_status_not_backed_by_results"


def test_pass_status_with_missing_results_is_not_a_pass():
    # Harness reported pass but returned fewer results than test cases.
    out = verify({"status": "pass", "test_results": _results(True)}, expected_count=3, attempt=1)
    assert out["decision"] == "retry"
    assert out["reason"] == "pass_status_not_backed_by_results"


def test_pass_with_zero_cases_is_not_a_pass():
    out = verify({"status": "pass", "test_results": []}, expected_count=0, attempt=1)
    assert out["decision"] != "pass"


def test_truthy_but_not_true_passed_flag_is_not_counted():
    results = [{"passed": "yes"}, {"passed": True}]
    out = verify({"status": "pass", "test_results": results}, expected_count=2, attempt=1)
    assert out["decision"] != "pass"


def test_unknown_status_is_not_a_pass():
    out = verify({"status": "weird", "test_results": _results(True)}, expected_count=1, attempt=1)
    assert out["decision"] == "retry"
    assert out["reason"] == "unknown_status"


def test_missing_status_is_not_a_pass():
    out = verify({}, expected_count=1, attempt=1)
    assert out["decision"] == "retry"
