"""Step 1.4 done-when checks: unknown id -> structured not_found error,
known ids validate against read_test_spec's contract shape, hidden
test cases never leak out."""
from pathlib import Path

import pytest

from agentic_coding_agent.agent.catalog import (
    CatalogLoadError,
    get_task,
    load_catalog,
    read_test_spec,
)

CATALOG_PATH = Path(__file__).resolve().parents[1] / "task_catalog.json"


@pytest.fixture(autouse=True)
def _real_catalog_present():
    if not CATALOG_PATH.exists():
        pytest.skip("task_catalog.json not present in this checkout (local-only file)")


def test_unknown_task_id_returns_structured_error():
    result = read_test_spec("this_task_does_not_exist", path=CATALOG_PATH)
    assert result == {
        "status": "error",
        "reason": "not_found",
        "message": "no task_id 'this_task_does_not_exist' in the catalog",
    }


def test_known_task_id_validates_against_contract_shape():
    result = read_test_spec("two_sum", path=CATALOG_PATH)
    assert set(result.keys()) == {
        "task_id",
        "prompt",
        "function_signature",
        "test_cases",
        "constraints",
    }
    assert result["task_id"] == "two_sum"
    assert result["prompt"]
    assert result["function_signature"]
    assert isinstance(result["test_cases"], list)
    assert len(result["test_cases"]) > 0
    for case in result["test_cases"]:
        assert set(case.keys()) == {"input", "expected_output"}


def test_hidden_cases_are_not_returned():
    full_entry = get_task("two_sum", path=CATALOG_PATH)
    hidden_count = sum(1 for c in full_entry["test_cases"] if not c.get("visible"))
    visible_count = sum(1 for c in full_entry["test_cases"] if c.get("visible"))
    assert hidden_count > 0, "fixture should include at least one hidden case"

    result = read_test_spec("two_sum", path=CATALOG_PATH)
    assert len(result["test_cases"]) == visible_count


def test_all_18_catalog_entries_load_without_error():
    # The main eval set is the 18 planned tasks; owner-only
    # "stress" entries are counted separately.
    catalog = load_catalog(path=CATALOG_PATH)
    assert sum(1 for e in catalog.values() if e.get("set", "main") == "main") == 18


def test_malformed_catalog_raises_catalog_load_error(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(CatalogLoadError):
        load_catalog(path=bad_file)
