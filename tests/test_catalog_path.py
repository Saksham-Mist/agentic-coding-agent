"""TASK_CATALOG_PATH selects the catalog, so a fresh clone (without the
private task_catalog.json) can run against the published example."""
from pathlib import Path

from agentic_coding_agent.agent import catalog

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "task_catalog.example.json"


def test_default_is_repo_root_catalog(monkeypatch):
    monkeypatch.delenv("TASK_CATALOG_PATH", raising=False)
    assert catalog.default_catalog_path() == ROOT / "task_catalog.json"


def test_relative_path_resolves_from_repo_root(monkeypatch):
    monkeypatch.setenv("TASK_CATALOG_PATH", "examples/task_catalog.example.json")
    assert catalog.default_catalog_path() == EXAMPLE


def test_absolute_path_is_used_as_is(monkeypatch, tmp_path):
    target = tmp_path / "mine.json"
    monkeypatch.setenv("TASK_CATALOG_PATH", str(target))
    assert catalog.default_catalog_path() == target


def test_example_catalog_is_valid_and_readable(monkeypatch):
    monkeypatch.setenv("TASK_CATALOG_PATH", "examples/task_catalog.example.json")
    loaded = catalog.load_catalog()
    assert list(loaded) == ["sum_list"]
    assert catalog.validate_entry(loaded["sum_list"]) == []
    spec = catalog.read_test_spec("sum_list")
    assert spec["function_signature"] == "def sum_list(nums: list[int]) -> int"
    visible = [c for c in loaded["sum_list"]["test_cases"] if c["visible"]]
    assert len(spec["test_cases"]) == len(visible) < len(loaded["sum_list"]["test_cases"])
