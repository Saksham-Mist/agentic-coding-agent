"""task_catalog.json loader + the read_test_spec tool.

See the design notes "Tool: read_test_spec" for the contract this
implements, and its "Harness modes" section for why test_cases here
are visible-only (hidden cases exist so the agent can't special-case
the expected output) and what arg_types/comparator/entry_point are
used for later, in the sandbox harness (Phase 2) — not exposed here.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[3] / "task_catalog.json"

_catalog_cache: dict[str, dict[str, Any]] | None = None
_catalog_path_cache: Path | None = None


class CatalogLoadError(Exception):
    """Raised when task_catalog.json is missing or malformed."""


def load_catalog(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load and cache task_catalog.json, keyed by task_id.

    Re-reads from disk only the first time, or if `path` differs from
    the last call — callers in tests pass their own `path` to avoid
    the cache entirely.
    """
    global _catalog_cache, _catalog_path_cache

    catalog_path = path or _DEFAULT_CATALOG_PATH
    if _catalog_cache is not None and _catalog_path_cache == catalog_path:
        return _catalog_cache

    if not catalog_path.exists():
        raise CatalogLoadError(f"catalog file not found: {catalog_path}")

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise CatalogLoadError(f"catalog file is not valid JSON: {e}") from e

    if not isinstance(raw, list):
        raise CatalogLoadError("catalog file must be a JSON array of task entries")

    by_id: dict[str, dict[str, Any]] = {}
    for entry in raw:
        task_id = entry.get("task_id")
        if not task_id:
            raise CatalogLoadError("catalog entry missing task_id")
        by_id[task_id] = entry

    _catalog_cache = by_id
    _catalog_path_cache = catalog_path
    return by_id


def get_task(task_id: str, path: Path | None = None) -> dict[str, Any] | None:
    """Return the raw catalog entry (all fields, visible + hidden test
    cases, entry_point, comparator, arg_types) for internal/harness
    use. Not what gets handed to the agent — see read_test_spec."""
    return load_catalog(path).get(task_id)


def read_test_spec(task_id: str, path: Path | None = None) -> dict[str, Any]:
    """The read_test_spec tool. Returns the contract shape on success,
    or a structured {status, reason, message} error — never raises."""
    try:
        catalog = load_catalog(path)
    except CatalogLoadError as e:
        return {"status": "error", "reason": "read_failed", "message": str(e)}

    entry = catalog.get(task_id)
    if entry is None:
        return {
            "status": "error",
            "reason": "not_found",
            "message": f"no task_id '{task_id}' in the catalog",
        }

    visible_cases = [
        {"input": case["input"], "expected_output": case["expected_output"]}
        for case in entry.get("test_cases", [])
        if case.get("visible", False)
    ]

    return {
        "task_id": entry["task_id"],
        "prompt": entry.get("prompt", ""),
        "function_signature": entry.get("function_signature", ""),
        "test_cases": visible_cases,
        "constraints": entry.get("constraints", ""),
    }
