"""task_catalog.json loader + the read_test_spec tool.

read_test_spec below is the tool the agent sees; this
implements, and its "Harness modes" section for why test_cases here
are visible-only (hidden cases exist so the agent can't special-case
the expected output) and what arg_types/comparator/entry_point are
used for later, in the sandbox harness (Phase 2) — not exposed here.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]


def default_catalog_path() -> Path:
    """TASK_CATALOG_PATH if set (relative paths are from the repo root),
    else task_catalog.json at the repo root. The real catalog holds the
    hidden test cases and isn't published, so a fresh clone points this
    at examples/task_catalog.example.json instead. Read on every call, so
    the API, the agent and the eval script all follow the same setting."""
    configured = os.environ.get("TASK_CATALOG_PATH")
    if configured:
        p = Path(configured)
        return p if p.is_absolute() else _REPO_ROOT / p
    return _REPO_ROOT / "task_catalog.json"

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

    catalog_path = path or default_catalog_path()
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


# --- step 5.1: catalog entry validation ---

COMPARATORS = {"exact", "unordered", "unordered_nested", "ops_sequence"}
ARG_TYPES = {"linked_list", "tree", "linked_list_list"}
RETURN_TYPES = {"linked_list", "tree"}
MIN_CASES = 8
# "main" = the 18-task eval set served publicly; "stress" = owner-only
# twisted variants used to exercise retries.
TASK_SETS = {"main", "stress"}


def task_set(entry: dict[str, Any]) -> str:
    return entry.get("set", "main")


def validate_entry(entry: dict[str, Any]) -> list[str]:
    """Returns a list of problems (empty = valid). Rules: prompt,
    signature and constraints filled; signature defines entry_point;
    known comparator/arg types; >= 8 cases with at least one visible,
    one hidden and one marked max_size; well-formed class-mode cases."""
    problems: list[str] = []
    for key in ("task_id", "entry_point", "comparator", "prompt", "function_signature",
                "constraints", "test_cases"):
        if key not in entry:
            problems.append(f"missing field: {key}")
    if problems:
        return problems

    ep = entry["entry_point"]
    if entry.get("reviewed") is False:
        problems.append("prompt/signature/constraints are an unreviewed draft (owner sets reviewed: true)")
    for key in ("prompt", "function_signature", "constraints"):
        if not str(entry[key]).strip():
            problems.append(f"{key} is empty")
    sig = entry["function_signature"]
    if sig.strip() and f"def {ep}(" not in sig and f"class {ep}" not in sig:
        problems.append(f"function_signature does not define entry_point {ep!r}")
    if task_set(entry) not in TASK_SETS:
        problems.append(f"unknown set {entry.get('set')!r}")
    if entry["comparator"] not in COMPARATORS:
        problems.append(f"unknown comparator {entry['comparator']!r}")
    for arg, t in (entry.get("arg_types") or {}).items():
        if t not in ARG_TYPES:
            problems.append(f"arg_types[{arg!r}] has unknown type {t!r}")
    if entry.get("return_type") not in (None, *RETURN_TYPES):
        problems.append(f"unknown return_type {entry['return_type']!r}")

    cases = entry["test_cases"]
    if len(cases) < MIN_CASES:
        problems.append(f"only {len(cases)} test cases (need >= {MIN_CASES})")
    for i, c in enumerate(cases):
        if not isinstance(c.get("input"), dict) or "expected_output" not in c \
                or not isinstance(c.get("visible"), bool):
            problems.append(f"case {i}: needs input (object), expected_output and visible (bool)")
            continue
        if entry["comparator"] == "ops_sequence":
            ops, args = c["input"].get("operations"), c["input"].get("arguments")
            if not isinstance(ops, list) or not isinstance(args, list) or len(ops) != len(args):
                problems.append(f"case {i}: class mode needs operations/arguments of equal length")
    if cases and not any(c.get("visible") for c in cases):
        problems.append("no visible case")
    if cases and all(c.get("visible") for c in cases):
        problems.append("no hidden case")
    if cases and not any(c.get("max_size") for c in cases):
        problems.append("no case marked max_size")
    return problems


def validate_catalog(path: Path | None = None) -> dict[str, list[str]]:
    """{task_id: problems} for every entry; all-empty lists = valid."""
    return {tid: validate_entry(e) for tid, e in load_catalog(path).items()}
