"""Build task_catalog.json test cases from reference solutions — step 5.1.

    python scripts/build_catalog.py two_sum          # build one task
    python scripts/build_catalog.py --all            # build every task that has files
    python scripts/build_catalog.py --check          # validate all 18 entries

Per task (the design notes "Filling in task_catalog.json"):
  task_catalog.json entry      prompt, function_signature, constraints (owner)
  reference/<task_id>.py       the owner's own accepted solution
  reference/cases/<task_id>.py the case inputs:
      VISIBLE  = [input, ...]     problem examples, shown to the agent
      EDGE     = [input, ...]     hand-picked edge cases (hidden)
      MAX      = [input, ...]     max-size cases (hidden, marked max_size)
      def generate(rng) -> [input, ...]   random cases (hidden)
      SEED     = int              fixed seed, so builds are reproducible
      brute    (optional)         an obviously-correct slow solution with the
                                  same interface; checked against the
                                  reference on every non-MAX case

Expected outputs are never hand-typed: they come from running the
reference through the same harness code the sandbox uses
(docker/sandbox/entrypoint.py), so conversion and comparison behave
exactly as they will when the agent is graded. Everything under
reference/ is gitignored (it's the answer key).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_coding_agent.agent.catalog import validate_entry  # noqa: E402

CATALOG = ROOT / "task_catalog.json"
REF_DIR = ROOT / "reference"
HARNESS = ROOT / "docker" / "sandbox" / "entrypoint.py"
SLOW_CASE_SECONDS = 1.0  # the sandbox runs all cases in one 10s window on 1 CPU


class BuildError(Exception):
    pass


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harness():
    return _load_module(HARNESS, "sandbox_harness")


def _run(h, namespace, entry, inputs):
    """Run `namespace[entry_point]` on inputs via the harness; returns
    (results, seconds per case)."""
    cases = [{"input": copy.deepcopy(i), "expected_output": None} for i in inputs]
    timings, results = [], []
    for case in cases:
        t0 = time.perf_counter()
        if entry["comparator"] == "ops_sequence":
            r = h._run_class_mode(namespace, entry["entry_point"], [case])
        else:
            r = h._run_function_mode(namespace, entry["entry_point"], [case], entry["comparator"],
                                     entry.get("arg_types"), entry.get("return_type"))
        timings.append(time.perf_counter() - t0)
        results.append(r[0])
    return results, timings


def build_task(task_id: str, catalog_path: Path = CATALOG, ref_dir: Path = REF_DIR,
               write: bool = True) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    entry = next((e for e in catalog if e["task_id"] == task_id), None)
    if entry is None:
        raise BuildError(f"{task_id}: not in {catalog_path.name}")
    for key in ("prompt", "function_signature", "constraints"):
        if not str(entry.get(key, "")).strip():
            raise BuildError(f"{task_id}: '{key}' is empty in {catalog_path.name} (owner fills this)")

    ref_file, cases_file = ref_dir / f"{task_id}.py", ref_dir / "cases" / f"{task_id}.py"
    for f in (ref_file, cases_file):
        if not f.exists():
            raise BuildError(f"{task_id}: missing {f.relative_to(ref_dir.parent)}")

    h = _harness()
    ns: dict[str, Any] = {"ListNode": h.ListNode, "TreeNode": h.TreeNode}
    exec(compile(ref_file.read_text(encoding="utf-8"), str(ref_file), "exec"), ns)
    ep = entry["entry_point"]
    if ep not in ns:
        raise BuildError(f"{task_id}: reference does not define {ep!r}")

    cm = _load_module(cases_file, f"cases_{task_id}")
    # A brute force for linked-list/tree tasks needs the harness's own
    # node classes (they aren't importable from the case file itself).
    cm.ListNode, cm.TreeNode = h.ListNode, h.TreeNode
    rng = random.Random(getattr(cm, "SEED", 0))
    groups = [("visible", getattr(cm, "VISIBLE", [])), ("edge", getattr(cm, "EDGE", [])),
              ("random", cm.generate(rng) if hasattr(cm, "generate") else []),
              ("max", getattr(cm, "MAX", []))]

    seen, inputs, kinds = set(), [], []
    for kind, items in groups:
        for i in items:
            key = json.dumps(i, sort_keys=True)
            if key not in seen:  # duplicates dropped; the first (e.g. visible) wins
                seen.add(key)
                inputs.append(i)
                kinds.append(kind)

    results, timings = _run(h, ns, entry, inputs)
    failed = [(i, r["error"]) for i, r in zip(inputs, results) if r.get("error")]
    if failed:
        i, err = failed[0]
        raise BuildError(f"{task_id}: reference raised on {json.dumps(i)[:200]}:\n{err[-600:]}")

    brute_checked = 0
    brute = getattr(cm, "brute", None)
    if brute is not None:
        bns = {"ListNode": h.ListNode, "TreeNode": h.TreeNode, ep: brute}
        check = [(i, r) for i, r, k in zip(inputs, results, kinds) if k != "max"]
        bres, _ = _run(h, bns, entry, [i for i, _ in check])
        def same(a, b):
            # Class mode replays a call sequence; its results compare
            # exactly (the harness's _compare covers function mode only
            # and fails closed on "ops_sequence").
            if entry["comparator"] == "ops_sequence":
                return a == b
            return h._compare(a, b, entry["comparator"])

        for (i, ref_r), b in zip(check, bres):
            if b.get("error") or not same(b["actual"], ref_r["actual"]):
                raise BuildError(
                    f"{task_id}: reference and brute disagree on {json.dumps(i)[:200]}\n"
                    f"  reference: {json.dumps(ref_r['actual'])[:200]}\n"
                    f"  brute:     {json.dumps(b.get('actual'))[:200]} {b.get('error') or ''}"
                )
        brute_checked = len(check)

    test_cases = []
    for i, r, k in zip(inputs, results, kinds):
        case = {"input": i, "expected_output": r["actual"], "visible": k == "visible"}
        if k == "max":
            case["max_size"] = True
        test_cases.append(case)

    new_entry = {**entry, "test_cases": test_cases}
    # Building is allowed before the owner reviews the drafted text; only
    # --check (the step 5.1 gate) requires reviewed entries.
    problems = [p for p in validate_entry(new_entry) if "unreviewed draft" not in p]
    if problems:
        raise BuildError(f"{task_id}: built entry is invalid: {'; '.join(problems)}")

    if write:
        out = [new_entry if e["task_id"] == task_id else e for e in catalog]
        tmp = catalog_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        tmp.replace(catalog_path)  # atomic: a crash never leaves a half-written catalog

    slowest = max(timings) if timings else 0.0
    return {
        "task_id": task_id,
        "cases": len(test_cases),
        "by_kind": {k: kinds.count(k) for k in ("visible", "edge", "random", "max")},
        "brute_checked": brute_checked,
        "slowest_case_seconds": round(slowest, 3),
        "total_seconds": round(sum(timings), 3),
        "warning": (f"slowest case {slowest:.2f}s locally; the sandbox runs all cases in 10s on 1 CPU"
                    if slowest > SLOW_CASE_SECONDS else None),
    }


def check(catalog_path: Path = CATALOG) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    bad = 0
    for e in catalog:
        problems = validate_entry(e)
        bad += bool(problems)
        print(f"{e['task_id']:38} {'OK' if not problems else '; '.join(problems)}")
    print(f"\n{len(catalog) - bad}/{len(catalog)} entries valid")
    return 0 if bad == 0 else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("task_ids", nargs="*")
    ap.add_argument("--all", action="store_true", help="build every task that has reference + cases files")
    ap.add_argument("--check", action="store_true", help="validate all entries, build nothing")
    args = ap.parse_args(argv)

    if args.check:
        return check()
    ids = args.task_ids
    if args.all:
        ids = sorted(p.stem for p in (REF_DIR / "cases").glob("*.py") if (REF_DIR / f"{p.stem}.py").exists())
    if not ids:
        ap.print_help()
        return 2
    status = 0
    for tid in ids:
        try:
            print(json.dumps(build_task(tid)))
        except BuildError as e:
            print(f"FAILED {e}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
