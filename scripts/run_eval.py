"""Eval harness — step 5.2.

    python scripts/run_eval.py                     # all runnable tasks, 3 repeats
    python scripts/run_eval.py --tasks two_sum valid_parentheses --repeats 1
    python scripts/run_eval.py --report <batch_id> # rebuild a past batch's table

Submits every (task, repeat) through the API with the owner token
(OWNER_TOKEN, tagged source=eval), follows each run to the end, records
it in eval_batches / eval_runs, and prints the results table. Repeats
are interleaved (all tasks once, then again), so a short outage hits
different tasks rather than every repeat of one task.

Every batch stores the exact command, git commit and model, so each
number can be traced back to what produced it.
The model recorded is the one this script's environment resolves
(GEMINI_MODEL or the code default); start the server from the same .env.
The default task list is whatever the API serves: catalog entries that
pass full validation (owner-reviewed, >= 8 cases, a max-size case).
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_coding_agent.agent.catalog import load_catalog, task_set, validate_entry  # noqa: E402
from agentic_coding_agent.agent.llm import DEFAULT_MODEL  # noqa: E402

DEFAULT_REPEATS = 3          # runs per task
RUN_TIMEOUT_SECONDS = 900    # well above 15 LLM steps + sandbox time
POLL_SECONDS = 2.0

# Difficulty per task, as planned when the catalog was chosen.
DIFFICULTY = {
    "two_sum": "Easy", "valid_parentheses": "Easy", "reverse_integer": "Easy",
    "merge_two_sorted_lists": "Easy", "maximum_subarray": "Easy", "climbing_stairs": "Easy",
    "best_time_to_buy_sell_stock": "Easy", "group_anagrams": "Medium",
    "longest_substring_without_repeating": "Medium", "three_sum": "Medium",
    "binary_tree_level_order_traversal": "Medium", "course_schedule": "Medium",
    "coin_change": "Medium", "number_of_islands": "Medium", "lru_cache": "Hard",
    "merge_k_sorted_lists": "Hard", "trapping_rain_water": "Hard", "word_ladder": "Hard",
    # stress set (twisted variants; difficulty is my estimate)
    "stairs_no_double_two": "Medium", "rain_water_with_drains": "Hard",
    "two_sum_value_pairs": "Medium", "anagram_groups_sorted": "Medium",
}


def runnable_tasks(which: str = "main") -> list[str]:
    return sorted(t for t, e in load_catalog().items()
                  if not validate_entry(e) and task_set(e) == which)


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def run_one(client: httpx.Client, task_id: str, token: str, poll_seconds: float,
            timeout: float) -> tuple[str | None, str | None]:
    """Submit one run and wait for it to finish. Returns (run_id, error)."""
    r = client.post("/runs", json={"task_id": task_id},
                    headers={"Authorization": f"Bearer {token}", "X-Run-Source": "eval"})
    if r.status_code != 202:
        return None, f"HTTP {r.status_code}: {r.text[:300]}"
    run_id = r.json()["run_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        g = client.get(f"/runs/{run_id}")
        if g.status_code == 200 and g.json()["status"] not in ("queued", "running"):
            return run_id, None
        time.sleep(poll_seconds)
    return run_id, f"timed out after {timeout:.0f}s waiting for the run"


def run_eval(client: httpx.Client, store, task_ids: list[str], repeats: int, token: str,
             label: str | None, command: str, model: str | None, commit: str | None,
             poll_seconds: float = POLL_SECONDS, timeout: float = RUN_TIMEOUT_SECONDS,
             log=print) -> str:
    batch_id = store.create_eval_batch(label, command, commit, model, repeats, task_ids)
    log(f"eval batch {batch_id}: {len(task_ids)} tasks x {repeats} repeats")
    for rep in range(1, repeats + 1):
        for task_id in task_ids:
            t0 = time.monotonic()
            run_id, error = run_one(client, task_id, token, poll_seconds, timeout)
            store.add_eval_run(batch_id, task_id, rep, run_id, error)
            status = error or client.get(f"/runs/{run_id}").json()["status"]
            log(f"  [{rep}/{repeats}] {task_id:36} {status}  ({time.monotonic() - t0:.0f}s)")
    store.finish_eval_batch(batch_id)
    return batch_id


def _avg(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate eval rows into per-task, per-difficulty and overall numbers."""
    by_task: dict[str, list[dict]] = {}
    for r in rows:
        by_task.setdefault(r["task_id"], []).append(r)

    def block(rs: list[dict]) -> dict[str, Any]:
        passed = [r for r in rs if r["status"] == "passed"]
        finished = [r for r in rs if r["status"] in ("passed", "incomplete", "error")]
        return {
            "runs": len(rs),
            "runs_passed": len(passed),
            "run_success_rate": round(len(passed) / len(rs), 3) if rs else None,
            "avg_steps_passed": _avg([r["steps"] for r in passed]),
            "avg_tokens_per_run": _avg([r["tokens"] for r in finished]),
            "avg_seconds_per_run": _avg([r["seconds"] for r in finished]),
            "not_finished": len(rs) - len(finished),
        }

    tasks = {}
    for t, rs in sorted(by_task.items()):
        tasks[t] = {**block(rs), "difficulty": DIFFICULTY.get(t, "?"),
                    "statuses": [r["status"] or f"error: {r['error']}" for r in sorted(rs, key=lambda r: r["repeat"])]}

    def task_counts(names: list[str]) -> dict[str, int]:
        return {
            "tasks": len(names),
            "tasks_passed_every_repeat": sum(tasks[t]["runs_passed"] == tasks[t]["runs"] for t in names),
            "tasks_passed_at_least_once": sum(tasks[t]["runs_passed"] > 0 for t in names),
        }

    by_difficulty = {}
    for d in ("Easy", "Medium", "Hard"):
        names = [t for t in tasks if tasks[t]["difficulty"] == d]
        if names:
            by_difficulty[d] = {**block([r for t in names for r in by_task[t]]), **task_counts(names)}
    return {"overall": {**block(rows), **task_counts(list(tasks))},
            "by_difficulty": by_difficulty, "tasks": tasks}


def _fmt(x, digits=1):
    return "-" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def render(batch: dict[str, Any], summary: dict[str, Any]) -> str:
    o = summary["overall"]
    lines = [
        f"Eval batch {batch['batch_id']}" + (f" ({batch['label']})" if batch.get("label") else ""),
        f"command: {batch['command']}",
        f"commit: {batch.get('git_commit') or '?'}   model: {batch.get('model') or '?'}   "
        f"repeats: {batch['repeats']}   started: {batch['started_at']:%Y-%m-%d %H:%M} UTC",
        "",
        "| Task | Difficulty | Passed | Avg steps (passed) | Avg tokens | Avg seconds | Runs |",
        "|---|---|---|---|---|---|---|",
    ]
    for t, s in summary["tasks"].items():
        lines.append(f"| {t} | {s['difficulty']} | {s['runs_passed']}/{s['runs']} | "
                     f"{_fmt(s['avg_steps_passed'])} | {_fmt(s['avg_tokens_per_run'], 0)} | "
                     f"{_fmt(s['avg_seconds_per_run'])} | {', '.join(s['statuses'])} |")
    lines += ["", "| Group | Tasks | Passed every repeat | Passed at least once | Runs passed | Avg steps (passed) | Avg tokens | Avg seconds |",
              "|---|---|---|---|---|---|---|---|"]
    for name, b in [*summary["by_difficulty"].items(), ("Overall", o)]:
        lines.append(f"| {name} | {b['tasks']} | {b['tasks_passed_every_repeat']}/{b['tasks']} | "
                     f"{b['tasks_passed_at_least_once']}/{b['tasks']} | {b['runs_passed']}/{b['runs']} "
                     f"({_fmt(b['run_success_rate'] and b['run_success_rate'] * 100, 0)}%) | "
                     f"{_fmt(b['avg_steps_passed'])} | {_fmt(b['avg_tokens_per_run'], 0)} | "
                     f"{_fmt(b['avg_seconds_per_run'])} |")
    if o["not_finished"]:
        lines.append(f"\nNote: {o['not_finished']} run(s) did not finish (submission error or timeout); "
                     "they count as not passed.")
    return "\n".join(lines)


def report(store, batch_id: str) -> str:
    batch = store.eval_batch(batch_id)
    if batch is None:
        raise SystemExit(f"no eval batch {batch_id}")
    return render(batch, summarize(store.eval_rows(batch_id)))


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    from agentic_coding_agent.db.store import RunStore

    ap = argparse.ArgumentParser(description="Run the eval through the API and print the results table.")
    ap.add_argument("--tasks", nargs="*", help="default: every task the API currently serves")
    ap.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    ap.add_argument("--set", dest="task_set", choices=["main", "stress"], default="main",
                    help="main = the 18-task eval (default); stress = owner-only twisted variants")
    ap.add_argument("--label", help="free-text label stored with the batch, e.g. 'milestone-10'")
    ap.add_argument("--report", metavar="BATCH_ID", help="print the table for a past batch and exit")
    args = ap.parse_args(argv)

    store = RunStore()
    try:
        if args.report:
            print(report(store, args.report))
            return 0
        token = os.environ.get("OWNER_TOKEN")
        if not token:
            print("OWNER_TOKEN is not set; the eval goes through the owner path", file=sys.stderr)
            return 2
        task_ids = args.tasks or runnable_tasks(args.task_set)
        not_ready = [t for t in task_ids if t not in runnable_tasks(args.task_set)]
        if not_ready:
            print(f"not runnable (not in catalog or not fully validated): {not_ready}", file=sys.stderr)
            return 2
        command = "python scripts/run_eval.py " + " ".join(shlex.quote(a) for a in (argv or sys.argv[1:]))
        base_url = os.environ.get("AGENT_API_URL", "http://127.0.0.1:8000")
        with httpx.Client(base_url=base_url, timeout=30) as client:
            batch_id = run_eval(client, store, task_ids, args.repeats, token, args.label,
                                command.strip(), os.environ.get("GEMINI_MODEL", DEFAULT_MODEL), git_commit())
        print()
        print(report(store, batch_id))
        return 0
    except httpx.ConnectError:
        print("cannot reach the API (is the server running?)", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
