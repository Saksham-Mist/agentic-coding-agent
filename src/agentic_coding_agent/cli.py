"""CLI — local runner (steps 3.5/3.6).

    python -m agentic_coding_agent.cli run two_sum
    python -m agentic_coding_agent.cli resume <run_id>

Runs the agent loop in-process and persists every step to Postgres
(DATABASE_URL), so a killed run can be resumed. Step 4.4 turns this
into a thin client over the API (POST /runs, GET /runs/{id}).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

from dotenv import load_dotenv

_HIDDEN_KEYS = ("seq", "ts", "kind", "code", "raw", "result", "feedback")


def _print_event(e: dict) -> None:
    detail = {k: v for k, v in e.items() if k not in _HIDDEN_KEYS}
    print(f"[{e['seq']:02d}] {e['kind']:<12} {json.dumps(detail, default=str)}", flush=True)


def _print_result(result: dict) -> int:
    if result.get("code"):
        print("\n--- final code ---\n" + result["code"])
    print("\n--- result ---")
    print(json.dumps({k: result.get(k) for k in
                      ("run_id", "task_id", "status", "reason", "steps", "tokens", "passed", "total")},
                     indent=2))
    return 0 if result["status"] == "passed" else 1


class _PrintingStore:
    """Wraps RunStore so each event is printed as soon as it's saved."""

    def __init__(self, store):
        self._store = store

    def append_event(self, run_id, event):
        self._store.append_event(run_id, event)
        _print_event(event)

    def __getattr__(self, name):
        return getattr(self._store, name)


def _cmd_run(args: argparse.Namespace) -> int:
    from .agent.graph import run_task
    from .agent.llm import GeminiClient
    from .db.store import RunStore

    store = RunStore()
    run_id = str(uuid.uuid4())
    print(f"run_id: {run_id}", flush=True)
    try:
        result = run_task(args.task_id, GeminiClient(), run_id=run_id, store=_PrintingStore(store))
    finally:
        store.close()
    return _print_result(result)


def _cmd_resume(args: argparse.Namespace) -> int:
    from .agent.graph import resume_run
    from .agent.llm import GeminiClient
    from .db.store import RunStore

    store = RunStore()
    try:
        before = len(store.load_events(args.run_id))
        print(f"resuming {args.run_id} ({before} saved events)", flush=True)
        result = resume_run(args.run_id, GeminiClient(), _PrintingStore(store))
    finally:
        store.close()
    return _print_result(result)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agentic-coding-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one catalog task end to end")
    run.add_argument("task_id")
    run.set_defaults(func=_cmd_run)

    resume = sub.add_parser("resume", help="resume a run from its last saved step")
    resume.add_argument("run_id")
    resume.set_defaults(func=_cmd_resume)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
