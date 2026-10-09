"""CLI — local runner (steps 3.5-3.7).

    python -m agentic_coding_agent.cli run two_sum
    python -m agentic_coding_agent.cli resume <run_id>

Runs the agent loop in-process, persists every step to Postgres
(DATABASE_URL) so a killed run can be resumed, and writes the
plain-text trace to stdout and logs/<run_id>.log (step 3.7). Step 4.4
turns this into a thin client over the API (POST /runs, GET /runs/{id}).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid

from dotenv import load_dotenv


def _print_result(result: dict, log_file) -> int:
    print("\n--- result ---")
    summary = {k: result.get(k) for k in
               ("run_id", "task_id", "status", "reason", "steps", "tokens", "passed", "total")}
    summary["log"] = str(log_file)
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "passed" else 1


def _cmd_run(args: argparse.Namespace) -> int:
    from .agent.graph import run_task
    from .agent.llm import GeminiClient
    from .agent.trace_log import TraceLogWriter
    from .db.store import RunStore

    store = RunStore()
    run_id = str(uuid.uuid4())
    log = TraceLogWriter(run_id)
    print(f"run_id: {run_id}\nlog: {log.path}\n", flush=True)
    try:
        result = run_task(args.task_id, GeminiClient(), run_id=run_id, store=store, log=log)
    finally:
        store.close()
    return _print_result(result, log.path)


def _cmd_resume(args: argparse.Namespace) -> int:
    from .agent.graph import resume_run
    from .agent.llm import GeminiClient
    from .agent.trace_log import TraceLogWriter
    from .db.store import RunStore

    store = RunStore()
    log = TraceLogWriter(args.run_id)
    try:
        before = len(store.load_events(args.run_id))
        print(f"resuming {args.run_id} ({before} saved events)\nlog: {log.path}\n", flush=True)
        result = resume_run(args.run_id, GeminiClient(), store, log=log)
    finally:
        store.close()
    return _print_result(result, log.path)


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
