"""CLI — step 3.5 local runner.

    python -m agentic_coding_agent.cli run two_sum

Runs the agent loop in-process for now. Step 4.4 turns this into a
thin client over the API (POST /runs, GET /runs/{id}).
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv


def _cmd_run(args: argparse.Namespace) -> int:
    from .agent.graph import run_task
    from .agent.llm import GeminiClient

    result = run_task(args.task_id, GeminiClient())
    for e in result["trace"]:
        detail = {k: v for k, v in e.items() if k not in ("seq", "ts", "kind", "code", "raw")}
        print(f"[{e['seq']:02d}] {e['kind']:<12} {json.dumps(detail, default=str)}")
    if result.get("code"):
        print("\n--- final code ---\n" + result["code"])
    print("\n--- result ---")
    print(json.dumps({k: result[k] for k in
                      ("run_id", "task_id", "status", "reason", "steps", "tokens", "passed", "total")},
                     indent=2))
    return 0 if result["status"] == "passed" else 1


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agentic-coding-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run one catalog task end to end")
    run.add_argument("task_id")
    run.set_defaults(func=_cmd_run)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
