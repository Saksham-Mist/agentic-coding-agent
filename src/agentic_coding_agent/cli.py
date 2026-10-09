"""CLI — thin client over the API (step 4.4).

    python -m agentic_coding_agent.cli run two_sum
    python -m agentic_coding_agent.cli status <run_id>
    python -m agentic_coding_agent.cli stats

Talks to AGENT_API_URL (default http://127.0.0.1:8000). Sends
OWNER_TOKEN as a bearer token when set, which skips the public rate
limit and tags runs source=cli. There is one submission path: the CLI
never runs the agent itself; the server does, and resumes unfinished
runs on startup (step 4.1).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import httpx
from dotenv import load_dotenv

from .agent.trace_log import render_event

DEFAULT_API_URL = "http://127.0.0.1:8000"
POLL_SECONDS = 1.0
RUN_TIMEOUT_SECONDS = 600  # well above 15 steps of LLM + sandbox time


def _headers(token: str | None, source: str | None = None) -> dict[str, str]:
    h = {}
    if token:
        h["Authorization"] = f"Bearer {token}"
        if source:
            h["X-Run-Source"] = source
    return h


def _print_error(r: httpx.Response) -> int:
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = r.text
    print(f"error: HTTP {r.status_code}: {json.dumps(detail)}", file=sys.stderr)
    if r.status_code == 429 and "retry-after" in r.headers:
        print(f"retry after {r.headers['retry-after']}s", file=sys.stderr)
    return 2


def _summary(body: dict[str, Any]) -> dict[str, Any]:
    return {k: body.get(k) for k in ("run_id", "task_id", "status", "steps", "tokens")}


def run(client: httpx.Client, task_id: str, token: str | None = None, source: str | None = None,
        poll_seconds: float = POLL_SECONDS, timeout: float = RUN_TIMEOUT_SECONDS) -> int:
    """POST /runs, then poll GET /runs/{id} and print each new trace
    event as it appears. Exit code: 0 passed, 1 finished but not
    passed, 2 request error or timeout."""
    r = client.post("/runs", json={"task_id": task_id}, headers=_headers(token, source))
    if r.status_code != 202:
        return _print_error(r)
    run_id = r.json()["run_id"]
    print(f"run_id: {run_id}  (queued)\n", flush=True)

    shown = 0
    deadline = time.monotonic() + timeout
    while True:
        g = client.get(f"/runs/{run_id}")
        if g.status_code != 200:
            return _print_error(g)
        body = g.json()
        for event in body["trace"][shown:]:
            print(render_event(event), flush=True)
        shown = len(body["trace"])
        if body["status"] not in ("queued", "running"):
            print("\n--- result ---")
            print(json.dumps(_summary(body), indent=2))
            return 0 if body["status"] == "passed" else 1
        if time.monotonic() > deadline:
            print(f"timed out waiting for {run_id}; it keeps running on the server", file=sys.stderr)
            return 2
        time.sleep(poll_seconds)


def status(client: httpx.Client, run_id: str) -> int:
    g = client.get(f"/runs/{run_id}")
    if g.status_code != 200:
        return _print_error(g)
    body = g.json()
    for event in body["trace"]:
        print(render_event(event))
    print("\n" + json.dumps(_summary(body), indent=2))
    return 0


def stats(client: httpx.Client) -> int:
    g = client.get("/stats")
    if g.status_code != 200:
        return _print_error(g)
    print(json.dumps(g.json(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="agentic-coding-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="submit a catalog task and follow it to the end")
    p_run.add_argument("task_id")
    p_status = sub.add_parser("status", help="show a run's trace and result")
    p_status.add_argument("run_id")
    sub.add_parser("stats", help="show aggregate stats")
    args = parser.parse_args(argv)

    base_url = os.environ.get("AGENT_API_URL", DEFAULT_API_URL)
    token = os.environ.get("OWNER_TOKEN") or None
    try:
        with httpx.Client(base_url=base_url, timeout=30) as client:
            if args.command == "run":
                return run(client, args.task_id, token)
            if args.command == "status":
                return status(client, args.run_id)
            return stats(client)
    except httpx.ConnectError:
        print(f"error: cannot reach the API at {base_url} (is the server running?)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
