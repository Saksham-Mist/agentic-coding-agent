"""Plain-text trace log — step 3.7.

Renders trace events as readable text (plan -> each step -> pass/fail
-> retry, the design notes) to stdout and to logs/<run_id>.log.
The same render_event() turns Postgres rows back into the log, so the
two are produced from the same events and can be checked against each
other (the design notes §7).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, TextIO

DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def render_event(e: dict[str, Any]) -> str:
    seq, kind = e["seq"], e["kind"]
    head = f"{e.get('ts', '')} [{seq:02d}]"
    tokens = (e.get("usage") or {}).get("total_tokens")
    tok = f"  tokens={tokens}" if tokens is not None else ""

    if kind == "plan":
        lines = [f"{head} PLAN  {e.get('approach', '')}{tok}"]
        lines += [f"    {i}. {s}" for i, s in enumerate(e.get("subtasks", []), 1)]
        return "\n".join(lines)
    if kind == "plan_error":
        return f"{head} PLAN ERROR  planner output was not valid JSON{tok}"
    if kind == "worker_code":
        retry = " (retry)" if e.get("is_retry") else ""
        lines = [f"{head} WRITE attempt {e.get('attempt')}{retry}{tok}"]
        if e.get("diagnosis"):
            lines.append(f"    diagnosis: {e['diagnosis']}")
        lines.append(_indent(e.get("code") or ""))
        return "\n".join(lines)
    if kind == "worker_error":
        return f"{head} WRITE attempt {e.get('attempt')}  ERROR no code in model output{tok}"
    if kind == "execute":
        return (f"{head} RUN   attempt {e.get('attempt')}  status={e.get('status')}  "
                f"passed {e.get('passed')}/{e.get('total')}")
    if kind == "verify":
        return (f"{head} CHECK attempt {e.get('attempt')}  decision={e.get('decision')}  "
                f"reason={e.get('reason')}")
    if kind == "resume":
        return (f"{head} RESUME at {e.get('resume_at')}  attempt={e.get('attempt')}  "
                f"steps={e.get('steps')}  tokens={e.get('tokens')}")
    if kind == "finish":
        return (f"{head} DONE  status={e.get('status')}  reason={e.get('reason')}  "
                f"steps={e.get('steps')}  tokens={e.get('tokens')}")
    return f"{head} {kind.upper()}"


def render_log(events: list[dict[str, Any]]) -> str:
    return "".join(render_event(e) + "\n" for e in events)


def log_path(run_id: str, log_dir: Path | str | None = None) -> Path:
    return Path(log_dir or DEFAULT_LOG_DIR) / f"{run_id}.log"


class TraceLogWriter:
    """Trace sink: appends each rendered event to logs/<run_id>.log and
    echoes it to `stream`. Appends, so a resumed run continues its file."""

    def __init__(self, run_id: str, log_dir: Path | str | None = None,
                 stream: TextIO | None = sys.stdout):
        self.path = log_path(run_id, log_dir)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = stream

    def __call__(self, event: dict[str, Any]) -> None:
        text = render_event(event) + "\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(text)
        if self._stream is not None:
            self._stream.write(text)
            self._stream.flush()


def fanout(*sinks: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None] | None:
    """Call sinks in order (Postgres first, then the log). None entries skipped."""
    active = [s for s in sinks if s is not None]
    if not active:
        return None

    def sink(event: dict[str, Any]) -> None:
        for s in active:
            s(event)
    return sink
