"""Per-run trace — step 3.2 minimal version.

An ordered list of events for one run, also emitted to the
`agentic_coding_agent.trace` logger. Step 3.7 adds the plain-text
stdout/.log writer and Postgres rows from these same events, so the
two never disagree (the design notes §7).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger("agentic_coding_agent.trace")


class Trace:
    """`sink(event)` is called for every event, before `event()`
    returns (step 3.6 passes RunStore.append_event, so each event is
    in Postgres before the run moves on). `events` seeds a resumed
    run with the events already persisted, so numbering continues."""

    def __init__(self, run_id: str, sink: Callable[[dict[str, Any]], None] | None = None,
                 events: list[dict[str, Any]] | None = None):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = list(events or [])
        self._sink = sink

    def event(self, kind: str, **data: Any) -> dict[str, Any]:
        entry = {
            "seq": len(self.events) + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **data,
        }
        self.events.append(entry)
        if self._sink is not None:
            self._sink(entry)
        logger.info("run=%s seq=%d kind=%s %s", self.run_id, entry["seq"], kind, data)
        return entry
