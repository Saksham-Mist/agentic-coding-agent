"""Per-run trace — step 3.2 minimal version.

An ordered list of events for one run, also emitted to the
`agentic_coding_agent.trace` logger. Step 3.7 adds the plain-text
stdout/.log writer and Postgres rows from these same events, so the
two never disagree (the design notes §7).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("agentic_coding_agent.trace")


class Trace:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.events: list[dict[str, Any]] = []

    def event(self, kind: str, **data: Any) -> dict[str, Any]:
        entry = {
            "seq": len(self.events) + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            **data,
        }
        self.events.append(entry)
        logger.info("run=%s seq=%d kind=%s %s", self.run_id, entry["seq"], kind, data)
        return entry
