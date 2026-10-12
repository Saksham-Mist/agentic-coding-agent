"""Planner node — step 3.2.

A separate LLM call (not folded into the Worker prompt) that turns a
task spec into an ordered plan. The plan is written to the trace
before anything executes, so "plan then execute" is an inspectable
step.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .trace import Trace

MAX_SUBTASKS = 5

SYSTEM_INSTRUCTION = (
    "You are the planning step of a coding agent. You do not write code. "
    "Given a programming task, produce a short ordered plan for writing a "
    "single Python solution. Respond with JSON only, no prose, in exactly "
    'this shape: {"approach": "<one or two sentences>", '
    '"subtasks": ["<step>", ...]}. Use at most '
    f"{MAX_SUBTASKS} subtasks; simple tasks need just one."
)


def _build_prompt(spec: dict[str, Any]) -> str:
    examples = json.dumps(spec.get("test_cases", []), indent=None)
    return (
        f"Task:\n{spec['prompt']}\n\n"
        f"Required signature:\n{spec['function_signature']}\n\n"
        f"Constraints:\n{spec['constraints']}\n\n"
        f"Example cases (input -> expected_output):\n{examples}\n"
    )


def _parse_plan(text: str) -> dict[str, Any] | None:
    # Models often wrap JSON in ```json fences despite instructions.
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, dict):
        return None
    subtasks = plan.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        return None
    if not all(isinstance(s, str) and s.strip() for s in subtasks):
        return None
    return {
        "approach": str(plan.get("approach", "")),
        "subtasks": subtasks[:MAX_SUBTASKS],
    }


def plan_task(spec: dict[str, Any], llm, trace: Trace) -> dict[str, Any]:
    """Returns {status: "ok", approach, subtasks, usage} or
    {status: "error", reason, message, usage}. Never raises on bad
    model output; the orchestrator (step 3.5) decides what to do."""
    result = llm.generate(_build_prompt(spec), system_instruction=SYSTEM_INSTRUCTION)
    usage = result["usage"]
    plan = _parse_plan(result["text"])

    if plan is None:
        trace.event("plan_error", raw=result["text"], usage=usage)
        return {
            "status": "error",
            "reason": "unparseable_plan",
            "message": "planner output was not the expected JSON shape",
            "usage": usage,
        }

    trace.event("plan", approach=plan["approach"], subtasks=plan["subtasks"], usage=usage)
    return {"status": "ok", **plan, "usage": usage}
