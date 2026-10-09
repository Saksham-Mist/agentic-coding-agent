"""Agent loop — step 3.5. LangGraph wiring with hard limits.

    plan -> write -> execute -> verify --retry--> write
                                       --pass---> END (passed)
                                       --give_up/limit--> END (incomplete)

Limits are enforced here, in orchestrator code the model can't reason
around (the design notes): retries per subtask (via the verifier), LLM
steps per run, and tokens per run. Every limit check happens before
an LLM call, so a run never makes a call it isn't allowed to.

The run ends as soon as the full test set passes, even if the plan
listed more subtasks: the worker writes the whole function each time,
so remaining subtasks wouldn't change the result (noted in step 3.3).
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .catalog import get_task, read_test_spec
from .planner import plan_task
from .trace import Trace
from .verifier import MAX_RETRIES, verify
from .worker import format_feedback, write_code
from ..sandbox.executor import execute_code


@dataclass(frozen=True)
class Limits:
    max_retries: int = int(os.environ.get("MAX_RETRIES_PER_SUBTASK", MAX_RETRIES))
    max_steps: int = int(os.environ.get("MAX_STEPS_PER_RUN", 15))
    max_tokens: int = int(os.environ.get("MAX_TOKENS_PER_RUN", 100_000))


class RunState(TypedDict, total=False):
    spec: dict[str, Any]
    task: dict[str, Any]
    plan: dict[str, Any]
    subtask_index: int
    attempt: int
    code: str | None
    feedback: str | None
    exec_result: dict[str, Any]
    steps: int
    tokens: int
    status: str          # running | passed | incomplete | error
    reason: str
    verdict: dict[str, Any]


def _limit_hit(state: RunState, limits: Limits) -> str | None:
    if state["steps"] >= limits.max_steps:
        return "step_limit"
    if state["tokens"] >= limits.max_tokens:
        return "token_limit"
    return None


def build_graph(llm, trace: Trace, limits: Limits, executor: Callable[..., dict] = execute_code):
    def stop(state: RunState, status: str, reason: str) -> RunState:
        trace.event("finish", status=status, reason=reason,
                    steps=state["steps"], tokens=state["tokens"])
        return {"status": status, "reason": reason}

    def plan_node(state: RunState) -> RunState:
        if (hit := _limit_hit(state, limits)):
            return stop(state, "incomplete", hit)
        try:
            result = plan_task(state["spec"], llm, trace)
        except Exception as e:  # API failure after the client's own retries
            return stop(state, "error", f"llm_error: {type(e).__name__}: {e}")
        update: RunState = {
            "steps": state["steps"] + 1,
            "tokens": state["tokens"] + result["usage"]["total_tokens"],
        }
        if result["status"] != "ok":
            return {**update, **stop({**state, **update}, "incomplete", result["reason"])}
        return {**update, "plan": result}

    def write_node(state: RunState) -> RunState:
        if (hit := _limit_hit(state, limits)):
            return stop(state, "incomplete", hit)
        try:
            result = write_code(
                state["spec"], state["plan"], state["subtask_index"], llm, trace,
                attempt=state["attempt"], feedback=state.get("feedback"),
            )
        except Exception as e:
            return stop(state, "error", f"llm_error: {type(e).__name__}: {e}")
        return {
            "steps": state["steps"] + 1,
            "tokens": state["tokens"] + result["usage"]["total_tokens"],
            "code": result.get("code"),
        }

    def execute_node(state: RunState) -> RunState:
        task = state["task"]
        cases = [{"input": c["input"], "expected_output": c["expected_output"]}
                 for c in task["test_cases"]]
        if state.get("code") is None:
            result = {"status": "error", "stdout": "", "stderr": "worker produced no code",
                      "test_results": []}
        else:
            result = executor(
                state["code"], "python", cases,
                entry_point=task["entry_point"],
                comparator=task.get("comparator", "exact"),
                arg_types=task.get("arg_types"),
                return_type=task.get("return_type"),
            )
        passed = sum(1 for r in result.get("test_results", []) if r.get("passed"))
        trace.event("execute", attempt=state["attempt"], status=result.get("status"),
                    passed=passed, total=len(cases))
        return {"exec_result": result}

    def verify_node(state: RunState) -> RunState:
        task = state["task"]
        verdict = verify(state["exec_result"], expected_count=len(task["test_cases"]),
                         attempt=state["attempt"], max_retries=limits.max_retries)
        trace.event("verify", attempt=state["attempt"], **verdict)

        if verdict["decision"] == "pass":
            return {"verdict": verdict, **stop(state, "passed", verdict["reason"])}
        if verdict["decision"] == "give_up":
            return {"verdict": verdict, **stop(state, "incomplete", verdict["reason"])}

        flags = [c.get("visible", False) for c in task["test_cases"]]
        return {
            "verdict": verdict,
            "attempt": state["attempt"] + 1,
            "feedback": format_feedback(state.get("code") or "", state["exec_result"], flags),
        }

    def finished(state: RunState) -> bool:
        return state.get("status", "running") != "running"

    g = StateGraph(RunState)
    g.add_node("plan", plan_node)
    g.add_node("write", write_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", lambda s: END if finished(s) else "write")
    g.add_conditional_edges("write", lambda s: END if finished(s) else "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges("verify", lambda s: END if finished(s) else "write")
    return g.compile()


def run_task(
    task_id: str,
    llm,
    limits: Limits | None = None,
    executor: Callable[..., dict] = execute_code,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run one task end to end. Returns {run_id, task_id, status,
    reason, steps, tokens, code, passed, total, trace}. Never raises."""
    limits = limits or Limits()
    run_id = run_id or str(uuid.uuid4())
    trace = Trace(run_id)

    spec = read_test_spec(task_id)
    task = get_task(task_id) if spec.get("status") != "error" else None
    if task is None:
        trace.event("finish", status="error", reason=spec.get("reason", "not_found"), steps=0, tokens=0)
        return {"run_id": run_id, "task_id": task_id, "status": "error",
                "reason": spec.get("reason", "not_found"), "steps": 0, "tokens": 0,
                "code": None, "passed": 0, "total": 0, "trace": trace.events}

    graph = build_graph(llm, trace, limits, executor)
    initial: RunState = {
        "spec": spec, "task": task, "subtask_index": 0, "attempt": 1,
        "code": None, "feedback": None, "steps": 0, "tokens": 0, "status": "running",
    }
    # Each attempt visits 3 nodes (write, execute, verify) plus 1 for
    # plan. LangGraph's default cap is 25 visits, which a 15-step run
    # would exceed; set it above what our own limits allow so they
    # always stop the run first.
    final = graph.invoke(initial, config={"recursion_limit": 3 * limits.max_steps + 10})

    verdict = final.get("verdict") or {}
    return {
        "run_id": run_id, "task_id": task_id,
        "status": final["status"], "reason": final.get("reason", ""),
        "steps": final["steps"], "tokens": final["tokens"], "code": final.get("code"),
        "passed": verdict.get("passed", 0), "total": len(task["test_cases"]),
        "trace": trace.events,
    }
