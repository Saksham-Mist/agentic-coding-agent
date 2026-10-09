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
from .trace_log import fanout
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
    resume_at: str       # plan | write | execute (step 3.6)
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
                    passed=passed, total=len(cases), result=result)
        return {"exec_result": result}

    def verify_node(state: RunState) -> RunState:
        task = state["task"]
        verdict = verify(state["exec_result"], expected_count=len(task["test_cases"]),
                         attempt=state["attempt"], max_retries=limits.max_retries)
        feedback = None
        if verdict["decision"] == "retry":
            flags = [c.get("visible", False) for c in task["test_cases"]]
            feedback = format_feedback(state.get("code") or "", state["exec_result"], flags)
        trace.event("verify", attempt=state["attempt"], feedback=feedback, **verdict)

        if verdict["decision"] == "pass":
            return {"verdict": verdict, **stop(state, "passed", verdict["reason"])}
        if verdict["decision"] == "give_up":
            return {"verdict": verdict, **stop(state, "incomplete", verdict["reason"])}
        return {"verdict": verdict, "attempt": state["attempt"] + 1, "feedback": feedback}

    def finished(state: RunState) -> bool:
        return state.get("status", "running") != "running"

    g = StateGraph(RunState)
    g.add_node("plan", plan_node)
    g.add_node("write", write_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)

    g.add_conditional_edges(START, lambda s: s.get("resume_at", "plan"))
    g.add_conditional_edges("plan", lambda s: END if finished(s) else "write")
    g.add_conditional_edges("write", lambda s: END if finished(s) else "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges("verify", lambda s: END if finished(s) else "write")
    return g.compile()


_LLM_EVENTS = {"plan", "plan_error", "worker_code", "worker_error"}


def _recursion_limit(limits: Limits) -> int:
    # Each attempt visits 3 nodes (write, execute, verify) plus 1 for
    # plan. LangGraph's default cap is 25 visits, which a 15-step run
    # would exceed; set it above what our own limits allow so they
    # always stop the run first.
    return 3 * limits.max_steps + 10


def _result(run_id, task_id, task, final, trace) -> dict[str, Any]:
    verdict = final.get("verdict") or {}
    return {
        "run_id": run_id, "task_id": task_id,
        "status": final["status"], "reason": final.get("reason", ""),
        "steps": final["steps"], "tokens": final["tokens"], "code": final.get("code"),
        "passed": verdict.get("passed", 0), "total": len(task["test_cases"]) if task else 0,
        "trace": trace.events,
    }


def _load_task(task_id: str):
    spec = read_test_spec(task_id)
    task = get_task(task_id) if spec.get("status") != "error" else None
    return spec, task


def run_task(
    task_id: str,
    llm,
    limits: Limits | None = None,
    executor: Callable[..., dict] = execute_code,
    run_id: str | None = None,
    store=None,
    log=None,
    source: str = "cli",
) -> dict[str, Any]:
    """Run one task end to end. With a `store` (db.store.RunStore),
    every event is persisted as it happens, so the run can be resumed
    with resume_run() after a crash. `log` is an extra event sink
    (trace_log.TraceLogWriter), called after the store so the log never
    shows an event Postgres doesn't have. Returns {run_id, task_id,
    status, reason, steps, tokens, code, passed, total, trace}."""
    limits = limits or Limits()
    run_id = run_id or str(uuid.uuid4())
    if store is not None:
        store.create_run(run_id, task_id, source=source)
    store_sink = (lambda e: store.append_event(run_id, e)) if store is not None else None
    trace = Trace(run_id, sink=fanout(store_sink, log))

    spec, task = _load_task(task_id)
    if task is None:
        reason = spec.get("reason", "not_found")
        trace.event("finish", status="error", reason=reason, steps=0, tokens=0)
        return _result(run_id, task_id, None,
                       {"status": "error", "reason": reason, "steps": 0, "tokens": 0}, trace)

    graph = build_graph(llm, trace, limits, executor)
    initial: RunState = {
        "spec": spec, "task": task, "subtask_index": 0, "attempt": 1,
        "code": None, "feedback": None, "steps": 0, "tokens": 0,
        "status": "running", "resume_at": "plan",
    }
    final = graph.invoke(initial, config={"recursion_limit": _recursion_limit(limits)})
    return _result(run_id, task_id, task, final, trace)


def _state_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild where a run stopped from its persisted events.

    - no plan saved            -> start at plan
    - plan, no code yet        -> write attempt 1
    - code saved, not verified -> re-execute that code (no LLM call)
    - last verify said retry   -> write the next attempt with its feedback
    """
    state: dict[str, Any] = {
        "steps": sum(1 for e in events if e["kind"] in _LLM_EVENTS),
        "tokens": sum(int((e.get("usage") or {}).get("total_tokens", 0)) for e in events),
        "subtask_index": 0, "attempt": 1, "code": None, "feedback": None,
    }
    plan_ev = next((e for e in events if e["kind"] == "plan"), None)
    if plan_ev is None:
        return {**state, "resume_at": "plan"}
    state["plan"] = {"approach": plan_ev.get("approach", ""), "subtasks": plan_ev["subtasks"]}

    worker_idx = max((i for i, e in enumerate(events)
                      if e["kind"] in ("worker_code", "worker_error")), default=None)
    if worker_idx is None:
        return {**state, "resume_at": "write"}

    worker_ev = events[worker_idx]
    state["attempt"] = worker_ev.get("attempt", 1)
    state["code"] = worker_ev.get("code")
    verify_ev = next((e for e in events[worker_idx + 1:] if e["kind"] == "verify"), None)
    if verify_ev is None:
        return {**state, "resume_at": "execute"}

    # verify said retry (pass/give_up also write a finish event, handled by the caller)
    return {**state, "resume_at": "write", "attempt": verify_ev["attempt"] + 1,
            "feedback": verify_ev.get("feedback")}


def resume_run(
    run_id: str,
    llm,
    store,
    limits: Limits | None = None,
    executor: Callable[..., dict] = execute_code,
    log=None,
) -> dict[str, Any]:
    """Continue a persisted run from its last saved event. A run that
    already finished returns its stored result without any calls."""
    limits = limits or Limits()
    row = store.get_run(run_id)
    if row is None:
        return {"run_id": run_id, "task_id": None, "status": "error", "reason": "run_not_found",
                "steps": 0, "tokens": 0, "code": None, "passed": 0, "total": 0, "trace": []}

    events = store.load_events(run_id)
    task_id = row["task_id"]
    spec, task = _load_task(task_id)
    trace = Trace(run_id, sink=fanout(lambda e: store.append_event(run_id, e), log),
                  events=events)

    finish_ev = next((e for e in events if e["kind"] == "finish"), None)
    if finish_ev is not None:
        last_verify = next((e for e in reversed(events) if e["kind"] == "verify"), {})
        last_code = next((e.get("code") for e in reversed(events) if e["kind"] == "worker_code"), None)
        return _result(run_id, task_id, task,
                       {"status": finish_ev["status"], "reason": finish_ev["reason"],
                        "steps": finish_ev["steps"], "tokens": finish_ev["tokens"],
                        "code": last_code, "verdict": last_verify}, trace)

    if task is None:
        reason = spec.get("reason", "not_found")
        trace.event("finish", status="error", reason=reason, steps=0, tokens=0)
        return _result(run_id, task_id, None,
                       {"status": "error", "reason": reason, "steps": 0, "tokens": 0}, trace)

    restored = _state_from_events(events)
    trace.event("resume", resume_at=restored["resume_at"], attempt=restored["attempt"],
                steps=restored["steps"], tokens=restored["tokens"])
    store.mark_running(run_id)

    graph = build_graph(llm, trace, limits, executor)
    initial: RunState = {"spec": spec, "task": task, "status": "running", **restored}
    final = graph.invoke(initial, config={"recursion_limit": _recursion_limit(limits)})
    return _result(run_id, task_id, task, final, trace)
