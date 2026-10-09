"""Worker node — step 3.3.

One LLM call per attempt: writes a complete Python solution for the
current subtask. On a retry, the prompt also carries the prior code
and its failing test results, so the correction is based on the
concrete failure (the design notes §2).

On a retry the model first writes a short diagnosis of why the last
attempt failed, then the fix, in the same call (a Reflexion-style
"reflect, then act" step at no extra quota cost). The diagnosis is
recorded in the trace so the reasoning behind each fix is visible.

Hidden test cases: the verifier runs them, but their inputs and
expected outputs never reach this prompt — only a count of how many
failed. Otherwise the retry loop would leak the hidden cases to the
model (the design notes "Harness modes").
"""
from __future__ import annotations

import json
import re
from typing import Any

from .trace import Trace

MAX_FAILURES_SHOWN = 5
MAX_STDERR_CHARS = 1500
MAX_DIAGNOSIS_CHARS = 600
MAX_CASE_ERROR_CHARS = 600

SYSTEM_INSTRUCTION = (
    "You are the coding step of an agent. Write one complete, "
    "self-contained Python solution that matches the required signature "
    "exactly. Use only the standard library. Do not print anything and do "
    "not read input; the solution is called directly by a test harness. "
    "If the task uses ListNode or TreeNode, those classes are already "
    "defined; do not redefine them. Respond with a single ```python code "
    "block. Only when asked for a diagnosis, put it before the code block."
)


def _format_spec(spec: dict[str, Any], plan: dict[str, Any], subtask_index: int) -> str:
    subtasks = "\n".join(
        f"{i + 1}. {s}{'  <-- current' if i == subtask_index else ''}"
        for i, s in enumerate(plan["subtasks"])
    )
    return (
        f"Task:\n{spec['prompt']}\n\n"
        f"Required signature:\n{spec['function_signature']}\n\n"
        f"Constraints:\n{spec['constraints']}\n\n"
        f"Example cases (input -> expected_output):\n"
        f"{json.dumps(spec.get('test_cases', []))}\n\n"
        f"Plan approach: {plan.get('approach', '')}\n"
        f"Plan subtasks:\n{subtasks}\n\n"
        "Write the full solution (not just the current subtask's part)."
    )


def format_feedback(prior_code: str, exec_result: dict[str, Any], visible_flags: list[bool]) -> str:
    """Builds the retry context from execute_code's output.

    `visible_flags[i]` says whether test_results[i] came from a visible
    case. Hidden failures are reported as a count only."""
    lines = [
        "Your previous attempt failed.",
        f"Previous code:\n```python\n{prior_code}\n```",
        f"Execution status: {exec_result.get('status')}",
    ]

    results = exec_result.get("test_results", [])
    visible_failures = [
        r for r, vis in zip(results, visible_flags) if vis and not r.get("passed")
    ]
    hidden_failed = sum(
        1 for r, vis in zip(results, visible_flags) if not vis and not r.get("passed")
    )

    for r in visible_failures[:MAX_FAILURES_SHOWN]:
        line = (
            f"FAILED input={json.dumps(r.get('input'))} "
            f"expected={json.dumps(r.get('expected'))} "
            f"actual={json.dumps(r.get('actual'))}"
        )
        # Per-case error text lives only on the case it came from, so it
        # is shown for visible cases and never for hidden ones.
        if r.get("error"):
            line += f"\n  raised: {r['error'].strip()[-MAX_CASE_ERROR_CHARS:]}"
        lines.append(line)
    if len(visible_failures) > MAX_FAILURES_SHOWN:
        lines.append(f"...and {len(visible_failures) - MAX_FAILURES_SHOWN} more visible failures")
    if hidden_failed:
        lines.append(
            f"{hidden_failed} hidden test(s) also failed (inputs not shown). "
            "Check edge cases and constraints."
        )

    # The shared stderr holds only load-time errors (syntax, missing
    # entry point), never per-case output, so it carries no test data.
    stderr = (exec_result.get("stderr") or "").strip()
    if stderr:
        lines.append(f"Error output (last {MAX_STDERR_CHARS} chars):\n{stderr[-MAX_STDERR_CHARS:]}")

    lines.append(
        "First write 'Diagnosis:' followed by 1-3 sentences on why the previous "
        "attempt failed. Then return the full corrected solution in a ```python block."
    )
    return "\n\n".join(lines)


def _extract_code(text: str) -> str | None:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    code = code.strip()
    return code or None


def _extract_diagnosis(text: str) -> str | None:
    """Text before the code fence, minus a leading 'Diagnosis:' label.
    Missing is fine: the fix is still used, the diagnosis is just empty."""
    fence = text.find("```")
    if fence <= 0:
        return None
    before = text[:fence].strip()
    before = re.sub(r"^\**\s*diagnosis\s*\**\s*:\s*\**", "", before, flags=re.IGNORECASE).strip()
    return before[:MAX_DIAGNOSIS_CHARS] or None


def write_code(
    spec: dict[str, Any],
    plan: dict[str, Any],
    subtask_index: int,
    llm,
    trace: Trace,
    attempt: int = 1,
    feedback: str | None = None,
) -> dict[str, Any]:
    """Returns {status: "ok", code, diagnosis, usage} or
    {status: "error", reason, message, usage}. Never raises on bad
    model output."""
    prompt = _format_spec(spec, plan, subtask_index)
    if feedback:
        prompt += "\n\n" + feedback

    result = llm.generate(prompt, system_instruction=SYSTEM_INSTRUCTION)
    usage = result["usage"]
    code = _extract_code(result["text"])

    if code is None:
        trace.event("worker_error", subtask=subtask_index, attempt=attempt,
                    raw=result["text"], usage=usage)
        return {"status": "error", "reason": "no_code",
                "message": "worker output contained no code", "usage": usage}

    diagnosis = _extract_diagnosis(result["text"]) if feedback else None
    trace.event("worker_code", subtask=subtask_index, attempt=attempt,
                is_retry=feedback is not None, diagnosis=diagnosis, code=code, usage=usage)
    return {"status": "ok", "code": code, "diagnosis": diagnosis, "usage": usage}
