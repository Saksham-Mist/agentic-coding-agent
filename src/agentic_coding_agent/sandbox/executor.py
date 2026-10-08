"""The execute_code tool — see the design notes "Tool: execute_code"
and the design notes' resource/timeout limits.

Fresh container per call, --network=none --cpus=1 --memory=512m,
killed at 10s wall-clock, always --rm. Every failure mode (timeout,
OOM, container error, malformed output) returns a structured status
field — nothing is raised as an exception the caller can't see.
"""
from __future__ import annotations

import json
import subprocess
import uuid

IMAGE = "agentic-sandbox:latest"
TIMEOUT_SECONDS = 10
CPUS = "1"
MEMORY = "512m"
# Caps total process count in the container. Without this, a fork
# bomb can spawn processes faster than the kernel reaps them even
# after `docker kill` sends SIGKILL to PID 1 — found by actually
# running one (see the design notes section 7): the container sat at ~2700
# PIDs and 99.7% CPU, unkillable, until the whole WSL2 VM was reset.
PIDS_LIMIT = "100"

# Docker's OOM-kill exit code (128 + SIGKILL's 9).
_OOM_EXIT_CODE = 137


def _empty_result(status: str, stderr: str) -> dict:
    return {"status": status, "stdout": "", "stderr": stderr, "test_results": []}


def execute_code(
    code: str,
    language: str,
    test_cases: list[dict],
    entry_point: str = "solution",
) -> dict:
    """Run `code` inside a fresh, resource-limited container and
    return the the design notes output shape."""
    if language != "python":
        return _empty_result("error", f"unsupported language: {language!r}")

    container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
    payload = json.dumps({
        "code": code,
        "entry_point": entry_point,
        "test_cases": test_cases,
    })

    cmd = [
        "docker", "run", "--rm", "-i",
        "--name", container_name,
        "--network=none",
        f"--cpus={CPUS}",
        f"--memory={MEMORY}",
        f"--pids-limit={PIDS_LIMIT}",
        IMAGE,
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=payload,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        # The container may still be running after the Python-side
        # timeout fires — docker run alone has no wall-clock flag, so
        # we kill it by name ourselves. --rm cleans it up once killed.
        subprocess.run(["docker", "kill", container_name], capture_output=True)
        return _empty_result("timeout", f"execution exceeded {TIMEOUT_SECONDS}s")

    if proc.returncode == _OOM_EXIT_CODE:
        return _empty_result("resource_exceeded", "container was OOM-killed")

    if proc.returncode != 0:
        return _empty_result(
            "error",
            f"container exited {proc.returncode}: {proc.stderr.strip()}",
        )

    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return _empty_result(
            "error", f"malformed harness output: {e}; raw: {proc.stdout!r}"
        )

    return result
