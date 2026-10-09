"""Public API — step 4.1.

    POST /runs        {task_id} -> 202 {run_id, status: queued}
    GET  /runs/{id}   status, counters and the public trace
    GET  /stats       success rate / steps / tokens / wall clock

Execution model (the design notes §9): POST only inserts a `queued` row
and returns; one background worker thread runs queued runs one at a
time, so v1 stays sequential. On startup the worker re-queues anything
left `queued` or `running`, and running ones resume from their last
saved step (step 3.6), so a server restart doesn't lose runs.

Run with:  uvicorn agentic_coding_agent.api.app:app
"""
from __future__ import annotations

import hmac
import logging
import os
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()

from ..agent.catalog import CatalogLoadError, get_task, load_catalog  # noqa: E402
from ..agent.graph import resume_run, run_task  # noqa: E402
from ..agent.trace_log import TraceLogWriter  # noqa: E402
from ..db.store import RunStore  # noqa: E402
from ..sandbox.executor import execute_code  # noqa: E402

logger = logging.getLogger("agentic_coding_agent.api")


class RunRequest(BaseModel):
    # Catalog ids are short snake_case names; anything else is rejected
    # before it reaches any code (step 4.2).
    task_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")


def _is_ready(entry: dict[str, Any]) -> bool:
    """A catalog entry can run only once it's filled in: a stub with no
    prompt or test cases would spend LLM calls and could never pass."""
    return bool(entry.get("prompt") and entry.get("function_signature") and entry.get("test_cases"))


def validate_task_id(task_id: str) -> None:
    """Step 4.2: POST /runs only runs ready entries of the fixed catalog
    (the design notes "Public abuse limits"). Raises 422 otherwise, before
    any run row exists or any LLM call is made."""
    try:
        catalog = load_catalog()
    except CatalogLoadError:
        raise HTTPException(status_code=503, detail={"reason": "catalog_unavailable"})
    ready = sorted(t for t, e in catalog.items() if _is_ready(e))
    entry = catalog.get(task_id)
    if entry is None:
        raise HTTPException(status_code=422, detail={
            "reason": "unknown_task_id",
            "message": f"'{task_id}' is not in the task catalog",
            "available_task_ids": ready,
        })
    if not _is_ready(entry):
        raise HTTPException(status_code=422, detail={
            "reason": "task_not_ready",
            "message": f"'{task_id}' is in the catalog but not filled in yet",
            "available_task_ids": ready,
        })


def public_trace(events: list[dict[str, Any]], task: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The trace as the public may see it. The stored trace contains
    every hidden test case (the full sandbox result) and anything the
    model's code printed; returning it would publish the answer key.
    Keep only visible test results, drop stdout/stderr and the
    internal retry feedback."""
    flags = [c.get("visible", False) for c in task["test_cases"]] if task else []
    out = []
    for e in events:
        e = dict(e)
        if e["kind"] == "execute" and isinstance(e.get("result"), dict):
            results = e["result"].get("test_results") or []
            e["result"] = {
                "status": e["result"].get("status"),
                "visible_test_results": [r for r, vis in zip(results, flags) if vis],
            }
        if e["kind"] == "verify":
            e.pop("feedback", None)
        out.append(e)
    return out


def seconds_until_utc_midnight(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return max(1, int((tomorrow - now).total_seconds()))


def client_ip(request: Request) -> str:
    """The caller's IP as uvicorn reports it. Behind nginx (Phase 6) run
    uvicorn with --proxy-headers --forwarded-allow-ips=127.0.0.1 so this
    is the visitor's address taken from nginx's header, which uvicorn
    only trusts when the request really came from nginx. Reading
    X-Forwarded-For here directly would let anyone dodge the limit by
    setting that header themselves."""
    return request.client.host if request.client else "unknown"


OWNER_SOURCES = {"cli", "eval"}


def owner_source(request: Request, owner_token: str | None) -> str | None:
    """Step 4.4: the owner's requests skip the rate limit
    (the design notes §9, "operator's own credentials").

    No Authorization header -> None (public caller). A bearer token that
    matches OWNER_TOKEN -> the run source ("cli", or "eval" via
    X-Run-Source). Any other Authorization header -> 401, so a wrong
    token fails loudly instead of quietly hitting the public limit."""
    header = request.headers.get("authorization")
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if not owner_token or scheme.lower() != "bearer" or not hmac.compare_digest(
        token.encode(), owner_token.encode()
    ):
        raise HTTPException(status_code=401, detail={"reason": "invalid_owner_token"})
    source = request.headers.get("x-run-source", "cli")
    return source if source in OWNER_SOURCES else "cli"


class RunWorker:
    """Single background thread: runs are executed strictly one at a time.

    stop() lets the run in progress finish, then exits without starting
    anything else; runs still queued stay `queued` in Postgres and are
    picked up at the next startup. (Before, stop() only queued a
    sentinel behind pending runs and waited 5s, so the thread kept
    working after shutdown -- in tests, into rows already deleted.)"""

    STOP_TIMEOUT_SECONDS = 120

    def __init__(self, llm_factory: Callable[[], Any], executor, store_factory, log_dir):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._llm_factory = llm_factory
        self._executor = executor
        self._store_factory = store_factory
        self._log_dir = log_dir
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="run-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        self._queue.put(None)  # wake the thread if it's waiting on an empty queue
        self._thread.join(timeout=self.STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning("run worker still busy after %ss; leaving it to finish",
                           self.STOP_TIMEOUT_SECONDS)

    def submit(self, run_id: str) -> None:
        self._queue.put(run_id)

    def _loop(self) -> None:
        store = self._store_factory()
        try:
            while not self._stopping.is_set():
                run_id = self._queue.get()
                if run_id is None or self._stopping.is_set():
                    break
                self._process(store, run_id)
        finally:
            store.close()

    def _process(self, store, run_id: str) -> None:
        try:
            row = store.get_run(run_id)
            if row is None or row["status"] not in ("queued", "running"):
                return
            if row["status"] == "queued":
                # Claim before starting: a duplicate queue entry or a second
                # server process loses the claim and skips the run.
                if not store.claim_queued(run_id):
                    return
                fresh = True
            else:
                # "running" here means a run interrupted by a restart; v1
                # runs one API process, so only this worker resumes it.
                fresh = False
            log = TraceLogWriter(run_id, log_dir=self._log_dir)
            llm = self._llm_factory()
            if fresh and not store.load_events(run_id):
                run_task(row["task_id"], llm, executor=self._executor, run_id=run_id,
                         store=store, log=log)
            else:
                resume_run(run_id, llm, store, executor=self._executor, log=log)
        except Exception as e:  # never let one run kill the worker thread
            logger.exception("run %s crashed", run_id)
            store.mark_error(run_id, f"worker_error: {type(e).__name__}: {e}")


def create_app(
    llm_factory: Callable[[], Any] | None = None,
    executor=execute_code,
    store_factory: Callable[[], Any] = RunStore,
    log_dir=None,
    run_source: str = "api",
    rate_limit: int | None = None,
    owner_token: str | None = None,
) -> FastAPI:
    """`run_source` tags runs created through POST /runs; tests pass
    "test" so their runs never count toward the public stats.
    `rate_limit` is runs per IP per UTC day (default
    RATE_LIMIT_RUNS_PER_DAY, else 5; the design notes)."""
    if rate_limit is None:
        rate_limit = int(os.environ.get("RATE_LIMIT_RUNS_PER_DAY", 5))
    if owner_token is None:
        owner_token = os.environ.get("OWNER_TOKEN") or None
    if llm_factory is None:
        from ..agent.llm import GeminiClient
        llm_factory = GeminiClient

    worker = RunWorker(llm_factory, executor, store_factory, log_dir)
    api_store = {"store": None}
    store_lock = threading.Lock()  # psycopg connections aren't safe to share across threads

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        api_store["store"] = store_factory()
        worker.start()
        for run in api_store["store"].list_unfinished():
            logger.info("recovering run %s (%s)", run["run_id"], run["status"])
            worker.submit(run["run_id"])
        yield
        worker.stop()
        api_store["store"].close()

    app = FastAPI(title="Agentic Coding Agent", lifespan=lifespan)
    app.state.worker = worker

    def store():
        return api_store["store"]

    @app.post("/runs", status_code=202)
    def create_run(req: RunRequest, request: Request) -> dict[str, Any]:
        # Order matters: a bad task_id is rejected without using up one
        # of the caller's runs, and a rate-limited request never creates
        # a row or reaches the worker, so it costs no tokens or sandbox
        # time (the design notes "Public abuse limits").
        validate_task_id(req.task_id)
        owner = owner_source(request, owner_token)
        ip = client_ip(request)
        run_id = str(uuid.uuid4())
        with store_lock:
            used = None if owner else store().consume_rate_limit(ip, rate_limit)
            if not owner and used is None:
                retry_after = seconds_until_utc_midnight()
                raise HTTPException(
                    status_code=429,
                    detail={
                        "reason": "rate_limited",
                        "message": f"limit of {rate_limit} runs per IP per day reached",
                        "limit": rate_limit,
                        "retry_after_seconds": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )
            store().create_queued(run_id, req.task_id, source=owner or run_source)
        worker.submit(run_id)
        return {"run_id": run_id, "status": "queued"}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        try:
            uuid.UUID(run_id)
        except ValueError:
            raise HTTPException(status_code=404, detail="run not found")
        with store_lock:
            row = store().get_run(run_id)
            if row is None:
                raise HTTPException(status_code=404, detail="run not found")
            events = store().load_events(run_id)
        task = get_task(row["task_id"])
        return {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "status": row["status"],
            "steps": row["steps_taken"],
            "tokens": row["tokens_used"],
            "result": row["result"],
            "trace": public_trace(events, task),
        }

    @app.get("/stats")
    def get_stats() -> dict[str, Any]:
        with store_lock:
            return store().stats()

    return app


app = create_app()
