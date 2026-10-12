"""Public API — step 4.1.

    POST /runs        {task_id} -> 202 {run_id, status: queued}
    GET  /runs/{id}   status, counters and the public trace
    GET  /stats       success rate / steps / tokens / wall clock
    GET  /tasks       the public task list, visible examples only
    GET  /            a one-page demo that drives the three calls above

Execution model: POST only inserts a `queued` row
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
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

load_dotenv()

from ..agent.catalog import CatalogLoadError, get_task, load_catalog, task_set, validate_entry  # noqa: E402
from ..agent.graph import resume_run, run_task  # noqa: E402
from ..agent.trace_log import TraceLogWriter  # noqa: E402
from ..db.store import RunStore  # noqa: E402
from ..sandbox.executor import execute_code  # noqa: E402

logger = logging.getLogger("agentic_coding_agent.api")

_DEMO_PAGE = Path(__file__).with_name("static") / "index.html"


class RunRequest(BaseModel):
    # Catalog ids are short snake_case names; anything else is rejected
    # before it reaches any code (step 4.2).
    task_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")


def _is_ready(entry: dict[str, Any]) -> bool:
    """A catalog entry is runnable only if it passes the full catalog
    validation (validate_entry): owner-reviewed prompt, >= 8 cases, a
    max-size case, etc. A weaker "has a prompt and some test cases"
    rule once let a 1-case entry with an unreviewed drafted prompt
    through, and would let the eval run on unapproved prompts."""
    return not validate_entry(entry)


def validate_task_id(task_id: str, owner: bool = False) -> None:
    """Step 4.2: POST /runs only runs ready entries of the fixed catalog
    (public abuse limits). Raises 422 otherwise, before
    any run row exists or any LLM call is made. Public callers only see
    the main set; owner-only "stress" tasks look unknown to them."""
    try:
        catalog = load_catalog()
    except CatalogLoadError:
        raise HTTPException(status_code=503, detail={"reason": "catalog_unavailable"})
    if not owner:
        catalog = {t: e for t, e in catalog.items() if task_set(e) == "main"}
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


def public_tasks(catalog: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """The tasks a public caller can run, as the demo page shows them:
    the problem text and only the visible examples (the same ones the
    model sees). Hidden cases are reduced to a count."""
    out = []
    for task_id, e in catalog.items():
        if task_set(e) != "main" or not _is_ready(e):
            continue
        cases = e["test_cases"]
        out.append({
            "task_id": task_id,
            "prompt": e["prompt"],
            "function_signature": e.get("function_signature"),
            "constraints": e.get("constraints"),
            "examples": [{"input": c["input"], "expected_output": c["expected_output"]}
                         for c in cases if c.get("visible")],
            "hidden_cases": sum(1 for c in cases if not c.get("visible")),
        })
    return out


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
    (the operator's own credentials).

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


class HeadAsGet:
    """Answer HEAD like GET minus the body. Uptime monitors check with
    HEAD, and FastAPI's GET routes otherwise return 405 to it. Done as
    middleware so the routes (and the API docs) stay GET-only."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "HEAD":
            return await self.app(scope, receive, send)

        async def send_headers_only(message):
            if message["type"] == "http.response.body":
                message = {"type": "http.response.body", "body": b"", "more_body": False}
                if not getattr(send_headers_only, "done", False):
                    send_headers_only.done = True
                    await send(message)
                return
            await send(message)

        await self.app({**scope, "method": "GET"}, receive, send_headers_only)


class RunWorker:
    """Single background thread: runs are executed strictly one at a time.

    stop() lets the run in progress finish, then exits without starting
    anything else; runs still queued stay `queued` in Postgres and are
    picked up at the next startup. (Before, stop() only queued a
    sentinel behind pending runs and waited 5s, so the thread kept
    working after shutdown -- in tests, into rows already deleted.)"""

    STOP_TIMEOUT_SECONDS = 120

    def __init__(self, llm_factory: Callable[[], Any], executor, store_factory, log_dir):
        self._queue: queue.Queue[tuple[str, bool] | None] = queue.Queue()
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

    def submit(self, run_id: str, recover: bool = False) -> None:
        """Queue a run. `recover=True` only from startup recovery: it is
        the one case where a `running` row means "interrupted, resume
        it" rather than "another worker is executing it right now"."""
        self._queue.put((run_id, recover))

    def _loop(self) -> None:
        store = self._store_factory()
        try:
            while not self._stopping.is_set():
                item = self._queue.get()
                if item is None or self._stopping.is_set():
                    break
                self._process(store, *item)
        finally:
            store.close()

    def _process(self, store, run_id: str, recover: bool = False) -> None:
        try:
            row = store.get_run(run_id)
            if row is None or row["status"] not in ("queued", "running"):
                return
            if row["status"] == "queued":
                # Claim before starting: a duplicate queue entry or a second
                # worker that also saw 'queued' loses the claim and skips.
                if not store.claim_queued(run_id):
                    return
                fresh = True
            elif recover:
                # Startup recovery: 'running' means the previous server died
                # mid-run. v1 runs one API process, so only this worker
                # resumes it (not claim-protected across processes).
                fresh = False
            else:
                # 'running' on an ordinary submission means someone else is
                # executing it right now. Resuming here would run it twice
                # (seen in testing: two workers' step rows collided).
                return
            log = TraceLogWriter(run_id, log_dir=self._log_dir)
            llm = self._llm_factory()
            if fresh and not store.load_events(run_id):
                run_task(row["task_id"], llm, executor=self._executor, run_id=run_id,
                         store=store, log=log)
            else:
                resume_run(run_id, llm, store, executor=self._executor, log=log)
        except Exception as e:  # never let one run kill the worker thread
            logger.exception("run %s crashed", run_id)
            try:
                store.mark_error(run_id, f"worker_error: {type(e).__name__}: {e}")
            except Exception:
                # e.g. the database is unreachable right now; the run stays
                # 'running' and is resumed on the next startup
                logger.exception("could not mark run %s as error", run_id)


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
    RATE_LIMIT_RUNS_PER_DAY, else 5)."""
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
            worker.submit(run["run_id"], recover=True)
        yield
        worker.stop()
        api_store["store"].close()

    app = FastAPI(title="Agentic Coding Agent", lifespan=lifespan)
    app.state.worker = worker
    app.add_middleware(HeadAsGet)

    def store():
        return api_store["store"]

    @app.get("/", include_in_schema=False)
    def demo_page() -> HTMLResponse:
        # One static page, no build step: it only calls the public API below.
        return HTMLResponse(_DEMO_PAGE.read_text(encoding="utf-8"))

    @app.get("/tasks")
    def list_tasks() -> list[dict[str, Any]]:
        try:
            return public_tasks(load_catalog())
        except CatalogLoadError:
            raise HTTPException(status_code=503, detail={"reason": "catalog_unavailable"})

    @app.post("/runs", status_code=202)
    def create_run(req: RunRequest, request: Request) -> dict[str, Any]:
        # Order matters: a bad task_id is rejected without using up one
        # of the caller's runs, and a rate-limited request never creates
        # a row or reaches the worker, so it costs no tokens or sandbox
        # time.
        # Owner check first: it decides which task sets are allowed.
        owner = owner_source(request, owner_token)
        validate_task_id(req.task_id, owner=bool(owner))
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
