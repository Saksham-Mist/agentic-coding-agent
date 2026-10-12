"""Step 5.2: the eval harness runs every (task, repeat) through the API
as the owner, records the batch, and produces the results table; the
table can be rebuilt later from Postgres. In-process API, fake LLM and
executor, real Postgres; everything created is deleted afterwards.
"""
import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "scripts" / "run_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL") or not (ROOT / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)

TOKEN = "test-owner-token-for-eval"


class FakeLLM:
    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 40}
        if "planning step" in (system_instruction or ""):
            return {"text": '{"approach": "a", "subtasks": ["s1"]}', "usage": usage}
        return {"text": "```python\ndef f(*a, **k):\n    return None\n```", "usage": usage}


def make_executor(fail_tasks):
    """Passes everything except runs whose entry point is in fail_tasks."""
    def executor(code, language, cases, entry_point=None, **kw):
        ok = entry_point not in fail_tasks
        return {"status": "pass" if ok else "fail", "stdout": "", "stderr": "",
                "test_results": [{"input": c["input"], "expected": c["expected_output"],
                                  "actual": c["expected_output"] if ok else None, "passed": ok}
                                 for c in cases]}
    return executor


@pytest.fixture
def env(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    created = {"batches": []}
    store = RunStore()

    def start(fail_tasks=()):
        app = create_app(llm_factory=FakeLLM, executor=make_executor(set(fail_tasks)),
                         log_dir=tmp_path, run_source="test", rate_limit=1, owner_token=TOKEN)
        c = TestClient(app, client=("203.0.113.90", 50000))
        c.__enter__()
        created.setdefault("clients", []).append(c)
        return c

    yield {"start": start, "store": store, "created": created}
    for c in created.get("clients", []):
        c.__exit__(None, None, None)
    for b in created["batches"]:
        for row in store.eval_rows(b):
            if row["run_id"]:
                store.delete_run(row["run_id"])
        store.delete_eval_batch(b)
    store.delete_rate_limit("203.0.113.90")
    store.close()


def test_eval_runs_every_task_and_repeat_and_prints_table(env):
    tasks = ["two_sum", "valid_parentheses"]
    client = env["start"](fail_tasks={"is_valid"})  # valid_parentheses' entry point fails
    logs = []
    batch = ev.run_eval(client, env["store"], tasks, 2, TOKEN, "test", "python scripts/run_eval.py --test",
                        "fake-model", "abc123", poll_seconds=0.2, log=logs.append)
    env["created"]["batches"].append(batch)

    rows = env["store"].eval_rows(batch)
    assert [(r["task_id"], r["repeat"]) for r in rows] == [
        ("two_sum", 1), ("two_sum", 2), ("valid_parentheses", 1), ("valid_parentheses", 2)]
    assert all(r["error"] is None and r["run_id"] for r in rows)

    # repeats are interleaved: all tasks once, then again
    order = [l.split("]")[1].split()[0] for l in logs if l.strip().startswith("[")]
    assert order == ["two_sum", "valid_parentheses", "two_sum", "valid_parentheses"]

    summary = ev.summarize(rows)
    assert summary["tasks"]["two_sum"]["runs_passed"] == 2
    assert summary["tasks"]["valid_parentheses"]["runs_passed"] == 0
    assert summary["overall"]["runs_passed"] == 2 and summary["overall"]["runs"] == 4
    assert summary["overall"]["tasks_passed_every_repeat"] == 1
    assert summary["by_difficulty"]["Easy"]["tasks"] == 2

    table = ev.report(env["store"], batch)
    assert "| two_sum | Easy | 2/2 |" in table
    assert "| valid_parentheses | Easy | 0/2 |" in table
    assert "command: python scripts/run_eval.py --test" in table
    assert "commit: abc123" in table and "model: fake-model" in table


def test_eval_runs_are_tagged_eval_and_skip_the_public_limit(env):
    # rate_limit=1 per IP: 3 runs only work because the owner path is used
    client = env["start"]()
    batch = ev.run_eval(client, env["store"], ["two_sum"], 3, TOKEN, None, "cmd", None, None,
                        poll_seconds=0.2, log=lambda *_: None)
    env["created"]["batches"].append(batch)
    rows = env["store"].eval_rows(batch)
    assert len(rows) == 3 and all(r["status"] == "passed" for r in rows)
    sources = {env["store"]._conn.execute("SELECT source FROM runs WHERE run_id = %s",
                                          (r["run_id"],)).fetchone()[0] for r in rows}
    assert sources == {"eval"}


def test_rejected_submission_is_recorded_not_raised(env):
    client = env["start"]()
    batch = ev.run_eval(client, env["store"], ["two_sum"], 1, "wrong-token", None, "cmd", None, None,
                        poll_seconds=0.2, log=lambda *_: None)
    env["created"]["batches"].append(batch)
    (row,) = env["store"].eval_rows(batch)
    assert row["run_id"] is None and row["error"].startswith("HTTP 401")
    summary = ev.summarize([row])
    assert summary["overall"]["runs_passed"] == 0 and summary["overall"]["not_finished"] == 1


def test_runnable_tasks_matches_api_rule():
    # Per set, so it holds however many main/stress entries are valid.
    from agentic_coding_agent.agent.catalog import load_catalog, task_set, validate_entry
    catalog = load_catalog()
    for which in ("main", "stress"):
        expected = sorted(t for t, e in catalog.items() if not validate_entry(e) and task_set(e) == which)
        assert ev.runnable_tasks(which) == expected
    assert ev.runnable_tasks() == ev.runnable_tasks("main")  # default is the main set


def test_eval_sets_are_separate(monkeypatch):
    from agentic_coding_agent.agent.catalog import load_catalog

    real = load_catalog()
    stress = {**real["two_sum"], "task_id": "zz_stress_copy", "set": "stress"}
    monkeypatch.setattr(ev, "load_catalog", lambda path=None: {**real, "zz_stress_copy": stress})
    assert "zz_stress_copy" not in ev.runnable_tasks()          # default = main
    assert "zz_stress_copy" in ev.runnable_tasks("stress")
    assert not set(ev.runnable_tasks("main")) & set(ev.runnable_tasks("stress"))  # never mixed
