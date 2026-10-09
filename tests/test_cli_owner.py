"""Step 4.4: the CLI is a thin client over the API, and the owner token
skips the public rate limit.

The CLI functions take an httpx.Client; FastAPI's TestClient is one, so
these run the real CLI code against an in-process API (fake LLM and
executor, real Postgres). The live check against uvicorn is separate.
"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()

from agentic_coding_agent import cli

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL")
    or not (Path(__file__).resolve().parents[1] / "task_catalog.json").exists(),
    reason="needs DATABASE_URL and the local-only task_catalog.json",
)

TOKEN = "test-owner-token-not-a-real-secret"


class FakeLLM:
    def generate(self, prompt, system_instruction=None):
        usage = {"prompt_tokens": 0, "response_tokens": 0, "thoughts_tokens": 0, "total_tokens": 50}
        if "planning step" in (system_instruction or ""):
            return {"text": '{"approach": "a", "subtasks": ["s1"]}', "usage": usage}
        return {"text": "```python\ndef two_sum(nums, target):\n    return [0, 1]\n```", "usage": usage}


def passing(code, language, cases, **kw):
    return {"status": "pass", "stdout": "", "stderr": "",
            "test_results": [{"input": c["input"], "expected": c["expected_output"],
                              "actual": c["expected_output"], "passed": True} for c in cases]}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """One API (limit 1/day, owner token set) shared by these tests."""
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app
    from agentic_coding_agent.db.store import RunStore

    ip = "203.0.113.60"
    store = RunStore()
    store.delete_rate_limit(ip)
    app = create_app(llm_factory=FakeLLM, executor=passing,
                     log_dir=tmp_path_factory.mktemp("logs"), run_source="test",
                     rate_limit=1, owner_token=TOKEN)
    created = []
    with TestClient(app, client=(ip, 50000)) as client:
        yield {"client": client, "store": store, "ip": ip, "created": created}
        # drain: wait for every run we started before deleting rows
        for run_id in created:
            cli.status(client, run_id)
    for run_id in created:
        store.delete_run(run_id)
    store.delete_rate_limit(ip)
    store.close()


def _source(store, run_id):
    return store._conn.execute("SELECT source FROM runs WHERE run_id = %s", (run_id,)).fetchone()[0]


def _slots_used(store, ip):
    row = store._conn.execute(
        "SELECT count FROM rate_limits WHERE ip = %s AND day = (now() AT TIME ZONE 'utc')::date", (ip,)
    ).fetchone()
    return row[0] if row else 0


def test_owner_token_skips_rate_limit_and_tags_cli(env):
    c, store = env["client"], env["store"]
    for _ in range(3):  # limit is 1/day; owner gets through every time
        r = c.post("/runs", json={"task_id": "two_sum"}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 202
        env["created"].append(r.json()["run_id"])
        assert _source(store, r.json()["run_id"]) == "cli"
    assert _slots_used(store, env["ip"]) == 0  # owner runs don't use public slots


def test_owner_can_tag_eval_runs(env):
    r = env["client"].post("/runs", json={"task_id": "two_sum"},
                           headers={"Authorization": f"Bearer {TOKEN}", "X-Run-Source": "eval"})
    assert r.status_code == 202
    env["created"].append(r.json()["run_id"])
    assert _source(env["store"], r.json()["run_id"]) == "eval"


@pytest.mark.parametrize("header", [f"Bearer {TOKEN}x", "Bearer ", "Basic abc", TOKEN])
def test_bad_owner_credentials_are_401_and_create_nothing(env, header):
    before = env["store"]._conn.execute("SELECT count(*) FROM runs").fetchone()[0]
    r = env["client"].post("/runs", json={"task_id": "two_sum"}, headers={"Authorization": header})
    assert r.status_code == 401
    assert env["store"]._conn.execute("SELECT count(*) FROM runs").fetchone()[0] == before


def test_public_caller_cannot_choose_source(env):
    # X-Run-Source only counts for the owner; a public caller stays public
    # (this also uses up the IP's single public slot for later tests).
    r = env["client"].post("/runs", json={"task_id": "two_sum"}, headers={"X-Run-Source": "eval"})
    assert r.status_code == 202
    env["created"].append(r.json()["run_id"])
    assert _source(env["store"], r.json()["run_id"]) == "test"


def test_cli_run_follows_trace_to_passed(env, capsys):
    code = cli.run(env["client"], "two_sum", token=TOKEN, poll_seconds=0.2)
    out = capsys.readouterr().out
    run_id = out.split("run_id: ")[1].split()[0]
    env["created"].append(run_id)
    assert code == 0
    assert "PLAN" in out and "WRITE attempt 1" in out and "DONE  status=passed" in out
    assert '"status": "passed"' in out


def test_cli_run_reports_rate_limit(env, capsys):
    # public slot was used in test_public_caller_cannot_choose_source
    code = cli.run(env["client"], "two_sum", token=None)
    err = capsys.readouterr().err
    assert code == 2
    assert "HTTP 429" in err and "retry after" in err


def test_cli_run_reports_unknown_task(env, capsys):
    code = cli.run(env["client"], "no_such_task", token=TOKEN)
    assert code == 2
    assert "unknown_task_id" in capsys.readouterr().err


def test_cli_stats(env, capsys):
    assert cli.stats(env["client"]) == 0
    assert "by_source" in capsys.readouterr().out


def test_server_without_owner_token_rejects_bearer(tmp_path):
    from fastapi.testclient import TestClient

    from agentic_coding_agent.api.app import create_app

    app = create_app(llm_factory=FakeLLM, executor=passing, log_dir=tmp_path,
                     run_source="test", rate_limit=1, owner_token="")
    with TestClient(app, client=("203.0.113.61", 50000)) as c:
        r = c.post("/runs", json={"task_id": "two_sum"}, headers={"Authorization": f"Bearer {TOKEN}"})
        assert r.status_code == 401
