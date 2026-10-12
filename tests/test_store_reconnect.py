"""The store survives the database dropping its connection.

Neon closes idle connections; before this fix, the API's long-lived
connection stayed dead and every request returned 500 until a restart
(found in production, 2026-10-12). Here the store's connection is killed
server-side with pg_terminate_backend, which is what that looks like.
Real Postgres in DATABASE_URL; rows created here are deleted.
"""
import os
import uuid

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(not os.environ.get("DATABASE_URL"), reason="needs DATABASE_URL")


def _direct_url():
    # Neon's pooler ("-pooler" host) multiplexes connections; go direct so the
    # store gets its own server process that we can terminate.
    return os.environ["DATABASE_URL"].replace("-pooler.", ".", 1)


def _store():
    """A store whose connection is tagged with a unique application_name.
    Neon fronts every connection with a proxy, so the backend_pid a client
    sees isn't the real server process; pg_stat_activity is how we find it."""
    from agentic_coding_agent.db.store import RunStore

    tag = f"reconnect-test-{uuid.uuid4().hex[:8]}"
    sep = "&" if "?" in _direct_url() else "?"
    store = RunStore(url=f"{_direct_url()}{sep}application_name={tag}")
    store.test_tag = tag
    return store


def _kill_connection_of(store):
    """Terminate the store's server process(es), as Neon does to idle ones."""
    with psycopg.connect(_direct_url(), autocommit=True) as admin:
        killed = admin.execute(
            "SELECT count(*) FILTER (WHERE pg_terminate_backend(pid)) FROM pg_stat_activity "
            "WHERE application_name = %s",
            (store.test_tag,),
        ).fetchone()[0]
    assert killed >= 1, "no server process found for the store's connection"
    return killed


def test_read_after_connection_killed():
    store = _store()
    try:
        _kill_connection_of(store)
        stats = store.stats()                      # would raise OperationalError before the fix
        assert "public" in stats and "by_source" in stats
    finally:
        store.close()


def test_writes_after_connection_killed():
    store = _store()
    run_id = str(uuid.uuid4())
    try:
        _kill_connection_of(store)
        store.create_queued(run_id, "two_sum", source="test")
        _kill_connection_of(store)
        assert store.claim_queued(run_id) is True
        _kill_connection_of(store)
        store.append_event(run_id, {"seq": 1, "kind": "plan", "approach": "a", "subtasks": ["s"],
                                    "usage": {"total_tokens": 7}})
        _kill_connection_of(store)
        assert [e["kind"] for e in store.load_events(run_id)] == ["plan"]
        assert store.get_run(run_id)["tokens_used"] == 7
    finally:
        store.delete_run(run_id)
        store.close()


def test_rate_limit_after_connection_killed():
    store = _store()
    ip = "203.0.113.70"
    try:
        store.delete_rate_limit(ip)
        _kill_connection_of(store)
        assert store.consume_rate_limit(ip, 5) == 1   # counted exactly once despite the reconnect
        assert store.consume_rate_limit(ip, 5) == 2
    finally:
        store.delete_rate_limit(ip)
        store.close()


def test_real_query_errors_are_not_retried():
    """A broken query must still raise: only dead connections are retried."""
    store = _store()
    try:
        with pytest.raises(psycopg.errors.UndefinedTable):
            store._execute("SELECT * FROM no_such_table")
        assert store.stats()  # connection still usable afterwards
    finally:
        store.close()
