import os
import threading

import pytest

TEST_DB = os.environ.get("RIFTWATCH_TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not TEST_DB, reason="RIFTWATCH_TEST_DATABASE_URL not set")


@pytest.fixture
def queue():
    from psycopg_pool import ConnectionPool

    from riftwatch.db import migrate
    from riftwatch.db.connection import connect
    from riftwatch.web.jobs import JobQueue

    with connect(TEST_DB) as conn:
        conn.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public")
        migrate.migrate(conn)
    pool = ConnectionPool(TEST_DB, min_size=1, max_size=6, open=True, kwargs={"autocommit": True})
    q = JobQueue(pool, workers=2, cooldown_s=60)
    yield q
    q.shutdown()
    pool.close()


def test_job_runs_and_reports_progress(queue):
    def work(params, progress):
        progress("half")
        return {"answer": params["x"] * 2}

    queue.register("demo", work)
    job, created = queue.submit("k", "demo", {"x": 21})
    assert created and job.status == "queued"
    assert queue.run_one()
    done = queue.get(job.id)
    assert done.status == "done" and done.result == {"answer": 42} and done.progress == ["half"]


def test_progress_keeps_only_the_last_lines(queue):
    queue.register("chatty", lambda p, progress: [progress(f"line {i}") for i in range(30)] and None)
    job, _ = queue.submit("k", "chatty", {})
    queue.run_one()
    progress = queue.get(job.id).progress
    assert len(progress) == 20 and progress[-1] == "line 29" and progress[0] == "line 10"


def test_failure_is_recorded_not_raised(queue):
    def boom(params, progress):
        raise RuntimeError("riot is down")

    queue.register("demo", boom)
    job, _ = queue.submit("k", "demo", {})
    assert queue.run_one()
    failed = queue.get(job.id)
    assert failed.status == "failed" and failed.error == "RuntimeError: riot is down"
    # A failed job doesn't block a retry.
    retry, created = queue.submit("k", "demo", {})
    assert created and retry.id != job.id


def test_same_key_shares_the_job_and_cooldown_applies(queue):
    queue.register("demo", lambda p, progress: "ok")
    first, _ = queue.submit("k", "demo", {})
    again, created = queue.submit("k", "demo", {})
    assert not created and again.id == first.id               # still queued
    queue.run_one()
    after, created = queue.submit("k", "demo", {})
    assert not created and after.id == first.id               # finished < 60 s ago
    queue.cooldown_s = 0
    _, created = queue.submit("k", "demo", {})
    assert created


def test_fair_scheduling_across_owners(queue):
    queue.register("demo", lambda p, progress: None)
    a1, _ = queue.submit("a1", "demo", {}, owner="alice")
    a2, _ = queue.submit("a2", "demo", {}, owner="alice")
    queue.submit("a3", "demo", {}, owner="alice")
    b1, _ = queue.submit("b1", "demo", {}, owner="bob")       # queued last
    assert queue.claim()[0] == a1.id          # nobody running yet: oldest first
    assert queue.claim()[0] == b1.id          # alice has one running, bob none: bob next
    assert queue.claim()[0] == a2.id


def test_concurrent_workers_never_take_the_same_job(queue):
    queue.register("demo", lambda p, progress: None)
    for i in range(12):
        queue.submit(f"k{i}", "demo", {}, owner=f"o{i}")
    taken, lock = [], threading.Lock()

    def grab():
        while (claimed := queue.claim()) is not None:
            with lock:
                taken.append(claimed[0])

    threads = [threading.Thread(target=grab) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(taken) == 12 and len(set(taken)) == 12


def test_stale_running_job_is_retried(queue):
    queue.register("demo", lambda p, progress: None)
    job, _ = queue.submit("k", "demo", {})
    assert queue.claim()[0] == job.id
    with queue.pool.connection() as conn:     # its worker died long ago
        conn.execute("UPDATE jobs SET heartbeat = now() - interval '10 minutes' WHERE id = %s", (job.id,))
    assert queue.claim()[0] == job.id


def test_only_registered_kinds_are_claimed(queue):
    queue.submit("k", "unknown_kind", {})
    assert queue.claim() is None


def test_background_workers_process_jobs(queue):
    queue.register("demo", lambda p, progress: p["n"] + 1)
    queue.start()
    job, _ = queue.submit("k", "demo", {"n": 1})
    assert queue.wait(job.id, timeout=10).result == 2
