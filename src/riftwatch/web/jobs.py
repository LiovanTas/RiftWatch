"""Background jobs for slow work (syncing from Riot), kept in Postgres.

Any number of server processes can run workers against the same table:

* a worker claims the next job with ``FOR UPDATE SKIP LOCKED``, so two workers never take
  the same one;
* "next" means: from whichever owner (visitor) has the fewest jobs running, oldest first --
  one visitor queuing several syncs can't make everyone else wait;
* a running job that stops sending heartbeats (its process died) goes back to the queue.

Jobs are stored as a *kind* plus JSON *params*; each process registers a handler per kind,
since a job may be picked up by a different process than the one that queued it.

Two rules keep a busy "Update" button from burning the shared Riot quota:
  * a key (e.g. one player's sync) has at most one job queued or running -- asking again
    returns that job;
  * a key that finished within ``cooldown_s`` returns the finished job instead of starting
    a new one, like op.gg's update cooldown.
"""

from __future__ import annotations

import os
import socket
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

Progress = Callable[[str], None]
Handler = Callable[[dict[str, Any], Progress], Any]

KEEP_PROGRESS = 20


@dataclass
class Job:
    id: int
    key: str
    kind: str
    status: str
    progress: list[str]
    result: Any
    error: str | None
    created: datetime | None
    started: datetime | None
    finished: datetime | None
    token: str = ""             # the public id; ``id`` never leaves the server

    def to_json(self) -> dict[str, Any]:
        def iso(t):
            return t.isoformat() if t else None
        return {"id": self.token, "kind": self.kind, "status": self.status,
                "progress": self.progress, "result": self.result, "error": self.error,
                "created": iso(self.created), "started": iso(self.started),
                "finished": iso(self.finished)}


_COLUMNS = "id, key, kind, status, progress, result, error, created, started, finished, token"


def _job(row) -> Job:
    return Job(*row)


class JobQueue:
    def __init__(self, pool: ConnectionPool, *, workers: int = 2, cooldown_s: float = 60.0,
                 stale_after_s: float = 120.0, poll_s: float = 0.25) -> None:
        self.pool = pool
        self.handlers: dict[str, Handler] = {}
        self.workers = workers
        self.cooldown_s = cooldown_s
        self.stale_after_s = stale_after_s
        self.poll_s = poll_s
        self.name = f"{socket.gethostname()}:{os.getpid()}"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def register(self, kind: str, handler: Handler) -> None:
        self.handlers[kind] = handler

    # -- queueing -------------------------------------------------------------------------

    def submit(self, key: str, kind: str, params: dict[str, Any], owner: str = "anonymous") -> tuple[Job, bool]:
        """Queue a job unless an equivalent one is active or just finished. Returns
        (job, created)."""
        with self.pool.connection() as conn, conn.transaction():
            # Serialise submits for one key so two clicks can't both create a job.
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (key,))
            row = conn.execute(
                f"""
                SELECT {_COLUMNS} FROM jobs
                 WHERE key = %s AND (status IN ('queued', 'running')
                       OR (status = 'done' AND finished > now() - make_interval(secs => %s)))
                 ORDER BY created DESC LIMIT 1
                """,
                (key, self.cooldown_s),
            ).fetchone()
            if row:
                return _job(row), False
            row = conn.execute(
                f"INSERT INTO jobs (key, kind, params, owner) VALUES (%s, %s, %s, %s) "
                f"RETURNING {_COLUMNS}",
                (key, kind, Jsonb(params), owner),
            ).fetchone()
            return _job(row), True

    def get(self, job_id: int | str) -> Job | None:
        """By internal id (int) or by public token (str), as the web API is given it."""
        column = "token" if isinstance(job_id, str) else "id"
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT {_COLUMNS} FROM jobs WHERE {column} = %s",
                               (job_id,)).fetchone()
        return _job(row) if row else None

    # -- working --------------------------------------------------------------------------

    def claim(self) -> tuple[int, str, dict[str, Any]] | None:
        """Take the next job, fairly. Also returns stale running jobs to the queue first."""
        with self.pool.connection() as conn, conn.transaction():
            conn.execute(
                """
                UPDATE jobs SET status = 'queued', worker = NULL
                 WHERE status = 'running' AND heartbeat < now() - make_interval(secs => %s)
                """,
                (self.stale_after_s,),
            )
            row = conn.execute(
                """
                WITH busy AS (
                    SELECT owner, count(*) AS n FROM jobs WHERE status = 'running' GROUP BY owner
                ), next AS (
                    SELECT j.id FROM jobs j LEFT JOIN busy b USING (owner)
                     WHERE j.status = 'queued' AND j.kind = ANY(%s)
                     ORDER BY coalesce(b.n, 0), j.created
                     LIMIT 1
                     FOR UPDATE OF j SKIP LOCKED
                )
                UPDATE jobs SET status = 'running', worker = %s, started = now(), heartbeat = now()
                  FROM next WHERE jobs.id = next.id
                RETURNING jobs.id, jobs.kind, jobs.params
                """,
                (list(self.handlers), self.name),
            ).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def _progress(self, job_id: int, message: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                """
                UPDATE jobs SET heartbeat = now(),
                       progress = (SELECT coalesce(jsonb_agg(x), '[]') FROM (
                           SELECT x FROM jsonb_array_elements(progress || to_jsonb(%s::text)) x
                           OFFSET greatest(jsonb_array_length(progress) + 1 - %s, 0)) t)
                 WHERE id = %s
                """,
                (message, KEEP_PROGRESS, job_id),
            )

    def _finish(self, job_id: int, *, result: Any = None, error: str | None = None) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE jobs SET status = %s, result = %s, error = %s, finished = now() WHERE id = %s",
                ("failed" if error else "done", Jsonb(result) if result is not None else None,
                 error, job_id),
            )

    def run_one(self) -> bool:
        """Claim and run one job. Returns False if there was nothing to do."""
        claimed = self.claim()
        if claimed is None:
            return False
        job_id, kind, params = claimed
        try:
            result = self.handlers[kind](params, lambda msg: self._progress(job_id, msg))
            self._finish(job_id, result=result)
        except Exception as exc:  # a job's failure is recorded, never raised into the worker
            self._progress(job_id, traceback.format_exc(limit=1).strip().splitlines()[-1])
            self._finish(job_id, error=f"{type(exc).__name__}: {exc}")
        return True

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.run_one():
                    self._stop.wait(self.poll_s)
            except Exception:      # database hiccup: back off, keep the worker alive
                self._stop.wait(2.0)

    def start(self) -> None:
        for i in range(self.workers):
            t = threading.Thread(target=self._worker, name=f"riftwatch-job-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def wait(self, job_id: int | str, timeout: float = 30.0) -> Job:
        """Block until a job finishes (tests and the CLI; the web never waits)."""
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.status in ("done", "failed") or time.monotonic() > deadline:
                return job
            time.sleep(0.05)

    def shutdown(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5)
