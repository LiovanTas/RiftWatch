"""Background jobs for slow work (syncing from Riot) so web requests never wait on it.

In-process: a small thread pool plus a job table in memory. That is enough for one server
process; running several would need the job table in Postgres instead.

Two rules keep a busy "Update" button from burning the shared Riot quota:
  * a key (e.g. one player's sync) has at most one job queued or running -- asking again
    returns that job;
  * a key that finished within ``cooldown_s`` returns the finished job instead of starting
    a new one, like op.gg's update cooldown.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

Progress = Callable[[str], None]


@dataclass
class Job:
    id: str
    key: str
    kind: str
    status: str = "queued"          # queued | running | done | failed
    progress: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    result: Any = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "kind": self.kind, "status": self.status,
            "progress": list(self.progress), "result": self.result, "error": self.error,
            "created": self.created, "started": self.started, "finished": self.finished,
        }


class JobManager:
    def __init__(self, workers: int = 2, cooldown_s: float = 60.0, keep: int = 500) -> None:
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="riftwatch-job")
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._ids = itertools.count(1)
        self.cooldown_s = cooldown_s
        self.keep = keep

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, key: str, kind: str, fn: Callable[[Progress], Any]) -> tuple[Job, bool]:
        """Start ``fn`` unless an equivalent job is active or just finished.
        Returns (job, created)."""
        with self._lock:
            existing = self._jobs.get(self._by_key.get(key, ""))
            if existing and (existing.status in ("queued", "running") or (
                    existing.status == "done"
                    and time.time() - (existing.finished or 0) < self.cooldown_s)):
                return existing, False
            job = Job(id=f"j{next(self._ids)}", key=key, kind=kind)
            self._jobs[job.id] = job
            self._by_key[key] = job.id
            self._trim()
        self._pool.submit(self._run, job, fn)
        return job, True

    def _run(self, job: Job, fn: Callable[[Progress], Any]) -> None:
        job.status, job.started = "running", time.time()
        try:
            job.result = fn(job.progress.append)
            job.status = "done"
        except Exception as exc:  # a job's failure is reported, never raised into the pool
            job.error = f"{type(exc).__name__}: {exc}"
            job.progress.append(traceback.format_exc(limit=1).strip().splitlines()[-1])
            job.status = "failed"
        finally:
            job.finished = time.time()

    def _trim(self) -> None:
        if len(self._jobs) <= self.keep:
            return
        finished = sorted((j for j in self._jobs.values() if j.finished),
                          key=lambda j: j.finished or 0)
        for j in finished[: len(self._jobs) - self.keep]:
            del self._jobs[j.id]
            if self._by_key.get(j.key) == j.id:
                del self._by_key[j.key]

    def wait(self, job_id: str, timeout: float = 30.0) -> Job:
        """Block until a job finishes (tests and the CLI; the web never waits)."""
        deadline = time.monotonic() + timeout
        while True:
            job = self._jobs[job_id]
            if job.status in ("done", "failed") or time.monotonic() > deadline:
                return job
            time.sleep(0.02)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
