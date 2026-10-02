import threading

from riftwatch.web.jobs import JobManager


def test_job_runs_and_reports_progress():
    jobs = JobManager()
    job, created = jobs.submit("k", "demo", lambda progress: (progress("half"), 42)[1])
    done = jobs.wait(job.id)
    assert created and done.status == "done" and done.result == 42
    assert list(done.progress) == ["half"] and done.finished >= done.started


def test_failure_is_reported_not_raised():
    jobs = JobManager()

    def boom(_progress):
        raise RuntimeError("riot is down")

    done = jobs.wait(jobs.submit("k", "demo", boom)[0].id)
    assert done.status == "failed" and done.error == "RuntimeError: riot is down"


def test_same_key_shares_a_running_job():
    jobs = JobManager()
    gate = threading.Event()
    first, created1 = jobs.submit("k", "demo", lambda p: gate.wait(5))
    second, created2 = jobs.submit("k", "demo", lambda p: None)
    assert created1 and not created2 and second is first
    gate.set()
    jobs.wait(first.id)


def test_cooldown_then_new_job():
    jobs = JobManager(cooldown_s=0.0)
    a = jobs.wait(jobs.submit("k", "demo", lambda p: 1)[0].id)
    b, created = jobs.submit("k", "demo", lambda p: 2)
    assert created and b.id != a.id
    assert JobManager(cooldown_s=60).submit("x", "d", lambda p: 1)[1] is True


def test_failed_job_can_be_retried_immediately():
    jobs = JobManager(cooldown_s=60)

    def boom(_p):
        raise ValueError("x")

    failed = jobs.wait(jobs.submit("k", "demo", boom)[0].id)
    retry, created = jobs.submit("k", "demo", lambda p: "ok")
    assert failed.status == "failed" and created and jobs.wait(retry.id).result == "ok"


def test_old_finished_jobs_are_trimmed():
    jobs = JobManager(keep=5, cooldown_s=0)
    ids = [jobs.wait(jobs.submit(f"k{i}", "demo", lambda p: None)[0].id).id for i in range(12)]
    jobs.submit("last", "demo", lambda p: None)
    assert len(jobs._jobs) <= 6
    assert jobs.get(ids[0]) is None and jobs.get(ids[-1]) is not None
