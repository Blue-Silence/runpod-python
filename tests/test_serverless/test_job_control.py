"""Pause/resume behavior through the real JobScaler loops, without network I/O."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from runpod.serverless import JobControl
from runpod.serverless.modules import rp_scale


class Progress:
    def __init__(self):
        self.jobs = set()

    def add(self, job):
        self.jobs.add(job["id"])

    def remove(self, job):
        self.jobs.remove(job["id"])

    def get_job_count(self):
        return len(self.jobs)


@pytest.fixture
async def worker(monkeypatch):
    monkeypatch.setattr(rp_scale, "JobsProgress", Progress)
    control = JobControl()
    scaler = rp_scale.JobScaler({"handler": lambda job: job, "job_control": control})
    scaler.jobs_handler = AsyncMock()
    tasks = []

    def start(coro):
        task = asyncio.create_task(coro)
        tasks.append(task)
        return task

    yield scaler, start

    scaler.kill_worker()
    tasks.extend(scaler.jobs_tasks.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def queue_job(scaler, job_id):
    job = {"id": job_id, "input": {}}
    await scaler.jobs_queue.put(job)
    scaler.job_progress.add(job)
    scaler._jobs_available.set()


async def wait(event):
    await asyncio.wait_for(event.wait(), timeout=1)


def test_public_controls_default_to_enabled_and_are_idempotent():
    control = JobControl()
    assert not control.polling_paused
    assert not control.dispatch_paused
    for _ in range(2):
        control.pause_polling()
        control.pause_dispatch()
    assert control.polling_paused
    assert control.dispatch_paused
    for _ in range(2):
        control.resume_dispatch()
        control.resume_polling()
    assert not control.polling_paused
    assert not control.dispatch_paused


@pytest.mark.asyncio
@pytest.mark.parametrize("from_thread", [False, True])
async def test_poll_pause_allows_queued_jobs_and_resume_wakes_intake(
    worker, from_thread
):
    scaler, start = worker
    fetched = asyncio.Event()
    handled = asyncio.Event()

    async def fetcher(_session, _needed):
        fetched.set()
        await asyncio.Event().wait()

    async def handler(_session, _config, _job):
        handled.set()

    scaler.jobs_fetcher = fetcher
    scaler.jobs_handler = handler
    scaler.job_control.pause_polling()
    await queue_job(scaler, "A")
    start(scaler.get_jobs(None))
    start(scaler.run_jobs(None))
    await wait(handled)
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert not fetched.is_set()

    if from_thread:
        timer = threading.Timer(0.02, scaler.job_control.resume_polling)
        timer.start()
        try:
            await wait(fetched)
        finally:
            timer.join()
    else:
        scaler.job_control.resume_polling()
        await wait(fetched)


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_before_fetch", [False, True])
async def test_dispatch_pause_retains_acquired_jobs_and_does_not_cancel_fetch(
    worker, pause_before_fetch
):
    scaler, start = worker
    fetching = asyncio.Event()
    release_fetch = asyncio.Event()
    handled = []
    fetch_count = 0

    async def fetcher(_session, _needed):
        nonlocal fetch_count
        fetch_count += 1
        fetching.set()
        await release_fetch.wait()
        return [{"id": "B", "input": {}}]

    async def handler(_session, _config, job):
        handled.append(job["id"])

    scaler.jobs_fetcher = fetcher
    scaler.jobs_handler = handler
    if pause_before_fetch:
        scaler.job_control.pause_dispatch()
    start(scaler.run_jobs(None))
    intake = start(scaler.get_jobs(None))
    await wait(fetching)
    scaler.job_control.pause_dispatch()
    scaler.job_control.pause_polling()
    release_fetch.set()
    await wait(scaler._jobs_available)
    await asyncio.sleep(0.02)

    assert not intake.done()
    assert handled == []
    assert scaler.jobs_queue.qsize() == 1
    assert scaler.job_progress.jobs == {"B"}
    assert not scaler.jobs_tasks
    assert fetch_count == 1

    scaler.job_control.resume_dispatch()
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert handled == ["B"]
    assert fetch_count == 1  # dispatch resume does not resume intake
    assert scaler.job_progress.get_job_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("from_thread", [False, True])
async def test_resume_starts_queued_b_after_a_finishes_while_paused(
    worker, from_thread
):
    """Regression: queue occupancy must never prevent dispatch from resuming."""
    scaler, start = worker
    started_a = asyncio.Event()
    finish_a = asyncio.Event()
    handled = []

    async def handler(_session, _config, job):
        handled.append(job["id"])
        if job["id"] == "A":
            started_a.set()
            await finish_a.wait()

    scaler.jobs_handler = handler
    await queue_job(scaler, "A")
    start(scaler.run_jobs(None))
    await wait(started_a)
    scaler.job_control.pause_polling()
    scaler.job_control.pause_dispatch()
    await queue_job(scaler, "B")
    finish_a.set()
    await wait(scaler._slot_freed)
    await asyncio.sleep(0.02)

    assert handled == ["A"]
    assert scaler.jobs_queue.qsize() == 1
    assert scaler.job_progress.jobs == {"B"}
    assert not scaler.jobs_tasks
    assert scaler.current_concurrency == 1

    if from_thread:
        timer = threading.Timer(0.02, scaler.job_control.resume_dispatch)
        timer.start()
        try:
            await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
        finally:
            timer.join()
    else:
        scaler.job_control.resume_dispatch()
        await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert handled == ["A", "B"]
    assert scaler.job_progress.get_job_count() == 0


@pytest.mark.asyncio
async def test_dispatch_resume_does_not_wait_for_running_job(worker):
    scaler, start = worker
    scaler.current_concurrency = 2
    scaler.jobs_queue = asyncio.Queue(maxsize=2)
    started_a = asyncio.Event()
    started_b = asyncio.Event()
    finish_a = asyncio.Event()

    async def handler(_session, _config, job):
        if job["id"] == "A":
            started_a.set()
            await finish_a.wait()
        else:
            started_b.set()

    scaler.jobs_handler = handler
    await queue_job(scaler, "A")
    start(scaler.run_jobs(None))
    await wait(started_a)
    scaler.job_control.pause_dispatch()
    await queue_job(scaler, "B")
    await asyncio.sleep(0.02)
    assert not started_b.is_set()
    scaler.job_control.resume_dispatch()
    await wait(started_b)
    assert "A" in scaler.jobs_tasks
    finish_a.set()
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("queued", [False, True])
async def test_shutdown_wakes_paused_loops_and_drains_held_jobs(worker, queued):
    scaler, start = worker
    scaler.job_control.pause_polling()
    scaler.job_control.pause_dispatch()
    scaler.jobs_fetcher = AsyncMock()
    if queued:
        await queue_job(scaler, "held")
    intake = start(scaler.get_jobs(None))
    dispatch = start(scaler.run_jobs(None))
    await asyncio.sleep(0.02)
    assert not intake.done()
    assert not dispatch.done()
    scaler.kill_worker()
    await asyncio.wait_for(asyncio.gather(intake, dispatch), timeout=1)
    scaler.jobs_fetcher.assert_not_called()
    assert scaler.jobs_handler.await_count == int(queued)
    assert scaler.job_progress.get_job_count() == 0


@pytest.mark.asyncio
async def test_stop_channel_remains_active_while_poll_and_dispatch_are_paused(worker):
    scaler, start = worker
    started = asyncio.Event()
    cancelled = asyncio.Event()
    send_stop = asyncio.Event()

    async def handler(_session, _config, _job):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def stop_fetcher(_session):
        await send_stop.wait()
        send_stop.clear()
        return ["A"]

    scaler.jobs_handler = handler
    scaler.stop_signals_fetcher = stop_fetcher
    await queue_job(scaler, "A")
    start(scaler.run_jobs(None))
    start(scaler.monitor_stop_signals(None))
    await wait(started)
    scaler.job_control.pause_polling()
    scaler.job_control.pause_dispatch()
    send_stop.set()
    await wait(cancelled)
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert scaler.job_progress.get_job_count() == 0


@pytest.mark.asyncio
async def test_repeated_pause_resume_keeps_every_queued_job(worker):
    scaler, start = worker
    handled = []

    async def handler(_session, _config, job):
        handled.append(job["id"])

    scaler.jobs_handler = handler
    start(scaler.run_jobs(None))
    for index in range(5):
        scaler.job_control.pause_dispatch()
        scaler.job_control.pause_dispatch()
        await queue_job(scaler, str(index))
        await asyncio.sleep(0.01)
        assert len(handled) == index
        scaler.job_control.resume_dispatch()
        scaler.job_control.resume_dispatch()
        await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert handled == [str(index) for index in range(5)]


@pytest.mark.asyncio
async def test_poll_resume_wakes_full_worker_wait_without_requiring_drain(
    worker, monkeypatch
):
    scaler, start = worker
    monkeypatch.setattr(rp_scale, "FULL_WAIT_SECONDS", 10)
    scaler.concurrency_modifier = lambda _: 2
    release = {job_id: asyncio.Event() for job_id in ("A", "B")}
    both_started = asyncio.Event()
    refetched = asyncio.Event()
    started = []
    fetch_sizes = []

    async def fetcher(_session, needed):
        fetch_sizes.append(needed)
        if len(fetch_sizes) == 1:
            return [{"id": job_id, "input": {}} for job_id in release]
        refetched.set()
        await asyncio.Event().wait()

    async def handler(_session, _config, job):
        started.append(job["id"])
        if len(started) == 2:
            both_started.set()
        await release[job["id"]].wait()

    scaler.jobs_fetcher = fetcher
    scaler.jobs_handler = handler
    start(scaler.get_jobs(None))
    start(scaler.run_jobs(None))
    await wait(both_started)
    release["A"].set()
    # Let intake observe the freed slot and continue its full-worker wait.
    await asyncio.sleep(0.02)
    assert fetch_sizes == [2]
    assert scaler.job_progress.jobs == {"B"}

    # Both changes happen before the intake loop runs again.
    scaler.job_control.pause_polling()
    scaler.job_control.resume_polling()
    await wait(refetched)
    assert fetch_sizes == [2, 1]
    assert "B" in scaler.jobs_tasks
