"""Return rejected jobs to the same bounded queue, with real handler adapters."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from runpod.serverless import RequeueJob
from runpod.serverless.modules import rp_job, rp_scale
from tests.test_serverless.test_job_control import queue_job, wait, worker


@pytest.fixture
def delivery(monkeypatch):
    result, stream = AsyncMock(), AsyncMock()
    monkeypatch.setattr(rp_job, "send_result", result)
    monkeypatch.setattr(rp_scale, "send_result", result)
    monkeypatch.setattr(rp_job, "stream_result", stream)
    return result, stream


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["sync", "async", "generator", "async_generator"])
async def test_requeue_has_no_terminal_result_and_resumes_with_same_job(
    worker, delivery, kind
):
    scaler, start = worker
    sent, streamed = delivery
    attempts = []

    def enter(job):
        attempts.append(job)
        if len(attempts) == 1:
            scaler.job_control.pause_dispatch()
            raise RequeueJob()
        return "done"

    async def async_handler(job):
        return enter(job)

    def generator(job):
        yield enter(job)

    async def async_generator(job):
        yield enter(job)

    scaler.config["handler"] = {
        "sync": enter,
        "async": async_handler,
        "generator": generator,
        "async_generator": async_generator,
    }[kind]
    scaler.jobs_handler = rp_job.handle_job
    queue = scaler.jobs_queue
    await queue_job(scaler, "same-id")
    start(scaler.run_jobs(None))
    await wait(scaler._slot_freed)
    join = start(queue.join())
    await asyncio.sleep(0.02)
    assert not join.done()
    assert queue.qsize() == 1
    assert scaler.job_progress.jobs == {"same-id"}
    assert not scaler.jobs_tasks
    sent.assert_not_awaited()
    streamed.assert_not_awaited()

    scaler.job_control.resume_dispatch()
    await asyncio.wait_for(join, timeout=1)
    assert scaler.jobs_queue is queue
    assert attempts[0] is attempts[1]
    assert len(attempts) == 2
    assert queue.empty()
    assert scaler.job_progress.get_job_count() == 0
    sent.assert_awaited_once()
    assert "error" not in sent.call_args.args[1]


@pytest.mark.asyncio
async def test_requeue_fits_with_fetch_in_flight_and_preserves_fifo(worker, delivery):
    scaler, start = worker
    sent, _ = delivery
    scaler.current_concurrency = 3
    scaler.jobs_queue = asyncio.Queue(maxsize=3)
    queue = scaler.jobs_queue
    fetching = asyncio.Event()
    release_fetch = asyncio.Event()
    reject_a = asyncio.Event()
    calls, completed, attempts = [], [], []

    async def fetcher(_session, needed):
        calls.append(needed)
        fetching.set()
        await release_fetch.wait()
        return [{"id": value, "input": {}} for value in ("B", "C")]

    async def handler(job):
        attempts.append(job["id"])
        if attempts == ["A"]:
            await reject_a.wait()
            raise RequeueJob()
        completed.append(job["id"])
        return job["id"]

    scaler.config["handler"] = handler
    scaler.jobs_handler = rp_job.handle_job
    scaler.jobs_fetcher = fetcher
    await queue_job(scaler, "A")
    a = start(scaler.handle_job(None, queue.get_nowait()))
    start(scaler.get_jobs(None))
    await wait(fetching)
    assert calls == [2]  # A still occupies capacity after leaving the queue.
    scaler.job_control.pause_polling()
    scaler.job_control.pause_dispatch()
    scaler._jobs_available.clear()
    release_fetch.set()
    await wait(scaler._jobs_available)
    assert queue.qsize() == 2
    reject_a.set()
    await asyncio.wait_for(a, timeout=1)
    assert queue.qsize() == 3
    assert scaler.job_progress.jobs == {"A", "B", "C"}
    sent.assert_not_awaited()

    start(scaler.run_jobs(None))
    scaler.job_control.resume_dispatch()
    await asyncio.wait_for(queue.join(), timeout=1)
    assert completed == ["B", "C", "A"]
    assert calls == [2]
    assert scaler.job_progress.get_job_count() == 0
    assert sent.await_count == 3


@pytest.mark.asyncio
async def test_repeated_requeue_balances_queue_join(worker, delivery):
    scaler, start = worker
    sent, _ = delivery
    attempts = 0

    async def handler(job):
        nonlocal attempts
        attempts += 1
        if attempts <= 3:
            scaler.job_control.pause_dispatch()
            raise RequeueJob()
        return "done"

    scaler.config["handler"] = handler
    scaler.jobs_handler = rp_job.handle_job
    await queue_job(scaler, "retry")
    start(scaler.run_jobs(None))
    for index in range(3):
        await wait(scaler._slot_freed)
        scaler._slot_freed.clear()
        assert attempts == index + 1
        assert scaler.jobs_queue.qsize() == 1
        assert scaler.job_progress.jobs == {"retry"}
        sent.assert_not_awaited()
        scaler.job_control.resume_dispatch()
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert attempts == 4
    sent.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_requeued_job_without_running_it_again(worker, delivery):
    scaler, start = worker
    sent, _ = delivery
    attempts = []

    async def handler(job):
        attempts.append(job["id"])
        scaler.job_control.pause_dispatch()
        raise RequeueJob()

    scaler.config["handler"] = handler
    scaler.jobs_handler = rp_job.handle_job
    await queue_job(scaler, "cancel-me")
    start(scaler.run_jobs(None))
    await wait(scaler._slot_freed)
    assert await scaler.stop_job("cancel-me")
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    scaler.job_control.resume_dispatch()
    await asyncio.sleep(0.02)
    assert attempts == ["cancel-me"]
    assert scaler.job_progress.get_job_count() == 0
    sent.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_id", ["A", "B", "C", "missing"])
async def test_queue_cancellation_preserves_other_jobs_and_order(
    worker, delivery, cancel_id
):
    scaler, start = worker
    scaler.current_concurrency = 3
    scaler.jobs_queue = asyncio.Queue(maxsize=3)
    completed = []

    async def handler(job):
        completed.append(job["id"])
        return "done"

    scaler.config["handler"] = handler
    scaler.jobs_handler = rp_job.handle_job
    for job_id in ("A", "B", "C"):
        await queue_job(scaler, job_id)
    assert await scaler.stop_job(cancel_id) == (cancel_id != "missing")
    start(scaler.run_jobs(None))
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    assert completed == [job_id for job_id in ("A", "B", "C") if job_id != cancel_id]
    assert scaler.job_progress.get_job_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("async_generator", [False, True])
async def test_cannot_requeue_after_streaming_output(worker, delivery, async_generator):
    scaler, start = worker
    sent, streamed = delivery

    def handler(job):
        yield "partial"
        raise RequeueJob()

    async def async_handler(job):
        yield "partial"
        raise RequeueJob()

    scaler.config["handler"] = async_handler if async_generator else handler
    scaler.jobs_handler = rp_job.handle_job
    await queue_job(scaler, "already-streamed")
    start(scaler.run_jobs(None))
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
    streamed.assert_awaited_once()
    sent.assert_awaited_once()
    assert "after producing output" in sent.call_args.args[1]["error"]
    assert scaler.jobs_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown", [False, True])
async def test_requeue_failure_reports_error_instead_of_blocking(
    worker, delivery, shutdown
):
    scaler, _ = worker
    sent, _ = delivery

    async def handler(job):
        raise RequeueJob()

    scaler.config["handler"] = handler
    scaler.jobs_handler = rp_job.handle_job
    await queue_job(scaler, "rejected")
    job = scaler.jobs_queue.get_nowait()
    if shutdown:
        scaler.kill_worker()
    else:
        # Deliberately violate the capacity contract, as an oversized fetch would.
        await queue_job(scaler, "excess")
    await asyncio.wait_for(scaler.handle_job(None, job), timeout=1)
    sent.assert_awaited_once()
    assert "error" in sent.call_args.args[1]
    assert "rejected" not in scaler.job_progress.jobs
    if not shutdown:
        assert await scaler.stop_job("excess")
    await asyncio.wait_for(scaler.jobs_queue.join(), timeout=1)
