"""Independent pause controls for one queue-based worker."""

import asyncio
from threading import Lock


class RequeueJob(Exception):
    """Return this job to the worker's queue before doing work or producing output.

    Pause dispatch before raising, and resume it when admission is possible.
    The job keeps its ID and progress tracking; this does not reset cloud TTLs.
    """


class JobControl:
    """Pause job acquisition or task dispatch without changing concurrency.

    Pass one instance as ``config["job_control"]`` to ``serverless.start``.
    Both operations are enabled by default. Methods may be called before
    startup, from a handler, or from another thread (for example a timer).

    Pausing does not cancel an acquisition already in flight or a task already
    dispatched. Acquired jobs remain queued. Shutdown overrides dispatch pause
    so the worker can drain held jobs using its normal shutdown behavior.
    These controls apply to JobScaler, not the direct HTTP API handler path.
    """

    def __init__(self):
        self._lock = Lock()
        self._polling_paused = False
        self._dispatch_paused = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._polling_changed = asyncio.Event()
        self._dispatch_changed = asyncio.Event()

    @property
    def polling_paused(self) -> bool:
        """Whether new job acquisitions are paused."""
        with self._lock:
            return self._polling_paused

    @property
    def dispatch_paused(self) -> bool:
        """Whether dispatch of queued jobs is paused."""
        with self._lock:
            return self._dispatch_paused

    def pause_polling(self) -> None:
        """Stop initiating acquisitions; retain results already in flight."""
        self._set_paused("_polling_paused", True, self._polling_changed)

    def resume_polling(self) -> None:
        """Allow acquisitions and wake the intake loop."""
        self._set_paused("_polling_paused", False, self._polling_changed)

    def pause_dispatch(self) -> None:
        """Stop dispatching queued jobs; leave dispatched tasks running."""
        self._set_paused("_dispatch_paused", True, self._dispatch_changed)

    def resume_dispatch(self) -> None:
        """Allow dispatch and wake the execution loop, even with queued jobs."""
        self._set_paused("_dispatch_paused", False, self._dispatch_changed)

    def _bind(self) -> None:
        """Bind lazily because workers are usually configured before asyncio.run."""
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._loop is not None and self._loop is not loop:
                raise RuntimeError("JobControl must belong to one worker event loop")
            self._loop = loop

    def _set_paused(self, attribute: str, paused: bool, changed: asyncio.Event):
        with self._lock:
            if getattr(self, attribute) == paused:
                return
            setattr(self, attribute, paused)
            if self._loop is None or self._loop.is_closed():
                return
            try:
                self._loop.call_soon_threadsafe(changed.set)
            except RuntimeError:
                # A timer may race the loop closing during worker teardown.
                if not self._loop.is_closed():
                    raise
