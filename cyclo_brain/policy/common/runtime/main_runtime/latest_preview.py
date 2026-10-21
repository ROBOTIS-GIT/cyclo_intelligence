"""Publish only the latest pending preview without blocking control ticks."""
import logging
import threading

import numpy as np


logger = logging.getLogger(__name__)


class LatestPreview:
    """One lazy daemon worker with a single replaceable pending callback.

    Reset and close discard queued work. An already running callback is allowed
    to finish; close waits at most 0.1 seconds for it.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = None
        self._thread = None
        self._closed = False

    def submit(self, callback, action, keys):
        """Copy and queue a preview; return False after close."""
        with self._condition:
            if self._closed:
                return False
            self._pending = (callback, np.array(action, copy=True), list(keys))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name='latest-preview', daemon=True,
                )
                self._thread.start()
            self._condition.notify()
        return True

    def reset(self):
        """Drop pending work while retaining the worker for the next session."""
        with self._condition:
            self._pending = None

    def close(self):
        """Stop accepting work and discard the queue without an unbounded wait."""
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=.1)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending is not None)
                if self._closed:
                    return
                callback, action, keys = self._pending
                self._pending = None
            try:
                callback(action, keys)
            except Exception:
                logger.exception('Failed to publish action preview')
