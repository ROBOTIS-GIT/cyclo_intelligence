"""Cross-process lock for dataset readers and mutators under ``/workspace``."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


DEFAULT_DATASET_LOCK_PATH = Path("/workspace/.cyclo_dataset.lock")


class DatasetLockBusyError(RuntimeError):
    """Raised when another dataset operation holds an incompatible lock."""


class DatasetOperationLock:
    """A small advisory ``flock`` shared by all dataset operation processes."""

    def __init__(
        self,
        *,
        exclusive: bool,
        blocking: bool = False,
        path: str | Path = DEFAULT_DATASET_LOCK_PATH,
    ) -> None:
        self.exclusive = bool(exclusive)
        self.blocking = bool(blocking)
        self.path = Path(path)
        self._descriptor: int | None = None

    def acquire(self) -> "DatasetOperationLock":
        if self._descriptor is not None:
            raise RuntimeError("Dataset operation lock is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            # The main container may create the file as root while the
            # one-off learner runs as uid 1000. Keep the lock reusable by
            # both processes regardless of their umask.
            try:
                os.chmod(self.path, 0o666)
            except PermissionError:
                pass
            operation = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
            if not self.blocking:
                operation |= fcntl.LOCK_NB
            try:
                fcntl.flock(descriptor, operation)
            except BlockingIOError as error:
                mode = "write" if self.exclusive else "read"
                raise DatasetLockBusyError(
                    f"Dataset is locked by another operation ({mode} lock unavailable)"
                ) from error
        except Exception:
            os.close(descriptor)
            raise
        self._descriptor = descriptor
        return self

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def __enter__(self) -> "DatasetOperationLock":
        return self.acquire()

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.release()


__all__ = [
    "DEFAULT_DATASET_LOCK_PATH",
    "DatasetLockBusyError",
    "DatasetOperationLock",
]
