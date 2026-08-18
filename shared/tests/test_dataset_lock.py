import pytest

from shared.io.dataset_lock import (
    DatasetLockBusyError,
    DatasetOperationLock,
)


def test_dataset_lock_allows_concurrent_readers_and_blocks_writer(tmp_path):
    path = tmp_path / "dataset.lock"
    first = DatasetOperationLock(exclusive=False, path=path).acquire()
    second = DatasetOperationLock(exclusive=False, path=path).acquire()
    try:
        with pytest.raises(DatasetLockBusyError):
            DatasetOperationLock(exclusive=True, path=path).acquire()
    finally:
        second.release()
        first.release()


def test_dataset_lock_writer_blocks_reader_until_release(tmp_path):
    path = tmp_path / "dataset.lock"
    writer = DatasetOperationLock(exclusive=True, path=path).acquire()
    try:
        with pytest.raises(DatasetLockBusyError):
            DatasetOperationLock(exclusive=False, path=path).acquire()
    finally:
        writer.release()

    with DatasetOperationLock(exclusive=False, path=path):
        pass
