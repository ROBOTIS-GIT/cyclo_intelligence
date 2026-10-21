import threading
import time

import numpy as np

from main_runtime.latest_preview import LatestPreview


def test_blocked_callback_does_not_block_submit_and_only_latest_pending_is_delivered():
    preview = LatestPreview()
    started = threading.Event()
    release = threading.Event()
    submitted = threading.Event()
    delivered = threading.Event()
    calls = []
    workers = []
    action = np.array([1.])
    keys = ['arm_left']

    def callback(values, selected):
        workers.append(threading.current_thread())
        if not calls:
            started.set()
            assert release.wait(5.)
        calls.append((values.copy(), list(selected)))
        if values[0] == 3.:
            delivered.set()

    def enqueue():
        assert preview.submit(callback, [2.], ['discarded'])
        latest_action = np.array([3.])
        latest_keys = ['arm_right']
        assert preview.submit(callback, latest_action, latest_keys)
        latest_action[:] = 999.
        latest_keys[:] = ['changed']
        submitted.set()

    producer = threading.Thread(target=enqueue, daemon=True)
    try:
        assert preview.submit(callback, action, keys)
        assert started.wait(1.)
        action[:] = 999.
        keys[:] = ['changed']
        producer.start()
        assert submitted.wait(1.)
        assert not release.is_set()
        release.set()
        assert delivered.wait(1.)
        assert [values.tolist() for values, _ in calls] == [[1.], [3.]]
        assert [selected for _, selected in calls] == [['arm_left'], ['arm_right']]
        assert workers[0] is workers[1]
        assert workers[0].daemon
    finally:
        release.set()
        producer.join(timeout=1.)
        preview.close()


def test_reset_discards_pending_work_and_allows_next_session():
    preview = LatestPreview()
    started = threading.Event()
    release = threading.Event()
    delivered = threading.Event()
    calls = []

    def callback(action, keys):
        if action[0] == 1:
            started.set()
            assert release.wait(5.)
        calls.append(int(action[0]))
        if action[0] == 3:
            delivered.set()

    try:
        preview.submit(callback, [1], [])
        assert started.wait(1.)
        preview.submit(callback, [2], [])
        preview.reset()
        preview.submit(callback, [3], [])
        release.set()
        assert delivered.wait(1.)
        assert calls == [1, 3]
    finally:
        release.set()
        preview.close()


def test_close_is_bounded_discards_pending_and_permanently_rejects_submissions():
    preview = LatestPreview()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    calls = []

    def callback(action, keys):
        started.set()
        assert release.wait(5.)
        calls.append(int(action[0]))
        finished.set()

    try:
        preview.submit(callback, [1], [])
        assert started.wait(1.)
        preview.submit(callback, [2], [])
        before = time.monotonic()
        preview.close()
        assert time.monotonic()-before < .5
        assert not preview.submit(callback, [3], [])
        preview.reset()
        assert not preview.submit(callback, [4], [])
        release.set()
        assert finished.wait(1.)
    finally:
        release.set()
        preview.close()
    assert calls == [1]


def test_callback_failure_is_logged_and_worker_recovers(caplog):
    preview = LatestPreview()
    failed = threading.Event()
    recovered = threading.Event()

    def callback(action, keys):
        if action[0] == 1:
            failed.set()
            raise RuntimeError('preview transport unavailable')
        recovered.set()

    try:
        preview.submit(callback, [1], [])
        assert failed.wait(1.)
        preview.submit(callback, [2], [])
        assert recovered.wait(1.)
        assert 'preview transport unavailable' in caplog.text
    finally:
        preview.close()


def test_close_from_callback_does_not_join_itself():
    preview = LatestPreview()
    closed = threading.Event()

    def callback(action, keys):
        preview.close()
        closed.set()

    try:
        preview.submit(callback, [1], [])
        assert closed.wait(1.)
        assert not preview.submit(callback, [2], [])
    finally:
        preview.close()


def test_close_before_first_submit_starts_no_worker():
    preview = LatestPreview()
    preview.close()
    called = threading.Event()
    assert not preview.submit(lambda *_: called.set(), [1], [])
    assert not called.is_set()
