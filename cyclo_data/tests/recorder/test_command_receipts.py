from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from cyclo_data.services.command_receipts import CommandReceipts


def request():
    return SimpleNamespace(request_id=str(uuid4()), command=1, task_info={'task': 'original'})


def response(success=True):
    return SimpleNamespace(success=success, message='ok' if success else 'disk full')


def test_pending_completed_failed_and_argument_mismatch():
    receipts = CommandReceipts()
    req = request()
    assert receipts.lookup(req.request_id, response()).command_state == 'unknown'
    assert receipts.reserve(req)
    assert receipts.lookup(req.request_id, response()).command_state == 'running'
    assert not receipts.reserve(deepcopy(req))
    changed = deepcopy(req)
    changed.task_info['task'] = 'different'
    assert receipts.lookup(req.request_id, response(), changed).command_state == 'rejected'
    receipts.complete(req.request_id, response(False))
    result = receipts.lookup(req.request_id, response())
    assert not result.success and result.message == 'disk full'
    assert result.command_state == 'completed'


def test_parallel_duplicate_reservation_occurs_once():
    receipts, req = CommandReceipts(), request()
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(lambda _: receipts.reserve(req), range(100))) == 1


def test_cache_is_bounded_but_keeps_pending_archive_and_unknown_after_restart():
    receipts = CommandReceipts(capacity=2)
    pending, complete, next_req = request(), request(), request()
    receipts.reserve(pending)
    receipts.archive(pending.request_id, 'running')
    receipts.complete(pending.request_id, response())
    receipts.reserve(complete)
    receipts.complete(complete.request_id, response())
    receipts.reserve(next_req)
    assert receipts.lookup(complete.request_id, response()).command_state == 'unknown'
    assert receipts.lookup(pending.request_id, response()).archive_state == 'running'
    assert len(receipts._entries) == 2
    assert CommandReceipts().lookup(pending.request_id, response()).command_state == 'unknown'


@pytest.mark.parametrize('value', ['', 'not-a-uuid', 'a' * 1000])
def test_invalid_identifiers_never_allocate(value):
    receipts = CommandReceipts()
    with pytest.raises(ValueError):
        receipts.lookup(value, response())
    assert not receipts._entries
