"""Lost recording responses are queried, never replayed, without ROS traffic."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from interfaces.srv import RecordingCommand
from orchestrator.internal.communication.cyclo_data_client import CallResult, CycloDataClient


@pytest.mark.parametrize('state', ['completed', 'running', 'unknown', 'rejected'])
@pytest.mark.parametrize('success', [True, False])
def test_lost_ack_queries_only_the_original_id(state, success):
    client = CycloDataClient.__new__(CycloDataClient)
    client._recording = Mock()
    requests = []

    def call(_service, request, timeout, label):
        requests.append(request)
        if len(requests) == 1:
            return CallResult(False, 'timeout')
        response = RecordingCommand.Response(
            success=success, message='receipt', request_id=request.request_id, command_state=state)
        return CallResult(success, 'receipt', response)

    client._call = call
    result = client.send_recording_command(RecordingCommand.Request.STOP_SEGMENT)
    assert [req.command for req in requests] == [RecordingCommand.Request.STOP_SEGMENT,
                                               RecordingCommand.Request.GET_RESULT]
    assert requests[0].request_id == requests[1].request_id == result.request_id
    assert result.outcome_unknown == (state not in ('completed', 'rejected'))
    assert result.success == (success and state == 'completed')


def test_wrong_receipt_id_or_missing_reply_remains_unconfirmed():
    for response in (None, SimpleNamespace(request_id='wrong', command_state='completed')):
        result = CycloDataClient._recording_result('expected', CallResult(True, 'ok', response))
        assert result.outcome_unknown and not result.success and result.response is None


def test_undiscovered_service_never_submits_or_creates_pending_ownership():
    client = CycloDataClient.__new__(CycloDataClient)
    client._recording = Mock()
    client._recording.wait_for_service.return_value = False
    client._call = Mock()
    result = client.send_recording_command(RecordingCommand.Request.START_SEGMENT)
    assert not result.success and not result.outcome_unknown
    client._call.assert_not_called()
