import pytest
from supervisor_api.autonomy_confirmation import ObservationDepartureConfirmation


def result(stamp, decision='next_observation', **extra):
    return {'action_allowed':True,'frame_stamp':stamp,'context_version':7,
            'decision':{'decision':decision,'mission_id':None},**extra}


def test_three_distinct_frames_required():
    gate=ObservationDepartureConfirmation()
    assert not gate.accept(result('a'),0,7)
    assert not gate.accept(result('a'),0,7)
    assert not gate.accept(result('b'),0,7)
    assert gate.accept(result('c'),0,7)


@pytest.mark.parametrize('interrupt', [None,result('x','wait'),result('x','unknown'),
    result('x','start'),result('x',error='bad'),result('x',action_allowed=False),
    result('x',context_version=6)])
def test_interruption_resets_confirmation(interrupt):
    gate=ObservationDepartureConfirmation()
    gate.accept(result('a'),0,7);gate.accept(result('b'),0,7)
    assert not gate.accept(interrupt,0,7)
    assert not gate.accept(result('c'),0,7)
    assert not gate.accept(result('d'),0,7)
    assert gate.accept(result('e'),0,7)


def test_operator_instruction_change_resets_confirmation():
    gate=ObservationDepartureConfirmation()
    gate.accept(result('a'),0,7);gate.accept(result('b'),0,7)
    assert not gate.accept(result('c'),1,7)
    assert not gate.accept(result('d'),1,7)
    assert gate.accept(result('e'),1,7)
