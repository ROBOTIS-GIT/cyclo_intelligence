import threading
from types import SimpleNamespace
from unittest.mock import Mock

from std_msgs.msg import Bool, String

from orchestrator.internal.communication.communicator import Communicator


def _communicator():
    communicator = object.__new__(Communicator)
    communicator.node = SimpleNamespace(
        create_subscription=Mock(side_effect=lambda *args: args),
        get_logger=Mock(return_value=Mock()),
    )
    communicator.robot_section = {
        'joystick': {
            'trigger_topic': '/leader/joystick_controller/tact_trigger',
            'middle_pedal_topic': '/leader/foot_switch/middle_pedal',
            'require_middle_pedal': True,
        },
    }
    communicator._require_middle_pedal = True
    communicator._middle_pedal_held = False
    communicator._joystick_lock = threading.Lock()
    communicator._joystick_handler = Mock()
    communicator.joystick_state = {'updated': False, 'mode': None}
    communicator._closed = False
    return communicator


def test_f2_subscribes_to_leader_trigger_and_middle_pedal():
    communicator = _communicator()

    communicator.init_subscribers()

    topics = [
        call.args[1]
        for call in communicator.node.create_subscription.call_args_list
    ]
    assert topics == [
        '/leader/joystick_controller/tact_trigger',
        '/leader/foot_switch/middle_pedal',
    ]


def test_middle_pedal_interlock_tracks_press_and_release():
    communicator = _communicator()
    assert communicator.recording_trigger_enabled() is False

    pressed = Bool()
    pressed.data = True
    communicator.middle_pedal_callback(pressed)
    assert communicator.recording_trigger_enabled() is True

    trigger = String()
    trigger.data = 'right'
    communicator.joystick_trigger_callback(trigger)
    communicator._joystick_handler.assert_called_once_with('right')

    released = Bool()
    released.data = False
    communicator.middle_pedal_callback(released)
    assert communicator.recording_trigger_enabled() is False
