"""ROS-independent cooperative pause state and XML admission checks."""
import threading
import time
from xml.etree import ElementTree as ET


class OperationCancelled(RuntimeError):
    pass


def wait_future(future, timeout, cancelled=None):
    deadline = time.monotonic() + timeout
    while not future.done():
        if cancelled is not None and cancelled.is_set():
            raise OperationCancelled('Operation stopped')
        if time.monotonic() >= deadline:
            raise TimeoutError('ROS response timed out')
        time.sleep(.03)
    return future.result()


def validate_tree(xml):
    """Admit only actions with a state-preserving actuator pause contract.

    The ordinary Run page can still run other nodes. Supervised autonomy must
    reject them before moving, since cancelling/reloading XML replays effects.
    """
    if not str(xml or '').strip():
        return
    root = ET.fromstring(xml)
    main = root.get('main_tree_to_execute')
    trees = [t for t in root.findall('BehaviorTree') if t.get('ID') == main]
    if not main or len(trees) != 1 or len(trees[0]) != 1:
        raise ValueError('Supervised task requires one valid main BehaviorTree')
    controls = {'Sequence', 'Loop'}
    leaves = {'Wait', 'Rotate', 'JointControl', 'HomeMeal', 'HomeLaundry', 'HomeTV', 'ArmStateGate', 'SendCommand', 'ArucoDock', 'DockRetreat', 'ReturnToWaypoint'}
    inference_loaded = inference_active = False
    backend = None
    for node in trees[0].iter():
        if node is trees[0]:
            continue
        kind = node.tag if node.tag in controls | leaves else node.get('ID', node.tag)
        if kind not in controls | leaves:
            raise ValueError(f'{kind} does not support confirmed pause/resume in supervised autonomy')
        if kind == 'SendCommand':
            if node.get('target', 'INFERENCE').upper() != 'INFERENCE':
                raise ValueError('SendCommand Docker operations do not support supervised pause')
            command = node.get('command', '').upper()
            if command == 'LOAD':
                model_backend = node.get('model', '').split(':')[0]
                if model_backend not in ('groot', 'lerobot') or not node.get('policy_path'):
                    raise ValueError('Supervised SendCommand LOAD requires a backend and policy_path')
                if backend and backend != model_backend:
                    raise ValueError('Supervised SendCommand cannot switch policy backends in one tree')
                backend = model_backend
                inference_loaded, inference_active = True, False
            elif command not in ('RESUME', 'STOP', 'CLEAR') or not inference_loaded:
                raise ValueError('Supervised SendCommand requires a preceding LOAD and supported command')
            else:
                inference_active = command == 'RESUME'
                if command == 'CLEAR':
                    inference_loaded = False
        if kind in leaves and len(node):
            raise ValueError(f'Action {kind} cannot contain child actions')
    if inference_active:
        raise ValueError('Supervised SendCommand must STOP or CLEAR policy output before tree completion')


class PauseControl:
    """A pause is acknowledged only after the worker stopped its actuator.

    No active operation means the pause is a gate for the next operation.
    Stop is latched for the lifetime of this runtime and always wins resume.
    """
    def __init__(self):
        self.condition = threading.Condition()
        self.stopped = threading.Event()
        self.requested = False
        self.acknowledged = False
        self.active = False
        self.error = None

    def check(self, cancelled=None):
        if self.stopped.is_set() or (cancelled is not None and cancelled.is_set()):
            raise OperationCancelled('Operation stopped')

    def begin(self, cancelled):
        with self.condition:
            self.check(cancelled)
            if self.active:
                raise RuntimeError('An autonomy operation is already active')
            self.active = True
            self.error = None

    def end(self, error=None):
        with self.condition:
            self.active = False
            self.error = error
            self.acknowledged = self.requested
            self.condition.notify_all()

    def request(self, paused, timeout=45):
        deadline = time.monotonic() + timeout
        with self.condition:
            self.check()
            if self.error is not None:
                raise RuntimeError(str(self.error))
            self.requested = paused
            if not self.active:
                self.acknowledged = paused
            self.condition.notify_all()
            while self.acknowledged != paused:
                self.check()
                if self.error is not None:
                    raise RuntimeError(str(self.error))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Actuator pause/resume was not confirmed')
                self.condition.wait(min(.1, remaining))
            self.check()
            if self.error is not None:
                raise RuntimeError(str(self.error))

    def wait_resume(self, cancelled):
        with self.condition:
            self.acknowledged = True
            self.condition.notify_all()
            while self.requested:
                self.check(cancelled)
                self.condition.wait(.03)
            self.check(cancelled)

    def resumed(self):
        with self.condition:
            self.check()
            self.acknowledged = False
            self.condition.notify_all()

    def stop(self):
        self.stopped.set()
        with self.condition:
            self.condition.notify_all()
