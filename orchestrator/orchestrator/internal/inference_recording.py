"""Backend-owned recording destination; callers serialize mutations."""

from shared.inference_recording import RECORDING_ROOT, allocate_folder, validate_folder


class InferenceRecordingSession:
    def __init__(self, root=RECORDING_ROOT):
        self.root = root
        self.session_id = ''
        self.identity = None

    def configure(self, robot_type, policy_id, policy_path):
        identity = (robot_type, policy_id, str(policy_path).strip().rstrip('/'))
        if identity != self.identity:
            self.session_id = ''
            self.identity = identity

    def select(self, session_id, robot_type):
        if session_id:
            validate_folder(self.root, session_id, robot_type)
        self.session_id = session_id

    def prepare(self, robot_type):
        if self.session_id:
            validate_folder(self.root, self.session_id, robot_type)
        else:
            self.session_id = allocate_folder(self.root)
        return self.session_id
