"""Frame-distinct confirmation gates, independent of ROS and HTTP."""

class MissionStartConfirmation:
    """Require three consecutive eligible decisions from distinct camera frames."""
    required = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.stamps = set()

    def accept(self, result, instruction_version):
        decision = result.get('decision') or {}
        selected = result.get('selected_mission') or {}
        stamp = result.get('frame_stamp')
        if (not result.get('action_allowed') or result.get('error')
                or decision.get('decision') != 'start' or not stamp
                or not selected.get('mission_id')
                or decision.get('mission_id') != selected.get('mission_id')):
            self.reset()
            return False
        key = (selected['mission_id'], selected.get('revision'), instruction_version)
        if key != self.key:
            self.reset()
            self.key = key
        self.stamps.add(stamp)
        return len(self.stamps) >= self.required


class ResumeConfirmation:
    def __init__(self, required=2):
        self.required = required
        self.stamps = set()

    def reset(self):
        self.stamps.clear()

    def accept(self, result):
        decision = result.get('decision') or {}
        if (result.get('error') or not result.get('action_allowed')
                or decision.get('decision') != 'continue' or not result.get('frame_stamp')):
            self.reset()
            return False
        self.stamps.add(result['frame_stamp'])
        return len(self.stamps) >= self.required


class ObservationDepartureConfirmation:
    """Require three consecutive eligible next decisions on distinct frames."""
    required = 3

    def __init__(self):
        self.reset()

    def reset(self):
        self.key = None
        self.stamps = set()

    def accept(self, result, instruction_version, context_version):
        decision = (result or {}).get('decision') or {}
        stamp = (result or {}).get('frame_stamp')
        if (not result or result.get('error') or not result.get('action_allowed')
                or decision.get('decision') != 'next_observation'
                or decision.get('mission_id') is not None or not stamp
                or result.get('context_version') != context_version):
            self.reset()
            return False
        key = (instruction_version, context_version)
        if key != self.key:
            self.reset()
            self.key = key
        self.stamps.add(stamp)
        return len(self.stamps) >= self.required
