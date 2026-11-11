"""Project the execution ledger onto the Worker contract, off the control tick.

The ControlLoop lock owns this object. Network serialization has a separate
lock, so a slow Worker never holds the lock needed to stop robot commands.
"""

from dataclasses import asdict, dataclass

from action_chunk_processing.tracked_buffer import (
    ExecutionSnapshot, PlanningEvent, PublicationEvent, TrackedActionBuffer,
)
from inference_context.execution import (
    ActionRecord, ExecutionContext, PlanningRecord, ResetRecord,
)
from inference_context.contract import ExecutionContract


def command_record(command, *, status="planned", emitted=None, **event):
    return ActionRecord(
        prediction_id=str(command.prediction_id), status=status, space="command",
        values=command.values if emitted is None else emitted,
        command_id=command.command_id, source_position=command.source_position,
        blend_weight=command.blend_weight, anchor_command_id=command.anchor_command_id,
        anchor_values=command.anchor_values,
        planned_values=command.values if emitted is not None and emitted != command.values else None,
        **event,
    )


@dataclass(frozen=True)
class FeedbackCapture:
    session_id: str
    generation: int
    revision: int
    phase: str
    after_event_id: int
    snapshot: ExecutionSnapshot
    feedback_schema: int = 1


class ExecutionFeedback:
    def __init__(self, initial: ExecutionContext, *, pending_command_count=None, **processing):
        if (initial.phase != "ready" or initial.actions or initial.planning or initial.resets
                or initial.after_event_id or initial.latest_event_id):
            raise ValueError("LOAD context must be an empty ready session")
        self.buffer = TrackedActionBuffer(**processing)
        self._pending_command_count = ExecutionContract(pending_command_count=pending_command_count).pending_command_count
        self.session_id = initial.session_id
        self.feedback_schema = initial.feedback_schema
        self.generation = initial.generation
        self.revision = initial.revision
        self.phase = initial.phase
        self._acknowledged_event = 0

    def reset(self, reason, phase):
        self.buffer.clear(reason)
        self.generation += 1
        self.phase = phase

    def capture(self):
        """Only take immutable references under the ControlLoop lock."""
        snapshot = self.buffer.snapshot(after_event_id=self._acknowledged_event,
                                        pending_limit=self._pending_command_count)
        if snapshot.in_flight is not None:
            raise RuntimeError("feedback snapshot must follow the publisher result")
        self.revision += 1
        return FeedbackCapture(self.session_id, self.generation, self.revision,
                               self.phase, self._acknowledged_event, snapshot, self.feedback_schema)

    @staticmethod
    def project(capture):
        """Validate/project outside the ControlLoop lock; no network dependency."""
        snapshot = capture.snapshot
        actions, planning, resets = [], [], []
        for event in snapshot.events:
            if isinstance(event, PublicationEvent):
                actions.append(command_record(
                    event.command, status=event.status, emitted=event.emitted_values,
                    event_id=event.event_id, recorded_s=event.recorded_s, reason=event.reason,
                ))
            elif isinstance(event, PlanningEvent):
                decision = asdict(event.decision)
                if capture.feedback_schema == 1:
                    decision.pop("command_start_id")
                planning.append(PlanningRecord(
                    **decision, event_id=event.event_id, recorded_s=event.recorded_s,
                ))
            else:
                resets.append(ResetRecord(event.event_id, event.reason, event.recorded_s))
        actions.extend(command_record(command) for command in snapshot.pending)
        return ExecutionContext(
            capture.session_id, capture.generation, capture.revision, capture.phase, tuple(actions),
            tuple(planning), tuple(resets), capture.after_event_id, snapshot.latest_event_id,
            capture.feedback_schema,
        )

    def acknowledge(self, context):
        if context.session_id == self.session_id and context.generation == self.generation:
            self._acknowledged_event = max(self._acknowledged_event, context.latest_event_id)
