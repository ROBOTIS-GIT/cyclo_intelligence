"""HTTP request validation for camera observation and task-scoped reasoning decisions."""
import json
import time
from copy import deepcopy
from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, Field
if __package__:
    from .mission_decisions import parse_decision, validate_target
else:
    from mission_decisions import parse_decision, validate_target


class StartRequest(BaseModel):
    model_id: str | None = Field(default=None, min_length=1, max_length=200)
    map_name: str | None = Field(default=None, min_length=1, max_length=128)


class ObserveRequest(BaseModel):
    previous_frame_stamp: str | None = Field(default=None, max_length=100)
    episode_id: str | None = Field(default=None, max_length=36)
    # This is the latest state known BEFORE capture, not state at inference completion.
    context: dict = Field(default_factory=dict)


class FrameMetadata(BaseModel):
    scenario: str | None = Field(default=None, min_length=1, max_length=300)
    group_id: str = Field(min_length=1, max_length=120)
    robot_type: str = Field(min_length=1, max_length=80)
    environment: str = Field(pattern='^(simulation|real|unknown)$')


class CaptureRequest(BaseModel):
    metadata: FrameMetadata | None = None
    context: dict = Field(default_factory=dict)


class ActivityEventRequest(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=120)
    detail: str = Field(default='', max_length=1000)


class SaveAnalysisRequest(BaseModel):
    analysis_id: str = Field(min_length=1, max_length=36)


def input_context(raw):
    if len(json.dumps(raw)) > 16000:
        raise HTTPException(422, 'Observation context is too large')
    context = {key: deepcopy(raw.get(key)) for key in ('pose', 'localized', 'client_sampled_at_ms', 'map_name')}
    context.update(source='ui_before_capture', received_at_ms=int(time.time() * 1000),
                   active_mission=None, recent_mission_results=None, runtime_source='unavailable')
    # Explicit hypothetical context for non-executing Collect previews only.
    if raw.get('preview_active_mission') is not None:
        preview = raw['preview_active_mission']
        if not isinstance(preview, dict) or not isinstance(preview.get('mission_id'), str) or not preview['mission_id']:
            raise HTTPException(422, 'Select a mission for the preview context')
        step = preview.get('step', 1)
        action = preview.get('action', '')
        if type(step) is not int or step < 1 or not isinstance(action, str) or len(action) > 120:
            raise HTTPException(422, 'Invalid preview step or action')
        status = preview.get('status', 'running')
        if status not in ('running', 'paused', 'returning'):
            raise HTTPException(422, 'Invalid preview mission status')
        context.update(active_mission={'mission_id': preview['mission_id'], 'status': status, 'step': step, 'action': action},
                       runtime_source='operator_supplied_preview')
    for key in ('reasoning_task', 'observation', 'candidate_mission_ids', 'allowed_mission_ids', 'recording_execution'):
        if key in raw:
            context[key] = deepcopy(raw[key])
    return context


class MissionDecisionRequest(BaseModel):
    expected_session_id: str | None = Field(default=None, max_length=36)
    map_name: str = Field(min_length=1, max_length=128)
    localized: bool = False
    previous_frame_stamp: str | None = Field(default=None, max_length=100)
    reasoning_task: Literal['mission_selection', 'mission_supervision', 'patrol_supervision'] = 'mission_selection'
    active_mission: dict | None = None
    operator_instruction: str | None = Field(default=None, max_length=1000)
    recent_missions: list[dict] = Field(default_factory=list, max_length=20)
    candidate_mission_ids: list[str] | None = Field(default=None, max_length=100)
    allowed_mission_ids: list[str] | None = Field(default=None, max_length=100)
    excluded_mission_ids: list[str] = Field(default_factory=list, max_length=100)
    observation: dict | None = None
    frame_captured_after: float | None = None


def validate_mission_decision(result, missions, request):
    if result.get('finish_reason') != 'stop':
        raise ValueError('Incomplete Qwen decision')
    raw = request.model_dump()
    decision = parse_decision(result.get('text', ''), raw)
    # Report occupied selection contexts as blocked, preserving preview diagnostics.
    blocked = 'A mission is already running' if decision['decision'] == 'start' and request.active_mission else None
    check_context = {**raw, 'active_mission': None} if blocked else raw
    validate_target(decision, check_context, missions)
    selected = next((m for m in missions if m['mission_id'] == decision['mission_id']), None)
    return {'decision': decision, 'selected_mission': selected,
            'action_allowed': blocked is None, 'block_reason': blocked}
