"""Observe the configured head camera with Qwen; never execute robot commands."""
import json
import hashlib
import os
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from uuid import uuid4
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__:
    from .reasoning_models import ReasoningModels, load_options, matches_loaded
    from .head_camera import HeadCameraCache, CameraUnavailable
    from .mission_decisions import SYSTEM as MISSION_SYSTEM, build_input, system_for, task_for, catalog_missions, PROMPT_VERSION
    from .reasoning_contract import (StartRequest, ObserveRequest, FrameMetadata, CaptureRequest,
        ActivityEventRequest, SaveAnalysisRequest, MissionDecisionRequest, input_context, validate_mission_decision)
    from .reasoning_capture_context import context_for_frame
    from .collection_observation_plan import live_record_scope
    from .operate_activity import OperateActivity
else:  # Direct module loading in standalone tests.
    from reasoning_models import ReasoningModels, load_options, matches_loaded
    from head_camera import HeadCameraCache, CameraUnavailable
    from mission_decisions import SYSTEM as MISSION_SYSTEM, build_input, system_for, task_for, catalog_missions, PROMPT_VERSION
    from reasoning_contract import (StartRequest, ObserveRequest, FrameMetadata, CaptureRequest,
        ActivityEventRequest, SaveAnalysisRequest, MissionDecisionRequest, input_context, validate_mission_decision)
    from reasoning_capture_context import context_for_frame
    from collection_observation_plan import live_record_scope
    from operate_activity import OperateActivity

from fastapi import APIRouter, HTTPException

QWEN_URL = f"http://127.0.0.1:{int(os.getenv('QWEN_PORT', '7110'))}"
CAMERA_TOPIC = os.getenv('QWEN_HEAD_CAMERA_TOPIC', '/zed/zed_node/left/image_rect_color/compressed')


def qwen_request(path, payload=None, timeout=5):
    request = Request(QWEN_URL + path, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get('detail', 'Qwen request failed')
        except (ValueError, AttributeError):
            detail = 'Qwen request failed'
        raise HTTPException(exc.code, detail) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(503, 'Qwen API is unavailable or timed out') from exc


HEAD_CAMERA = HeadCameraCache(CAMERA_TOPIC)


def capture_head():
    try:
        return HEAD_CAMERA.snapshot()
    except CameraUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


def shutdown_head_camera():
    HEAD_CAMERA.close()




class ReasoningService:
    def __init__(self, start_container, stop_container, collection_store=None, models=None, context_provider=None):
        self.collection_store = collection_store
        self.context_provider = context_provider
        self.models = models or ReasoningModels(collection_store.catalog_provider if collection_store else lambda _: [])
        self.selection = None
        self.start_container = start_container
        self.stop_container = stop_container
        self.lifecycle_lock = threading.Lock()
        self.generation = 0
        self.prepare_lock = threading.Lock()
        self.observe_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.state = {'phase': 'idle', 'error': None}
        self.analysis_lock = threading.RLock()
        self.analyses = OrderedDict()
        self.activity = OperateActivity()

    def update(self, **values):
        with self.state_lock:
            self.state.update(values)

    def status(self):
        with self.state_lock:
            return {**self.state, 'camera_topic': CAMERA_TOPIC}

    def current(self, generation):
        with self.state_lock:
            return generation == self.generation

    def update_current(self, generation, **values):
        with self.state_lock:
            if generation == self.generation:
                self.state.update(values)

    def start(self, request=None):
        request = request or StartRequest()
        selection = self.models.resolve(request.model_id, request.map_name) if request.model_id else None
        with self.state_lock:
            if self.state['phase'] == 'stopping':
                raise HTTPException(409, 'Qwen is stopping; wait before starting again')
            if self.state['phase'] in {'starting', 'loading'}:
                if self.selection != selection:
                    raise HTTPException(409, 'Wait for model loading or stop reasoning before changing the model')
                return {**self.state, 'camera_topic': CAMERA_TOPIC}
            if self.observe_lock.locked() or not self.prepare_lock.acquire(blocking=False):
                raise HTTPException(409, 'Previous Qwen request is finishing; retry shortly')
            self.generation += 1
            generation = self.generation
            self.selection = selection
            self.state.update(phase='starting', error=None, session_id=str(uuid4()),
                              selected_model_id=selection['id'] if selection else None,
                              model_name=selection['model_name'] if selection else None,
                              model=None, model_revision=None, adapter=None)
        threading.Thread(target=self.prepare, args=(generation,), daemon=True).start()
        return self.status()

    def prepare(self, generation=None):
        if generation is None:
            generation = self.generation
        try:
            # Stop cannot overtake container creation/start and leave it running.
            with self.lifecycle_lock:
                if not self.current(generation):
                    return
                self.start_container()
            deadline = time.monotonic() + 60
            while self.current(generation):
                try:
                    health = qwen_request('/health')
                    break
                except HTTPException:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(1)
            if not self.current(generation):
                return
            selection = self.selection
            if not health.get('ready') or (selection and not matches_loaded(health, selection)):
                self.update_current(generation, phase='loading')
                qwen_request('/load', load_options(selection) if selection else {}, timeout=600)
            if not self.current(generation):
                return
            health = qwen_request('/health')
            if not health.get('ready'):
                raise HTTPException(503, 'Qwen model did not become ready')
            if selection and not matches_loaded(health, selection):
                raise HTTPException(503, 'Qwen loaded a different model; stop and retry the selected model')
            self.update_current(generation, phase='ready', model=health.get('model'), model_revision=health.get('revision'), quantization=health.get('quantization'), adapter=health.get('adapter'), error=None)
        except Exception as exc:
            self.update_current(generation, phase='error', error=str(getattr(exc, 'detail', exc)))
        finally:
            self.prepare_lock.release()

    def stop(self):
        with self.state_lock:
            if self.state['phase'] == 'stopping':
                raise HTTPException(409, 'Qwen shutdown is already in progress')
            self.generation += 1
            generation = self.generation
            self.state.update(phase='stopping', error=None)
        try:
            with self.lifecycle_lock:
                self.stop_container()
            self.update_current(generation, phase='stopped', error=None)
        except Exception as exc:
            detail = str(getattr(exc, 'detail', exc))
            self.update_current(generation, phase='error', error=detail)
            raise HTTPException(503, f'Qwen container shutdown failed: {detail}') from exc
        return self.status()

    def observe(self, request):
        # Collect can preview idle or explicitly hypothetical running context.
        # It never turns a decision into a robot command.
        context = input_context(request.context)
        context['operator_instruction'] = None
        if context['active_mission'] is None:
            context['runtime_source'] = 'collect_idle_preview'
        if request.episode_id and self.collection_store:
            episode = self.collection_store.get(request.episode_id)
            if context['map_name'] and context['map_name'] != episode['map_name']:
                raise HTTPException(409, 'Episode belongs to a different map. Select a matching episode.')
            context['map_name'] = episode['map_name']
            events = episode.get('instructions', [])
            instruction = next((e for e in reversed(events) if e['kind'] == 'operator_instruction'), None)
            context['operator_instruction'] = instruction.get('text') if instruction else None
        if not context['map_name']:
            raise HTTPException(409, 'Load a map to preview its Design missions.')
        decision_request = MissionDecisionRequest(
            map_name=context['map_name'], localized=bool(context['localized']),
            previous_frame_stamp=request.previous_frame_stamp,
            operator_instruction=context['operator_instruction'], active_mission=context['active_mission'],
            reasoning_task=context.get('reasoning_task', 'mission_selection'), observation=context.get('observation'),
            candidate_mission_ids=context.get('candidate_mission_ids'), allowed_mission_ids=context.get('allowed_mission_ids'))
        return self._decision(decision_request, preview_context=context, episode_id=request.episode_id)

    def decide(self, request):
        if not request.localized:
            raise HTTPException(409, 'Localize on the selected map before starting autonomy')
        return self._decision(request)

    def _decision(self, request, preview_context=None, episode_id=None):
        preview = preview_context is not None
        if not self.collection_store:
            raise HTTPException(503, 'Mission catalog unavailable')
        if not self.observe_lock.acquire(blocking=False):
            raise HTTPException(409, 'A Qwen request is already in progress')
        try:
            with self.state_lock:
                if self.state['phase'] != 'ready':
                    raise HTTPException(409, 'Qwen is not ready')
                generation = self.generation
                selection = self.selection
                if request.expected_session_id and request.expected_session_id != self.state.get('session_id'):
                    raise HTTPException(412, 'Reasoning session changed. Stop and run again.')
                if selection and selection['map_name'] != request.map_name:
                    raise HTTPException(412, 'The loaded model belongs to another map')
            full_catalog = deepcopy(self.collection_store.catalog_provider(request.map_name))
            started = time.monotonic()
            frame = capture_head()
            raw = request.model_dump()
            if preview:
                preview_context = context_for_frame(preview_context, frame, self.context_provider, normalized=True)
                if preview_context.get('runtime_source') == 'unverified':
                    raise HTTPException(409, 'Waiting for a frame with matching execution context')
                if preview_context.get('map_name') != request.map_name:
                    raise HTTPException(409, 'Camera context belongs to a different map')
                preview_context = live_record_scope(self.collection_store, preview_context)
                raw.update({k: preview_context[k] for k in ('active_mission', 'observation', 'candidate_mission_ids', 'allowed_mission_ids', 'operator_instruction', 'recent_missions', 'context_version', 'runtime_source', 'mission_catalog_snapshot') if k in preview_context})
                raw['reasoning_task'] = task_for(preview_context)
                request = MissionDecisionRequest(**raw)
            elif self.context_provider:
                captured = context_for_frame(raw, frame, self.context_provider, normalized=True)
                if captured.get('runtime_source') == 'unverified':
                    raise HTTPException(409, 'Waiting for a frame with matching execution context')
                if captured.get('runtime_source') == 'autonomy':
                    if (task_for(captured) != task_for(raw) or captured.get('active_mission') != raw.get('active_mission')
                            or captured.get('map_name') != request.map_name):
                        raise HTTPException(409, 'Execution context changed before capture')
                    raw.update(captured)
                    request = MissionDecisionRequest(**raw)
            input_catalog = (raw.get('mission_catalog_snapshot')
                             if raw.get('runtime_source') == 'autonomy' else None)
            if input_catalog is None:
                input_catalog = full_catalog
            try:
                content = build_input(raw, input_catalog)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            missions, context = content['missions'], content['context']
            active = context.get('active_mission')
            if active and (active.get('mission_id') not in {m['mission_id'] for m in full_catalog}
                           or active.get('status') not in ('running', 'paused', 'returning')):
                raise HTTPException(422, 'Invalid active mission context')
            if not self.current(generation): raise HTTPException(409, 'Reasoning stopped')
            if request.frame_captured_after is not None and frame.get('captured_at', 0) <= request.frame_captured_after:
                raise HTTPException(409, 'Waiting for a frame captured after arrival')
            if frame['frame_stamp'] == request.previous_frame_stamp:
                raise HTTPException(409, 'Waiting for a fresh camera frame')
            payload = {
                'system_prompt': system_for(raw),
                'prompt': json.dumps(content, ensure_ascii=False),
                'image_base64': frame['image_base64'], 'max_new_tokens': 192,
                'enable_thinking': False, 'do_sample': False, 'json_object': True,
            }
            if preview:
                payload['attention_preview'] = True
            try:
                result = qwen_request('/generate', payload, timeout=100)
            except Exception as exc:
                if not preview:
                    raise
                result = {'error': str(getattr(exc, 'detail', exc)), 'text': '', 'finish_reason': 'error'}
            if not self.current(generation): raise HTTPException(409, 'Reasoning stopped')
            error = None; validated = {'action_allowed': False, 'decision': None, 'selected_mission': None}
            try:
                if selection and not matches_loaded(result, selection):
                    raise ValueError('Inference model differs from the selected model')
                if time.monotonic() - started > 15: raise ValueError('Decision expired; waiting for a fresh frame')
                if full_catalog != self.collection_store.catalog_provider(request.map_name): raise ValueError('Mission catalog changed during inference')
                if not preview and self.context_provider and raw.get('context_version') is not None:
                    try:
                        latest = self.context_provider(snapshot_time=None)
                    except Exception as exc:
                        raise ValueError('Execution context is unavailable after inference') from exc
                    if not latest or latest.get('context_version') != raw['context_version']:
                        raise ValueError('Execution context changed during inference')
                checked = validate_mission_decision(result, missions, request)
                selected = checked['selected_mission']
                if selected and selected['mission_id'] not in {m['mission_id'] for m in full_catalog}:
                    raise ValueError('Selected mission is no longer in the current Design catalog')
                validated = checked
            except (ValueError, TypeError) as exc:
                error = result.get('error') or str(exc)
            status = self.status()
            output = {**frame, **result, **validated, 'error': error,
                      'selected_model_id': status.get('selected_model_id'), 'model_name': status.get('model_name'),
                      'session_id': status.get('session_id'),
                      'model': status.get('model'), 'model_revision': status.get('model_revision'),
                      'adapter': status.get('adapter'), 'quantization': status.get('quantization'),
                      'map_name': request.map_name, 'input_context': context, 'missions': missions,
                      'reasoning_task': task_for(raw), 'context_version': raw.get('context_version'),
                      'localized': request.localized, 'preview_only': preview,
                      'decision_id': str(uuid4()), 'decision_received_at_ms': int(time.time()*1000)}
            if preview:
                # Eligibility is diagnostic only. Even a valid start is never
                # executable through Collect, regardless of localization.
                output['decision_valid'] = error is None
                output['would_allow_action'] = validated['action_allowed'] and request.localized
                output['action_allowed'] = False
                analysis_id = str(uuid4())
                config = {k: v for k, v in payload.items() if k != 'image_base64'}
                config.update(prompt_version=PROMPT_VERSION, model=status.get('model'),
                              model_revision=status.get('model_revision'), adapter=status.get('adapter'),
                              quantization=status.get('quantization'),
                              preprocessing_version='qwen_service_default_v1',
                              min_pixels=256 * 32 * 32, max_pixels=768 * 32 * 32)
                input_missions = catalog_missions(input_catalog)
                snapshot_id = hashlib.sha256(json.dumps(input_missions, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                observed_live_id = hashlib.sha256(json.dumps(full_catalog, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                preview_context['mission_catalog_snapshot'] = {'missions': deepcopy(input_missions)}
                capture_info = {'source': 'analyzed_frame', 'source_episode_id': episode_id,
                                'source_catalog_snapshot_id': snapshot_id,
                                'observed_live_catalog_snapshot_id': observed_live_id}
                with self.analysis_lock:
                    self.analyses[analysis_id] = {
                        'frame': deepcopy(frame), 'context': deepcopy(preview_context), 'capture_info': capture_info,
                        'prediction': {'prediction': {k: deepcopy(v) for k, v in output.items() if k not in ('image_base64', 'attention_preview')},
                                       'config': config, 'inference_completed_at_ms': int(time.time() * 1000)},
                        'expires': time.monotonic() + 1800,
                    }
                    while len(self.analyses) > 16 or sum(len(v['frame']['image_base64']) for v in self.analyses.values()) > 64 * 1024 * 1024:
                        self.analyses.popitem(last=False)
                output['analysis_id'] = analysis_id
            if not preview:
                try:
                    decision = output.get('decision') or {}
                    selected = output.get('selected_mission') or {}
                    observed = context.get('observation') or {}
                    active = context.get('active_mission') or {}
                    title = 'start proposal' if decision.get('decision') == 'start' else decision.get('decision') or 'rejected'
                    self.activity.append(request.map_name, {
                        'kind': 'decision', 'title': f'AI · {title}',
                        'detail': output.get('block_reason') or decision.get('reason') or error or '',
                        'decision': decision.get('decision'), 'mission_id': decision.get('mission_id'),
                        'mission_name': selected.get('mission_name'), 'action_allowed': output.get('action_allowed'),
                        'model_name': output.get('model_name'), 'session_id': output.get('session_id'),
                        'decision_id': output.get('decision_id'), 'frame_stamp': frame.get('frame_stamp'),
                        'reasoning_task': task_for(raw), 'phase': observed.get('phase'),
                        'context_version': raw.get('context_version'),
                        'observation_spot_id': observed.get('at_spot_id') or observed.get('spot_id'),
                        'target_spot_id': observed.get('target_spot_id'),
                        'active_mission_id': active.get('mission_id'), 'active_mission_status': active.get('status'),
                        'execution_dispatched': False,
                        'raw_response': result.get('text', '')[:8000],
                        'validation_error': error, 'prompt_version': PROMPT_VERSION,
                    })
                except (OSError, ValueError):
                    pass  # Journal failures must never change a robot decision.
            return output
        finally:
            self.observe_lock.release()

    def capture_frame(self, eid, request):
        if self.collection_store is None:
            raise HTTPException(503, 'Collection storage unavailable')
        ep = self.collection_store.episode(eid)
        if ep['ended_at']:
            raise HTTPException(410, 'Start an episode before capturing a frame')
        frame = capture_head()
        context = context_for_frame(request.context, frame, self.context_provider)
        if context['map_name'] != ep['map_name']:
            raise HTTPException(409, 'Map changed. Start an episode for the current map.')
        active = context.get('active_mission')
        missions = self.collection_store.current_catalog(ep['map_name'])['missions']
        if active and active['mission_id'] not in {m['mission_id'] for m in missions}:
            raise HTTPException(422, 'Preview mission is not in the current Design catalog')
        row = self.collection_store.capture(eid, frame, context, metadata=request.metadata.model_dump(exclude_none=True) if request.metadata else None)
        return {'episode': self.collection_store.get(eid), 'saved_frame': row}

    def finish_episode(self, eid, request):
        if self.collection_store is None:
            raise HTTPException(503, 'Collection storage unavailable')
        # Keep other writers from ending/deleting the episode between capture and close.
        with self.collection_store.lock:
            if len(self.collection_store.records(eid, 'observations')) >= 2000:
                raise HTTPException(409, 'Episode frame limit reached; no final frame was saved.')
            result = self.capture_frame(eid, request)
            result['episode'] = self.collection_store.end(eid)
            return result

    def save_analyzed_frame(self, eid, request):
        if self.collection_store is None:
            raise HTTPException(503, 'Collection storage unavailable')
        with self.analysis_lock:
            analysis = self.analyses.get(request.analysis_id)
            if not analysis or analysis['expires'] < time.monotonic():
                raise HTTPException(410, 'Analyzed frame expired. Run reasoning again and save its result.')
            analysis = deepcopy(analysis)
        ep = self.collection_store.episode(eid)
        if analysis['context']['map_name'] != ep['map_name']:
            raise HTTPException(409, 'Analyzed frame belongs to a different map. Use a matching episode.')
        snapshot = analysis['capture_info'].get('observed_live_catalog_snapshot_id',
                                                  analysis['capture_info']['source_catalog_snapshot_id'])
        current = self.collection_store.current_catalog(ep['map_name'])
        if snapshot and snapshot != current['snapshot_id']:
            raise HTTPException(409, 'Design missions changed since analysis. Run reasoning again before saving the frame.')
        row = self.collection_store.capture(eid, analysis['frame'], analysis['context'],
            capture_info=analysis['capture_info'], prediction=analysis['prediction'], analysis_id=request.analysis_id)
        return {'episode': self.collection_store.get(eid), 'saved_frame': row}


def create_router(start_container, stop_container, collection_store=None, service=None):
    service = service or ReasoningService(start_container, stop_container, collection_store)
    router = APIRouter(tags=['reasoning', 'collection'])

    @router.get('/reasoning/models')
    def models(map_name: str):
        return {'models': service.models.list(map_name)}

    @router.post('/reasoning/start')
    def start(request: StartRequest | None = None):
        return service.start(request)

    @router.post('/reasoning/stop')
    def stop():
        return service.stop()

    @router.get('/reasoning/status')
    def status():
        return service.status()

    @router.get('/reasoning/activity')
    def activity(map_name: str, limit: int = 100):
        try:
            return {'events': service.activity.recent(map_name, max(1, min(limit, 500)))}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post('/reasoning/activity')
    def record_activity(request: ActivityEventRequest):
        try:
            return service.activity.append(request.map_name, {
                'kind': 'operation', 'title': request.title, 'detail': request.detail,
            })
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post('/reasoning/observe')
    def observe(request: ObserveRequest):
        return service.observe(request)

    @router.post('/reasoning/decide')
    def decide(request: MissionDecisionRequest):
        return service.decide(request)

    @router.post('/collection/episodes/{eid}/capture')
    def capture(eid: str, request: CaptureRequest):
        return service.capture_frame(eid, request)

    @router.post('/collection/episodes/{eid}/finish')
    def finish(eid: str, request: CaptureRequest):
        return service.finish_episode(eid, request)

    @router.post('/collection/episodes/{eid}/save-analysis')
    def save_analysis(eid: str, request: SaveAnalysisRequest):
        return service.save_analyzed_frame(eid, request)

    return router
