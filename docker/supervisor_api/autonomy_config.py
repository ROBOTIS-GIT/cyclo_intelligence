"""Map-scoped observation plans and immutable Design mission execution snapshots."""
from copy import deepcopy
import math
import json
import threading
from typing import Literal
import xml.etree.ElementTree as ET

from fastapi import HTTPException
from pydantic import BaseModel, Field
from . import navigation_missions as missions, navigation_spots as spots


class ObservationPoint(BaseModel):
    spot_id: str = Field(min_length=1, max_length=128)
    departure_policy: Literal['reasoning', 'timed'] = 'reasoning'
    dwell_seconds: float = Field(default=15, ge=2, le=3600, allow_inf_nan=False)
    mission_ids: list[str] = Field(default_factory=list, max_length=100)
    # Optional saved posture, independent of a task mission's route.
    arrival_bt: str = Field(default='', max_length=50000)


class ObservationPlan(BaseModel):
    schema_version: Literal[2] = 2
    revision: int = Field(default=0, ge=0)
    mode: Literal['fixed', 'patrol'] = 'fixed'
    route_closed: bool = True  # Older patrol plans were always cyclic.
    settle_seconds: float = Field(default=1, ge=0.2, le=30, allow_inf_nan=False)
    interval_seconds: float = Field(default=1, ge=0.1, le=60, allow_inf_nan=False)
    start_confirmations: int = Field(default=3, ge=1, le=10)
    resume_confirmations: int = Field(default=2, ge=1, le=10)
    mission_start_policy: Literal['confirm_at_arrival'] = 'confirm_at_arrival'
    resume_policy: Literal['return_to_trigger_spot'] = 'return_to_trigger_spot'
    observations: list[ObservationPoint] = Field(default_factory=list, max_length=100)
    excluded_mission_ids: list[str] = Field(default_factory=list, max_length=100)


def finite_pose(pose):
    if pose.get('frame_id', 'map') != 'map':
        raise ValueError('Observation and mission poses must use the map frame')
    if not all(math.isfinite(float(pose.get(k, 0))) for k in ('x', 'y', 'yaw')):
        raise ValueError('Pose coordinates must be finite')
    return pose


class AutonomyRepository:
    def __init__(self):
        self.lock = threading.RLock()

    def path(self, map_name):
        # Use the stricter mission name validator (rejects . and ..).
        name = missions._validate_map_name(map_name)
        path = spots.SPOTS_ROOT / 'maps' / name / 'autonomy.json'
        if not path.resolve().is_relative_to((spots.SPOTS_ROOT / 'maps').resolve()):
            raise HTTPException(400, 'Observation path escapes the map storage')
        return path

    def read(self, map_name):
        with self.lock:
            path = self.path(map_name)
            if not path.exists():
                return ObservationPlan()
            value = json.loads(path.read_text())
            if value.get('schema_version', 1) == 1:
                # V1 [] meant all tasks. Expand that meaning once in the v2 view;
                # the next explicit Save persists it without rewriting datasets.
                excluded = set(value.get('excluded_mission_ids', []))
                allowed = [m['mission_id'] for m in missions.collection_catalog(map_name)
                           if m['mission_id'] not in excluded]
                for point in value.get('observations', []):
                    if not point.get('mission_ids'):
                        point['mission_ids'] = list(allowed)
                value['schema_version'] = 2
            return ObservationPlan.model_validate(value)

    def resolve(self, map_name, plan):
        available = {s.id: spots._serialize_spot(s) for s in spots._list_spots(map_name)}
        catalog = missions.collection_catalog(map_name)
        mids = {m['mission_id'] for m in catalog}
        resolved = []
        seen = set()
        for item in plan.observations:
            if item.spot_id in seen:
                raise ValueError('Each observation spot may appear only once')
            seen.add(item.spot_id)
            if item.spot_id not in available:
                raise ValueError(f'Observation spot was deleted: {item.spot_id}')
            if set(item.mission_ids) - mids:
                raise ValueError('An observation references a deleted mission. Update its mission selection.')
            if item.arrival_bt:
                try: ET.fromstring(item.arrival_bt)
                except ET.ParseError as exc: raise ValueError('Invalid observation posture BT XML') from exc
            spot = available[item.spot_id]
            finite_pose(spot['pose'])
            resolved.append({**item.model_dump(), 'label': spot['label'], 'pose': spot['pose']})
        return resolved, catalog

    def save(self, map_name, plan):
        with self.lock:
            current = self.read(map_name)
            if current.revision != plan.revision:
                raise HTTPException(409, 'Observation settings changed. Reload and try again.')
            self.resolve(map_name, plan)
            value = plan.model_copy(update={'revision': current.revision + 1})
            # Preserve legacy observation identity even when it is removed from
            # the patrol plan. Record must still offer it as a saved observation.
            observation_ids = {p.spot_id for p in [*current.observations, *plan.observations]}
            with spots._SPOT_MUTATION_LOCK:
                for spot in spots._list_spots(map_name):
                    if spot.id in observation_ids and spot.metadata.get('role') != 'observation':
                        spots.update_spot(spot.id, spots.SpotUpdateRequest(map_name=map_name,
                            metadata={**spot.metadata, 'role': 'observation'}))
                self._write_plan(map_name, value)
            return value

    def _write_plan(self, map_name, plan):
        path = self.path(map_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.json.tmp')
        temp.write_text(plan.model_dump_json(indent=2) + '\n')
        temp.replace(path)

    def write_observation_spot(self, map_name, request, spot_id=None):
        with self.lock, spots._SPOT_MUTATION_LOCK:
            current = self.read(map_name)
            if current.revision != request.plan_revision:
                raise HTTPException(409, 'Observation settings changed. Refresh and try again.')
            label = request.label.strip()
            if not label:
                raise HTTPException(422, 'Observation name is required')
            try:
                finite_pose(request.pose.model_dump())
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            metadata = {'coordinate_space': 'map', 'role': 'observation'}
            if spot_id:
                existing = next((s for s in spots._list_spots(map_name) if s.id == spot_id), None)
                if existing is None:
                    raise HTTPException(404, 'Observation spot not found')
                if existing.metadata.get('role') != 'observation' and not any(
                        p.spot_id == spot_id for p in current.observations):
                    raise HTTPException(422, 'This is not an observation spot')
                metadata = {**existing.metadata, **metadata}
                saved = spots.update_spot(spot_id, spots.SpotUpdateRequest(
                    map_name=map_name, label=label, pose=request.pose, metadata=metadata))
            else:
                saved = spots.create_spot(spots.SpotCreateRequest(
                    map_name=map_name, label=label, pose=request.pose, metadata=metadata))
            self._write_plan(map_name, current.model_copy(update={'revision': current.revision + 1}))
            return saved

    def delete_spot(self, map_name, spot_id, plan_revision):
        # Remove the reference first: a process interruption can leave an unused
        # spot, but never a saved plan pointing at a spot this operation deleted.
        with self.lock, spots._SPOT_MUTATION_LOCK:
            current = self.read(map_name)
            if current.revision != plan_revision:
                raise HTTPException(409, 'Observation settings changed. Refresh before deleting.')
            stored = spots._list_spots(map_name)
            if not any(s.id == spot_id for s in stored):
                raise HTTPException(404, 'Saved spot not found')
            remaining = [p for p in current.observations if p.spot_id != spot_id]
            changed = len(remaining) != len(current.observations)
            if changed:
                self._write_plan(map_name, current.model_copy(update={
                    'observations': remaining, 'revision': current.revision + 1}))
            try:
                spots.delete_spot(spot_id, map_name=map_name)
            except Exception:
                if changed: self._write_plan(map_name, current)
                raise

    def view(self, map_name):
        plan = self.read(map_name)
        return {'plan': plan.model_dump(), 'spots': [spots._serialize_spot(s) for s in spots._list_spots(map_name)],
                'missions': missions.collection_catalog(map_name)}

    def mission(self, map_name, selected):
        # Snapshot all selected BTs under the same lock used by Design writes.
        with missions._MISSION_MUTATION_LOCK:
            manifest = missions._read_manifest(map_name, selected['mission_name'])
            if (not manifest.exists or manifest.revision != selected['revision']
                    or manifest.metadata.get('reasoning_mission_id') != selected['mission_id']):
                raise ValueError('Mission changed since the decision. Observe again.')
            waypoints = {w.id: w.model_dump() for w in manifest.waypoints}
            flow = manifest.metadata.get('mission_flow') or {}
            edges = flow.get('edges', [])
            if not edges:
                ids = list(waypoints) if len(waypoints) == 1 else []
            else:
                if any(e['source'] not in waypoints or e['target'] not in waypoints for e in edges):
                    raise ValueError('Mission route references a missing waypoint')
                outgoing = {}
                for edge in edges:
                    if edge['source'] in outgoing:
                        raise ValueError('Branched routes are not supported by autonomous execution')
                    outgoing[edge['source']] = edge['target']
                starts = [wid for wid in waypoints if wid in outgoing and wid not in outgoing.values()]
                current = starts[0] if len(starts) == 1 else edges[0]['source']
                ids, used = [], set()
                while current:
                    ids.append(current)
                    if current in used:
                        if current != ids[0]: raise ValueError('Route loops back into an intermediate waypoint')
                        break
                    used.add(current)
                    current = outgoing.get(current)
                if len(used) != len(set(outgoing) | set(outgoing.values())):
                    raise ValueError('Mission route is disconnected')
            if not ids:
                raise ValueError('Mission has no executable route')
            result = []
            for wid in ids:
                wp = deepcopy(waypoints[wid]); finite_pose(wp['pose'])
                path = wp.get('local_bt')
                wp['xml'] = ''
                if path:
                    bt = missions._resolve_bt_path(map_name, selected['mission_name'], path)
                    wp['xml'] = bt.read_text(encoding='utf-8')
                    parsed = ET.fromstring(wp['xml'])
                    if any(n.tag == 'ReturnToWaypoint' or n.get('ID') == 'ReturnToWaypoint' for n in parsed.iter()):
                        snapshot = [{k: w[k] for k in ('id', 'label', 'pose')} for w in waypoints.values()]
                        parsed.set('mission_waypoints', json.dumps(snapshot, allow_nan=False))
                        wp['xml'] = ET.tostring(parsed, encoding='unicode')
                result.append(wp)
            return result
