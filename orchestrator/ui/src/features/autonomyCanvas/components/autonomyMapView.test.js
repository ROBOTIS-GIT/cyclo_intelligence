// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { renderHook } from '@testing-library/react';
import { autonomyMapView, useAutonomyMapView } from './autonomyMapView';
import { recordMissionOverlay } from '../lib/recordMissionOverlay';

const observation = { spot_id: 'watch-table', label: 'Table view', pose: { x: 1, y: 2 } };
const source = { ...observation, id: observation.spot_id };
const config = { draft: { mode: 'patrol', observations: [{ spot_id: 'watch-table' }, { spot_id: 'watch-sink' }] },
  spots: [source, { id: 'watch-sink', label: 'Sink view', pose: { x: 3, y: 2 } }] };
const mission = { active: true, mode: 'mission', mission: { mission_id: 'clean' }, phase: 'executing',
  mission_route: [{ id: 'task-table' }, { id: 'task-sink' }],
  runner: { phase: 'navigation', activeSpotId: 'task-sink' }, observation,
  resume_checkpoint: { return_spot_id: 'watch-table' }, config };

test('mission uses the same separate child overlay as Record while preserving observations', () => {
  const view = autonomyMapView(mission);
  expect(view.spots).toEqual(config.spots);
  expect(view.activeWaypointId).toBeNull();
  expect(view.missionOverlay).toEqual(recordMissionOverlay({
    cycle: { isRunning: true, pinnedSpot: source, pinnedMissionId: 'clean' },
    mission: { missionId: 'clean', mapName: 'house' }, mapName: 'house', activeWaypointId: 'task-sink',
    routeView: { orderedSpots: mission.mission_route },
  }));
  expect(view.missionOverlay.pathSpots).toEqual([source, ...mission.mission_route, source]);
  expect(view.navigationActive).toBe(true);
});

test.each(['return-navigation', 'return-posture'])('return focuses the observation and removes completed task during %s', phase => {
  const view = autonomyMapView({ ...mission, phase: 'returning', runner: { phase } });
  expect(view.spots).toEqual(config.spots);
  expect(view.activeWaypointId).toBe('watch-table');
  expect(view.route.order).toHaveLength(2);
  expect(view.missionOverlay).toBeNull();
  expect(view.navigationActive).toBe(phase === 'return-navigation');
});

test('paused return and retained mission metadata never pulse or show the cancelled Nav2 plan', () => {
  const view = autonomyMapView({ ...mission, paused: true, phase: 'paused', runner: { phase: 'return-navigation' } });
  expect(view.activeWaypointId).toBeNull();
  expect(view.navigationActive).toBe(false);
  expect(view.missionOverlay).toBeNull();
});

test('execution snapshot supplies observation pose before config hydration', () => {
  expect(autonomyMapView({ ...mission, phase: 'returning', config: null }).spots).toEqual([source]);
  expect(autonomyMapView({ ...mission, config: null }).missionOverlay.pathSpots).toEqual([source, ...mission.mission_route, source]);
});

test.each([true, false, undefined])('patrol draws the saved closing edge (route_closed=%s)', closed => {
  const view = autonomyMapView({ config: { ...config, draft: { ...config.draft, route_closed: closed } } });
  expect(view.route.order.map(point => point.id)).toEqual(['watch-table', 'watch-sink']);
  expect(view.route.closed).toBe(closed !== false);
  expect(view.activeWaypointId).toBeNull();
  expect(autonomyMapView({ config: { ...config, draft: { ...config.draft, mode: 'fixed' } } }).route).toEqual({ closed: false, order: [] });
});

test('unused saved observations stay visible and a single spot has no closing edge', () => {
  const view = autonomyMapView({ config: { ...config, spots: [...config.spots,
    { id: 'unused', metadata: { role: 'observation' } }, { id: 'task' }],
    draft: { ...config.draft, observations: [{ spot_id: 'watch-table' }] } } });
  expect(view.spots.map(spot => spot.id)).toEqual(['watch-table', 'unused']);
  expect(view.route).toEqual({ closed: false, order: [{ id: 'watch-table', order: 1 }] });
});

test.each(['idle', 'stopped', 'stopping', 'interrupted', 'error'])('%s clears stale execution highlights and child route', phase => {
  const view = autonomyMapView({ ...mission, active: false, phase });
  expect(view.activeWaypointId).toBeNull();
  expect(view.missionOverlay).toBeNull();
  expect(view.navigationActive).toBe(false);
});

test('normalized execution controls rendering even when compatibility fields are stale', () => {
  const execution = { lifecycle: 'active', owns_execution: true, control: 'running', activity: 'patrol_navigation', target_spot_id: 'watch-sink' };
  const state = { active: true, phase: 'positioning', config, observation, execution };
  expect(autonomyMapView(state)).toMatchObject({ navigationActive: true, activeWaypointId: 'watch-sink' });
  expect(autonomyMapView({ ...state, execution: { ...execution, control: 'paused' } })).toMatchObject({ navigationActive: false, activeWaypointId: null });
  expect(autonomyMapView({ ...mission, execution: { ...execution, lifecycle: 'failed' } }).missionOverlay).toBeNull();
});

test('closed mission traversal retains its return edge without duplicate waypoint badges', () => {
  const route = [...mission.mission_route, mission.mission_route[0]];
  const overlay = autonomyMapView({ ...mission, mission_route: route }).missionOverlay;
  expect(overlay.spots).toHaveLength(2);
  expect(overlay.order).toHaveLength(2);
  expect(overlay.pathSpots).toEqual([source, ...route, source]);
});


const patrolExecution = { lifecycle: 'active', owns_execution: true, control: 'running',
  activity: 'patrol_navigation', target_spot_id: 'watch-table' };

test('map hook keeps geometry references through unrelated ROS renders and reasoning status changes', () => {
  const state = { ...mission, mode: 'patrol', execution: patrolExecution };
  const utils = renderHook(({ value }) => useAutonomyMapView(value), { initialProps: { value: state } });
  const view = utils.result.current;
  utils.rerender({ value: { ...state, result: { decision_id: 'new' }, events: [{ title: 'Thought' }],
    config: { ...config, error: 'Unrelated status' },
    execution: { ...patrolExecution, revision: 2 }, runner: { ...mission.runner, progress: 50 } } });
  expect(utils.result.current).toBe(view);
  expect(utils.result.current.spots).toBe(view.spots);
  expect(utils.result.current.route).toBe(view.route);
  utils.unmount();
});

test('map hook updates execution targets and pause highlights', () => {
  const state = { ...mission, mode: 'patrol', execution: patrolExecution };
  const utils = renderHook(({ value }) => useAutonomyMapView(value), { initialProps: { value: state } });
  const view = utils.result.current;
  utils.rerender({ value: { ...state, execution: { ...patrolExecution, target_spot_id: 'watch-sink' } } });
  expect(utils.result.current).not.toBe(view);
  expect(utils.result.current.activeWaypointId).toBe('watch-sink');
  utils.rerender({ value: { ...state, execution: { ...patrolExecution, control: 'paused' } } });
  expect(utils.result.current.activeWaypointId).toBeNull();
  expect(utils.result.current.navigationActive).toBe(false);
  utils.unmount();
});

test('map hook updates changed plan, observation geometry and mission routes', () => {
  const utils = renderHook(({ value }) => useAutonomyMapView(value), { initialProps: { value: mission } });
  const view = utils.result.current;
  const changedConfig = { ...config, draft: { ...config.draft, route_closed: false } };
  utils.rerender({ value: { ...mission, config: changedConfig } });
  expect(utils.result.current).not.toBe(view);
  expect(utils.result.current.route.closed).toBe(false);
  const moved = { ...source, pose: { x: 9, y: 2 } };
  utils.rerender({ value: { ...mission, config: { ...changedConfig, spots: [moved, config.spots[1]] } } });
  expect(utils.result.current.spots[0]).toBe(moved);
  expect(utils.result.current.missionOverlay.pathSpots[0]).toBe(moved);
  const changedRoute = [{ id: 'new-task', pose: { x: 2, y: 5 } }];
  utils.rerender({ value: { ...mission, mission_route: changedRoute } });
  expect(utils.result.current.missionOverlay.spots).toEqual(changedRoute);
  utils.unmount();
});
