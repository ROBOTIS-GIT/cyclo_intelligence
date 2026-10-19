import { withMissionWaypointContext } from './missionWaypointContext';
const target = { id: 'start', label: 'Dock & start', pose: { x: 1, y: 2, yaw: 0 } };
const xml = '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><ReturnToWaypoint waypoint="start"/></BehaviorTree></root>';
test('injects the loaded pose snapshot with escaped labels without modifying the input', () => {
  const result = withMissionWaypointContext(xml, [target]);
  const doc = new DOMParser().parseFromString(result, 'text/xml');
  expect(JSON.parse(doc.documentElement.getAttribute('mission_waypoints'))).toEqual([
    { ...target, pose: { ...target.pose, frame_id: 'map' } },
  ]);
  expect(xml).not.toContain('mission_waypoints');
});
test('leaves existing trees unchanged', () => {
  const wait = '<root><BehaviorTree><Wait duration="1"/></BehaviorTree></root>';
  expect(withMissionWaypointContext(wait, [target])).toBe(wait);
});
test('closed routes do not duplicate a waypoint target', () => {
  const result = withMissionWaypointContext(xml, [target, target]);
  const doc = new DOMParser().parseFromString(result, 'text/xml');
  expect(JSON.parse(doc.documentElement.getAttribute('mission_waypoints'))).toHaveLength(1);
});
test('rejects non-finite and conflicting destinations before issuing a BT request', () => {
  expect(() => withMissionWaypointContext(xml, [{ ...target, pose: { x: NaN, y: 2 } }])).toThrow();
  expect(() => withMissionWaypointContext(xml, [target, { ...target, pose: { x: 9, y: 2, yaw: 0 } }])).toThrow();
});
