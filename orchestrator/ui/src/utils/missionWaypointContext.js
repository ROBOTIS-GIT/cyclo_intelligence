// Runtime-only mission poses for navigation actions in a local BT.
// Saved XML remains reusable; execution uses the same loaded mission snapshot.
export function withMissionWaypointContext(xml, waypoints = []) {
  if (!xml || !xml.includes('ReturnToWaypoint')) return xml;
  const doc = new DOMParser().parseFromString(xml, 'text/xml');
  if (doc.getElementsByTagName('parsererror').length) return xml;
  if (!doc.querySelector('ReturnToWaypoint, Action[ID="ReturnToWaypoint"]')) return xml;
  const byId = new Map();
  for (const spot of waypoints) {
    const pose = spot.pose || {};
    const target = { id: spot.id, label: spot.label || spot.id,
      pose: { frame_id: pose.frame_id || 'map', x: Number(pose.x), y: Number(pose.y), yaw: Number(pose.yaw ?? 0) } };
    if (!target.id || target.pose.frame_id !== 'map'
      || !['x', 'y', 'yaw'].every(key => Number.isFinite(target.pose[key]))) {
      throw new Error('ReturnToWaypoint requires valid map-frame mission waypoints');
    }
    if (byId.has(target.id) && JSON.stringify(byId.get(target.id).pose) !== JSON.stringify(target.pose)) {
      throw new Error('Conflicting mission waypoint poses');
    }
    byId.set(target.id, target);
  }
  doc.documentElement.setAttribute('mission_waypoints', JSON.stringify([...byId.values()]));
  return new XMLSerializer().serializeToString(doc);
}
