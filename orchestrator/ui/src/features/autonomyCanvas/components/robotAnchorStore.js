// Screen-coordinate updates should move the thought bubble without re-rendering
// the map scene, HUD, waypoint editor, or mission overlay.
export function createRobotAnchorStore() {
  let value = null;
  const listeners = new Set();
  return {
    getSnapshot: () => value,
    subscribe: listener => { listeners.add(listener); return () => listeners.delete(listener); },
    set: next => {
      if (value === next || (value && next && value.x === next.x && value.y === next.y
        && value.width === next.width && value.height === next.height
        && ["left", "right", "top", "bottom"].every(key => value.robotBounds?.[key] === next.robotBounds?.[key]))) return;
      value = next;
      listeners.forEach(listener => listener());
    },
  };
}
