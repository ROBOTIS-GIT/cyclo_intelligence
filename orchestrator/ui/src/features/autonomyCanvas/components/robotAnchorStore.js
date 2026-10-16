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
