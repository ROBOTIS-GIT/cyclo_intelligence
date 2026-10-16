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

import * as THREE from "three";

// Reuse the map's marker shapes, but remove all hit targets: runtime mission
// geometry must never edit or select an Observation with a colliding ID.
export function addRouteOverlay(layer, overlay, scale, palette, { marker, badge, line }) {
  if (!overlay) return;
  const colors = { ...palette, idle: overlay.color ?? palette.idle, selected: overlay.activeColor ?? palette.selected };
  const addReadonly = object => {
    if (!object) return;
    object.traverse(child => {
      const { pulse, pulseBase } = child.userData;
      child.userData = { pulse, pulseBase };
      child.raycast = () => {};
    });
    layer.add(object);
  };
  const points = (overlay.pathSpots || []).filter(spot => spot?.pose).map(spot =>
    new THREE.Vector3(Number(spot.pose.x), Number(spot.pose.y), 0.025));
  if (points.length > 1) addReadonly(line(points, colors.idle, scale));
  const spots = new Map((overlay.spots || []).map(spot => [spot.id, spot]));
  spots.forEach(spot => addReadonly(marker(spot, false, scale, spot.id === overlay.activeWaypointId, colors)));
  (overlay.order || []).forEach(({ id, order }) =>
    addReadonly(badge(spots.get(id), order, id === overlay.activeWaypointId, scale, colors)));
}
