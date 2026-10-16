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

// A fixed 72 x 72 CSS-pixel area stays clickable at every map zoom level.
// Name pills provide their own projected rectangle with additional padding.
export function nearestWaypointHit(points, x, y) {
  let nearest = null;
  let distance = Infinity;
  for (const point of points) {
    const dx = point.x - x;
    const dy = point.y - y;
    const squared = dx * dx + dy * dy;
    if (
      Math.abs(dx) <= (point.width ?? 72) / 2 &&
      Math.abs(dy) <= (point.height ?? 72) / 2 &&
      Number.isFinite(squared) &&
      squared < distance
    ) {
      nearest = point.id;
      distance = squared;
    }
  }
  return nearest;
}

// Project marker and label hit areas using the current camera and canvas viewport.
export function waypointHitAreas(layers, camera, rect) {
  const candidates = [];
  layers.updateWorldMatrix(true, true);
  layers.traverse((object) => {
    if (!object.userData?.waypointHitAnchor && !object.userData?.waypointHitLabel) return;
    for (let parent = object; parent; parent = parent.parent) {
      if (!parent.visible) return;
    }
    const projected = object.getWorldPosition(new THREE.Vector3()).project(camera);
    if (
      projected.z < -1 ||
      projected.z > 1 ||
      Math.abs(projected.x) > 1 ||
      Math.abs(projected.y) > 1
    )
      return;
    const candidate = {
      id: object.userData.spotId,
      x: rect.left + ((projected.x + 1) * rect.width) / 2,
      y: rect.top + ((1 - projected.y) * rect.height) / 2,
    };
    if (object.userData.waypointHitLabel) {
      // Sprite.center shifts the name pill above the waypoint.
      // Project its actual billboard corners, including parent scale.
      const center = object
        .getWorldPosition(new THREE.Vector3())
        .applyMatrix4(camera.matrixWorldInverse);
      const scale = object.getWorldScale(new THREE.Vector3());
      const corners = [
        [0, 0],
        [1, 1],
      ].map(([x, y]) => {
        const corner = center.clone();
        corner.x += (x - object.center.x) * scale.x;
        corner.y += (y - object.center.y) * scale.y;
        corner.applyMatrix4(camera.projectionMatrix);
        return {
          x: rect.left + ((corner.x + 1) * rect.width) / 2,
          y: rect.top + ((1 - corner.y) * rect.height) / 2,
        };
      });
      candidate.x = (corners[0].x + corners[1].x) / 2;
      candidate.y = (corners[0].y + corners[1].y) / 2;
      candidate.width = Math.max(72, Math.abs(corners[1].x - corners[0].x) + 16);
      candidate.height = Math.max(40, Math.abs(corners[1].y - corners[0].y) + 16);
    }
    candidates.push(candidate);
  });
  return candidates;
}
