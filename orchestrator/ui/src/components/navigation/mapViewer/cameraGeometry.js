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
// Author: Howon Kim, Seongwoo Kim

"use client";

import { RENDER_INTERVAL_HIDDEN_MS, RENDER_INTERVAL_ACTIVE_MS, RENDER_INTERVAL_IDLE_MS, CAMERA_NEAR, CAMERA_FAR, BT_FOCUS_VISIBLE_HEIGHT_MIN, BT_FOCUS_VISIBLE_HEIGHT_MAX, BT_FOCUS_WAYPOINT_NDC_X } from './constants';
import * as THREE from 'three';
export function mapRenderIntervalMs({
  hidden = false,
  active = false,
  interactive = false,
  renderProfile = "default"
} = {}) {
  if (hidden) return RENDER_INTERVAL_HIDDEN_MS;
  if (active && (renderProfile === "autonomy") && !interactive) return 33;
  return active ? RENDER_INTERVAL_ACTIVE_MS : RENDER_INTERVAL_IDLE_MS;
}
export function fitCameraToMap(camera, controls, meta, roll = 0) {
  if (!meta) return;
  const width = meta.width * meta.resolution;
  const height = meta.height * meta.resolution;
  const center = new THREE.Vector3(meta.originX + (width / 2), meta.originY + (height / 2), 0);
  const maxDim = Math.max(width, height, 1);
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const distanceForHeight = height / (2 * Math.tan(halfFov));
  const distanceForWidth = width / (2 * Math.tan(halfFov) * Math.max(camera.aspect, 0.1));
  const distance = Math.max(distanceForHeight, distanceForWidth, maxDim) * 1.12;
  camera.up.set(Math.sin(roll), Math.cos(roll), 0);
  camera.position.set(center.x, center.y, distance);
  camera.lookAt(center);
  camera.near = CAMERA_NEAR;
  camera.far = Math.max(CAMERA_FAR, maxDim * 10);
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}
export function applyTopViewRoll(camera, controls, roll) {
  camera.up.set(Math.sin(roll), Math.cos(roll), 0);
  camera.lookAt(controls.target);
  controls.update();
}
export function focusCameraToWaypoint(camera, controls, meta, pose, roll = 0) {
  var _a, _b;
  if (!meta || !pose) return;
  const width = meta.width * meta.resolution;
  const height = meta.height * meta.resolution;
  const maxDim = Math.max(width, height, 1);
  const visibleHeight = Math.max(BT_FOCUS_VISIBLE_HEIGHT_MIN, Math.min(maxDim * 0.68, BT_FOCUS_VISIBLE_HEIGHT_MAX));
  const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
  const distance = visibleHeight / (2 * Math.tan(halfFov));
  const halfWidth = distance * Math.tan(halfFov) * Math.max(camera.aspect, 0.1);
  const x = Number(((_a = pose.x) !== null) && (_a !== void 0) ? _a : 0);
  const y = Number(((_b = pose.y) !== null) && (_b !== void 0) ? _b : 0);
  const screenRight = new THREE.Vector3(Math.cos(roll), -Math.sin(roll), 0);
  const target = new THREE.Vector3(x, y, 0).addScaledVector(screenRight, -BT_FOCUS_WAYPOINT_NDC_X * halfWidth);
  camera.up.set(Math.sin(roll), Math.cos(roll), 0);
  camera.position.set(target.x, target.y, Math.max(distance, CAMERA_NEAR * 10));
  camera.lookAt(target);
  camera.near = CAMERA_NEAR;
  camera.far = Math.max(CAMERA_FAR, maxDim * 10);
  camera.updateProjectionMatrix();
  controls.target.copy(target);
  controls.update();
}
