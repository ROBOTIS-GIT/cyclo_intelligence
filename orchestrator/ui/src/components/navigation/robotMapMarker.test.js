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
import { createSg2TopDownMarker, updateSg2TopDownMarker, usesSg2TopDownMarker } from "./robotMapMarker";
import { SG2_BASE_BOUNDS, SG2_TOP_DOWN_BOUNDS, SG2_TOP_DOWN_PARTS } from "./sg2TopDownGeometry";

const poseAt = (x, y, yaw) => ({ position: { x, y }, orientation: { x: 0, y: 0, z: Math.sin(yaw / 2), w: Math.cos(yaw / 2) } });

test("uses the SG2 silhouette only for a known SG2 in the visible Autonomy footprint layer", () => {
  const options = { renderProfile: "autonomy", robotType: "ffw_sg2_rev1", showFootprint: true };
  expect(usesSg2TopDownMarker(options)).toBe(true);
  expect(usesSg2TopDownMarker({ ...options, robotType: "ffw_sg2_rev1_follower" })).toBe(true);
  expect(usesSg2TopDownMarker({ ...options, robotType: "ffw_f1" })).toBe(false);
  expect(usesSg2TopDownMarker({ ...options, robotType: "" })).toBe(false);
  expect(usesSg2TopDownMarker({ ...options, renderProfile: "default" })).toBe(false);
  expect(usesSg2TopDownMarker({ ...options, showFootprint: false })).toBe(false);
});

test("keeps source mesh scale and the off-centre base_link origin in a small shaded visual", () => {
  const base = SG2_TOP_DOWN_PARTS.find(({ name }) => name === "base").points;
  expect(Math.min(...base.map(([x]) => x))).toBe(SG2_BASE_BOUNDS.minX);
  expect(Math.max(...base.map(([x]) => x))).toBe(SG2_BASE_BOUNDS.maxX);
  expect(Math.max(...base.map(([, y]) => y)) - Math.min(...base.map(([, y]) => y))).toBeCloseTo(0.602, 4);
  expect(SG2_BASE_BOUNDS.maxX - SG2_BASE_BOUNDS.minX).toBeCloseTo(0.6275, 4);
  const marker = createSg2TopDownMarker();
  expect(marker.children).toHaveLength(2);
  const mesh = marker.children[0];
  expect(mesh.geometry.getAttribute("position").count).toBeLessThan(3000);
  expect(mesh.material.vertexColors).toBe(true);
  expect(mesh.material.map).toBeNull();
  mesh.geometry.computeBoundingBox();
  expect(mesh.geometry.boundingBox.min.x).toBeCloseTo(SG2_TOP_DOWN_BOUNDS.minX);
  expect(mesh.geometry.boundingBox.max.x).toBeCloseTo(SG2_TOP_DOWN_BOUNDS.maxX);
  expect(marker.userData.screenAnchorBounds).toEqual(SG2_TOP_DOWN_BOUNDS);
  const shadow = marker.children[1];
  expect(shadow.material.depthWrite).toBe(false);
  expect(shadow.geometry.getAttribute("color").itemSize).toBe(4);
  expect(shadow.geometry.getAttribute("position").count).toBeLessThan(6000);
  expect([...mesh.geometry.getAttribute("color").array].every(Number.isFinite)).toBe(true);
  marker.children.forEach(child => { child.geometry.dispose(); child.material.dispose(); });
});

test.each([0, Math.PI / 2, Math.PI, -Math.PI / 2])("keeps +X forward when the pose yaw is %s", (yaw) => {
  const marker = createSg2TopDownMarker();
  const mesh = marker.children[0];
  const geometry = mesh.geometry;
  updateSg2TopDownMarker(marker, poseAt(3, -2, yaw));
  marker.updateMatrixWorld(true);
  const forward = marker.localToWorld(new THREE.Vector3(0.2246, 0, 0));
  expect(forward.x).toBeCloseTo(3 + 0.2246 * Math.cos(yaw));
  expect(forward.y).toBeCloseTo(-2 + 0.2246 * Math.sin(yaw));
  expect(marker.position.x).toBe(3);
  expect(marker.position.y).toBe(-2);
  updateSg2TopDownMarker(marker, poseAt(-4, 5, yaw + 0.2));
  expect(marker.children[0]).toBe(mesh);
  expect(mesh.geometry).toBe(geometry);
  expect(marker.visible).toBe(true);
  updateSg2TopDownMarker(marker, null);
  expect(marker.visible).toBe(false);
  updateSg2TopDownMarker(marker, poseAt(NaN, 0, 0));
  expect(marker.visible).toBe(false);
  marker.children.forEach(child => { child.geometry.dispose(); child.material.dispose(); });
});
