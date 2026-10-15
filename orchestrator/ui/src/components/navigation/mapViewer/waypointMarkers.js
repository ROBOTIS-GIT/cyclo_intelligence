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

import { WAYPOINT_CENTER_RADIUS, WAYPOINT_HEADING_LENGTH, WAYPOINT_HEADING_SHAFT_WIDTH, WAYPOINT_HEADING_HEAD_WIDTH, WAYPOINT_HEADING_HEAD_LENGTH, WAYPOINT_ROTATE_HIT_RADIUS, WAYPOINT_ROTATE_HIT_OFFSET, markerPalette, WAYPOINT_BODY_HIT_RADIUS, WAYPOINT_SELECTED_HALO_OUTER_RADIUS, WAYPOINT_SELECTED_HALO_INNER_RADIUS, WAYPOINT_RING_INNER_RADIUS, WAYPOINT_RING_OUTER_RADIUS, WAYPOINT_LABEL_SCALE_Y, WAYPOINT_LABEL_OFFSET_Y } from './constants';
import * as THREE from 'three';
import { yawFromPose } from '../../../utils/navigationTf';
import { makeWaypointLabelSprite } from './labelSprites';
export function waypointHeadingShape(shaftWidth, headWidth, headLength) {
  const start = WAYPOINT_CENTER_RADIUS * 0.32;
  const tip = WAYPOINT_HEADING_LENGTH;
  const headBase = Math.max(start, tip - headLength);
  const shaftHalf = shaftWidth / 2;
  const headHalf = headWidth / 2;
  const shape = new THREE.Shape();
  shape.moveTo(start, -shaftHalf);
  shape.lineTo(headBase, -shaftHalf);
  shape.lineTo(headBase, -headHalf);
  shape.lineTo(tip, 0);
  shape.lineTo(headBase, headHalf);
  shape.lineTo(headBase, shaftHalf);
  shape.lineTo(start, shaftHalf);
  shape.closePath();
  return shape;
}
export function makeWaypointHeadingArrow() {
  const group = new THREE.Group();
  const arrow = new THREE.Mesh(new THREE.ShapeGeometry(waypointHeadingShape(WAYPOINT_HEADING_SHAFT_WIDTH, WAYPOINT_HEADING_HEAD_WIDTH, WAYPOINT_HEADING_HEAD_LENGTH)), new THREE.MeshBasicMaterial({
    color: 0x111827,
    transparent: true,
    opacity: 0.94,
    side: THREE.DoubleSide
  }));
  arrow.position.z = 0.04;
  group.add(arrow);
  // The visible arrow remains unchanged. A transparent target around its tip
  // makes rotation easy to grab without stealing drag input from the marker
  // center. It also reaches slightly beneath the fixed waypoint label.
  const hitArea = new THREE.Mesh(new THREE.CircleGeometry(WAYPOINT_ROTATE_HIT_RADIUS, 24), new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    colorWrite: false,
    depthWrite: false,
    side: THREE.DoubleSide
  }));
  hitArea.position.set(WAYPOINT_HEADING_LENGTH + WAYPOINT_ROTATE_HIT_OFFSET, 0, 0.06);
  hitArea.userData.waypointRotateHitArea = true;
  group.add(hitArea);
  return group;
}
export function makePoseMarker(pose, color, z) {
  var _a, _b, _c, _d;
  const group = new THREE.Group();
  const x = Number(((_b = (_a = pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null) && (_b !== void 0) ? _b : 0);
  const y = Number(((_d = (_c = pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null) && (_d !== void 0) ? _d : 0);
  const yaw = yawFromPose(pose);
  group.position.set(x, y, z);
  group.rotation.z = yaw;
  const body = new THREE.Mesh(new THREE.CircleGeometry(0.13, 24), new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.86
  }));
  group.add(body);
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(0.26, 0);
  arrowShape.lineTo(-0.1, 0.13);
  arrowShape.lineTo(-0.04, 0);
  arrowShape.lineTo(-0.1, -0.13);
  arrowShape.closePath();
  const arrow = new THREE.Mesh(new THREE.ShapeGeometry(arrowShape), new THREE.MeshBasicMaterial({
    color: markerPalette.poseArrow
  }));
  arrow.position.z = 0.01;
  group.add(arrow);
  return group;
}
export function makeSpotMarker(spot, selected = false, scale = 1, active = false, palette = markerPalette, showLabel = true) {
  var _a, _b, _c, _d;
  const pose = (spot === null) || (spot === void 0) ? void 0 : spot.pose;
  if (!pose) return null;
  const x = Number(((_a = pose.x) !== null) && (_a !== void 0) ? _a : 0);
  const y = Number(((_b = pose.y) !== null) && (_b !== void 0) ? _b : 0);
  const yaw = Number(((_c = pose.yaw) !== null) && (_c !== void 0) ? _c : 0);
  const pal = palette;
  const color = active ? pal.selected : selected ? pal.selected : pal.idle;
  const group = new THREE.Group();
  // Flush with the map plane: just enough lift to avoid z-fighting, so the
  // marker never parallax-floats off its true position under the
  // perspective camera.
  group.position.set(x, y, 0.01);
  group.scale.setScalar(scale);
  group.userData = {
    spotId: spot.id,
    dragAction: "move",
    waypointHitAnchor: true
  };
  const marker = new THREE.Group();
  marker.rotation.z = yaw;
  marker.userData = {
    spotId: spot.id,
    dragAction: "move"
  };
  // Fill the complete visual footprint with one invisible pointer target.
  // The visible center and ring otherwise leave a dead annulus between them,
  // which makes small waypoints unnecessarily difficult to select.
  const bodyHitArea = new THREE.Mesh(new THREE.CircleGeometry(WAYPOINT_BODY_HIT_RADIUS, 32), new THREE.MeshBasicMaterial({
    transparent: true,
    opacity: 0,
    colorWrite: false,
    depthWrite: false,
    side: THREE.DoubleSide
  }));
  bodyHitArea.position.z = 0.005;
  bodyHitArea.userData = {
    spotId: spot.id,
    dragAction: "move",
    waypointBodyHitArea: true
  };
  marker.add(bodyHitArea);
  if (active) {
    // Pulsing halo for the waypoint the mission is currently working on;
    // the animation loop drives opacity + scale from userData.pulse.
    const pulseBase = 0.5;
    const pulse = new THREE.Mesh(new THREE.RingGeometry(WAYPOINT_SELECTED_HALO_OUTER_RADIUS + 0.5, WAYPOINT_SELECTED_HALO_OUTER_RADIUS + 1.5, 44), new THREE.MeshBasicMaterial({
      color: pal.selected,
      transparent: true,
      opacity: pulseBase,
      side: THREE.DoubleSide
    }));
    pulse.userData = {
      spotId: spot.id,
      dragAction: "move",
      pulse: true,
      pulseBase
    };
    marker.add(pulse);
  }
  if (selected || active) {
    const halo = new THREE.Mesh(new THREE.RingGeometry(WAYPOINT_SELECTED_HALO_INNER_RADIUS, WAYPOINT_SELECTED_HALO_OUTER_RADIUS, 40), new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.34,
      side: THREE.DoubleSide
    }));
    halo.userData = {
      spotId: spot.id,
      dragAction: "move"
    };
    marker.add(halo);
  }
  const ring = new THREE.Mesh(new THREE.RingGeometry(WAYPOINT_RING_INNER_RADIUS, WAYPOINT_RING_OUTER_RADIUS, 40), new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.95,
    side: THREE.DoubleSide
  }));
  ring.userData = {
    spotId: spot.id,
    dragAction: "move"
  };
  marker.add(ring);
  const center = new THREE.Mesh(new THREE.CircleGeometry(WAYPOINT_CENTER_RADIUS, 32), new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: selected ? 0.95 : 0.78
  }));
  center.position.z = 0.01;
  center.userData = {
    spotId: spot.id,
    dragAction: "move"
  };
  marker.add(center);
  const heading = makeWaypointHeadingArrow();
  heading.traverse(child => {
    child.userData = {
      ...child.userData,
      spotId: spot.id,
      dragAction: "rotate"
    };
  });
  marker.add(heading);
  group.add(marker);
  const label = String(((_d = spot.label) !== null) && (_d !== void 0) ? _d : spot.id);
  if (label && showLabel) {
    const {
      sprite,
      aspect
    } = makeWaypointLabelSprite(label, {
      selected
    });
    // Anchor the sprite at the waypoint and shift its visual center in
    // screen space. Unlike a world-Y position offset, this stays directly
    // above the waypoint when the camera rolls with a rotated map.
    sprite.position.set(0, 0, 0.04);
    sprite.scale.set(WAYPOINT_LABEL_SCALE_Y * aspect, WAYPOINT_LABEL_SCALE_Y, 1); // auto width, no squish
    sprite.center.set(0.5, 0.5 - (WAYPOINT_LABEL_OFFSET_Y / WAYPOINT_LABEL_SCALE_Y));
    sprite.userData = {
      spotId: spot.id,
      dragAction: "move",
      waypointHitLabel: true
    };
    // Keep the full name pill interactive. The dedicated rotate target sits
    // above it on the z-axis, so overlapping arrow input still wins.
    group.add(sprite);
  }
  return group;
}
