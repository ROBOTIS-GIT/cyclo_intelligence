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

import * as THREE from 'three';
import { markerPalette, WAYPOINT_RING_OUTER_RADIUS, WAYPOINT_HEADING_LENGTH } from './constants';
export function makeMissionRouteLine(points, color, scale) {
  if (points.length < 2) return null;
  const group = new THREE.Group();
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineDashedMaterial({
    color,
    dashSize: 3 * scale,
    gapSize: 2 * scale,
    transparent: true,
    opacity: 0.72
  });
  const line = new THREE.Line(geometry, material);
  // LineDashedMaterial renders solid until per-vertex distances exist.
  line.computeLineDistances();
  group.add(line);
  const arrowLength = 3.2 * scale;
  const arrowHalfWidth = 1.8 * scale;
  const arrowShape = new THREE.Shape();
  arrowShape.moveTo(arrowLength / 2, 0);
  arrowShape.lineTo(-arrowLength / 2, arrowHalfWidth);
  arrowShape.lineTo(-arrowLength / 2, -arrowHalfWidth);
  arrowShape.closePath();
  const arrowGeometry = new THREE.ShapeGeometry(arrowShape);
  for (let i = 0; i < (points.length - 1); i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    // Legs too short to fit an arrowhead between the endpoint markers
    // stay dash-only.
    if (Math.hypot(dx, dy) < (arrowLength * 2.5)) continue;
    const arrow = new THREE.Mesh(arrowGeometry, new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide
    }));
    arrow.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, 0.025);
    arrow.rotation.z = Math.atan2(dy, dx);
    group.add(arrow);
  }
  return group;
}
export function makeRouteBadgeSprite(text, selected = false, palette = markerPalette) {
  const pal = palette;
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(64, 64, 52, 0, Math.PI * 2);
    ctx.fillStyle = selected ? `#${pal.selected.toString(16).padStart(6, "0")}` : `#${pal.idle.toString(16).padStart(6, "0")}`;
    ctx.fill();
    ctx.lineWidth = 6;
    ctx.strokeStyle = pal.badgeStroke;
    ctx.stroke();
    ctx.fillStyle = selected ? pal.badgeTextSelected : pal.badgeTextIdle;
    ctx.font = '700 56px "IBM Plex Mono", ui-monospace, monospace';
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(text), 64, 68);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true
  }));
}
export function makeMissionRouteBadge(spot, order, selected = false, scale = 1, palette = markerPalette) {
  var _a, _b;
  const pose = (spot === null) || (spot === void 0) ? void 0 : spot.pose;
  if (!pose) return null;
  const x = Number(((_a = pose.x) !== null) && (_a !== void 0) ? _a : 0);
  const y = Number(((_b = pose.y) !== null) && (_b !== void 0) ? _b : 0);
  const sprite = makeRouteBadgeSprite(order, selected, palette);
  const markerRadius = WAYPOINT_RING_OUTER_RADIUS * scale;
  const arrowRadius = WAYPOINT_HEADING_LENGTH * scale;
  const size = Math.max(0.34, markerRadius * 2.2);
  const clearance = Math.max(0.04, scale * 0.5);
  const radialOffset = Math.max(markerRadius, arrowRadius) + (size / 2) + clearance;
  const diagonalOffset = radialOffset / Math.SQRT2;
  // Keep route order at a stable screen-space bottom-left position. The
  // waypoint name stays above, while the radial clearance prevents overlap
  // with both the marker ring and a heading arrow pointed toward the badge.
  sprite.position.set(x, y, 0.05);
  sprite.scale.set(size, size, 1);
  sprite.center.set(0.5 + (diagonalOffset / size), 0.5 + (diagonalOffset / size));
  sprite.userData = {
    spotId: spot.id,
    dragAction: "move",
    missionRouteBadge: true
  };
  return sprite;
}
