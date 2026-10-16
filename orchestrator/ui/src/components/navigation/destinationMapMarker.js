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

export function destinationLabel(destination) {
  const label = String(destination?.label || destination?.id || "Destination").trim();
  return label.length > 24 ? `${label.slice(0, 23)}…` : label;
}

// A billboard keeps the flag and name upright when the user rotates the map.
// Its anchor is the execution waypoint itself, with the same screen-space
// clearance as the ordinary waypoint labels. It has no pointer hit target.
export function createDestinationMapMarker(destination, scale = 1) {
  if (!destination?.pose) return null;
  const label = destinationLabel(destination);
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const font = '700 40px "Nunito", "Hanken Grotesk", sans-serif';
  ctx.font = font;
  canvas.width = Math.ceil(ctx.measureText(label).width) + 106;
  canvas.height = 86;
  ctx.font = font;
  ctx.beginPath();
  ctx.roundRect(2, 2, canvas.width - 4, 72, 24);
  ctx.fillStyle = "#fff8ed";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = "#b96447";
  ctx.stroke();
  // Short pennant and pole, with a small point tying the badge to the waypoint.
  ctx.beginPath();
  ctx.moveTo(27, 24);
  ctx.lineTo(52, 24);
  ctx.lineTo(44, 34);
  ctx.lineTo(52, 44);
  ctx.lineTo(27, 44);
  ctx.closePath();
  ctx.fillStyle = "#c67551";
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(26, 22);
  ctx.lineTo(26, 56);
  ctx.strokeStyle = "#87553f";
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(canvas.width / 2 - 8, 73);
  ctx.lineTo(canvas.width / 2, 83);
  ctx.lineTo(canvas.width / 2 + 8, 73);
  ctx.fillStyle = "#b96447";
  ctx.fill();
  ctx.fillStyle = "#75442f";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(label, 65, 39);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture, transparent: true, depthTest: false, depthWrite: false,
  }));
  const height = 7 * scale;
  sprite.position.set(destination.pose.x, destination.pose.y, 0.05);
  sprite.scale.set(height * canvas.width / canvas.height, height, 1);
  sprite.center.set(0.5, -0.8);
  sprite.renderOrder = 20;
  sprite.userData = { destinationId: destination.id, destinationScope: destination.scope };
  sprite.raycast = () => {};
  return sprite;
}
