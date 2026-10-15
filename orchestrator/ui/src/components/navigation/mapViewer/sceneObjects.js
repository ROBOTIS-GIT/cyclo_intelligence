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
import { gridMeta } from './gridGeometry';
import { makeOccupancyTexture } from './occupancyTexture';
import { yawFromPose } from '../../../utils/navigationTf';
import { MAP_DISPLAY_ROTATION } from './constants';
export function disposeObject(object) {
  object.traverse(child => {
    var _a;
    if ((child instanceof THREE.Mesh) || (child instanceof THREE.Points) || (child instanceof THREE.Line) || (child instanceof THREE.Sprite)) {
      ((_a = child.geometry) === null) || (_a === void 0) ? void 0 : _a.dispose();
      const material = child.material;
      if (Array.isArray(material)) {
        material.forEach(item => {
          const texture = item.map;
          // Textures flagged `retain` are cached across layer rebuilds
          // (annotation regions) and disposed by their cache instead.
          if (texture && !texture.userData.retain) texture.dispose();
          item.dispose();
        });
      } else {
        const texture = (material === null) || (material === void 0) ? void 0 : material.map;
        if (texture && !texture.userData.retain) texture.dispose();
        (material === null) || (material === void 0) ? void 0 : material.dispose();
      }
    }
  });
}
export function makeGridPlane(grid, mode, z, framePose = null, highlightedCells = null, refined = false) {
  var _a, _b, _c, _d;
  const meta = gridMeta(grid);
  const texture = makeOccupancyTexture(grid, mode === "map" ? 255 : 170, mode, highlightedCells, refined);
  if (!meta || !texture) return null;
  const width = meta.width * meta.resolution;
  const height = meta.height * meta.resolution;
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: (mode !== "map") || refined,
    opacity: mode === "map" ? 1 : 0.82,
    depthWrite: mode === "map",
    side: THREE.DoubleSide
  });
  const mesh = new THREE.Mesh(geometry, material);
  const originYaw = meta.originYaw;
  const originCos = Math.cos(originYaw);
  const originSin = Math.sin(originYaw);
  const gridCenterX = (meta.originX + originCos * (width / 2)) - (originSin * (height / 2));
  const gridCenterY = meta.originY + (originSin * (width / 2)) + (originCos * (height / 2));
  const frameYaw = framePose ? yawFromPose(framePose) : 0;
  const frameCos = Math.cos(frameYaw);
  const frameSin = Math.sin(frameYaw);
  const frameX = Number(((_b = (_a = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _a === void 0 ? void 0 : _a.x) !== null) && (_b !== void 0) ? _b : 0);
  const frameY = Number(((_d = (_c = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _c === void 0 ? void 0 : _c.y) !== null) && (_d !== void 0) ? _d : 0);
  mesh.position.set((frameX + frameCos * gridCenterX) - (frameSin * gridCenterY), frameY + (frameSin * gridCenterX) + (frameCos * gridCenterY), z);
  mesh.rotation.z = frameYaw + originYaw + MAP_DISPLAY_ROTATION;
  mesh.userData.mapTexture = texture;
  return mesh;
}
export function makeGridTexturePlane(grid, texture, z) {
  const meta = gridMeta(grid);
  if (!meta || !texture) return null;
  const width = meta.width * meta.resolution;
  const height = meta.height * meta.resolution;
  const geometry = new THREE.PlaneGeometry(width, height);
  const material = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity: 1,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  const mesh = new THREE.Mesh(geometry, material);
  const originYaw = meta.originYaw;
  const originCos = Math.cos(originYaw);
  const originSin = Math.sin(originYaw);
  const gridCenterX = (meta.originX + originCos * (width / 2)) - (originSin * (height / 2));
  const gridCenterY = meta.originY + (originSin * (width / 2)) + (originCos * (height / 2));
  mesh.position.set(gridCenterX, gridCenterY, z);
  mesh.rotation.z = originYaw + MAP_DISPLAY_ROTATION;
  mesh.userData.mapTexture = texture;
  mesh.userData.annotation = true;
  return mesh;
}
export function makeLine(points, color, lineWidth = 2) {
  if (points.length < 2) return null;
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color,
    linewidth: lineWidth
  });
  return new THREE.Line(geometry, material);
}
