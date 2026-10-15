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

import { gridMeta, mapAreaPreviewCellIndices, mapPointToGridCell, boundedFreeRegion, gridCellToMapPoint } from './gridGeometry';
import { OCC_ROUNDED, drawRoundedOccupancyLayer } from './occupancyDrawing';
import { rgbaArrayFromHex, pastelizeHexColor, hexColorString, makeAnnotationLabelSprite } from './labelSprites';
import * as THREE from 'three';
import { mapAreaGridMeta } from '../../../utils/mapAreaGeometry';
import { makeGridTexturePlane } from './sceneObjects';
export function makeAnnotationRegionTexture(grid, region, colorString, alpha = 164) {
  const meta = gridMeta(grid);
  if (!meta || !region) return null;
  const scale = Math.max(1, Math.min(OCC_ROUNDED.scale, Math.floor(OCC_ROUNDED.maxDim / Math.max(meta.width, meta.height)) || 1));
  const mask = new Uint8Array(meta.width * meta.height);
  region.cells.forEach(index => {
    const gridX = index % meta.width;
    const gridY = Math.floor(index / meta.width);
    const canvasX = meta.width - 1 - gridX;
    const canvasY = meta.height - 1 - gridY;
    mask[canvasX + (canvasY * meta.width)] = 1;
  });
  const canvas = document.createElement("canvas");
  canvas.width = meta.width * scale;
  canvas.height = meta.height * scale;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  drawRoundedOccupancyLayer(ctx, mask, meta.width, meta.height, scale, 1, rgbaArrayFromHex(colorString, alpha));
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.generateMipmaps = false;
  texture.flipY = false;
  texture.needsUpdate = true;
  return texture;
}
export function makeEditorAreaPreview(selection, grid, excludedCells = null) {
  const meta = mapAreaGridMeta(grid);
  if (!meta) return null;
  const cells = mapAreaPreviewCellIndices(grid, selection, excludedCells);
  if (!cells.length) return null;

  // Merge adjacent cells into one quad per horizontal run. This shows the
  // exact free, unclaimed cells that will be saved without rebuilding a
  // full-map canvas texture on every pointer move.
  const positions = [];
  const indices = [];
  const originCos = Math.cos(meta.originYaw);
  const originSin = Math.sin(meta.originYaw);
  const addVertex = (cellX, cellY) => {
    const localX = cellX * meta.resolution;
    const localY = cellY * meta.resolution;
    positions.push((meta.originX + originCos * localX) - (originSin * localY), meta.originY + (originSin * localX) + (originCos * localY), 0.82);
  };
  const addRun = (y, xStart, xEnd) => {
    const vertex = positions.length / 3;
    addVertex(xStart, y);
    addVertex(xEnd + 1, y);
    addVertex(xEnd + 1, y + 1);
    addVertex(xStart, y + 1);
    indices.push(vertex, vertex + 1, vertex + 2, vertex, vertex + 2, vertex + 3);
  };
  let runY = -1;
  let runStart = -1;
  let runEnd = -1;
  const flushRun = () => {
    if (runStart >= 0) addRun(runY, runStart, runEnd);
  };
  cells.forEach(index => {
    const x = index % meta.width;
    const y = Math.floor(index / meta.width);
    if ((y !== runY) || (x !== runEnd + 1)) {
      flushRun();
      runY = y;
      runStart = x;
    }
    runEnd = x;
  });
  flushRun();
  if (!positions.length) return null;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingSphere();
  return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: 0x6d1f2a,
    transparent: true,
    opacity: 0.3,
    depthWrite: false,
    side: THREE.DoubleSide
  }));
}
export const annotationTextureCache = new WeakMap();
export function cachedAnnotationRegionTexture(annotation, grid, region, colorString, selected) {
  const hit = annotationTextureCache.get(annotation);
  if (hit && (hit.grid === grid) && (hit.selected === selected)) {
    return hit.texture;
  }
  const texture = makeAnnotationRegionTexture(grid, region, colorString, selected ? 210 : 164);
  if (texture) {
    texture.userData.retain = true;
    if (((hit === null) || (hit === void 0) ? void 0 : hit.texture) && (hit.texture !== texture)) hit.texture.dispose();
    annotationTextureCache.set(annotation, {
      grid,
      selected,
      texture
    });
  }
  return texture;
}
export function makeMapAnnotationRegion(annotation, grid, coveredCells = null, selected = false) {
  var _a, _b, _c, _d;
  const meta = gridMeta(grid);
  if (!meta) return null;
  const pose = ((_a = annotation === null || annotation === void 0 ? void 0 : annotation.pose) !== null) && (_a !== void 0) ? _a : {};
  const fallbackCell = mapPointToGridCell(meta, Number(((_b = pose.x) !== null) && (_b !== void 0) ? _b : NaN), Number(((_c = pose.y) !== null) && (_c !== void 0) ? _c : NaN));
  const regionSpec = ((_d = annotation === null || annotation === void 0 ? void 0 : annotation.region) !== null) && (_d !== void 0) ? _d : fallbackCell ? {
    seed_cell: fallbackCell
  } : null;
  const region = boundedFreeRegion(grid, regionSpec, coveredCells);
  if (!region) return null;
  // Pastelize whatever color is stored: legacy annotations carry the old
  // deep palette, and areas must stay a light background wash either way.
  const colorString = pastelizeHexColor(hexColorString((annotation === null) || (annotation === void 0) ? void 0 : annotation.color, "#6D1F2A"));
  const texture = cachedAnnotationRegionTexture(annotation, grid, region, colorString, selected);
  // Flush with the map, and strictly BELOW the waypoint/route layer
  // (0.01-0.05) so labeled areas never cover mission markers.
  const plane = makeGridTexturePlane(grid, texture, 0.004);
  if (!plane) return null;
  const group = new THREE.Group();
  group.add(plane);
  if (coveredCells) {
    region.cells.forEach(index => coveredCells.add(index));
  }
  const label = String(((annotation === null) || (annotation === void 0) ? void 0 : annotation.label) || "Area").trim();
  if (label) {
    const {
      sprite,
      aspect
    } = makeAnnotationLabelSprite(label, colorString);
    const center = gridCellToMapPoint(meta, region.centroid.x, region.centroid.y);
    const labelHeight = Math.max(0.24, Math.min(0.72, meta.resolution * 11));
    sprite.position.set(center.x, center.y, 0.008);
    sprite.scale.set(labelHeight * aspect, labelHeight, 1);
    group.add(sprite);
  }
  return group;
}
