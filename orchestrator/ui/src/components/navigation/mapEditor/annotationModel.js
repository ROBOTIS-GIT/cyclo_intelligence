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

import { makeAnnotationId, normalizeAnnotationColor } from './annotationIdentity';
export function normalizeRegionCells(cells, width = null, height = null) {
  if (!Array.isArray(cells)) return [];
  const seen = new Set();
  const normalized = [];
  cells.forEach(cell => {
    if (!cell) return;
    const x = Math.floor(Number(cell.x));
    const y = Math.floor(Number(cell.y));
    if (!Number.isFinite(x) || !Number.isFinite(y) || (x < 0) || (y < 0)) return;
    if (width && (x >= width)) return;
    if (height && (y >= height)) return;
    const key = `${x}:${y}`;
    if (seen.has(key)) return;
    seen.add(key);
    normalized.push({
      x,
      y
    });
  });
  normalized.sort((a, b) => (a.y - b.y) || (a.x - b.x));
  return normalized;
}
export function normalizeAnnotationRegion(region) {
  var _a, _b, _c, _d, _e;
  const seed = ((_a = region === null || region === void 0 ? void 0 : region.seed_cell) !== null) && (_a !== void 0) ? _a : {};
  const x = Number(((_b = seed.x) !== null) && (_b !== void 0) ? _b : NaN);
  const y = Number(((_c = seed.y) !== null) && (_c !== void 0) ? _c : NaN);
  if (!Number.isFinite(x) || !Number.isFinite(y) || (x < 0) || (y < 0)) return null;
  const normalized = {
    seed_cell: {
      x: Math.floor(x),
      y: Math.floor(y)
    },
    cell_count: Number.isFinite(Number((region === null) || (region === void 0) ? void 0 : region.cell_count)) ? Math.max(0, Math.floor(Number(region.cell_count))) : null,
    width: Number.isFinite(Number(((_d = region === null || region === void 0 ? void 0 : region.width) !== null) && (_d !== void 0) ? _d : NaN)) ? Math.max(0, Math.floor(Number(region.width))) : null,
    height: Number.isFinite(Number(((_e = region === null || region === void 0 ? void 0 : region.height) !== null) && (_e !== void 0) ? _e : NaN)) ? Math.max(0, Math.floor(Number(region.height))) : null
  };
  const bounds = (region === null) || (region === void 0) ? void 0 : region.bounds;
  if (bounds) {
    const xMin = Math.floor(Number(bounds.x_min));
    const yMin = Math.floor(Number(bounds.y_min));
    const xMax = Math.floor(Number(bounds.x_max));
    const yMax = Math.floor(Number(bounds.y_max));
    if ([xMin, yMin, xMax, yMax].every(Number.isFinite)) {
      normalized.bounds = {
        x_min: Math.max(0, Math.min(xMin, xMax)),
        y_min: Math.max(0, Math.min(yMin, yMax)),
        x_max: Math.max(0, Math.max(xMin, xMax)),
        y_max: Math.max(0, Math.max(yMin, yMax))
      };
    }
  }
  const cells = normalizeRegionCells((region === null) || (region === void 0) ? void 0 : region.cells, normalized.width, normalized.height);
  if (cells.length) {
    normalized.cells = cells;
    normalized.cell_count = cells.length;
  }
  return normalized;
}
export function normalizeAnnotation(annotation) {
  var _a, _b, _c, _d, _e, _f;
  const pose = ((_a = annotation === null || annotation === void 0 ? void 0 : annotation.pose) !== null) && (_a !== void 0) ? _a : {};
  const label = String(((_b = annotation === null || annotation === void 0 ? void 0 : annotation.label) !== null) && (_b !== void 0) ? _b : "Area").trim() || "Area";
  const normalized = {
    id: String(((_c = annotation === null || annotation === void 0 ? void 0 : annotation.id) !== null) && (_c !== void 0) ? _c : makeAnnotationId()),
    label: label.slice(0, 80),
    color: normalizeAnnotationColor((annotation === null) || (annotation === void 0) ? void 0 : annotation.color),
    pose: {
      frame_id: String(((_d = pose.frame_id) !== null) && (_d !== void 0) ? _d : "map"),
      x: Number(((_e = pose.x) !== null) && (_e !== void 0) ? _e : 0),
      y: Number(((_f = pose.y) !== null) && (_f !== void 0) ? _f : 0),
      yaw: Number((pose.yaw !== null) && (pose.yaw !== void 0) ? pose.yaw : 0)
    }
  };
  const region = normalizeAnnotationRegion((annotation === null) || (annotation === void 0) ? void 0 : annotation.region);
  if (region) normalized.region = region;
  return normalized;
}
