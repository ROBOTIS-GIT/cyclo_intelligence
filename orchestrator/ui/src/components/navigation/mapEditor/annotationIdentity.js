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

import { ANNOTATION_COLORS } from './constants';
export let annotationIdCounter = 0;
export function makeAnnotationId() {
  annotationIdCounter += 1;
  return `mark_${Date.now().toString(36)}_${annotationIdCounter.toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
export function normalizeAnnotationColor(value) {
  const color = String(value || "").trim();
  return /^#[0-9A-Fa-f]{6}$/.test(color) ? color.toUpperCase() : ANNOTATION_COLORS[0].value;
}
export function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs(hp % 2 - 1));
  const [r1, g1, b1] = hp < 1 ? [c, x, 0] : hp < 2 ? [x, c, 0] : hp < 3 ? [0, c, x] : hp < 4 ? [0, x, c] : hp < 5 ? [x, 0, c] : [c, 0, x];
  const m = l - (c / 2);
  const toHex = value => Math.round((value + m) * 255).toString(16).padStart(2, "0").toUpperCase();
  return `#${toHex(r1)}${toHex(g1)}${toHex(b1)}`;
}
export const SAFE_ANNOTATION_HUE_SPANS = [[0, 15], [170, 360]];
export const SAFE_ANNOTATION_HUE_TOTAL = SAFE_ANNOTATION_HUE_SPANS.reduce((total, [start, end]) => total + (end - start), 0);
export function safeAnnotationHue(seed) {
  let t = (seed % SAFE_ANNOTATION_HUE_TOTAL + SAFE_ANNOTATION_HUE_TOTAL) % SAFE_ANNOTATION_HUE_TOTAL;
  for (const [start, end] of SAFE_ANNOTATION_HUE_SPANS) {
    const length = end - start;
    if (t < length) return start + t;
    t -= length;
  }
  return SAFE_ANNOTATION_HUE_SPANS[SAFE_ANNOTATION_HUE_SPANS.length - 1][0];
}
export function chooseAnnotationColor(existingAnnotations) {
  const used = new Set((existingAnnotations || []).map(annotation => normalizeAnnotationColor(annotation.color)));
  const available = ANNOTATION_COLORS.filter(color => !used.has(color.value));
  if (available.length) {
    return available[Math.floor(Math.random() * available.length)].value;
  }
  for (let attempt = 0; attempt < 48; attempt += 1) {
    const hue = safeAnnotationHue((Math.random() * SAFE_ANNOTATION_HUE_TOTAL) + (used.size * 137.508) + (attempt * 29));
    const generated = hslToHex(hue, 0.52, 0.32);
    if (!used.has(generated)) return generated;
  }
  return hslToHex(safeAnnotationHue(used.size * 137.508), 0.52, 0.32);
}
export function nextAutoAreaLabel(annotations) {
  let max = 0;
  (annotations || []).forEach(annotation => {
    const match = /^Area (\d+)$/.exec(String(((annotation === null) || (annotation === void 0) ? void 0 : annotation.label) || "").trim());
    if (match) max = Math.max(max, Number(match[1]));
  });
  return `Area ${max + 1}`;
}
