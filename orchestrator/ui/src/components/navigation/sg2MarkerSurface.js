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

// Bake a soft studio light into the small map mesh once. No scene lights,
// shadow maps, textures, or shading work are needed when the robot moves.
const LIGHT = new THREE.Vector2(0.55, 0.83).normalize();
const WHITE = new THREE.Color("#ffffff");
const SHADOW = new THREE.Color("#30302e");
const centerOf = points => points.reduce(([x, y], p) => [x + p[0] / points.length, y + p[1] / points.length], [0, 0]);
const insetRing = (points, center, scale) => points.map(([x, y]) => [center[0] + (x - center[0]) * scale, center[1] + (y - center[1]) * scale]);

function buffer(alpha = false) {
  const positions = [], colors = [];
  return {
    triangle(points, shades, z) {
      points.forEach(([x, y], i) => {
        positions.push(x, y, z);
        const { color, opacity = 1 } = shades[i];
        colors.push(color.r, color.g, color.b);
        if (alpha) colors.push(opacity);
      });
    },
    geometry() {
      const result = new THREE.BufferGeometry();
      result.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      result.setAttribute("color", new THREE.Float32BufferAttribute(colors, alpha ? 4 : 3));
      result.computeBoundingSphere();
      return result;
    },
  };
}

function bevelPart(surface, { name, points, color, inset }, z) {
  const center = centerOf(points);
  const base = new THREE.Color(color);
  const dark = Math.max(base.r, base.g, base.b) < 0.2;
  const detail = name.startsWith("camera-") || name === "front-bumper";
  const ring = insetRing(points, center, detail ? 0.88 : Math.min(inset, dark ? 0.90 : 0.95));
  const lightAt = ([x, y]) => {
    const dx = x - center[0], dy = y - center[1];
    return (dx * LIGHT.x + dy * LIGHT.y) / (Math.hypot(dx, dy) || 1);
  };
  const faceShade = p => ({ color: base.clone().multiplyScalar(0.86 + 0.10 * lightAt(p)) });
  const crown = { color: base.clone().lerp(WHITE, dark ? 0.022 : 0.13) };
  for (let i = 0; i < points.length; i += 1) {
    const j = (i + 1) % points.length;
    const a = points[i], b = points[j];
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const diffuse = (dy * LIGHT.x - dx * LIGHT.y) / (Math.hypot(dx, dy) || 1);
    const outer = { color: base.clone().multiplyScalar(0.40 + 0.22 * Math.max(0, diffuse)) };
    const rim = { color: base.clone().multiplyScalar(0.76 + 0.21 * diffuse)
      .lerp(WHITE, Math.max(0, diffuse) * (dark ? 0.065 : 0.18)) };
    surface.triangle([a, b, ring[j]], [outer, outer, rim], z);
    surface.triangle([a, ring[j], ring[i]], [outer, rim, rim], z);
    // These authored silhouettes are convex; a fan adds a smooth raised crown
    // instead of the broad, uniformly filled panels of the old icon.
    surface.triangle([center, ring[i], ring[j]], [crown, faceShade(ring[i]), faceShade(ring[j])], z + 0.0001);
  }
}

function contactShadow(shadow, points, z, ground = false) {
  const center = centerOf(points);
  const shifted = points.map(([x, y]) => [x - (ground ? 0.014 : 0.008), y - (ground ? 0.014 : 0.008)]);
  const c = centerOf(shifted);
  const scales = ground ? [0.94, 1.02, 1.09, 1.16] : [0.86, 1.0, 1.08, 1.15];
  const opacity = ground ? [0.20, 0.13, 0.05, 0] : [0.23, 0.17, 0.06, 0];
  const shade = n => ({ color: SHADOW, opacity: opacity[n] });
  const rings = scales.map(scale => insetRing(shifted, c, scale));
  for (let i = 0; i < points.length; i += 1) {
    const j = (i + 1) % points.length;
    shadow.triangle([center, rings[0][i], rings[0][j]], [shade(0), shade(0), shade(0)], z);
    for (let n = 0; n < rings.length - 1; n += 1) {
      shadow.triangle([rings[n][i], rings[n + 1][i], rings[n + 1][j]], [shade(n), shade(n + 1), shade(n + 1)], z);
      shadow.triangle([rings[n][i], rings[n + 1][j], rings[n][j]], [shade(n), shade(n + 1), shade(n)], z);
    }
  }
}

export function createSg2Surfaces(parts) {
  const surface = buffer(), shadow = buffer(true);
  parts.forEach((part, layer) => {
    const z = layer * 0.002;
    bevelPart(surface, part, z);
    if (["base", "lift", "torso", "head"].includes(part.name) || /-arm-[246]$/.test(part.name)) {
      contactShadow(shadow, part.points, z - 0.0005, part.name === "base");
    }
  });
  return { surface: surface.geometry(), shadow: shadow.geometry() };
}
