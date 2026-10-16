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

// Camera lifetime follows reasoning, including paused mission supervision.
export function thoughtRunning(state = {}) {
  const terminal = ["idle", "stopping", "stopped", "error", "failed", "interrupted"];
  if (terminal.includes(state.phase) || terminal.includes(state.execution?.lifecycle)) return false;
  return !!state.active || ["starting", "loading"].includes(state.phase) || ["starting", "active"].includes(state.execution?.lifecycle);
}

const clamp = (value, min, max) => Math.max(min, Math.min(value, max));

// Prefer the space above the robot. Near an edge, use a clear side instead of
// clamping the cloud on top of the robot. All dimensions are viewport CSS pixels.
export function thoughtPlacement(anchor, width, height, previousSide) {
  if (!anchor || !(anchor.width > 0) || !(anchor.height > 0)) return null;
  const gap = 44;
  const viewport = { left: 16, top: Math.min(88, anchor.height * .15),
    right: Math.max(17, anchor.width - 80), bottom: Math.max(17, anchor.height - 16) };
  const robot = anchor.robotBounds || { left: anchor.x - 24, right: anchor.x + 24,
    top: anchor.y, bottom: anchor.y + 48 };
  const center = { x: (robot.left + robot.right) / 2, y: (robot.top + robot.bottom) / 2 };
  const wantedWidth = Math.min(width, viewport.right - viewport.left);
  const wantedHeight = Math.min(height, viewport.bottom - viewport.top);
  const areas = [
    { side: "above", ...viewport, bottom: Math.min(viewport.bottom, robot.top - gap), bias: 0 },
    { side: "left", ...viewport, right: Math.min(viewport.right, robot.left - gap), bias: 30 },
    { side: "right", ...viewport, left: Math.max(viewport.left, robot.right + gap), bias: 30 },
    { side: "below", ...viewport, top: Math.max(viewport.top, robot.bottom + gap), bias: 55 },
  ];
  const candidates = areas.filter(area => area.right - area.left >= 148 && area.bottom - area.top >= 160)
    .map(area => {
      const w = Math.min(wantedWidth, area.right - area.left);
      const h = Math.min(wantedHeight, area.bottom - area.top);
      let x = clamp(center.x - w / 2, area.left, area.right - w);
      let y = clamp(center.y - h / 2, area.top, area.bottom - h);
      if (area.side === "above") y = area.bottom - h;
      if (area.side === "below") y = area.top;
      if (area.side === "left") x = area.right - w;
      if (area.side === "right") x = area.left;
      return { x, y, width: w, height: h, side: area.side,
        cost: (wantedWidth - w) * .7 + (wantedHeight - h) * 1.6 + area.bias
          - (previousSide === area.side ? 28 : 0) };
    }).sort((a, b) => a.cost - b.cost);
  // Very small map viewports cannot fit both the camera and marker. Dock the
  // content inside the viewport and omit a misleading connector in that case.
  if (!candidates.length) return { x: viewport.left, y: viewport.top,
    width: wantedWidth, height: wantedHeight, side: "docked", circles: [] };
  const result = candidates[0];
  const horizontal = result.side === "left" || result.side === "right";
  const end = horizontal
    ? { x: result.side === "left" ? robot.left - 10 : robot.right + 10, y: center.y }
    : { x: center.x, y: result.side === "above" ? robot.top - 10 : robot.bottom + 10 };
  const start = horizontal
    ? { x: result.side === "left" ? result.x + result.width : result.x,
      y: clamp(center.y, result.y + 32, result.y + result.height - 32) }
    : { x: clamp(center.x, result.x + 32, result.x + result.width - 32),
      y: result.side === "above" ? result.y + result.height : result.y };
  return { ...result, circles: [11, 7, 4].map((r, index) => {
    const t = [0, .5, 1][index];
    return { x: start.x + (end.x - start.x) * t, y: start.y + (end.y - start.y) * t, r };
  }) };
}
