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

import { yawFromPose } from "./navigationTf";

// Map editing works with both the raw PGM description (`width`, `height`, ...)
// and ROS OccupancyGrid (`info.width`, `info.height`, ...). Keeping their world
// to-cell conversion here prevents the drag preview and saved Area from using
// subtly different rounding rules.
export function mapAreaGridMeta(source) {
    const info = source?.info ?? source;
    const width = Math.floor(Number(info?.width));
    const height = Math.floor(Number(info?.height));
    const resolution = Number(info?.resolution ?? 1);
    if (!Number.isFinite(width) || width <= 0 ||
        !Number.isFinite(height) || height <= 0 ||
        !Number.isFinite(resolution) || resolution <= 0) {
        return null;
    }
    const origin = info?.origin ?? {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
    };
    const rawOriginX = Number(origin.position?.x ?? 0);
    const rawOriginY = Number(origin.position?.y ?? 0);
    return {
        width,
        height,
        resolution,
        originX: Number.isFinite(rawOriginX) ? rawOriginX : 0,
        originY: Number.isFinite(rawOriginY) ? rawOriginY : 0,
        originYaw: yawFromPose(origin),
    };
}

export function mapPointToAreaGridCell(source, x, y) {
    const meta = mapAreaGridMeta(source);
    if (!meta || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    const dx = x - meta.originX;
    const dy = y - meta.originY;
    const localX = Math.cos(meta.originYaw) * dx + Math.sin(meta.originYaw) * dy;
    const localY = -Math.sin(meta.originYaw) * dx + Math.cos(meta.originYaw) * dy;
    const cellX = Math.floor(localX / meta.resolution);
    const cellY = Math.floor(localY / meta.resolution);
    if (cellX < 0 || cellX >= meta.width || cellY < 0 || cellY >= meta.height) {
        return null;
    }
    return { x: cellX, y: cellY };
}

export function mapAreaSelectionBounds(source, selection) {
    if (!selection) return null;
    const start = mapPointToAreaGridCell(source, selection.startX, selection.startY);
    const end = mapPointToAreaGridCell(source, selection.endX, selection.endY);
    if (!start || !end) return null;
    return {
        x_min: Math.min(start.x, end.x),
        y_min: Math.min(start.y, end.y),
        x_max: Math.max(start.x, end.x),
        y_max: Math.max(start.y, end.y),
    };
}
