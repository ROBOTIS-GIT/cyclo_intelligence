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
import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
// @ts-ignore
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mapAreaGridMeta, mapAreaSelectionBounds, mapPointToAreaGridCell } from "../../utils/mapAreaGeometry";
import { buildTfFramePoses, normalizeFrameId, orientationFromYaw, poseForScanFrame, poseForScanFrameAtBasePose, poseForTfAxesFrame, yawFromPose, } from "../../utils/navigationTf";
import { roomSegmentationParams, segmentFreeRooms } from "../../utils/roomSegmentation";
const CAMERA_NEAR = 0.05;
const CAMERA_FAR = 2000;
const MAP_DISPLAY_ROTATION = Math.PI;
const CLICK_DRAG_THRESHOLD_PX = 8;
const TF_AXIS_LENGTH = 0.2;
const WAYPOINT_RING_INNER_RADIUS = 2.25;
const WAYPOINT_RING_OUTER_RADIUS = 3;
const WAYPOINT_SELECTED_HALO_INNER_RADIUS = 3.28;
const WAYPOINT_SELECTED_HALO_OUTER_RADIUS = 3.72;
const WAYPOINT_CENTER_RADIUS = 1.25;
const WAYPOINT_BODY_HIT_RADIUS = WAYPOINT_SELECTED_HALO_OUTER_RADIUS;
const WAYPOINT_HEADING_LENGTH = 4.4;
const WAYPOINT_HEADING_SHAFT_WIDTH = 1.12;
const WAYPOINT_HEADING_HEAD_LENGTH = 1.05;
const WAYPOINT_HEADING_HEAD_WIDTH = 2.84;
const WAYPOINT_ROTATE_HIT_RADIUS = 1.4;
const WAYPOINT_ROTATE_HIT_OFFSET = 0.1;
const WAYPOINT_LABEL_SCALE_Y = 6;
const WAYPOINT_LABEL_BG_ALPHA = 0.5;
const WAYPOINT_LABEL_SELECTED_BG_ALPHA = 0.5;
// Keep the label just outside the heading arrow's full rotation radius. This
// leaves the rotate handle unobstructed without visually detaching the name
// from its waypoint marker.
const WAYPOINT_LABEL_CLEARANCE = 1.5;
const WAYPOINT_LABEL_OFFSET_Y = (
    WAYPOINT_HEADING_LENGTH + WAYPOINT_LABEL_SCALE_Y / 2 + WAYPOINT_LABEL_CLEARANCE
);
const BT_FOCUS_VISIBLE_HEIGHT_MIN = 5;
const BT_FOCUS_VISIBLE_HEIGHT_MAX = 11;
const BT_FOCUS_WAYPOINT_NDC_X = -0.75;
// UI-only decimation. Nav2 still consumes every LaserScan ray; the browser
// projects half of them to reduce point geometry and local-costmap highlighting.
const SCAN_VISUALIZATION_STRIDE = 2;
// Interaction renders at ~60fps: panning/rotating a full-viewport map at
// 30fps reads as visible judder. Idle/hidden tiers keep the CPU savings —
// interaction is transient, so the 60fps bursts are cheap overall.
const RENDER_INTERVAL_ACTIVE_MS = 16;
const RENDER_INTERVAL_IDLE_MS = 100;
const RENDER_INTERVAL_HIDDEN_MS = 500;
const GLOBAL_COSTMAP_FULL_UPLOAD_RATIO = 0.25;

export function mapRenderIntervalMs({ hidden = false, active = false } = {}) {
    if (hidden)
        return RENDER_INTERVAL_HIDDEN_MS;
    return active ? RENDER_INTERVAL_ACTIVE_MS : RENDER_INTERVAL_IDLE_MS;
}

function forEachVisualizedScanRange(ranges, callback) {
    let lastVisited = -1;
    for (let index = 0; index < ranges.length; index += SCAN_VISUALIZATION_STRIDE) {
        callback(ranges[index], index);
        lastVisited = index;
    }
    // Preserve the far edge of the scanner's field of view when stride does
    // not naturally land on the final ray.
    const lastIndex = ranges.length - 1;
    if (lastIndex >= 0 && lastVisited !== lastIndex) {
        callback(ranges[lastIndex], lastIndex);
    }
}
function gridMeta(grid) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k;
    const info = grid === null || grid === void 0 ? void 0 : grid.info;
    const width = Number((_a = info === null || info === void 0 ? void 0 : info.width) !== null && _a !== void 0 ? _a : 0);
    const height = Number((_b = info === null || info === void 0 ? void 0 : info.height) !== null && _b !== void 0 ? _b : 0);
    const resolution = Number((_c = info === null || info === void 0 ? void 0 : info.resolution) !== null && _c !== void 0 ? _c : 0);
    const originX = Number((_f = (_e = (_d = info === null || info === void 0 ? void 0 : info.origin) === null || _d === void 0 ? void 0 : _d.position) === null || _e === void 0 ? void 0 : _e.x) !== null && _f !== void 0 ? _f : 0);
    const originY = Number((_j = (_h = (_g = info === null || info === void 0 ? void 0 : info.origin) === null || _g === void 0 ? void 0 : _g.position) === null || _h === void 0 ? void 0 : _h.y) !== null && _j !== void 0 ? _j : 0);
    const originYaw = yawFromPose((_k = info === null || info === void 0 ? void 0 : info.origin) !== null && _k !== void 0 ? _k : null);
    if (!width || !height || !resolution)
        return null;
    return { width, height, resolution, originX, originY, originYaw };
}
function scanCellsForGrid(grid, scan, scanPose, framePose) {
    var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l, _m, _o;
    const meta = gridMeta(grid);
    if (!meta || !((_a = scan === null || scan === void 0 ? void 0 : scan.ranges) === null || _a === void 0 ? void 0 : _a.length) || !scanPose)
        return null;
    const frameYaw = framePose ? yawFromPose(framePose) : 0;
    const frameCos = Math.cos(frameYaw);
    const frameSin = Math.sin(frameYaw);
    const frameX = Number((_c = (_b = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _b === void 0 ? void 0 : _b.x) !== null && _c !== void 0 ? _c : 0);
    const frameY = Number((_e = (_d = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _d === void 0 ? void 0 : _d.y) !== null && _e !== void 0 ? _e : 0);
    const originYaw = meta.originYaw;
    const originCos = Math.cos(originYaw);
    const originSin = Math.sin(originYaw);
    const scanX = Number((_g = (_f = scanPose.position) === null || _f === void 0 ? void 0 : _f.x) !== null && _g !== void 0 ? _g : 0);
    const scanY = Number((_j = (_h = scanPose.position) === null || _h === void 0 ? void 0 : _h.y) !== null && _j !== void 0 ? _j : 0);
    const scanYaw = yawFromPose(scanPose);
    const min = Number((_k = scan.range_min) !== null && _k !== void 0 ? _k : 0.02);
    const max = Number((_l = scan.range_max) !== null && _l !== void 0 ? _l : 20);
    const angleMin = Number((_m = scan.angle_min) !== null && _m !== void 0 ? _m : 0);
    const inc = Number((_o = scan.angle_increment) !== null && _o !== void 0 ? _o : 0);
    const cells = new Set();
    forEachVisualizedScanRange(scan.ranges, (range, index) => {
        const r = Number(range);
        if (!Number.isFinite(r) || r < min || r > max)
            return;
        const angle = scanYaw + angleMin + inc * index;
        const mapX = scanX + Math.cos(angle) * r;
        const mapY = scanY + Math.sin(angle) * r;
        const frameDx = mapX - frameX;
        const frameDy = mapY - frameY;
        const gridFrameX = frameCos * frameDx + frameSin * frameDy;
        const gridFrameY = -frameSin * frameDx + frameCos * frameDy;
        const originDx = gridFrameX - meta.originX;
        const originDy = gridFrameY - meta.originY;
        const localX = originCos * originDx + originSin * originDy;
        const localY = -originSin * originDx + originCos * originDy;
        const cellX = Math.floor(localX / meta.resolution);
        const cellY = Math.floor(localY / meta.resolution);
        if (cellX < 0 || cellX >= meta.width || cellY < 0 || cellY >= meta.height)
            return;
        for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
                const x = cellX + dx;
                const y = cellY + dy;
                if (x < 0 || x >= meta.width || y < 0 || y >= meta.height)
                    continue;
                cells.add(x + y * meta.width);
            }
        }
    });
    return cells;
}
// Warm, theme-aware occupancy palette (see design handoff, turn 6).
const OCC_COLORS = {
    unknown: [227, 221, 207, 220], // #E3DDCF, blends into paper scene bg
    free: [250, 248, 243, 255],    // #FAF8F3
    wall: [42, 38, 32, 255],       // #2A2620 warm ink
    // Local costmap in muted sand-amber (the UI's warning family) at low
    // opacity: semantically "caution zone", warm enough to sit apart from
    // the neutral-gray global costmap, quiet enough to stay under markers.
    lethal: [172, 116, 58, 100],   // desaturated --mc-warning
    inflate: [188, 148, 96, 70],   // lighter sand
};
// Rounded occupancy rendering keeps the exact cells, then only shaves exposed
// outer corners. This avoids the old blur/threshold path where 1-cell edits
// could disappear visually.
const OCC_ROUNDED = {
    scale: 4,
    maxDim: 2048,
    radiusCells: 0.38,
};
function occRgba([r, g, b, a]) {
    return `rgba(${r},${g},${b},${a / 255})`;
}
const OCC_CELL_UNKNOWN = 0;
const OCC_CELL_FREE = 1;
const OCC_CELL_WALL = 2;
function occupancyCellType(value) {
    if (value === 0)
        return OCC_CELL_FREE;
    if (value > 0)
        return OCC_CELL_WALL;
    return OCC_CELL_UNKNOWN;
}
function sameOccupancyCell(cells, w, h, type, x, y) {
    return x >= 0 && x < w && y >= 0 && y < h && cells[x + y * w] === type;
}
function addCornerCut(ctx, x0, y0, x1, y1, r, corner) {
    if (corner === "tl") {
        ctx.moveTo(x0, y0);
        ctx.lineTo(x0 + r, y0);
        ctx.quadraticCurveTo(x0, y0, x0, y0 + r);
    }
    else if (corner === "tr") {
        ctx.moveTo(x1, y0);
        ctx.lineTo(x1, y0 + r);
        ctx.quadraticCurveTo(x1, y0, x1 - r, y0);
    }
    else if (corner === "br") {
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - r, y1);
        ctx.quadraticCurveTo(x1, y1, x1, y1 - r);
    }
    else {
        ctx.moveTo(x0, y1);
        ctx.lineTo(x0, y1 - r);
        ctx.quadraticCurveTo(x0, y1, x0 + r, y1);
    }
    ctx.closePath();
}
function drawRoundedOccupancyLayer(targetCtx, cells, w, h, scale, type, color) {
    const layer = document.createElement("canvas");
    layer.width = w * scale;
    layer.height = h * scale;
    const ctx = layer.getContext("2d");
    if (!ctx)
        return;
    ctx.fillStyle = occRgba(color);
    for (let y = 0; y < h; y += 1) {
        let runStart = -1;
        for (let x = 0; x <= w; x += 1) {
            const matches = x < w && cells[x + y * w] === type;
            if (matches && runStart < 0) {
                runStart = x;
            }
            else if (!matches && runStart >= 0) {
                ctx.fillRect(runStart * scale, y * scale, (x - runStart) * scale, scale);
                runStart = -1;
            }
        }
    }
    const r = Math.min(scale * OCC_ROUNDED.radiusCells, scale / 2);
    if (r > 0.1) {
        ctx.globalCompositeOperation = "destination-out";
        ctx.fillStyle = "#000";
        ctx.beginPath();
        for (let y = 0; y < h; y += 1) {
            for (let x = 0; x < w; x += 1) {
                if (cells[x + y * w] !== type)
                    continue;
                const left = sameOccupancyCell(cells, w, h, type, x - 1, y);
                const right = sameOccupancyCell(cells, w, h, type, x + 1, y);
                const top = sameOccupancyCell(cells, w, h, type, x, y - 1);
                const bottom = sameOccupancyCell(cells, w, h, type, x, y + 1);
                const x0 = x * scale;
                const y0 = y * scale;
                const x1 = x0 + scale;
                const y1 = y0 + scale;
                if (!left && !top)
                    addCornerCut(ctx, x0, y0, x1, y1, r, "tl");
                if (!right && !top)
                    addCornerCut(ctx, x0, y0, x1, y1, r, "tr");
                if (!right && !bottom)
                    addCornerCut(ctx, x0, y0, x1, y1, r, "br");
                if (!left && !bottom)
                    addCornerCut(ctx, x0, y0, x1, y1, r, "bl");
            }
        }
        ctx.fill();
        ctx.globalCompositeOperation = "source-over";
    }
    targetCtx.drawImage(layer, 0, 0);
    return layer;
}
// Display-only "floor plan" refinement (viewer mode; the editor renders the raw
// grid so pixel edits stay faithful). Geometry is never altered — only how the
// grid is drawn: lidar speckles are despeckled, small holes filled, walls get a
// uniform stroke, and unknown space turns transparent so the known floor floats
// as a silhouette with a soft shadow on the scene background.
const OCC_REFINE = {
    minWallCells: 6,     // wall specks smaller than this become floor
    minFreeCells: 24,    // stray free islands smaller than this become unknown
    holeFillCells: 24,   // enclosed unknown holes smaller than this become floor
    minGridCells: 400,   // skip refinement for tiny grids
    wallThickenPasses: 1,
};
// Per-room pastel tints (refined viewer only). Hues stay off green/orange so
// rooms never read as marker or area colors; the tint sits under annotations.
const ROOM_TINTS = ["#D7E3F0", "#F0DCE3", "#E4DEF0", "#D9EAE8", "#DFE6EF", "#EFDCE9"];
const ROOM_TINT_ALPHA = 0.5;
function occRefineComponents(cells, w, h, { type, minSize, replacement, interiorOnly = false }) {
    const labels = new Uint8Array(w * h);
    const queue = new Int32Array(w * h);
    const component = [];
    for (let start = 0; start < w * h; start += 1) {
        if (cells[start] !== type || labels[start])
            continue;
        let head = 0;
        let tail = 0;
        queue[tail] = start;
        tail += 1;
        labels[start] = 1;
        component.length = 0;
        let touchesBorder = false;
        while (head < tail) {
            const index = queue[head];
            head += 1;
            component.push(index);
            const x = index % w;
            const y = (index / w) | 0;
            if (x === 0 || y === 0 || x === w - 1 || y === h - 1)
                touchesBorder = true;
            if (x > 0 && cells[index - 1] === type && !labels[index - 1]) {
                labels[index - 1] = 1;
                queue[tail] = index - 1;
                tail += 1;
            }
            if (x < w - 1 && cells[index + 1] === type && !labels[index + 1]) {
                labels[index + 1] = 1;
                queue[tail] = index + 1;
                tail += 1;
            }
            if (y > 0 && cells[index - w] === type && !labels[index - w]) {
                labels[index - w] = 1;
                queue[tail] = index - w;
                tail += 1;
            }
            if (y < h - 1 && cells[index + w] === type && !labels[index + w]) {
                labels[index + w] = 1;
                queue[tail] = index + w;
                tail += 1;
            }
        }
        if (component.length < minSize && (!interiorOnly || !touchesBorder)) {
            for (let i = 0; i < component.length; i += 1)
                cells[component[i]] = replacement;
        }
    }
}
function occThickenWalls(cells, w, h, passes) {
    // Thicken OUTWARD only (into unknown): room interiors keep their exact
    // edges, so walls read as a clean stroke without visually shrinking rooms.
    for (let pass = 0; pass < passes; pass += 1) {
        const source = new Uint8Array(cells);
        for (let y = 0; y < h; y += 1) {
            for (let x = 0; x < w; x += 1) {
                if (source[x + y * w] !== OCC_CELL_WALL)
                    continue;
                for (let dy = -1; dy <= 1; dy += 1) {
                    for (let dx = -1; dx <= 1; dx += 1) {
                        const nx = x + dx;
                        const ny = y + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h)
                            continue;
                        const index = nx + ny * w;
                        if (cells[index] === OCC_CELL_UNKNOWN)
                            cells[index] = OCC_CELL_WALL;
                    }
                }
            }
        }
    }
}
function occSilhouetteShadow(cells, w, h, scale) {
    const silhouette = document.createElement("canvas");
    silhouette.width = w;
    silhouette.height = h;
    const sctx = silhouette.getContext("2d");
    if (!sctx)
        return null;
    const image = sctx.createImageData(w, h);
    for (let i = 0; i < w * h; i += 1) {
        if (cells[i] !== OCC_CELL_UNKNOWN)
            image.data[i * 4 + 3] = 255;
    }
    sctx.putImageData(image, 0, 0);
    const big = document.createElement("canvas");
    big.width = w * scale;
    big.height = h * scale;
    const bctx = big.getContext("2d");
    if (!bctx)
        return null;
    bctx.imageSmoothingEnabled = true;
    bctx.filter = `blur(${Math.max(2, scale * 1.6)}px)`;
    bctx.drawImage(silhouette, 0, 0, big.width, big.height);
    bctx.filter = "none";
    bctx.globalCompositeOperation = "source-in";
    bctx.fillStyle = "rgba(28,26,23,0.26)";
    bctx.fillRect(0, 0, big.width, big.height);
    return big;
}
function makeOccupancyTexture(grid, alpha, mode, highlightedCells = null, refined = false) {
    var _a;
    const meta = gridMeta(grid);
    if (!meta || !grid.data || grid.data.length < meta.width * meta.height)
        return null;
    const palette = OCC_COLORS;
    const w = meta.width;
    const h = meta.height;

    // ---- Rounded exact-cell rendering for the base map ----
    if (mode === "map") {
        const scale = Math.max(1, Math.min(OCC_ROUNDED.scale, Math.floor(OCC_ROUNDED.maxDim / Math.max(w, h)) || 1));
        const cells = new Uint8Array(w * h);
        for (let y = 0; y < h; y += 1) {
            for (let x = 0; x < w; x += 1) {
                const srcIndex = (w - 1 - x) + (h - 1 - y) * w; // same flip as before
                const value = (_a = grid.data[srcIndex]) !== null && _a !== void 0 ? _a : -1;
                cells[x + y * w] = occupancyCellType(value);
            }
        }

        const refine = refined && w * h >= OCC_REFINE.minGridCells;
        if (refine) {
            // Despeckle: wall specks inside the floor become floor, stray free
            // islands vanish, and small enclosed unknown holes are filled in.
            occRefineComponents(cells, w, h, { type: OCC_CELL_WALL, minSize: OCC_REFINE.minWallCells, replacement: OCC_CELL_FREE });
            occRefineComponents(cells, w, h, { type: OCC_CELL_FREE, minSize: OCC_REFINE.minFreeCells, replacement: OCC_CELL_UNKNOWN });
            occRefineComponents(cells, w, h, { type: OCC_CELL_UNKNOWN, minSize: OCC_REFINE.holeFillCells, replacement: OCC_CELL_FREE, interiorOnly: true });
            occThickenWalls(cells, w, h, OCC_REFINE.wallThickenPasses);
        }

        const out = document.createElement("canvas");
        out.width = w * scale; out.height = h * scale;
        const ctx = out.getContext("2d");
        if (!ctx)
            return null;
        let roomLabels = null;
        if (refine) {
            // Room segmentation for the pastel tint (2+ rooms only).
            const freeMask = new Uint8Array(w * h);
            for (let i = 0; i < w * h; i += 1)
                freeMask[i] = cells[i] === OCC_CELL_FREE ? 1 : 0;
            const segmented = segmentFreeRooms(freeMask, w, h, roomSegmentationParams(meta.resolution));
            if (segmented.roomCount >= 2)
                roomLabels = segmented.labels;
        }
        if (refine) {
            // Floating silhouette: unknown stays transparent, the known floor
            // casts a soft shadow onto the warm scene background.
            const shadow = occSilhouetteShadow(cells, w, h, scale);
            if (shadow)
                ctx.drawImage(shadow, scale * 1.2, scale * 1.6);
        }
        else {
            ctx.fillStyle = occRgba(palette.unknown);
            ctx.fillRect(0, 0, out.width, out.height);
        }
        const freeLayer = drawRoundedOccupancyLayer(ctx, cells, w, h, scale, OCC_CELL_FREE, palette.free);
        if (roomLabels && freeLayer) {
            const tintSmall = document.createElement("canvas");
            tintSmall.width = w;
            tintSmall.height = h;
            const tctx = tintSmall.getContext("2d");
            if (tctx) {
                const tintPalette = ROOM_TINTS.map((hex) => rgbaArrayFromHex(hex, 255));
                const tintImage = tctx.createImageData(w, h);
                for (let i = 0; i < w * h; i += 1) {
                    const label = roomLabels[i];
                    if (label <= 0)
                        continue;
                    const [r, g, b] = tintPalette[(label - 1) % tintPalette.length];
                    tintImage.data[i * 4] = r;
                    tintImage.data[i * 4 + 1] = g;
                    tintImage.data[i * 4 + 2] = b;
                    tintImage.data[i * 4 + 3] = 255;
                }
                tctx.putImageData(tintImage, 0, 0);
                const tintBig = document.createElement("canvas");
                tintBig.width = w * scale;
                tintBig.height = h * scale;
                const btx = tintBig.getContext("2d");
                if (btx) {
                    btx.imageSmoothingEnabled = true;
                    btx.drawImage(tintSmall, 0, 0, tintBig.width, tintBig.height);
                    // Clip the tint to the rounded free-space silhouette.
                    btx.globalCompositeOperation = "destination-in";
                    btx.drawImage(freeLayer, 0, 0);
                    ctx.globalAlpha = ROOM_TINT_ALPHA;
                    ctx.drawImage(tintBig, 0, 0);
                    ctx.globalAlpha = 1;
                }
            }
        }
        drawRoundedOccupancyLayer(ctx, cells, w, h, scale, OCC_CELL_WALL, palette.wall);

        const texture = new THREE.CanvasTexture(out);
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.magFilter = THREE.LinearFilter;
        texture.minFilter = THREE.LinearFilter;
        texture.generateMipmaps = false;
        texture.flipY = false;
        texture.needsUpdate = true;
        return texture;
    }

    // ---- costmaps: crisp per-cell (safety-accurate), warm colors ----
    const data = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
            const srcIndex = (w - 1 - x) + (h - 1 - y) * w;
            const dstIndex = (x + y * w) * 4;
            const value = (_a = grid.data[srcIndex]) !== null && _a !== void 0 ? _a : -1;
            let r = 118, g = 118, b = 118, a = alpha;
            if (mode === "globalCostmap") {
                if (value <= 0) { r = 0; g = 0; b = 0; a = 0; }
                else {
                    // Neutral gray at ~1/3 the old opacity — a hint under the
                    // mission markers rather than a poster over the map.
                    const normalized = Math.min(Math.max(value, 0), 100) / 100;
                    const gray = Math.round(200 - normalized * 130);
                    r = gray; g = gray; b = gray;
                    a = Math.round(40 + normalized * 70);
                }
            }
            else if (mode === "localCostmap") {
                if (highlightedCells === null || highlightedCells === void 0 ? void 0 : highlightedCells.has(srcIndex)) {
                    [r, g, b, a] = palette.lethal;
                }
                else if (value <= 20) { a = 0; }
                else if (value < 70) { [r, g, b, a] = palette.lethal; }
                else { [r, g, b, a] = palette.inflate; }
            }
            data[dstIndex] = r;
            data[dstIndex + 1] = g;
            data[dstIndex + 2] = b;
            data[dstIndex + 3] = a;
        }
    }
    const texture = new THREE.DataTexture(
        data,
        w,
        h,
        THREE.RGBAFormat,
        THREE.UnsignedByteType,
    );
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.magFilter = THREE.NearestFilter;
    texture.minFilter = THREE.NearestFilter;
    texture.generateMipmaps = false;
    texture.flipY = false;
    if (mode === "globalCostmap") {
        texture.userData.globalCostmapFullUploadPending = false;
        texture.onUpdate = () => {
            texture.userData.globalCostmapFullUploadPending = false;
        };
    }
    texture.needsUpdate = true;
    return texture;
}

function writeGlobalCostmapPixel(data, index, value) {
    if (value <= 0) {
        data[index] = 0;
        data[index + 1] = 0;
        data[index + 2] = 0;
        data[index + 3] = 0;
        return;
    }
    const normalized = Math.min(Math.max(value, 0), 100) / 100;
    const gray = Math.round(200 - normalized * 130);
    data[index] = gray;
    data[index + 1] = gray;
    data[index + 2] = gray;
    data[index + 3] = Math.round(40 + normalized * 70);
}

export function updateGlobalCostmapTexture(texture, grid, updateRegion) {
    const meta = gridMeta(grid);
    const image = texture?.image;
    const textureData = image?.data;
    if (!meta || !textureData || !updateRegion || !texture?.addUpdateRange)
        return false;
    const x = Number(updateRegion.x);
    const y = Number(updateRegion.y);
    const width = Number(updateRegion.width);
    const height = Number(updateRegion.height);
    if (
        !Number.isInteger(x) || !Number.isInteger(y) ||
        !Number.isInteger(width) || !Number.isInteger(height) ||
        x < 0 || y < 0 || width <= 0 || height <= 0 ||
        x + width > meta.width || y + height > meta.height ||
        image.width !== meta.width || image.height !== meta.height ||
        textureData.length < meta.width * meta.height * 4
    ) {
        return false;
    }

    const useFullUpload = (
        width * height > meta.width * meta.height * GLOBAL_COSTMAP_FULL_UPLOAD_RATIO
    );
    const fullUploadPending = Boolean(
        texture.userData?.globalCostmapFullUploadPending
    );
    if (useFullUpload) {
        texture.clearUpdateRanges?.();
        texture.userData.globalCostmapFullUploadPending = true;
    }

    for (let gridY = y; gridY < y + height; gridY += 1) {
        const textureY = meta.height - 1 - gridY;
        const textureX = meta.width - x - width;
        for (let gridX = x; gridX < x + width; gridX += 1) {
            const sourceIndex = gridY * meta.width + gridX;
            const targetX = meta.width - 1 - gridX;
            const targetIndex = (textureY * meta.width + targetX) * 4;
            writeGlobalCostmapPixel(textureData, targetIndex, grid.data[sourceIndex] ?? -1);
        }
        if (!useFullUpload && !fullUploadPending) {
            texture.addUpdateRange(
                (textureY * meta.width + textureX) * 4,
                width * 4,
            );
        }
    }
    texture.needsUpdate = true;
    return true;
}
function disposeObject(object) {
    object.traverse((child) => {
        var _a;
        if (child instanceof THREE.Mesh ||
            child instanceof THREE.Points ||
            child instanceof THREE.Line ||
            child instanceof THREE.Sprite) {
            (_a = child.geometry) === null || _a === void 0 ? void 0 : _a.dispose();
            const material = child.material;
            if (Array.isArray(material)) {
                material.forEach((item) => {
                    const texture = item.map;
                    // Textures flagged `retain` are cached across layer rebuilds
                    // (annotation regions) and disposed by their cache instead.
                    if (texture && !texture.userData.retain)
                        texture.dispose();
                    item.dispose();
                });
            }
            else {
                const texture = material === null || material === void 0 ? void 0 : material.map;
                if (texture && !texture.userData.retain)
                    texture.dispose();
                material === null || material === void 0 ? void 0 : material.dispose();
            }
        }
    });
}
function makeGridPlane(grid, mode, z, framePose = null, highlightedCells = null, refined = false) {
    var _a, _b, _c, _d;
    const meta = gridMeta(grid);
    const texture = makeOccupancyTexture(grid, mode === "map" ? 255 : 170, mode, highlightedCells, refined);
    if (!meta || !texture)
        return null;
    const width = meta.width * meta.resolution;
    const height = meta.height * meta.resolution;
    const geometry = new THREE.PlaneGeometry(width, height);
    const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: mode !== "map" || refined,
        opacity: mode === "map" ? 1 : 0.82,
        depthWrite: mode === "map",
        side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    const originYaw = meta.originYaw;
    const originCos = Math.cos(originYaw);
    const originSin = Math.sin(originYaw);
    const gridCenterX = meta.originX + originCos * (width / 2) - originSin * (height / 2);
    const gridCenterY = meta.originY + originSin * (width / 2) + originCos * (height / 2);
    const frameYaw = framePose ? yawFromPose(framePose) : 0;
    const frameCos = Math.cos(frameYaw);
    const frameSin = Math.sin(frameYaw);
    const frameX = Number((_b = (_a = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _a === void 0 ? void 0 : _a.x) !== null && _b !== void 0 ? _b : 0);
    const frameY = Number((_d = (_c = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _c === void 0 ? void 0 : _c.y) !== null && _d !== void 0 ? _d : 0);
    mesh.position.set(frameX + frameCos * gridCenterX - frameSin * gridCenterY, frameY + frameSin * gridCenterX + frameCos * gridCenterY, z);
    mesh.rotation.z = frameYaw + originYaw + MAP_DISPLAY_ROTATION;
    mesh.userData.mapTexture = texture;
    return mesh;
}
function makeGridTexturePlane(grid, texture, z) {
    const meta = gridMeta(grid);
    if (!meta || !texture)
        return null;
    const width = meta.width * meta.resolution;
    const height = meta.height * meta.resolution;
    const geometry = new THREE.PlaneGeometry(width, height);
    const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 1,
        depthWrite: false,
        side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    const originYaw = meta.originYaw;
    const originCos = Math.cos(originYaw);
    const originSin = Math.sin(originYaw);
    const gridCenterX = meta.originX + originCos * (width / 2) - originSin * (height / 2);
    const gridCenterY = meta.originY + originSin * (width / 2) + originCos * (height / 2);
    mesh.position.set(gridCenterX, gridCenterY, z);
    mesh.rotation.z = originYaw + MAP_DISPLAY_ROTATION;
    mesh.userData.mapTexture = texture;
    mesh.userData.annotation = true;
    return mesh;
}
function mapPointToGridCell(meta, x, y) {
    if (!meta || !Number.isFinite(x) || !Number.isFinite(y))
        return null;
    const dx = x - meta.originX;
    const dy = y - meta.originY;
    const localX = Math.cos(meta.originYaw) * dx + Math.sin(meta.originYaw) * dy;
    const localY = -Math.sin(meta.originYaw) * dx + Math.cos(meta.originYaw) * dy;
    const cellX = Math.floor(localX / meta.resolution);
    const cellY = Math.floor(localY / meta.resolution);
    if (cellX < 0 || cellX >= meta.width || cellY < 0 || cellY >= meta.height)
        return null;
    return { x: cellX, y: cellY };
}
function gridCellToMapPoint(meta, cellX, cellY) {
    const localX = (cellX + 0.5) * meta.resolution;
    const localY = (cellY + 0.5) * meta.resolution;
    return {
        x: meta.originX + Math.cos(meta.originYaw) * localX - Math.sin(meta.originYaw) * localY,
        y: meta.originY + Math.sin(meta.originYaw) * localX + Math.cos(meta.originYaw) * localY,
    };
}
function boundedFreeRegion(grid, regionSpec, excludedCells = null) {
    const meta = gridMeta(grid);
    if (!meta || !grid.data || !regionSpec)
        return null;
    if (Array.isArray(regionSpec.cells) && regionSpec.cells.length) {
        const cells = [];
        const seen = new Set();
        let sumX = 0;
        let sumY = 0;
        regionSpec.cells.forEach((cell) => {
            const x = Math.floor(Number(cell === null || cell === void 0 ? void 0 : cell.x));
            const y = Math.floor(Number(cell === null || cell === void 0 ? void 0 : cell.y));
            if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= meta.width || y >= meta.height)
                return;
            const index = x + y * meta.width;
            if (seen.has(index) || grid.data[index] !== 0)
                return;
            if (excludedCells === null || excludedCells === void 0 ? void 0 : excludedCells.has(index))
                return;
            seen.add(index);
            cells.push(index);
            sumX += x;
            sumY += y;
        });
        if (!cells.length)
            return null;
        return {
            cells,
            centroid: {
                x: sumX / cells.length,
                y: sumY / cells.length,
            },
        };
    }
    const seed = regionSpec.seed_cell;
    const bounds = regionSpec.bounds || (seed ? {
        x_min: seed.x,
        y_min: seed.y,
        x_max: seed.x,
        y_max: seed.y,
    } : null);
    if (!bounds)
        return null;
    const rawXMin = Math.floor(Number(bounds.x_min));
    const rawXMax = Math.floor(Number(bounds.x_max));
    const rawYMin = Math.floor(Number(bounds.y_min));
    const rawYMax = Math.floor(Number(bounds.y_max));
    if (![rawXMin, rawXMax, rawYMin, rawYMax].every(Number.isFinite))
        return null;
    const xMin = Math.max(0, Math.min(rawXMin, rawXMax));
    const xMax = Math.min(meta.width - 1, Math.max(rawXMin, rawXMax));
    const yMin = Math.max(0, Math.min(rawYMin, rawYMax));
    const yMax = Math.min(meta.height - 1, Math.max(rawYMin, rawYMax));
    const cells = [];
    let sumX = 0;
    let sumY = 0;
    for (let y = yMin; y <= yMax; y += 1) {
        for (let x = xMin; x <= xMax; x += 1) {
            const index = x + y * meta.width;
            if (grid.data[index] !== 0)
                continue;
            if (excludedCells === null || excludedCells === void 0 ? void 0 : excludedCells.has(index))
                continue;
            cells.push(index);
            sumX += x;
            sumY += y;
        }
    }
    if (!cells.length)
        return null;
    return {
        cells,
        centroid: {
            x: sumX / cells.length,
            y: sumY / cells.length,
        },
    };
}
export function mapAreaPreviewCellIndices(grid, selection, excludedCells = null) {
    const bounds = mapAreaSelectionBounds(grid, selection);
    if (!bounds)
        return [];
    return boundedFreeRegion(grid, { bounds }, excludedCells)?.cells ?? [];
}
function rgbaArrayFromHex(color, alpha = 150) {
    const value = hexColorString(color, "#6D1F2A");
    return [
        Number.parseInt(value.slice(1, 3), 16),
        Number.parseInt(value.slice(3, 5), 16),
        Number.parseInt(value.slice(5, 7), 16),
        alpha,
    ];
}
function makeAnnotationRegionTexture(grid, region, colorString, alpha = 164) {
    const meta = gridMeta(grid);
    if (!meta || !region)
        return null;
    const scale = Math.max(1, Math.min(OCC_ROUNDED.scale, Math.floor(OCC_ROUNDED.maxDim / Math.max(meta.width, meta.height)) || 1));
    const mask = new Uint8Array(meta.width * meta.height);
    region.cells.forEach((index) => {
        const gridX = index % meta.width;
        const gridY = Math.floor(index / meta.width);
        const canvasX = meta.width - 1 - gridX;
        const canvasY = meta.height - 1 - gridY;
        mask[canvasX + canvasY * meta.width] = 1;
    });
    const canvas = document.createElement("canvas");
    canvas.width = meta.width * scale;
    canvas.height = meta.height * scale;
    const ctx = canvas.getContext("2d");
    if (!ctx)
        return null;
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
function makeEditorAreaPreview(selection, grid, excludedCells = null) {
    const meta = mapAreaGridMeta(grid);
    if (!meta)
        return null;
    const cells = mapAreaPreviewCellIndices(grid, selection, excludedCells);
    if (!cells.length)
        return null;

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
        positions.push(
            meta.originX + originCos * localX - originSin * localY,
            meta.originY + originSin * localX + originCos * localY,
            0.82,
        );
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
        if (runStart >= 0)
            addRun(runY, runStart, runEnd);
    };
    cells.forEach((index) => {
        const x = index % meta.width;
        const y = Math.floor(index / meta.width);
        if (y !== runY || x !== runEnd + 1) {
            flushRun();
            runY = y;
            runStart = x;
        }
        runEnd = x;
    });
    flushRun();
    if (!positions.length)
        return null;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setIndex(indices);
    geometry.computeBoundingSphere();
    return new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
        color: 0x6d1f2a,
        transparent: true,
        opacity: 0.3,
        depthWrite: false,
        side: THREE.DoubleSide,
    }));
}
function makeLine(points, color, lineWidth = 2) {
    if (points.length < 2)
        return null;
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color, linewidth: lineWidth });
    return new THREE.Line(geometry, material);
}
// Mission route rendering: dashed legs plus a mid-segment arrowhead per leg,
// so the travel direction reads at a glance. Sizes follow the waypoint
// marker scale (map resolution) like every other route element.
function makeMissionRouteLine(points, color, scale) {
    if (points.length < 2)
        return null;
    const group = new THREE.Group();
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineDashedMaterial({
        color,
        dashSize: 3 * scale,
        gapSize: 2 * scale,
        transparent: true,
        opacity: 0.72,
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
    for (let i = 0; i < points.length - 1; i += 1) {
        const a = points[i];
        const b = points[i + 1];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        // Legs too short to fit an arrowhead between the endpoint markers
        // stay dash-only.
        if (Math.hypot(dx, dy) < arrowLength * 2.5)
            continue;
        const arrow = new THREE.Mesh(arrowGeometry, new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.9,
            side: THREE.DoubleSide,
        }));
        arrow.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, 0.025);
        arrow.rotation.z = Math.atan2(dy, dx);
        group.add(arrow);
    }
    return group;
}
// Warm marker palette (turn 9 shapes + 11b colors + 12 labels).
const MARKER_COLORS = {
    idle: 0x5b8266,          // sage — waypoints/behavior actions (walls stay ink)
    selected: 0xc96442,      // clay
    control: 0x1c1a17,       // ink — behavior control chips
    action: 0x5b8266,        // sage
    decorator: 0xb4762f,     // amber
    badgeTextIdle: "#ffffff",
    badgeTextSelected: "#ffffff",
    badgeStroke: "rgba(250,248,243,0.95)",
    labelBg: "rgba(243,241,234,0.92)",
    labelBorder: "rgba(226,221,209,1)",
    labelText: "#1c1a17",
    labelSelBg: "rgba(244,229,220,0.96)",
    labelSelBorder: "#c96442",
    labelSelText: "#b5563a",
    poseArrow: "#f3f1ea",
};
const markerPalette = MARKER_COLORS;

function makeTextSprite(text, { width = 256, height = 64, fontSize = 22, backgroundAlpha = 0.92 } = {}) {
    const pal = markerPalette;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (ctx) {
        const r = height / 2; // pill radius
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.roundRect(2, 2, width - 4, height - 4, r);
        ctx.fillStyle = pal.labelBg.replace("0.92", String(backgroundAlpha));
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = pal.labelBorder;
        ctx.stroke();
        ctx.fillStyle = pal.labelText;
        ctx.font = `600 ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, width / 2, height / 2 + 1);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
}

function hexColorString(color, fallback = "#C96442") {
    const match = String(color || "").trim().match(/^#?([0-9A-Fa-f]{6})$/);
    return match ? `#${match[1].toUpperCase()}` : fallback;
}
// Mix a hex color toward white so any stored area color renders as a pastel
// wash. Already-pastel palette entries only get slightly lighter; the legacy
// deep palette (navy, wine, ...) lands in the same family.
function pastelizeHexColor(hex, mix = 0.45) {
    const num = parseInt(hex.slice(1), 16);
    const lift = (channel) => Math.round(channel + (255 - channel) * mix);
    const r = lift((num >> 16) & 255);
    const g = lift((num >> 8) & 255);
    const b = lift(num & 255);
    return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0").toUpperCase()}`;
}
function makeAnnotationLabelSprite(text, color) {
    const label = String(text || "Area");
    const fontSize = 21;
    const font = `700 ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
    const measure = document.createElement("canvas").getContext("2d");
    measure.font = font;
    const textW = Math.ceil(measure.measureText(label).width);
    const height = 44;
    const padX = 10;
    const width = Math.max(72, textW + padX * 2);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (ctx) {
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = "#000000";
        ctx.font = font;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(label, width / 2, height / 2 + 1);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    return { sprite, aspect: width / height };
}

// Auto-width glass pill + downward tail; clay tint when selected. Returns
// { sprite, aspect } so the caller scales without squishing long names.
function makeWaypointLabelSprite(text, { fontSize = 56, selected = false } = {}) {
    const pal = markerPalette;
    const font = `${selected ? 700 : 600} ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
    const measure = document.createElement("canvas").getContext("2d");
    measure.font = font;
    const textW = Math.ceil(measure.measureText(String(text)).width);
    const pad = Math.round(fontSize * 0.7);
    const pillH = Math.round(fontSize * 1.9);
    const tailH = Math.round(fontSize * 0.38);
    const width = Math.max(textW + pad * 2, pillH);
    const height = pillH + tailH;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (ctx) {
        const bgAlpha = selected ? WAYPOINT_LABEL_SELECTED_BG_ALPHA : WAYPOINT_LABEL_BG_ALPHA;
        const bg = (selected ? pal.labelSelBg : pal.labelBg)
            .replace(/,\s*[\d.]+\s*\)$/, `,${bgAlpha})`);
        const border = selected ? pal.labelSelBorder : pal.labelBorder;
        ctx.clearRect(0, 0, width, height);
        ctx.beginPath();
        ctx.roundRect(3, 3, width - 6, pillH - 6, pillH / 2);
        ctx.fillStyle = bg;
        ctx.fill();
        ctx.lineWidth = 3;
        ctx.strokeStyle = border;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(width / 2 - tailH, pillH - 4);
        ctx.lineTo(width / 2, pillH + tailH - 4);
        ctx.lineTo(width / 2 + tailH, pillH - 4);
        ctx.closePath();
        ctx.fillStyle = bg;
        ctx.fill();
        ctx.strokeStyle = border;
        ctx.stroke();
        ctx.fillStyle = bg;
        ctx.fillRect(width / 2 - tailH + 2, pillH - 7, tailH * 2 - 4, 6);
        ctx.fillStyle = selected ? pal.labelSelText : pal.labelText;
        ctx.font = font;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(text), width / 2, pillH / 2 + 2);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
        map: texture,
        transparent: true,
        depthWrite: false,
    }));
    return { sprite, aspect: width / height };
}
function waypointHeadingShape(shaftWidth, headWidth, headLength) {
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
function makeWaypointHeadingArrow() {
    const group = new THREE.Group();
    const arrow = new THREE.Mesh(new THREE.ShapeGeometry(waypointHeadingShape(WAYPOINT_HEADING_SHAFT_WIDTH, WAYPOINT_HEADING_HEAD_WIDTH, WAYPOINT_HEADING_HEAD_LENGTH)), new THREE.MeshBasicMaterial({
        color: 0x111827,
        transparent: true,
        opacity: 0.94,
        side: THREE.DoubleSide,
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
        side: THREE.DoubleSide,
    }));
    hitArea.position.set(WAYPOINT_HEADING_LENGTH + WAYPOINT_ROTATE_HIT_OFFSET, 0, 0.06);
    hitArea.userData.waypointRotateHitArea = true;
    group.add(hitArea);
    return group;
}
function makePoseMarker(pose, color, z) {
    var _a, _b, _c, _d;
    const group = new THREE.Group();
    const x = Number((_b = (_a = pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null && _b !== void 0 ? _b : 0);
    const y = Number((_d = (_c = pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null && _d !== void 0 ? _d : 0);
    const yaw = yawFromPose(pose);
    group.position.set(x, y, z);
    group.rotation.z = yaw;
    const body = new THREE.Mesh(new THREE.CircleGeometry(0.13, 24), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.86 }));
    group.add(body);
    const arrowShape = new THREE.Shape();
    arrowShape.moveTo(0.26, 0);
    arrowShape.lineTo(-0.1, 0.13);
    arrowShape.lineTo(-0.04, 0);
    arrowShape.lineTo(-0.1, -0.13);
    arrowShape.closePath();
    const arrow = new THREE.Mesh(new THREE.ShapeGeometry(arrowShape), new THREE.MeshBasicMaterial({ color: markerPalette.poseArrow }));
    arrow.position.z = 0.01;
    group.add(arrow);
    return group;
}
function makeSpotMarker(spot, selected = false, scale = 1, active = false) {
    var _a, _b, _c, _d;
    const pose = spot === null || spot === void 0 ? void 0 : spot.pose;
    if (!pose)
        return null;
    const x = Number((_a = pose.x) !== null && _a !== void 0 ? _a : 0);
    const y = Number((_b = pose.y) !== null && _b !== void 0 ? _b : 0);
    const yaw = Number((_c = pose.yaw) !== null && _c !== void 0 ? _c : 0);
    const pal = markerPalette;
    const color = active ? pal.selected : (selected ? pal.selected : pal.idle);
    const group = new THREE.Group();
    // Flush with the map plane: just enough lift to avoid z-fighting, so the
    // marker never parallax-floats off its true position under the
    // perspective camera.
    group.position.set(x, y, 0.01);
    group.scale.setScalar(scale);
    group.userData = { spotId: spot.id, dragAction: "move" };
    const marker = new THREE.Group();
    marker.rotation.z = yaw;
    marker.userData = { spotId: spot.id, dragAction: "move" };
    // Fill the complete visual footprint with one invisible pointer target.
    // The visible center and ring otherwise leave a dead annulus between them,
    // which makes small waypoints unnecessarily difficult to select.
    const bodyHitArea = new THREE.Mesh(new THREE.CircleGeometry(WAYPOINT_BODY_HIT_RADIUS, 32), new THREE.MeshBasicMaterial({
        transparent: true,
        opacity: 0,
        colorWrite: false,
        depthWrite: false,
        side: THREE.DoubleSide,
    }));
    bodyHitArea.position.z = 0.005;
    bodyHitArea.userData = {
        spotId: spot.id,
        dragAction: "move",
        waypointBodyHitArea: true,
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
            side: THREE.DoubleSide,
        }));
        pulse.userData = { spotId: spot.id, dragAction: "move", pulse: true, pulseBase };
        marker.add(pulse);
    }
    if (selected || active) {
        const halo = new THREE.Mesh(new THREE.RingGeometry(WAYPOINT_SELECTED_HALO_INNER_RADIUS, WAYPOINT_SELECTED_HALO_OUTER_RADIUS, 40), new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.34,
            side: THREE.DoubleSide,
        }));
        halo.userData = { spotId: spot.id, dragAction: "move" };
        marker.add(halo);
    }
    const ring = new THREE.Mesh(new THREE.RingGeometry(WAYPOINT_RING_INNER_RADIUS, WAYPOINT_RING_OUTER_RADIUS, 40), new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.95,
        side: THREE.DoubleSide,
    }));
    ring.userData = { spotId: spot.id, dragAction: "move" };
    marker.add(ring);
    const center = new THREE.Mesh(new THREE.CircleGeometry(WAYPOINT_CENTER_RADIUS, 32), new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: selected ? 0.95 : 0.78,
    }));
    center.position.z = 0.01;
    center.userData = { spotId: spot.id, dragAction: "move" };
    marker.add(center);
    const heading = makeWaypointHeadingArrow();
    heading.traverse((child) => {
        child.userData = { ...child.userData, spotId: spot.id, dragAction: "rotate" };
    });
    marker.add(heading);
    group.add(marker);
    const label = String((_d = spot.label) !== null && _d !== void 0 ? _d : spot.id);
    if (label) {
        const { sprite, aspect } = makeWaypointLabelSprite(label, { selected });
        // Anchor the sprite at the waypoint and shift its visual center in
        // screen space. Unlike a world-Y position offset, this stays directly
        // above the waypoint when the camera rolls with a rotated map.
        sprite.position.set(0, 0, 0.04);
        sprite.scale.set(WAYPOINT_LABEL_SCALE_Y * aspect, WAYPOINT_LABEL_SCALE_Y, 1); // auto width, no squish
        sprite.center.set(0.5, 0.5 - WAYPOINT_LABEL_OFFSET_Y / WAYPOINT_LABEL_SCALE_Y);
        sprite.userData = { spotId: spot.id, dragAction: "move" };
        // Keep the full name pill interactive. The dedicated rotate target sits
        // above it on the z-axis, so overlapping arrow input still wins.
        group.add(sprite);
    }
    return group;
}
function makeRouteBadgeSprite(text, selected = false) {
    const pal = markerPalette;
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.beginPath();
        ctx.arc(64, 64, 52, 0, Math.PI * 2);
        ctx.fillStyle = selected
            ? `#${pal.selected.toString(16).padStart(6, "0")}`
            : `#${pal.idle.toString(16).padStart(6, "0")}`;
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
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
}
function makeMissionRouteBadge(spot, order, selected = false, scale = 1) {
    var _a, _b;
    const pose = spot === null || spot === void 0 ? void 0 : spot.pose;
    if (!pose)
        return null;
    const x = Number((_a = pose.x) !== null && _a !== void 0 ? _a : 0);
    const y = Number((_b = pose.y) !== null && _b !== void 0 ? _b : 0);
    const sprite = makeRouteBadgeSprite(order, selected);
    const markerRadius = WAYPOINT_RING_OUTER_RADIUS * scale;
    const arrowRadius = WAYPOINT_HEADING_LENGTH * scale;
    const size = Math.max(0.34, markerRadius * 2.2);
    const clearance = Math.max(0.04, scale * 0.5);
    const radialOffset = Math.max(markerRadius, arrowRadius) + size / 2 + clearance;
    const diagonalOffset = radialOffset / Math.SQRT2;
    // Keep route order at a stable screen-space bottom-left position. The
    // waypoint name stays above, while the radial clearance prevents overlap
    // with both the marker ring and a heading arrow pointed toward the badge.
    sprite.position.set(x, y, 0.05);
    sprite.scale.set(size, size, 1);
    sprite.center.set(
        0.5 + diagonalOffset / size,
        0.5 + diagonalOffset / size,
    );
    sprite.userData = { spotId: spot.id, dragAction: "move", missionRouteBadge: true };
    return sprite;
}
// Rasterizing an annotation region is the most expensive part of a layers
// rebuild (supersampled full-map canvas + blur), so cache textures per
// annotation object. The editor keeps unchanged annotations reference-identical
// across edits, so only the annotation actually being brushed re-rasterizes.
const annotationTextureCache = new WeakMap();
function cachedAnnotationRegionTexture(annotation, grid, region, colorString, selected) {
    const hit = annotationTextureCache.get(annotation);
    if (hit && hit.grid === grid && hit.selected === selected) {
        return hit.texture;
    }
    const texture = makeAnnotationRegionTexture(grid, region, colorString, selected ? 210 : 164);
    if (texture) {
        texture.userData.retain = true;
        if ((hit === null || hit === void 0 ? void 0 : hit.texture) && hit.texture !== texture)
            hit.texture.dispose();
        annotationTextureCache.set(annotation, { grid, selected, texture });
    }
    return texture;
}
function makeMapAnnotationRegion(annotation, grid, coveredCells = null, selected = false) {
    var _a, _b, _c, _d;
    const meta = gridMeta(grid);
    if (!meta)
        return null;
    const pose = (_a = annotation === null || annotation === void 0 ? void 0 : annotation.pose) !== null && _a !== void 0 ? _a : {};
    const fallbackCell = mapPointToGridCell(meta, Number((_b = pose.x) !== null && _b !== void 0 ? _b : NaN), Number((_c = pose.y) !== null && _c !== void 0 ? _c : NaN));
    const regionSpec = (_d = annotation === null || annotation === void 0 ? void 0 : annotation.region) !== null && _d !== void 0 ? _d : (fallbackCell ? { seed_cell: fallbackCell } : null);
    const region = boundedFreeRegion(grid, regionSpec, coveredCells);
    if (!region)
        return null;
    // Pastelize whatever color is stored: legacy annotations carry the old
    // deep palette, and areas must stay a light background wash either way.
    const colorString = pastelizeHexColor(
        hexColorString(annotation === null || annotation === void 0 ? void 0 : annotation.color, "#6D1F2A"),
    );
    const texture = cachedAnnotationRegionTexture(annotation, grid, region, colorString, selected);
    // Flush with the map, and strictly BELOW the waypoint/route layer
    // (0.01-0.05) so labeled areas never cover mission markers.
    const plane = makeGridTexturePlane(grid, texture, 0.004);
    if (!plane)
        return null;
    const group = new THREE.Group();
    group.add(plane);
    if (coveredCells) {
        region.cells.forEach((index) => coveredCells.add(index));
    }
    const label = String((annotation === null || annotation === void 0 ? void 0 : annotation.label) || "Area").trim();
    if (label) {
        const { sprite, aspect } = makeAnnotationLabelSprite(label, colorString);
        const center = gridCellToMapPoint(meta, region.centroid.x, region.centroid.y);
        const labelHeight = Math.max(0.24, Math.min(0.72, meta.resolution * 11));
        sprite.position.set(center.x, center.y, 0.008);
        sprite.scale.set(labelHeight * aspect, labelHeight, 1);
        group.add(sprite);
    }
    return group;
}
function behaviorNodeColor(category, selected = false) {
    const pal = markerPalette;
    if (selected)
        return pal.selected;
    switch (category) {
        case "control":
            return pal.control;
        case "decorator":
            return pal.decorator;
        case "action":
        default:
            return pal.action;
    }
}
function makeBehaviorNodeMarker(node, selected = false) {
    var _a, _b, _c, _d, _e;
    const pose = node === null || node === void 0 ? void 0 : node.pose;
    if (!pose)
        return null;
    const x = Number((_a = pose.x) !== null && _a !== void 0 ? _a : 0);
    const y = Number((_b = pose.y) !== null && _b !== void 0 ? _b : 0);
    const yaw = Number((_c = pose.yaw) !== null && _c !== void 0 ? _c : 0);
    const color = behaviorNodeColor(node.category, selected);
    const group = new THREE.Group();
    group.position.set(x, y, 0.28);
    group.rotation.z = yaw;
    group.userData.behaviorNodeId = node.id;
    const body = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.28), new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: selected ? 0.95 : 0.82,
        side: THREE.DoubleSide,
    }));
    body.userData.behaviorNodeId = node.id;
    group.add(body);
    const outline = makeLine([
        new THREE.Vector3(-0.26, -0.16, 0.02),
        new THREE.Vector3(0.26, -0.16, 0.02),
        new THREE.Vector3(0.26, 0.16, 0.02),
        new THREE.Vector3(-0.26, 0.16, 0.02),
        new THREE.Vector3(-0.26, -0.16, 0.02),
    ], 0xf3f1ea, 2);
    if (outline) {
        outline.userData.behaviorNodeId = node.id;
        group.add(outline);
    }
    const port = new THREE.Mesh(new THREE.CircleGeometry(0.045, 16), new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.92,
    }));
    port.position.set(-0.16, 0, 0.03);
    port.userData.behaviorNodeId = node.id;
    group.add(port);
    const label = String((_e = (_d = node.label) !== null && _d !== void 0 ? _d : node.tag) !== null && _e !== void 0 ? _e : node.id);
    if (label) {
        const sprite = makeTfLabelSprite(label);
        sprite.position.set(0, 0.38, 0.06);
        sprite.scale.set(0.56, 0.14, 1);
        sprite.userData.behaviorNodeId = node.id;
        group.add(sprite);
    }
    return group;
}
function makeTfAxes(pose, label) {
    var _a, _b, _c, _d, _e, _f;
    const group = new THREE.Group();
    group.position.set(Number((_b = (_a = pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null && _b !== void 0 ? _b : 0), Number((_d = (_c = pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null && _d !== void 0 ? _d : 0), Number((_f = (_e = pose.position) === null || _e === void 0 ? void 0 : _e.z) !== null && _f !== void 0 ? _f : 0) + 0.08);
    group.rotation.z = yawFromPose(pose);
    const xAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(TF_AXIS_LENGTH, 0, 0)], 0xef4444);
    const yAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, TF_AXIS_LENGTH, 0)], 0x22c55e);
    const zAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, TF_AXIS_LENGTH)], 0x3b82f6);
    if (xAxis)
        group.add(xAxis);
    if (yAxis)
        group.add(yAxis);
    if (zAxis)
        group.add(zAxis);
    const sprite = makeTfLabelSprite(label);
    sprite.position.set(0, -TF_AXIS_LENGTH * 0.85, 0.012);
    group.add(sprite);
    return group;
}
function makeFootprintMarker(footprint, framePose) {
    var _a, _b, _c, _d, _e, _f, _g, _h;
    const sourcePoints = (_b = (_a = footprint.polygon) === null || _a === void 0 ? void 0 : _a.points) !== null && _b !== void 0 ? _b : [];
    const polygonPoints = sourcePoints
        .map((point) => {
        var _a, _b, _c;
        return ({
            x: Number((_a = point.x) !== null && _a !== void 0 ? _a : 0),
            y: Number((_b = point.y) !== null && _b !== void 0 ? _b : 0),
            z: Number((_c = point.z) !== null && _c !== void 0 ? _c : 0),
        });
    })
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
    if (polygonPoints.length < 3)
        return null;
    const group = new THREE.Group();
    const yaw = framePose ? yawFromPose(framePose) : 0;
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const frameX = Number((_d = (_c = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _c === void 0 ? void 0 : _c.x) !== null && _d !== void 0 ? _d : 0);
    const frameY = Number((_f = (_e = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _e === void 0 ? void 0 : _e.y) !== null && _f !== void 0 ? _f : 0);
    const frameZ = Number((_h = (_g = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _g === void 0 ? void 0 : _g.z) !== null && _h !== void 0 ? _h : 0);
    const transformPoint = (point) => (new THREE.Vector3(frameX + cos * point.x - sin * point.y, frameY + sin * point.x + cos * point.y, frameZ + point.z + 0.18));
    const points = polygonPoints.map(transformPoint);
    points.push(points[0].clone());
    const line = makeLine(points, 0x38bdf8, 3);
    if (line)
        group.add(line);
    const shape = new THREE.Shape();
    polygonPoints.forEach((point, index) => {
        if (index === 0)
            shape.moveTo(point.x, point.y);
        else
            shape.lineTo(point.x, point.y);
    });
    shape.closePath();
    const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
        side: THREE.DoubleSide,
    }));
    fill.position.set(frameX, frameY, frameZ + 0.17);
    fill.rotation.z = yaw;
    group.add(fill);
    return group;
}
function makeTfLabelSprite(text) {
    const sprite = makeTextSprite(text);
    sprite.scale.set(TF_AXIS_LENGTH * 1.8, TF_AXIS_LENGTH * 0.45, 1);
    return sprite;
}
function fitCameraToMap(camera, controls, meta, roll = 0) {
    if (!meta)
        return;
    const width = meta.width * meta.resolution;
    const height = meta.height * meta.resolution;
    const center = new THREE.Vector3(meta.originX + width / 2, meta.originY + height / 2, 0);
    const maxDim = Math.max(width, height, 1);
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const distanceForHeight = height / (2 * Math.tan(halfFov));
    const distanceForWidth = width / (2 * Math.tan(halfFov) * Math.max(camera.aspect, 0.1));
    const distance = Math.max(distanceForHeight, distanceForWidth, maxDim) * 1.12;
    camera.up.set(Math.sin(roll), Math.cos(roll), 0);
    camera.position.set(center.x, center.y, distance);
    camera.lookAt(center);
    camera.near = CAMERA_NEAR;
    camera.far = Math.max(CAMERA_FAR, maxDim * 10);
    camera.updateProjectionMatrix();
    controls.target.copy(center);
    controls.update();
}
function applyTopViewRoll(camera, controls, roll) {
    camera.up.set(Math.sin(roll), Math.cos(roll), 0);
    camera.lookAt(controls.target);
    controls.update();
}
function focusCameraToWaypoint(camera, controls, meta, pose, roll = 0) {
    var _a, _b;
    if (!meta || !pose)
        return;
    const width = meta.width * meta.resolution;
    const height = meta.height * meta.resolution;
    const maxDim = Math.max(width, height, 1);
    const visibleHeight = Math.max(BT_FOCUS_VISIBLE_HEIGHT_MIN, Math.min(maxDim * 0.68, BT_FOCUS_VISIBLE_HEIGHT_MAX));
    const halfFov = THREE.MathUtils.degToRad(camera.fov / 2);
    const distance = visibleHeight / (2 * Math.tan(halfFov));
    const halfWidth = distance * Math.tan(halfFov) * Math.max(camera.aspect, 0.1);
    const x = Number((_a = pose.x) !== null && _a !== void 0 ? _a : 0);
    const y = Number((_b = pose.y) !== null && _b !== void 0 ? _b : 0);
    const screenRight = new THREE.Vector3(Math.cos(roll), -Math.sin(roll), 0);
    const target = new THREE.Vector3(x, y, 0).addScaledVector(screenRight, -BT_FOCUS_WAYPOINT_NDC_X * halfWidth);
    camera.up.set(Math.sin(roll), Math.cos(roll), 0);
    camera.position.set(target.x, target.y, Math.max(distance, CAMERA_NEAR * 10));
    camera.lookAt(target);
    camera.near = CAMERA_NEAR;
    camera.far = Math.max(CAMERA_FAR, maxDim * 10);
    camera.updateProjectionMatrix();
    controls.target.copy(target);
    controls.update();
}
function BtCanvasNode({ className = "", children, tone = "default" }) {
    const styles = {
        default: { color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", borderColor: "var(--mc-text)" },
        active: { color: "var(--mc-accent-fg)", backgroundColor: "var(--mc-accent)", borderColor: "var(--mc-accent)" },
        muted: { color: "var(--mc-text-muted)", backgroundColor: "var(--mc-surface-2)", borderColor: "var(--mc-success)" },
    };
    return (<div className={`absolute min-w-[112px] h-10 px-3 border rounded-md flex items-center justify-center text-xs font-semibold shadow-sm ${className}`} style={styles[tone] || styles.default}>
      {children}
    </div>);
}
function WaypointBtFocusLayer({ layer, onClose }) {
    const spot = layer === null || layer === void 0 ? void 0 : layer.spot;
    if (!spot)
        return null;
    return (<section className="absolute inset-0 z-20 pointer-events-none" aria-label="Waypoint Task focus canvas">
      {onClose ? (<button type="button" className="absolute inset-y-0 left-0 w-[25%] pointer-events-auto cursor-pointer border-0 p-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px]" style={{
            background: "linear-gradient(90deg, rgba(28,26,23,0.08), rgba(28,26,23,0))",
            outlineColor: "var(--mc-accent)",
        }} aria-label="Back to Map from waypoint context" title="Back to Map" onClick={(event) => {
            event.stopPropagation();
            onClose();
        }}/>) : (<div className="absolute inset-y-0 left-0 w-[25%] pointer-events-none" style={{
            background: "linear-gradient(90deg, rgba(28,26,23,0.08), rgba(28,26,23,0))",
        }}/>)}
      <div className="absolute inset-y-0 right-0 w-[75%] pointer-events-auto overflow-hidden" style={{
            color: "var(--mc-text)",
            backgroundColor: "var(--mc-surface)",
            borderLeft: "1px solid var(--mc-border-strong)",
        }}>
        <div className="h-full min-h-0 p-4">
          <div className="relative min-h-0 overflow-hidden rounded-md border" style={{
            borderColor: "var(--mc-border)",
            height: "100%",
        }}>
            {layer.editor ? layer.editor : (<>
            <div className="absolute inset-0 opacity-80" style={{
            backgroundImage: "linear-gradient(rgba(28,26,23,0.13) 1px, transparent 1px), linear-gradient(90deg, rgba(28,26,23,0.13) 1px, transparent 1px)",
            backgroundSize: "28px 28px",
        }}/>
            <svg className="absolute inset-0 h-full w-full" aria-hidden="true">
              <line x1="20%" y1="49%" x2="42%" y2="49%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round"/>
              <line x1="56%" y1="38%" x2="76%" y2="28%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round"/>
              <line x1="56%" y1="60%" x2="76%" y2="72%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round"/>
            </svg>
            <BtCanvasNode className="left-[8%] top-[43%]" tone={spot.linked_bt_tree ? "active" : "muted"}>
              {spot.linked_bt_tree ? "Task" : "New Task"}
            </BtCanvasNode>
            <BtCanvasNode className="left-[38%] top-[43%]">
              Sequence
            </BtCanvasNode>
            <BtCanvasNode className="left-[68%] top-[22%]" tone="active">
              Navigate
            </BtCanvasNode>
            <BtCanvasNode className="left-[68%] top-[66%]" tone="default">
              Task
            </BtCanvasNode>
            </>)}
          </div>
        </div>
      </div>
    </section>);
}
// Scene background per theme (warm-minimal). Light uses warm paper; dark uses a
// warm near-black so the occupancy palette (free ~245, occupied ~28) still reads
// as a clean floor-plan without retuning makeOccupancyTexture.
// Keep these values aligned with the Mission Canvas `--mc-canvas` theme token.
const SCENE_BG = 0xefece3;

export function MapViewer({ map, globalCostmap, localCostmap, scan, scanPose = null, pose, plan, goalPose, footprint, tf, showMap, showGlobalCostmap, showLocalCostmap, showScan, showGlobalPlan, showGoalPose, showTf, showRobotModel, interactionDisabled, interactionMode, editorActive, editorPaintOnDrag = true, editorAreaSelection = false, editorBrush = null, mapRefined = true, viewKey, waitingLabel = "Waiting for /map", fitContainer = false, spots = [], selectedSpotId = "", activeWaypointId = "", missionFollowRobot = false, behaviorNodes = [], selectedBehaviorNodeId = "", behaviorPreviewNode = null, missionRouteOrder = [], missionRouteClosed = false, missionRouteMode = false, selectedMissionRouteSourceId = "", mapAnnotations = [], selectedMapAnnotationId = "", btLayer = null, onBtLayerClose, onSpotClick, onBehaviorNodeClick, onMissionRouteSpotClick, onMissionRouteMapClick, onSpotPoseChange, onBehaviorNodePoseChange, onEditorMapPoint, onEditorMapArea, onMapClick, onMapPose, }) {
    const btSpotId = btLayer?.spot?.id;
    const btSpotPose = btLayer?.spot?.pose;
    const btSpotTree = btLayer?.spot?.linked_bt_tree;
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const rendererRef = useRef(null);
    const cameraRef = useRef(null);
    const controlsRef = useRef(null);
    const layersRef = useRef(null);
    const mapLayerRef = useRef(null);
    const animationFrameRef = useRef(null);
    const fitMapKeyRef = useRef(null);
    const viewRollRef = useRef(0);
    const viewRotateDragRef = useRef(null);
    const btFocusActiveRef = useRef(false);
    const followRobotRef = useRef(false);
    const followPoseRef = useRef(null);
    const lastMotionPoseRef = useRef(null);
    const renderActiveUntilRef = useRef(0);
    const renderInteractionActiveRef = useRef(false);
    // True while an active-waypoint halo is pulsing: the animation needs the
    // 60fps cadence or main-thread contention (BT split view) makes the
    // 10fps idle ticks visibly stutter.
    const pulseActiveRef = useRef(false);
    pulseActiveRef.current = Boolean(activeWaypointId);
    const latestFootprintRef = useRef(null);
    const tfSyncedFootprintRef = useRef(null);
    const [dragPreviewPose, setDragPreviewPose] = useState(null);
    const [nodeDragPreview, setNodeDragPreview] = useState(null);
    const [editorAreaPreview, setEditorAreaPreview] = useState(null);
    const [mapDragActive, setMapDragActive] = useState(false);
    const [viewerError, setViewerError] = useState(null);
    // Freeze each LaserScan in display coordinates until the next scan or map geometry change.
    const scanProjectionRef = useRef(null);
    const raycasterRef = useRef(new THREE.Raycaster());
    const pointerRef = useRef(new THREE.Vector2());
    const pointerDownRef = useRef(null);
    const editorPaintPointerRef = useRef(null);
    const editorMovePendingRef = useRef(null);
    const editorMoveRafRef = useRef(null);
    const editorAreaDragRef = useRef(null);
    const editorBrushLayerRef = useRef(null);
    // Split layer groups so high-frequency signals (TF pose, scan) never force
    // a rebuild of the expensive layers (costmap pixel planes, marker sprites).
    const costmapLayerRef = useRef(null);
    const globalCostmapTextureRef = useRef(null);
    const navPathLayerRef = useRef(null);
    const liveLayerRef = useRef(null);
    const nodeDragRef = useRef(null);
    useEffect(() => {
        const containerEl = containerRef.current;
        if (!containerEl || rendererRef.current)
            return;
        let scene = null;
        let camera = null;
        let renderer = null;
        let mapLayer = null;
        let layers = null;
        let controls = null;
        let resizeObserver = null;
        let resize = null;
        let handleControlsStart = null;
        let handleControlsEnd = null;
        try {
            scene = new THREE.Scene();
            scene.background = new THREE.Color(SCENE_BG);
            sceneRef.current = scene;
            camera = new THREE.PerspectiveCamera(48, 1, CAMERA_NEAR, CAMERA_FAR);
            camera.up.set(0, 1, 0);
            camera.position.set(0, 0, 10);
            cameraRef.current = camera;
            renderer = new THREE.WebGLRenderer({ antialias: true });
            // Cap the pixel ratio: full DPR on hi-DPI displays nearly doubles
            // the pixels rendered per frame for no visible gain on a map view,
            // and the full-width editor canvas made that cost noticeable.
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
            renderer.setClearColor(SCENE_BG, 1);
            renderer.domElement.className = "block w-full h-full cursor-grab";
            containerEl.appendChild(renderer.domElement);
            rendererRef.current = renderer;
            mapLayer = new THREE.Group();
            scene.add(mapLayer);
            mapLayerRef.current = mapLayer;
            layers = new THREE.Group();
            scene.add(layers);
            layersRef.current = layers;
            // Dedicated group for the pointer-following editor brush ring — kept
            // outside `layers` so hover tracking never rebuilds the heavy layers.
            const brushLayer = new THREE.Group();
            scene.add(brushLayer);
            editorBrushLayerRef.current = brushLayer;
            const costmapLayer = new THREE.Group();
            scene.add(costmapLayer);
            costmapLayerRef.current = costmapLayer;
            const navPathLayer = new THREE.Group();
            scene.add(navPathLayer);
            navPathLayerRef.current = navPathLayer;
            const liveLayer = new THREE.Group();
            scene.add(liveLayer);
            liveLayerRef.current = liveLayer;
            controls = new OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.08;
            controls.enableRotate = false;
            controls.mouseButtons = {
                LEFT: THREE.MOUSE.PAN,
                MIDDLE: THREE.MOUSE.DOLLY,
                RIGHT: THREE.MOUSE.PAN,
            };
            controls.screenSpacePanning = true;
            controlsRef.current = controls;
            handleControlsStart = () => {
                renderInteractionActiveRef.current = true;
            };
            handleControlsEnd = () => {
                renderInteractionActiveRef.current = false;
                renderActiveUntilRef.current = performance.now() + 400;
            };
            controls.addEventListener?.("start", handleControlsStart);
            controls.addEventListener?.("end", handleControlsEnd);
            resize = () => {
                const width = containerEl.clientWidth || 1;
                const height = containerEl.clientHeight || 1;
                renderer.setSize(width, height, false);
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderActiveUntilRef.current = performance.now() + 250;
            };
            resize();
            if (typeof ResizeObserver !== "undefined") {
                resizeObserver = new ResizeObserver(resize);
                resizeObserver.observe(containerEl);
            }
            else {
                window.addEventListener("resize", resize);
            }
            const pulseStart = performance.now();
            // Adapt rendering to actual activity. Static maps render at 10fps,
            // motion and interaction use 30fps, and background tabs drop to 2fps.
            let lastFrameAt = 0;
            const animate = () => {
                animationFrameRef.current = requestAnimationFrame(animate);
                const now = performance.now();
                const followTarget = followPoseRef.current;
                const followCameraMoving = Boolean(
                    followRobotRef.current &&
                    followTarget &&
                    Math.hypot(
                        followTarget.x - controls.target.x,
                        followTarget.y - controls.target.y,
                    ) > 0.002
                );
                const renderActive = (
                    renderInteractionActiveRef.current ||
                    followCameraMoving ||
                    pulseActiveRef.current ||
                    now < renderActiveUntilRef.current
                );
                const frameInterval = mapRenderIntervalMs({
                    hidden: document.visibilityState === "hidden",
                    active: renderActive,
                });
                if (now - lastFrameAt < frameInterval) return;
                lastFrameAt = now;
                const elapsed = (now - pulseStart) / 1000;
                // Pulse the active-waypoint halo (userData.pulse), and smoothly pan
                // the camera to keep the robot centered while it navigates.
                const wave = 0.5 + 0.5 * Math.sin(elapsed * 3.2);
                const activeLayers = layersRef.current;
                if (activeLayers && pulseActiveRef.current) {
                    activeLayers.traverse((object) => {
                        if (object.userData && object.userData.pulse && object.material) {
                            object.material.opacity = object.userData.pulseBase * (0.35 + 0.65 * wave);
                            object.scale.setScalar(1 + 0.16 * wave);
                        }
                    });
                }
                if (followRobotRef.current && followPoseRef.current) {
                    const target = followPoseRef.current;
                    const dx = (target.x - controls.target.x) * 0.08;
                    const dy = (target.y - controls.target.y) * 0.08;
                    if (Math.abs(dx) > 1e-4 || Math.abs(dy) > 1e-4) {
                        controls.target.x += dx;
                        controls.target.y += dy;
                        camera.position.x += dx;
                        camera.position.y += dy;
                    }
                }
                controls.update();
                renderer.render(scene, camera);
            };
            animate();
            setViewerError(null);
        }
        catch (error) {
            console.error("Navigation map viewer failed to initialize:", error);
            setViewerError(error instanceof Error ? error.message : "Map viewer failed to initialize");
            controls === null || controls === void 0 ? void 0 : controls.dispose();
            if (mapLayer)
                disposeObject(mapLayer);
            if (layers)
                disposeObject(layers);
            renderer === null || renderer === void 0 ? void 0 : renderer.dispose();
            if ((renderer === null || renderer === void 0 ? void 0 : renderer.domElement.parentNode) === containerEl) {
                containerEl.removeChild(renderer.domElement);
            }
            sceneRef.current = null;
            rendererRef.current = null;
            cameraRef.current = null;
            controlsRef.current = null;
            mapLayerRef.current = null;
            layersRef.current = null;
            editorBrushLayerRef.current = null;
            return undefined;
        }
        return () => {
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
            else if (resize) {
                window.removeEventListener("resize", resize);
            }
            if (animationFrameRef.current != null) {
                cancelAnimationFrame(animationFrameRef.current);
                animationFrameRef.current = null;
            }
            if (controls && handleControlsStart)
                controls.removeEventListener?.("start", handleControlsStart);
            if (controls && handleControlsEnd)
                controls.removeEventListener?.("end", handleControlsEnd);
            controls === null || controls === void 0 ? void 0 : controls.dispose();
            if (mapLayer)
                disposeObject(mapLayer);
            if (layers)
                disposeObject(layers);
            if (editorBrushLayerRef.current)
                disposeObject(editorBrushLayerRef.current);
            if (costmapLayerRef.current)
                disposeObject(costmapLayerRef.current);
            if (navPathLayerRef.current)
                disposeObject(navPathLayerRef.current);
            if (liveLayerRef.current)
                disposeObject(liveLayerRef.current);
            renderer === null || renderer === void 0 ? void 0 : renderer.dispose();
            if ((renderer === null || renderer === void 0 ? void 0 : renderer.domElement.parentNode) === containerEl) {
                containerEl.removeChild(renderer.domElement);
            }
            sceneRef.current = null;
            rendererRef.current = null;
            cameraRef.current = null;
            controlsRef.current = null;
            mapLayerRef.current = null;
            layersRef.current = null;
            editorBrushLayerRef.current = null;
            costmapLayerRef.current = null;
            globalCostmapTextureRef.current = null;
            navPathLayerRef.current = null;
            liveLayerRef.current = null;
        };
    }, []);
    // Editor brush ring — geometry rebuilt only when the brush spec changes.
    useEffect(() => {
        const brushLayer = editorBrushLayerRef.current;
        if (!brushLayer)
            return;
        disposeObject(brushLayer);
        brushLayer.clear();
        const meta = gridMeta(map);
        if (!editorBrush || !meta)
            return;
        const radius = (Math.max(1, Number(editorBrush.sizeCells) || 1) * meta.resolution) / 2;
        const color = new THREE.Color(editorBrush.color || "#5B8266");
        const group = new THREE.Group();
        // Ink/cream contrast underlay so a paper-colored ring stays visible on the map.
        const underlay = new THREE.Mesh(new THREE.RingGeometry(radius * 0.78, radius * 1.08, 40), new THREE.MeshBasicMaterial({
            color: 0x1c1a17,
            transparent: true,
            opacity: 0.32,
            depthWrite: false,
            side: THREE.DoubleSide,
        }));
        group.add(underlay);
        const ring = new THREE.Mesh(new THREE.RingGeometry(radius * 0.84, radius, 40), new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.9,
            depthWrite: false,
            side: THREE.DoubleSide,
        }));
        group.add(ring);
        const fill = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.84, 40), new THREE.MeshBasicMaterial({
            color,
            transparent: true,
            opacity: 0.12,
            depthWrite: false,
        }));
        group.add(fill);
        group.position.set(0, 0, 0.9);
        group.visible = false;
        brushLayer.add(group);
    }, [editorBrush, map]);
    // Follow the pointer via ref mutation — zero React renders per move.
    useEffect(() => {
        const renderer = rendererRef.current;
        const camera = cameraRef.current;
        if (!renderer || !camera || !editorBrush)
            return undefined;
        const meta = gridMeta(map);
        if (!meta)
            return undefined;
        const brushGroup = () => {
            var _a;
            return ((_a = editorBrushLayerRef.current) === null || _a === void 0 ? void 0 : _a.children[0]) || null;
        };
        const handleMove = (event) => {
            const group = brushGroup();
            if (!group)
                return;
            const rect = renderer.domElement.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0)
                return;
            pointerRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointerRef.current.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
            raycasterRef.current.setFromCamera(pointerRef.current, camera);
            const point = new THREE.Vector3();
            const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
            if (!raycasterRef.current.ray.intersectPlane(plane, point)) {
                group.visible = false;
                return;
            }
            const width = meta.width * meta.resolution;
            const height = meta.height * meta.resolution;
            const inside = point.x >= meta.originX &&
                point.x <= meta.originX + width &&
                point.y >= meta.originY &&
                point.y <= meta.originY + height;
            group.visible = inside;
            if (inside)
                group.position.set(point.x, point.y, 0.9);
        };
        const handleLeave = () => {
            const group = brushGroup();
            if (group)
                group.visible = false;
        };
        renderer.domElement.addEventListener("pointermove", handleMove);
        renderer.domElement.addEventListener("pointerleave", handleLeave);
        return () => {
            renderer.domElement.removeEventListener("pointermove", handleMove);
            renderer.domElement.removeEventListener("pointerleave", handleLeave);
            handleLeave();
        };
    }, [editorBrush, map]);
    useEffect(() => {
        const renderer = rendererRef.current;
        const controls = controlsRef.current;
        if (!renderer)
            return;
        const cursor = interactionDisabled
            ? "cursor-wait"
            : editorActive
                ? (editorAreaSelection ? "cursor-crosshair" : "cursor-cell")
                : interactionMode === "view"
                    ? mapDragActive ? "cursor-grabbing" : "cursor-grab"
                    : "cursor-crosshair";
        renderer.domElement.className = `block w-full h-full ${cursor}`;
        if (controls) {
            controls.enabled = !interactionDisabled && !editorActive && interactionMode === "view";
        }
    }, [editorActive, editorAreaSelection, interactionDisabled, interactionMode, mapDragActive]);
    useEffect(() => {
        const camera = cameraRef.current;
        const controls = controlsRef.current;
        const meta = gridMeta(map);
        if (!camera || !controls || !meta)
            return;
        if (btSpotPose) {
            focusCameraToWaypoint(camera, controls, meta, btSpotPose, viewRollRef.current);
            btFocusActiveRef.current = true;
            return;
        }
        if (btFocusActiveRef.current) {
            btFocusActiveRef.current = false;
        }
    }, [
        btSpotId,
        btSpotPose,
        btSpotTree,
        map,
        viewKey,
    ]);
    // While the runner is navigating, glide the camera to follow the robot; the
    // animation loop reads these refs each frame (no per-pose re-render).
    useEffect(() => {
        followRobotRef.current = missionFollowRobot;
        if (missionFollowRobot)
            renderActiveUntilRef.current = performance.now() + 400;
    }, [missionFollowRobot]);
    useEffect(() => {
        const nextPosition = (pose === null || pose === void 0 ? void 0 : pose.position)
            ? { x: Number(pose.position.x), y: Number(pose.position.y) }
            : null;
        followPoseRef.current = nextPosition;
        const nextMotionPose = nextPosition
            ? { ...nextPosition, yaw: yawFromPose(pose) }
            : null;
        const previous = lastMotionPoseRef.current;
        if (
            nextMotionPose &&
            (!previous ||
                Math.hypot(nextMotionPose.x - previous.x, nextMotionPose.y - previous.y) >= 0.01 ||
                Math.abs(Math.atan2(
                    Math.sin(nextMotionPose.yaw - previous.yaw),
                    Math.cos(nextMotionPose.yaw - previous.yaw),
                )) >= 0.01)
        ) {
            renderActiveUntilRef.current = performance.now() + 350;
        }
        lastMotionPoseRef.current = nextMotionPose;
    }, [pose]);
    useEffect(() => {
        const interactionActive = Boolean(
            mapDragActive || dragPreviewPose || nodeDragPreview || editorAreaPreview
        );
        renderInteractionActiveRef.current = interactionActive;
        if (!interactionActive)
            renderActiveUntilRef.current = performance.now() + 400;
    }, [dragPreviewPose, editorAreaPreview, mapDragActive, nodeDragPreview]);
    useEffect(() => {
        latestFootprintRef.current = footprint;
    }, [footprint]);
    useEffect(() => {
        tfSyncedFootprintRef.current = latestFootprintRef.current;
    }, [tf]);
    // The map is a static layer. Keep its texture alive while TF, pose, scan,
    // plans, and robot markers update in the dynamic layer below.
    useEffect(() => {
        const mapLayer = mapLayerRef.current;
        if (!mapLayer)
            return;
        disposeObject(mapLayer);
        mapLayer.clear();
        if (!showMap || !map)
            return;
        const mapPlane = makeGridPlane(map, "map", 0, null, null, mapRefined);
        if (mapPlane)
            mapLayer.add(mapPlane);
    }, [map, showMap, mapRefined]);
    useEffect(() => {
        const scene = sceneRef.current;
        const layers = layersRef.current;
        const camera = cameraRef.current;
        const controls = controlsRef.current;
        if (!scene || !layers || !camera || !controls)
            return;
        disposeObject(layers);
        layers.clear();
        const meta = gridMeta(map);
        const waypointScale = meta?.resolution && meta.resolution > 0 ? meta.resolution : 1;
        if (dragPreviewPose === null || dragPreviewPose === void 0 ? void 0 : dragPreviewPose.position) {
            const previewX = Number(dragPreviewPose.position.x ?? 0);
            const previewY = Number(dragPreviewPose.position.y ?? 0);
            const previewYaw = yawFromPose(dragPreviewPose);
            if (interactionMode === "spot") {
                layers.add(makeSpotMarker({
                    id: "__waypoint_preview__",
                    label: "Waypoint",
                    pose: { x: previewX, y: previewY, yaw: previewYaw },
                }, true, waypointScale));
            }
            else if (interactionMode === "behavior" && behaviorPreviewNode) {
                layers.add(makeBehaviorNodeMarker({
                    id: "__behavior_preview__",
                    tag: behaviorPreviewNode.tag,
                    label: behaviorPreviewNode.label || behaviorPreviewNode.tag,
                    category: behaviorPreviewNode.category || "action",
                    pose: { x: previewX, y: previewY, yaw: previewYaw },
                }, true));
            }
            else {
                layers.add(makePoseMarker(dragPreviewPose, interactionMode === "initial" ? 0x5b8266 : 0xc96442, 0.2));
            }
        }
        const spotById = new Map(spots.map((spot) => [spot.id, spot]));
        const coveredAnnotationCells = new Set();
        mapAnnotations.forEach((annotation) => {
            const marker = makeMapAnnotationRegion(annotation, map, coveredAnnotationCells, annotation.id === selectedMapAnnotationId);
            if (marker)
                layers.add(marker);
        });
        if (editorAreaPreview) {
            const preview = makeEditorAreaPreview(editorAreaPreview, map, coveredAnnotationCells);
            if (preview)
                layers.add(preview);
        }
        spots.forEach((spot) => {
            const preview = (nodeDragPreview === null || nodeDragPreview === void 0 ? void 0 : nodeDragPreview.type) === "spot" && nodeDragPreview.id === spot.id
                ? nodeDragPreview
                : null;
            const marker = makeSpotMarker(preview
                ? {
                    ...spot,
                    pose: {
                        ...spot.pose,
                        x: preview.x,
                        y: preview.y,
                        yaw: preview.yaw,
                    },
                }
                : spot, spot.id === selectedSpotId, waypointScale, spot.id === activeWaypointId);
            if (marker)
                layers.add(marker);
        });
        if (missionRouteOrder.length >= 2) {
            const routePoints = [...missionRouteOrder]
                .sort((a, b) => a.order - b.order)
                .map(({ id }) => spotById.get(id))
                .filter((spot) => spot === null || spot === void 0 ? void 0 : spot.pose)
                .map((spot) => new THREE.Vector3(Number(spot.pose.x ?? 0), Number(spot.pose.y ?? 0), 0.02));
            if (missionRouteClosed && routePoints.length > 1) {
                routePoints.push(routePoints[0].clone());
            }
            const routeLine = makeMissionRouteLine(routePoints, markerPalette.idle, waypointScale);
            if (routeLine)
                layers.add(routeLine);
        }
        missionRouteOrder.forEach(({ id, order }) => {
            const spot = spotById.get(id);
            const badge = makeMissionRouteBadge(spot, order, id === selectedMissionRouteSourceId, waypointScale);
            if (badge)
                layers.add(badge);
        });
        behaviorNodes.forEach((node) => {
            const preview = (nodeDragPreview === null || nodeDragPreview === void 0 ? void 0 : nodeDragPreview.type) === "behaviorNode" && nodeDragPreview.id === node.id
                ? nodeDragPreview
                : null;
            const marker = makeBehaviorNodeMarker(preview
                ? {
                    ...node,
                    pose: {
                        ...node.pose,
                        x: preview.x,
                        y: preview.y,
                        yaw: preview.yaw,
                    },
                }
                : node, node.id === selectedBehaviorNodeId);
            if (marker)
                layers.add(marker);
        });
        const fitKey = viewKey !== null && viewKey !== void 0 ? viewKey : "default";
        if (meta && fitMapKeyRef.current !== fitKey && !btSpotPose) {
            fitCameraToMap(camera, controls, meta, viewRollRef.current);
            fitMapKeyRef.current = fitKey;
        }
    }, [
        dragPreviewPose,
        editorAreaPreview,
        nodeDragPreview,
        interactionMode,
        missionRouteOrder,
        missionRouteClosed,
        mapAnnotations,
        selectedMapAnnotationId,
        selectedMissionRouteSourceId,
        behaviorPreviewNode,
        map,
        behaviorNodes,
        btSpotId,
        btSpotPose,
        selectedBehaviorNodeId,
        selectedSpotId,
        activeWaypointId,
        spots,
        viewKey,
    ]);
    // Global costmap plane. A full snapshot builds one DataTexture; subsequent
    // dirty rectangles patch only their affected rows in CPU memory and WebGL.
    // Geometry changes or a client resync still rebuild the complete plane.
    useEffect(() => {
        const group = costmapLayerRef.current;
        if (!group)
            return;
        if (!showGlobalCostmap || !globalCostmap) {
            disposeObject(group);
            group.clear();
            globalCostmapTextureRef.current = null;
            return;
        }
        const meta = gridMeta(globalCostmap);
        const geometryKey = meta
            ? `${meta.width}:${meta.height}:${meta.resolution}:${meta.originX}:${meta.originY}:${meta.originYaw}`
            : null;
        const current = globalCostmapTextureRef.current;
        if (
            current &&
            geometryKey &&
            current.geometryKey === geometryKey &&
            globalCostmap.updateRegion &&
            updateGlobalCostmapTexture(
                current.texture,
                globalCostmap,
                globalCostmap.updateRegion,
            )
        ) {
            return;
        }

        disposeObject(group);
        group.clear();
        globalCostmapTextureRef.current = null;
        const plane = makeGridPlane(globalCostmap, "globalCostmap", 0.03, null, null);
        if (plane) {
            group.add(plane);
            globalCostmapTextureRef.current = {
                geometryKey,
                texture: plane.userData.mapTexture,
            };
        }
    }, [globalCostmap, showGlobalCostmap]);
    // Planner output: global plan line + goal marker (updates at plan rate).
    useEffect(() => {
        var _a, _b;
        const group = navPathLayerRef.current;
        if (!group)
            return;
        disposeObject(group);
        group.clear();
        if (showGlobalPlan && ((_a = plan === null || plan === void 0 ? void 0 : plan.poses) === null || _a === void 0 ? void 0 : _a.length)) {
            const points = plan.poses
                .map((p) => { var _c; return (_c = p.pose) === null || _c === void 0 ? void 0 : _c.position; })
                .filter((p) => !!p)
                .map((p) => { var _c, _d; return new THREE.Vector3(Number((_c = p.x) !== null && _c !== void 0 ? _c : 0), Number((_d = p.y) !== null && _d !== void 0 ? _d : 0), 0.09); });
            const planLine = makeLine(points, 0x0e7fd1, 3);
            if (planLine)
                group.add(planLine);
        }
        if (showGoalPose && ((_b = goalPose === null || goalPose === void 0 ? void 0 : goalPose.pose) === null || _b === void 0 ? void 0 : _b.position)) {
            group.add(makePoseMarker(goalPose.pose, 0xc96442, 0.14));
        }
    }, [plan, showGlobalPlan, goalPose, showGoalPose]);
    // High-frequency live overlay: robot pose marker, footprint, scan points,
    // TF axes, and the (small) local costmap. Rebuilding this group is cheap;
    // everything expensive lives in the groups above and is untouched by TF.
    useEffect(() => {
        var _a, _b, _c, _d, _e, _f, _g, _j, _k, _l, _o, _p, _q, _r, _s, _t, _u;
        const group = liveLayerRef.current;
        if (!group)
            return;
        disposeObject(group);
        group.clear();
        const meta = gridMeta(map);
        const mapKey = meta ? `${meta.width}:${meta.height}:${meta.resolution}:${meta.originX}:${meta.originY}` : null;
        const tfSyncedFootprint = tfSyncedFootprintRef.current;
        const tfFramePoses = buildTfFramePoses(tf, "map");
        const tfFramePoseByName = new Map(tfFramePoses.map(({ frame, pose: framePose }) => [frame, framePose]));
        const robotX = Number((_b = (_a = pose === null || pose === void 0 ? void 0 : pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null && _b !== void 0 ? _b : 0);
        const robotY = Number((_d = (_c = pose === null || pose === void 0 ? void 0 : pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null && _d !== void 0 ? _d : 0);
        if (showLocalCostmap && localCostmap) {
            const localFrame = normalizeFrameId((_e = localCostmap.header) === null || _e === void 0 ? void 0 : _e.frame_id);
            const localFramePose = localFrame && localFrame !== "map"
                ? (_f = tfFramePoseByName.get(localFrame)) !== null && _f !== void 0 ? _f : null
                : null;
            const scanFrame = normalizeFrameId((_g = scan === null || scan === void 0 ? void 0 : scan.header) === null || _g === void 0 ? void 0 : _g.frame_id) || "base_link";
            const resolvedScanPose = scanPose
                ? poseForScanFrameAtBasePose(scanFrame, tfFramePoseByName, scanPose)
                : poseForScanFrame(scanFrame, tfFramePoseByName, pose);
            const scanCells = scanCellsForGrid(localCostmap, scan, resolvedScanPose, localFramePose);
            const plane = makeGridPlane(localCostmap, "localCostmap", 0.1, localFramePose, scanCells);
            if (plane)
                group.add(plane);
        }
        if (showScan && ((_j = scan === null || scan === void 0 ? void 0 : scan.ranges) === null || _j === void 0 ? void 0 : _j.length)) {
            const scanFrame = normalizeFrameId((_l = scan.header) === null || _l === void 0 ? void 0 : _l.frame_id) || "base_link";
            const resolvedScanPose = scanPose
                ? poseForScanFrameAtBasePose(scanFrame, tfFramePoseByName, scanPose)
                : poseForScanFrame(scanFrame, tfFramePoseByName, pose);
            // Keep the legacy Run projection frozen for the lifetime of one
            // LaserScan. Mapping supplies an explicit scan-time pose; include
            // only that pose in the cache key so a late matching odometry
            // sample can correct the same scan without making live Run scans
            // drift with subsequent robot-pose updates.
            const projectionPoseKey = scanPose?.position && resolvedScanPose?.position
                ? `${Number(resolvedScanPose.position.x ?? 0)}:${Number(resolvedScanPose.position.y ?? 0)}:${yawFromPose(resolvedScanPose)}`
                : "legacy";
            let points = ((_k = scanProjectionRef.current) === null || _k === void 0 ? void 0 : _k.scan) === scan &&
                scanProjectionRef.current.mapKey === mapKey &&
                scanProjectionRef.current.projectionPoseKey === projectionPoseKey
                ? scanProjectionRef.current.points
                : null;
            if (!points) {
                const scanX = Number((_p = (_o = resolvedScanPose === null || resolvedScanPose === void 0 ? void 0 : resolvedScanPose.position) === null || _o === void 0 ? void 0 : _o.x) !== null && _p !== void 0 ? _p : robotX);
                const scanY = Number((_r = (_q = resolvedScanPose === null || resolvedScanPose === void 0 ? void 0 : resolvedScanPose.position) === null || _q === void 0 ? void 0 : _q.y) !== null && _r !== void 0 ? _r : robotY);
                const scanYaw = yawFromPose(resolvedScanPose);
                const min = Number((_s = scan.range_min) !== null && _s !== void 0 ? _s : 0.02);
                const max = Number((_t = scan.range_max) !== null && _t !== void 0 ? _t : 20);
                const angleMin = Number((_u = scan.angle_min) !== null && _u !== void 0 ? _u : 0);
                const inc = Number(scan.angle_increment ?? 0);
                points = [];
                forEachVisualizedScanRange(scan.ranges, (range, index) => {
                    const r = Number(range);
                    if (!Number.isFinite(r) || r < min || r > max)
                        return;
                    const angle = scanYaw + angleMin + inc * index;
                    points.push(scanX + Math.cos(angle) * r, scanY + Math.sin(angle) * r, 0.11);
                });
                scanProjectionRef.current = { scan, mapKey, projectionPoseKey, points };
            }
            const geometry = new THREE.BufferGeometry();
            geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
            const material = new THREE.PointsMaterial({ color: 0x22c55e, size: 0.1, sizeAttenuation: true });
            group.add(new THREE.Points(geometry, material));
        }
        if (showTf && (tf === null || tf === void 0 ? void 0 : tf.transforms?.length)) {
            const framePoses = tfFramePoses.slice(0, 80);
            if (framePoses.length > 0) {
                framePoses.forEach(({ frame, pose: framePose }) => {
                    group.add(makeTfAxes(
                        poseForTfAxesFrame(frame, framePose, pose),
                        frame,
                    ));
                });
            }
            else {
                group.add(makeTfAxes({
                    position: { x: robotX, y: robotY, z: 0 },
                    orientation: pose === null || pose === void 0 ? void 0 : pose.orientation,
                }, "base_link"));
            }
        }
        const footprintPoints = tfSyncedFootprint === null || tfSyncedFootprint === void 0 ? void 0 : tfSyncedFootprint.polygon?.points;
        if (showRobotModel && (footprintPoints === null || footprintPoints === void 0 ? void 0 : footprintPoints.length)) {
            const footprintFrame = normalizeFrameId(tfSyncedFootprint.header?.frame_id);
            const tfFootprintFramePose = footprintFrame && footprintFrame !== "map"
                ? tfFramePoseByName.get(footprintFrame) ?? null
                : null;
            const footprintFramePose = poseForTfAxesFrame(footprintFrame, tfFootprintFramePose, pose);
            const footprintMarker = makeFootprintMarker(tfSyncedFootprint, footprintFramePose);
            if (footprintMarker)
                group.add(footprintMarker);
        }
        if (pose === null || pose === void 0 ? void 0 : pose.position) {
            group.add(makePoseMarker(pose, showRobotModel && (footprintPoints === null || footprintPoints === void 0 ? void 0 : footprintPoints.length) ? 0x60a5fa : 0x007acc, 0.16));
        }
    }, [tf, pose, scan, scanPose, localCostmap, showLocalCostmap, showScan, showTf, showRobotModel, map]);
    useEffect(() => {
        const renderer = rendererRef.current;
        const camera = cameraRef.current;
        const controls = controlsRef.current;
        if (!renderer || !camera || !controls)
            return;
        const mapPointFromEvent = (event) => {
            const meta = gridMeta(map);
            if (!meta)
                return null;
            const rect = renderer.domElement.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0)
                return null;
            pointerRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointerRef.current.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
            raycasterRef.current.setFromCamera(pointerRef.current, camera);
            const point = new THREE.Vector3();
            const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
            if (!raycasterRef.current.ray.intersectPlane(plane, point))
                return null;
            if (!Number.isFinite(point.x) || !Number.isFinite(point.y))
                return null;
            // Validate in the map's local grid frame. Axis-aligned world bounds
            // reject valid points (and accept invalid ones) when origin yaw is
            // non-zero.
            if (!mapPointToAreaGridCell(map, point.x, point.y))
                return null;
            return point;
        };
        const targetFromEvent = (event) => {
            const layers = layersRef.current;
            if (!layers || (typeof onSpotClick !== "function" && typeof onBehaviorNodeClick !== "function"))
                return null;
            const rect = renderer.domElement.getBoundingClientRect();
            if (rect.width <= 0 || rect.height <= 0)
                return null;
            pointerRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            pointerRef.current.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
            raycasterRef.current.setFromCamera(pointerRef.current, camera);
            const hits = raycasterRef.current.intersectObjects(layers.children, true);
            for (const hit of hits) {
                let object = hit.object;
                while (object) {
                    if (object.userData && object.userData.behaviorNodeId && typeof onBehaviorNodeClick === "function")
                        return { type: "behaviorNode", id: object.userData.behaviorNodeId, dragAction: "move" };
                    if (object.userData && object.userData.spotId && typeof onSpotClick === "function")
                        return {
                            type: "spot",
                            id: object.userData.spotId,
                            dragAction: object.userData.dragAction || "move",
                        };
                    object = object.parent;
                }
            }
            return null;
        };
        const draggableTargetPose = (target) => {
            var _a;
            if ((target === null || target === void 0 ? void 0 : target.type) === "spot") {
                const spot = spots.find((item) => item.id === target.id);
                if (!spot || typeof onSpotPoseChange !== "function")
                    return null;
                const spotPose = spot.pose;
                if (!spotPose)
                    return null;
                return {
                    x: Number((_a = spotPose.x) !== null && _a !== void 0 ? _a : 0),
                    y: Number(spotPose.y !== null && spotPose.y !== void 0 ? spotPose.y : 0),
                    yaw: Number(spotPose.yaw !== null && spotPose.yaw !== void 0 ? spotPose.yaw : 0),
                };
            }
            if ((target === null || target === void 0 ? void 0 : target.type) === "behaviorNode") {
                const node = behaviorNodes.find((item) => item.id === target.id);
                if (!node || node.id !== selectedBehaviorNodeId || typeof onBehaviorNodePoseChange !== "function")
                    return null;
                const nodePose = node.pose;
                if (!nodePose)
                    return null;
                return {
                    x: Number(nodePose.x !== null && nodePose.x !== void 0 ? nodePose.x : 0),
                    y: Number(nodePose.y !== null && nodePose.y !== void 0 ? nodePose.y : 0),
                    yaw: Number(nodePose.yaw !== null && nodePose.yaw !== void 0 ? nodePose.yaw : 0),
                };
            }
            return null;
        };
        const previewPoseFromDrag = (start, point, clientX, clientY) => {
            const moved = Math.hypot(clientX - start.clientX, clientY - start.clientY);
            const yaw = moved > CLICK_DRAG_THRESHOLD_PX
                ? Math.atan2(point.y - start.mapY, point.x - start.mapX)
                : yawFromPose(pose);
            return {
                position: { x: start.mapX, y: start.mapY, z: 0 },
                orientation: orientationFromYaw(yaw),
            };
        };
        const paintEditorPoint = (event, phase = "paint") => {
            if (typeof onEditorMapPoint !== "function")
                return false;
            const point = mapPointFromEvent(event);
            if (!point)
                return false;
            event.preventDefault();
            event.stopImmediatePropagation();
            onEditorMapPoint(point.x, point.y, phase);
            return true;
        };
        // Coalesce paint moves to one commit per animation frame — pointermove can
        // fire far faster than strokes need, and each commit re-rasterizes the
        // stroked annotation.
        const flushEditorMove = () => {
            if (editorMoveRafRef.current != null) {
                cancelAnimationFrame(editorMoveRafRef.current);
                editorMoveRafRef.current = null;
            }
            const pending = editorMovePendingRef.current;
            editorMovePendingRef.current = null;
            if (pending)
                paintEditorPoint(pending, "move");
        };
        const queueEditorMove = (event) => {
            editorMovePendingRef.current = {
                clientX: event.clientX,
                clientY: event.clientY,
                preventDefault: () => { },
                stopImmediatePropagation: () => { },
            };
            if (editorMoveRafRef.current == null) {
                editorMoveRafRef.current = requestAnimationFrame(() => {
                    editorMoveRafRef.current = null;
                    const pending = editorMovePendingRef.current;
                    editorMovePendingRef.current = null;
                    if (pending)
                        paintEditorPoint(pending, "move");
                });
            }
        };
        const handlePointerDown = (event) => {
            if (event.button === 2) {
                if (interactionDisabled || editorActive || interactionMode !== "view")
                    return;
                event.preventDefault();
                event.stopImmediatePropagation();
                viewRotateDragRef.current = {
                    clientX: event.clientX,
                    roll: viewRollRef.current,
                };
                // The custom roll drag bypasses OrbitControls, so raise the
                // interaction flag ourselves — otherwise the render loop stays
                // at the idle frame rate and the rotation looks choppy.
                renderInteractionActiveRef.current = true;
                renderer.domElement.setPointerCapture(event.pointerId);
                return;
            }
            if (event.button !== 0)
                return;
            if (editorActive) {
                if (interactionDisabled) {
                    pointerDownRef.current = null;
                    setDragPreviewPose(null);
                    return;
                }
                if (editorAreaSelection && typeof onEditorMapArea === "function") {
                    const point = mapPointFromEvent(event);
                    if (!point)
                        return;
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    editorAreaDragRef.current = {
                        pointerId: event.pointerId,
                        clientX: event.clientX,
                        clientY: event.clientY,
                        startX: point.x,
                        startY: point.y,
                        endX: point.x,
                        endY: point.y,
                    };
                    setEditorAreaPreview(null);
                    renderer.domElement.setPointerCapture(event.pointerId);
                    return;
                }
                if (paintEditorPoint(event, "start") && editorPaintOnDrag) {
                    editorPaintPointerRef.current = event.pointerId;
                    renderer.domElement.setPointerCapture(event.pointerId);
                }
                return;
            }
            if (!interactionDisabled && interactionMode === "view") {
                const target = targetFromEvent(event);
                if (target) {
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    if (missionRouteMode && target.type === "spot" && typeof onMissionRouteSpotClick === "function") {
                        onMissionRouteSpotClick(target.id);
                    }
                    else if (target.type === "behaviorNode") {
                        onBehaviorNodeClick(target.id);
                    }
                    else {
                        onSpotClick(target.id);
                    }
                    const point = mapPointFromEvent(event);
                    const targetPose = point && !missionRouteMode ? draggableTargetPose(target) : null;
                    if (point && targetPose) {
                        nodeDragRef.current = {
                            type: target.type,
                            id: target.id,
                            dragAction: target.dragAction || "move",
                            clientX: event.clientX,
                            clientY: event.clientY,
                            offsetX: point.x - targetPose.x,
                            offsetY: point.y - targetPose.y,
                            x: targetPose.x,
                            y: targetPose.y,
                            yaw: targetPose.yaw,
                            dragging: false,
                        };
                        renderer.domElement.setPointerCapture(event.pointerId);
                    }
                    else {
                        nodeDragRef.current = null;
                    }
                    pointerDownRef.current = null;
                    setDragPreviewPose(null);
                    return;
                }
            }
            if (interactionDisabled || interactionMode === "view") {
                if (!interactionDisabled && interactionMode === "view" && missionRouteMode && typeof onMissionRouteMapClick === "function") {
                    onMissionRouteMapClick();
                }
                else if (!interactionDisabled && interactionMode === "view" && typeof onMapClick === "function") {
                    const point = mapPointFromEvent(event);
                    if (point) {
                        onMapClick(point.x, point.y);
                    }
                }
                if (!interactionDisabled && interactionMode === "view") {
                    setMapDragActive(true);
                }
                pointerDownRef.current = null;
                setDragPreviewPose(null);
                return;
            }
            const point = mapPointFromEvent(event);
            if (!point) {
                pointerDownRef.current = null;
                setDragPreviewPose(null);
                return;
            }
            const pointerDown = {
                clientX: event.clientX,
                clientY: event.clientY,
                mapX: point.x,
                mapY: point.y,
            };
            pointerDownRef.current = pointerDown;
            setDragPreviewPose(previewPoseFromDrag(pointerDown, point, event.clientX, event.clientY));
            renderer.domElement.setPointerCapture(event.pointerId);
        };
        const handlePointerMove = (event) => {
            const editorAreaDrag = editorAreaDragRef.current;
            if (editorAreaDrag && editorAreaDrag.pointerId === event.pointerId) {
                const point = mapPointFromEvent(event);
                if (!point)
                    return;
                event.preventDefault();
                event.stopImmediatePropagation();
                editorAreaDrag.endX = point.x;
                editorAreaDrag.endY = point.y;
                setEditorAreaPreview({
                    startX: editorAreaDrag.startX,
                    startY: editorAreaDrag.startY,
                    endX: point.x,
                    endY: point.y,
                });
                return;
            }
            if (editorPaintPointerRef.current === event.pointerId) {
                event.preventDefault();
                queueEditorMove(event);
                return;
            }
            const nodeDrag = nodeDragRef.current;
            if (nodeDrag) {
                const point = mapPointFromEvent(event);
                if (!point)
                    return;
                event.preventDefault();
                event.stopImmediatePropagation();
                const moved = Math.hypot(event.clientX - nodeDrag.clientX, event.clientY - nodeDrag.clientY);
                const nextX = nodeDrag.dragAction === "rotate"
                    ? nodeDrag.x
                    : point.x - nodeDrag.offsetX;
                const nextY = nodeDrag.dragAction === "rotate"
                    ? nodeDrag.y
                    : point.y - nodeDrag.offsetY;
                const nextYaw = nodeDrag.dragAction === "rotate"
                    ? Math.atan2(point.y - nodeDrag.y, point.x - nodeDrag.x)
                    : nodeDrag.yaw;
                nodeDrag.x = nextX;
                nodeDrag.y = nextY;
                nodeDrag.yaw = nextYaw;
                nodeDrag.dragging = nodeDrag.dragging || moved > CLICK_DRAG_THRESHOLD_PX;
                if (nodeDrag.dragging) {
                    setNodeDragPreview({
                        type: nodeDrag.type,
                        id: nodeDrag.id,
                        x: nextX,
                        y: nextY,
                        yaw: nextYaw,
                    });
                }
                return;
            }
            const viewRotateDrag = viewRotateDragRef.current;
            if (viewRotateDrag) {
                event.preventDefault();
                event.stopImmediatePropagation();
                const nextRoll = viewRotateDrag.roll - (event.clientX - viewRotateDrag.clientX) * 0.01;
                viewRollRef.current = nextRoll;
                applyTopViewRoll(camera, controls, nextRoll);
                return;
            }
            if (interactionDisabled || interactionMode === "view")
                return;
            const pointerDown = pointerDownRef.current;
            if (!pointerDown)
                return;
            const point = mapPointFromEvent(event);
            if (!point)
                return;
            setDragPreviewPose(previewPoseFromDrag(pointerDown, point, event.clientX, event.clientY));
        };
        const handlePointerUp = (event) => {
            setMapDragActive(false);
            const editorAreaDrag = editorAreaDragRef.current;
            if (editorAreaDrag && editorAreaDrag.pointerId === event.pointerId) {
                event.preventDefault();
                event.stopImmediatePropagation();
                const point = mapPointFromEvent(event);
                const endX = point ? point.x : editorAreaDrag.endX;
                const endY = point ? point.y : editorAreaDrag.endY;
                editorAreaDragRef.current = null;
                setEditorAreaPreview(null);
                if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                    renderer.domElement.releasePointerCapture(event.pointerId);
                }
                onEditorMapArea(editorAreaDrag.startX, editorAreaDrag.startY, endX, endY);
                return;
            }
            if (editorPaintPointerRef.current === event.pointerId) {
                event.preventDefault();
                event.stopImmediatePropagation();
                editorPaintPointerRef.current = null;
                flushEditorMove();
                if (typeof onEditorMapPoint === "function")
                    onEditorMapPoint(0, 0, "end");
                if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                    renderer.domElement.releasePointerCapture(event.pointerId);
                }
                return;
            }
            if (nodeDragRef.current) {
                event.preventDefault();
                event.stopImmediatePropagation();
                const nodeDrag = nodeDragRef.current;
                nodeDragRef.current = null;
                setNodeDragPreview(null);
                if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                    renderer.domElement.releasePointerCapture(event.pointerId);
                }
                if (nodeDrag.dragging) {
                    if (nodeDrag.type === "spot" && typeof onSpotPoseChange === "function") {
                        onSpotPoseChange(nodeDrag.id, nodeDrag.x, nodeDrag.y, nodeDrag.yaw);
                    }
                    else if (nodeDrag.type === "behaviorNode" && typeof onBehaviorNodePoseChange === "function") {
                        onBehaviorNodePoseChange(nodeDrag.id, nodeDrag.x, nodeDrag.y, nodeDrag.yaw);
                    }
                }
                return;
            }
            if (viewRotateDragRef.current) {
                event.preventDefault();
                event.stopImmediatePropagation();
                viewRotateDragRef.current = null;
                renderInteractionActiveRef.current = false;
                renderActiveUntilRef.current = performance.now() + 400;
                if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                    renderer.domElement.releasePointerCapture(event.pointerId);
                }
                return;
            }
            if (interactionDisabled || interactionMode === "view" || event.button !== 0)
                return;
            const pointerDown = pointerDownRef.current;
            pointerDownRef.current = null;
            setDragPreviewPose(null);
            if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                renderer.domElement.releasePointerCapture(event.pointerId);
            }
            if (!pointerDown)
                return;
            const point = mapPointFromEvent(event);
            if (!point)
                return;
            const moved = Math.hypot(event.clientX - pointerDown.clientX, event.clientY - pointerDown.clientY);
            const yaw = moved > CLICK_DRAG_THRESHOLD_PX
                ? Math.atan2(point.y - pointerDown.mapY, point.x - pointerDown.mapX)
                : yawFromPose(pose);
            onMapPose(pointerDown.mapX, pointerDown.mapY, yaw);
        };
        const handlePointerCancel = (event) => {
            const hadPaintPointer = editorPaintPointerRef.current === event.pointerId;
            editorPaintPointerRef.current = null;
            editorAreaDragRef.current = null;
            if (viewRotateDragRef.current) {
                renderInteractionActiveRef.current = false;
                renderActiveUntilRef.current = performance.now() + 400;
            }
            viewRotateDragRef.current = null;
            nodeDragRef.current = null;
            pointerDownRef.current = null;
            setMapDragActive(false);
            setEditorAreaPreview(null);
            setNodeDragPreview(null);
            setDragPreviewPose(null);
            if (hadPaintPointer) {
                flushEditorMove();
                if (typeof onEditorMapPoint === "function")
                    onEditorMapPoint(0, 0, "end");
            }
            if (renderer.domElement.hasPointerCapture(event.pointerId)) {
                renderer.domElement.releasePointerCapture(event.pointerId);
            }
        };
        const handlePointerLeave = () => {
            setMapDragActive(false);
        };
        const handleContextMenu = (event) => {
            event.preventDefault();
        };
        renderer.domElement.addEventListener("pointerdown", handlePointerDown, { capture: true });
        renderer.domElement.addEventListener("pointermove", handlePointerMove);
        renderer.domElement.addEventListener("pointerup", handlePointerUp);
        renderer.domElement.addEventListener("pointercancel", handlePointerCancel);
        renderer.domElement.addEventListener("pointerleave", handlePointerLeave);
        renderer.domElement.addEventListener("contextmenu", handleContextMenu);
        return () => {
            renderer.domElement.removeEventListener("pointerdown", handlePointerDown, { capture: true });
            renderer.domElement.removeEventListener("pointermove", handlePointerMove);
            renderer.domElement.removeEventListener("pointerup", handlePointerUp);
            renderer.domElement.removeEventListener("pointercancel", handlePointerCancel);
            renderer.domElement.removeEventListener("pointerleave", handlePointerLeave);
            renderer.domElement.removeEventListener("contextmenu", handleContextMenu);
            if (editorMoveRafRef.current != null) {
                cancelAnimationFrame(editorMoveRafRef.current);
                editorMoveRafRef.current = null;
            }
            editorMovePendingRef.current = null;
        };
    }, [
        behaviorNodes,
        editorActive,
        editorAreaSelection,
        editorPaintOnDrag,
        interactionDisabled,
        interactionMode,
        map,
        missionRouteMode,
        onBehaviorNodeClick,
        onBehaviorNodePoseChange,
        onEditorMapArea,
        onEditorMapPoint,
        onMapPose,
        onMapClick,
        onMissionRouteMapClick,
        onMissionRouteSpotClick,
        onSpotClick,
        onSpotPoseChange,
        pose,
        selectedBehaviorNodeId,
        selectedSpotId,
        spots,
    ]);
    return (<div className={`relative border rounded-md min-h-0 overflow-hidden ${fitContainer ? "h-full w-full" : ""}`} style={{
            ...(fitContainer
                ? { height: "100%", width: "100%" }
                : { aspectRatio: "1 / 1" }),
            backgroundColor: "var(--vscode-editor-background)",
            borderColor: "var(--vscode-panel-border)",
        }}>
      <div ref={containerRef} className="h-full w-full"/>
      {viewerError && (<div className="absolute inset-0 flex items-center justify-center p-4 text-center text-sm pointer-events-none" style={{ color: "var(--vscode-descriptionForeground)" }}>
          Map viewer unavailable: {viewerError}
        </div>)}
      {!viewerError && showMap && !map && (<div className="absolute inset-0 flex items-center justify-center text-sm pointer-events-none" style={{ color: "var(--vscode-descriptionForeground)" }}>
          {waitingLabel}
        </div>)}
      <WaypointBtFocusLayer layer={btLayer} onClose={onBtLayerClose}/>
    </div>);
}
