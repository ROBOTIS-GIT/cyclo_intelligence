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
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MdRedo, MdUndo } from "react-icons/md";
import { mapPointToAreaGridCell } from "../../utils/mapAreaGeometry";
import { getMapAnnotations, getPgmFiles, getPgmImage, saveMapAnnotations, savePgmImage } from "../../utils/navigationApi";
import { yawFromPose } from "../../utils/navigationTf";
const FREE_VALUE = 254;
const OCCUPIED_VALUE = 0;
const UNKNOWN_VALUE = 205; // ROS map_server convention for unexplored cells
const FREE_THRESHOLD = 250;
const OCCUPIED_THRESHOLD = 50;
const DEFAULT_BRUSH_SIZE_CELLS = 1;
const MAX_BRUSH_SIZE_CELLS = 10;
export const BRUSH_SIZE_OPTIONS = [
    { label: "Small", value: 1 },
    { label: "Medium", value: 3 },
    { label: "Large", value: 6 },
    { label: "XL", value: 10 },
];
export const EDIT_TOOLS = [
    { id: "erase_black", label: "Clear Space" },
    { id: "draw_black", label: "Add Obstacle" },
    { id: "draw_unknown", label: "Mark Unknown" },
];
const PIXEL_TOOL_IDS = new Set(EDIT_TOOLS.map((tool) => tool.id));
export const ANNOTATION_TOOL = { id: "label_marker", label: "Area" };
export const ANNOTATION_EXTEND_TOOL = { id: "extend_area", label: "Extend" };
export const ANNOTATION_ERASE_TOOL = { id: "erase_area", label: "Erase Area" };
// Area colors deliberately avoid green and orange/amber hues — those belong to
// the marker language (sage = waypoints/actions, amber = decorators + local
// costmap, clay = selection) and areas must not read as markers. Pastel base
// tones keep areas as a quiet background wash under mission markers; the
// viewer additionally pastelizes stored colors so legacy dark areas match.
const ANNOTATION_COLORS = [
    { id: "dark_brown", label: "Warm sand", value: "#CBB9A4" },
    { id: "red_wine", label: "Rose", value: "#D9A4AB" },
    { id: "navy", label: "Powder blue", value: "#A9BFD9" },
    { id: "gray", label: "Silver gray", value: "#BFC3C9" },
    { id: "plum", label: "Lavender", value: "#C0ABD6" },
    { id: "teal", label: "Soft teal", value: "#9CC5C7" },
    { id: "indigo", label: "Periwinkle", value: "#A9AEDB" },
    { id: "berry", label: "Dusty berry", value: "#CFA3BE" },
];
function decodePgmPixels(image) {
    const binary = window.atob(image.pixels_base64);
    const pixels = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        pixels[index] = binary.charCodeAt(index);
    }
    return pixels;
}
function encodePgmPixels(pixels) {
    let binary = "";
    const chunkSize = 0x8000;
    for (let index = 0; index < pixels.length; index += chunkSize) {
        const chunk = pixels.subarray(index, index + chunkSize);
        binary += String.fromCharCode.apply(null, Array.from(chunk));
    }
    return window.btoa(binary);
}
function applyPgmBrush(pixels, width, height, pixelX, pixelY, operation, brushSizeCells) {
    const value = operation === "erase_black"
        ? FREE_VALUE
        : operation === "draw_unknown"
            ? UNKNOWN_VALUE
            : OCCUPIED_VALUE;
    const offset = Math.floor(brushSizeCells / 2);
    const startX = pixelX - offset;
    const startY = pixelY - offset;
    for (let y = startY; y < startY + brushSizeCells; y += 1) {
        if (y < 0 || y >= height)
            continue;
        for (let x = startX; x < startX + brushSizeCells; x += 1) {
            if (x < 0 || x >= width)
                continue;
            pixels[x + y * width] = value;
        }
    }
}
function paintPgmPixels(pixels, width, height, pixelX, pixelY, operation, brushSizeCells) {
    const next = new Uint8Array(pixels);
    applyPgmBrush(next, width, height, pixelX, pixelY, operation, brushSizeCells);
    return next;
}
function paintPgmPixelSegment(pixels, width, height, startPixel, endPixel, operation, brushSizeCells) {
    const next = new Uint8Array(pixels);
    let x0 = startPixel.pixelX;
    let y0 = startPixel.pixelY;
    const x1 = endPixel.pixelX;
    const y1 = endPixel.pixelY;
    const dx = Math.abs(x1 - x0);
    const sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0);
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    while (true) {
        applyPgmBrush(next, width, height, x0, y0, operation, brushSizeCells);
        if (x0 === x1 && y0 === y1)
            break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    return next;
}
function annotationGridBounds(annotation, image) {
    const region = normalizeAnnotationRegion(annotation === null || annotation === void 0 ? void 0 : annotation.region);
    if (!region)
        return null;
    if ((region.width && region.width !== image.width) || (region.height && region.height !== image.height))
        return null;
    const bounds = region.bounds || {
        x_min: region.seed_cell.x,
        y_min: region.seed_cell.y,
        x_max: region.seed_cell.x,
        y_max: region.seed_cell.y,
    };
    return {
        x_min: Math.max(0, Math.min(bounds.x_min, bounds.x_max)),
        y_min: Math.max(0, Math.min(bounds.y_min, bounds.y_max)),
        x_max: Math.min(image.width - 1, Math.max(bounds.x_min, bounds.x_max)),
        y_max: Math.min(image.height - 1, Math.max(bounds.y_min, bounds.y_max)),
    };
}
function cellKey(cell) {
    return `${cell.x}:${cell.y}`;
}
function normalizeRegionCells(cells, width = null, height = null) {
    if (!Array.isArray(cells))
        return [];
    const seen = new Set();
    const normalized = [];
    cells.forEach((cell) => {
        if (!cell)
            return;
        const x = Math.floor(Number(cell.x));
        const y = Math.floor(Number(cell.y));
        if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0)
            return;
        if (width && x >= width)
            return;
        if (height && y >= height)
            return;
        const key = `${x}:${y}`;
        if (seen.has(key))
            return;
        seen.add(key);
        normalized.push({ x, y });
    });
    normalized.sort((a, b) => a.y - b.y || a.x - b.x);
    return normalized;
}
function pgmPixelToGridCell(image, pixel) {
    return {
        x: pixel.pixelX,
        y: image.height - 1 - pixel.pixelY,
    };
}
function gridCellToPgmIndex(image, cell) {
    const pgmY = image.height - 1 - cell.y;
    if (cell.x < 0 || cell.x >= image.width || pgmY < 0 || pgmY >= image.height)
        return -1;
    return cell.x + pgmY * image.width;
}
function annotationCells(annotation, image, pixels = null) {
    const region = normalizeAnnotationRegion(annotation === null || annotation === void 0 ? void 0 : annotation.region);
    if (!region)
        return [];
    if ((region.width && region.width !== image.width) || (region.height && region.height !== image.height))
        return [];
    if (Array.isArray(region.cells) && region.cells.length) {
        return normalizeRegionCells(region.cells, image.width, image.height);
    }
    const bounds = annotationGridBounds(annotation, image);
    if (!bounds)
        return [];
    const cells = [];
    for (let y = bounds.y_min; y <= bounds.y_max; y += 1) {
        for (let x = bounds.x_min; x <= bounds.x_max; x += 1) {
            const cell = { x, y };
            const index = pixels ? gridCellToPgmIndex(image, cell) : -1;
            if (pixels && (index < 0 || pixels[index] < FREE_THRESHOLD))
                continue;
            cells.push(cell);
        }
    }
    return cells;
}
function cellsToRegion(cells, image) {
    const normalizedCells = normalizeRegionCells(cells, image.width, image.height);
    if (!normalizedCells.length)
        return null;
    let xMin = normalizedCells[0].x;
    let xMax = normalizedCells[0].x;
    let yMin = normalizedCells[0].y;
    let yMax = normalizedCells[0].y;
    let sumX = 0;
    let sumY = 0;
    normalizedCells.forEach((cell) => {
        xMin = Math.min(xMin, cell.x);
        xMax = Math.max(xMax, cell.x);
        yMin = Math.min(yMin, cell.y);
        yMax = Math.max(yMax, cell.y);
        sumX += cell.x;
        sumY += cell.y;
    });
    const seedCell = {
        x: Math.floor(sumX / normalizedCells.length),
        y: Math.floor(sumY / normalizedCells.length),
    };
    const centroidPgmY = image.height - 1 - (sumY / normalizedCells.length);
    const centroid = pgmPixelToMapPoint(image, sumX / normalizedCells.length, centroidPgmY);
    return {
        pose: { frame_id: "map", x: centroid.x, y: centroid.y, yaw: 0 },
        region: {
            seed_cell: seedCell,
            bounds: { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax },
            cells: normalizedCells,
            cell_count: normalizedCells.length,
            width: image.width,
            height: image.height,
        },
    };
}
// Fast-path region builder for stroke sessions: assumes `cells` are already
// deduped/in-range (session Maps guarantee this), so it skips the O(n log n)
// sort + revalidation that cellsToRegion pays on every brush move.
function regionFromCells(cells, image) {
    if (!cells.length)
        return null;
    let xMin = cells[0].x;
    let xMax = cells[0].x;
    let yMin = cells[0].y;
    let yMax = cells[0].y;
    let sumX = 0;
    let sumY = 0;
    cells.forEach((cell) => {
        if (cell.x < xMin)
            xMin = cell.x;
        if (cell.x > xMax)
            xMax = cell.x;
        if (cell.y < yMin)
            yMin = cell.y;
        if (cell.y > yMax)
            yMax = cell.y;
        sumX += cell.x;
        sumY += cell.y;
    });
    const centroidX = sumX / cells.length;
    const centroidY = sumY / cells.length;
    const centroidPgmY = image.height - 1 - centroidY;
    const centroid = pgmPixelToMapPoint(image, centroidX, centroidPgmY);
    return {
        pose: { frame_id: "map", x: centroid.x, y: centroid.y, yaw: 0 },
        region: {
            seed_cell: { x: Math.floor(centroidX), y: Math.floor(centroidY) },
            bounds: { x_min: xMin, y_min: yMin, x_max: xMax, y_max: yMax },
            cells,
            cell_count: cells.length,
            width: image.width,
            height: image.height,
        },
    };
}
function coveredAnnotationCellSet(annotations, image, pixels, excludeId = "") {
    const covered = new Set();
    annotations.forEach((annotation) => {
        if (excludeId && annotation.id === excludeId)
            return;
        annotationCells(annotation, image, pixels).forEach((cell) => covered.add(cellKey(cell)));
    });
    return covered;
}
function visibleAnnotationCellSnapshots(annotations, image, pixels) {
    const covered = new Set();
    return annotations
        .map((annotation) => {
        const visibleCells = annotationCells(annotation, image, pixels)
            .filter((cell) => !covered.has(cellKey(cell)));
        visibleCells.forEach((cell) => covered.add(cellKey(cell)));
        return { annotation, cells: visibleCells };
    })
        .filter((snapshot) => snapshot.cells.length > 0);
}
function annotationWithCells(annotation, cells, image) {
    const geometry = cellsToRegion(cells, image);
    if (!geometry)
        return null;
    return normalizeAnnotation({
        ...annotation,
        pose: geometry.pose,
        region: geometry.region,
    });
}
function brushCellsForSegment(pixels, image, startPixel, endPixel, brushSizeCells, coveredCells = new Set()) {
    const cells = new Map();
    const addBrush = (pixelX, pixelY) => {
        const offset = Math.floor(brushSizeCells / 2);
        for (let y = pixelY - offset; y < pixelY - offset + brushSizeCells; y += 1) {
            if (y < 0 || y >= image.height)
                continue;
            for (let x = pixelX - offset; x < pixelX - offset + brushSizeCells; x += 1) {
                if (x < 0 || x >= image.width)
                    continue;
                if (pixels[x + y * image.width] < FREE_THRESHOLD)
                    continue;
                const cell = pgmPixelToGridCell(image, { pixelX: x, pixelY: y });
                const key = cellKey(cell);
                if (coveredCells.has(key))
                    continue;
                cells.set(key, cell);
            }
        }
    };
    let x0 = startPixel.pixelX;
    let y0 = startPixel.pixelY;
    const x1 = endPixel.pixelX;
    const y1 = endPixel.pixelY;
    const dx = Math.abs(x1 - x0);
    const sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0);
    const sy = y0 < y1 ? 1 : -1;
    let err = dx + dy;
    while (true) {
        addBrush(x0, y0);
        if (x0 === x1 && y0 === y1)
            break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
    return Array.from(cells.values());
}
function selectedFreePgmRegion(pixels, image, startPixel, endPixel, existingAnnotations = []) {
    if (!pixels || !image || !startPixel || !endPixel)
        return null;
    const xMin = Math.max(0, Math.min(startPixel.pixelX, endPixel.pixelX));
    const xMax = Math.min(image.width - 1, Math.max(startPixel.pixelX, endPixel.pixelX));
    const yMin = Math.max(0, Math.min(startPixel.pixelY, endPixel.pixelY));
    const yMax = Math.min(image.height - 1, Math.max(startPixel.pixelY, endPixel.pixelY));
    // Precompute the covered pgm-index set ONCE — checking every rect pixel
    // against every annotation's cell list is O(rect × cells) and freezes the
    // UI on large selections.
    const covered = new Set();
    (existingAnnotations || []).forEach((annotation) => {
        annotationCells(annotation, image, pixels).forEach((cell) => {
            const index = gridCellToPgmIndex(image, cell);
            if (index >= 0)
                covered.add(index);
        });
    });
    let count = 0;
    let sumX = 0;
    let sumY = 0;
    for (let y = yMin; y <= yMax; y += 1) {
        for (let x = xMin; x <= xMax; x += 1) {
            const index = x + y * image.width;
            if (pixels[index] < FREE_THRESHOLD)
                continue;
            if (covered.has(index))
                continue;
            count += 1;
            sumX += x;
            sumY += y;
        }
    }
    const gridYMin = image.height - 1 - yMax;
    const gridYMax = image.height - 1 - yMin;
    return {
        count,
        centroidPixelX: count ? sumX / count : (xMin + xMax) / 2,
        centroidPixelY: count ? sumY / count : (yMin + yMax) / 2,
        bounds: {
            x_min: xMin,
            y_min: gridYMin,
            x_max: xMax,
            y_max: gridYMax,
        },
        seedCell: {
            x: Math.floor((xMin + xMax) / 2),
            y: Math.floor((gridYMin + gridYMax) / 2),
        },
    };
}
function pgmPixelsToGrid(image, pixels) {
    const resolution = Number(image.resolution ?? 1) || 1;
    const origin = image.origin ?? {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
    };
    const data = new Array(image.width * image.height);
    for (let pgmY = 0; pgmY < image.height; pgmY += 1) {
        for (let x = 0; x < image.width; x += 1) {
            const value = pixels[x + pgmY * image.width];
            const occupancy = value <= OCCUPIED_THRESHOLD
                ? 100
                : value >= FREE_THRESHOLD
                    ? 0
                    : -1;
            const gridY = image.height - 1 - pgmY;
            data[x + gridY * image.width] = occupancy;
        }
    }
    return {
        header: { frame_id: "map" },
        info: {
            resolution,
            width: image.width,
            height: image.height,
            origin,
        },
        data,
    };
}
function mapPointToPgmPixel(image, x, y) {
    const cell = mapPointToAreaGridCell(image, x, y);
    if (!cell)
        return null;
    return { pixelX: cell.x, pixelY: image.height - 1 - cell.y };
}
function pgmPixelToMapPoint(image, pixelX, pixelY) {
    const resolution = Number(image.resolution ?? 1) || 1;
    const origin = image.origin ?? {
        position: { x: 0, y: 0, z: 0 },
        orientation: { x: 0, y: 0, z: 0, w: 1 },
    };
    const originX = Number(origin.position?.x ?? 0);
    const originY = Number(origin.position?.y ?? 0);
    const originYaw = yawFromPose(origin);
    const localX = (pixelX + 0.5) * resolution;
    const localY = (image.height - 1 - pixelY + 0.5) * resolution;
    return {
        x: originX + Math.cos(originYaw) * localX - Math.sin(originYaw) * localY,
        y: originY + Math.sin(originYaw) * localX + Math.cos(originYaw) * localY,
    };
}
function normalizeBrushSize(value) {
    if (!Number.isFinite(value))
        return DEFAULT_BRUSH_SIZE_CELLS;
    return Math.min(Math.max(Math.floor(value), 1), MAX_BRUSH_SIZE_CELLS);
}
function mapNameFromPgmPath(path) {
    return path.split("/").pop().replace(/\.pgm$/i, "");
}
let annotationIdCounter = 0;
function makeAnnotationId() {
    annotationIdCounter += 1;
    return `mark_${Date.now().toString(36)}_${annotationIdCounter.toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}
function normalizeAnnotationColor(value) {
    const color = String(value || "").trim();
    return /^#[0-9A-Fa-f]{6}$/.test(color) ? color.toUpperCase() : ANNOTATION_COLORS[0].value;
}
function hslToHex(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const hp = h / 60;
    const x = c * (1 - Math.abs((hp % 2) - 1));
    const [r1, g1, b1] = hp < 1 ? [c, x, 0]
        : hp < 2 ? [x, c, 0]
            : hp < 3 ? [0, c, x]
                : hp < 4 ? [0, x, c]
                    : hp < 5 ? [x, 0, c]
                        : [c, 0, x];
    const m = l - c / 2;
    const toHex = (value) => Math.round((value + m) * 255).toString(16).padStart(2, "0").toUpperCase();
    return `#${toHex(r1)}${toHex(g1)}${toHex(b1)}`;
}
// Hue bands the generated fallback colors may use: reds/browns [0,15), then
// [170,360) (cyans/blues/purples/magentas). The gap [15,170) — orange, amber,
// yellow, and every green — is reserved for the marker language.
const SAFE_ANNOTATION_HUE_SPANS = [[0, 15], [170, 360]];
const SAFE_ANNOTATION_HUE_TOTAL = SAFE_ANNOTATION_HUE_SPANS.reduce((total, [start, end]) => total + (end - start), 0);
function safeAnnotationHue(seed) {
    let t = ((seed % SAFE_ANNOTATION_HUE_TOTAL) + SAFE_ANNOTATION_HUE_TOTAL) % SAFE_ANNOTATION_HUE_TOTAL;
    for (const [start, end] of SAFE_ANNOTATION_HUE_SPANS) {
        const length = end - start;
        if (t < length)
            return start + t;
        t -= length;
    }
    return SAFE_ANNOTATION_HUE_SPANS[SAFE_ANNOTATION_HUE_SPANS.length - 1][0];
}
function chooseAnnotationColor(existingAnnotations) {
    const used = new Set((existingAnnotations || []).map((annotation) => normalizeAnnotationColor(annotation.color)));
    const available = ANNOTATION_COLORS.filter((color) => !used.has(color.value));
    if (available.length) {
        return available[Math.floor(Math.random() * available.length)].value;
    }
    for (let attempt = 0; attempt < 48; attempt += 1) {
        const hue = safeAnnotationHue(Math.random() * SAFE_ANNOTATION_HUE_TOTAL + used.size * 137.508 + attempt * 29);
        const generated = hslToHex(hue, 0.52, 0.32);
        if (!used.has(generated))
            return generated;
    }
    return hslToHex(safeAnnotationHue(used.size * 137.508), 0.52, 0.32);
}
export function nextAutoAreaLabel(annotations) {
    let max = 0;
    (annotations || []).forEach((annotation) => {
        const match = /^Area (\d+)$/.exec(String((annotation === null || annotation === void 0 ? void 0 : annotation.label) || "").trim());
        if (match)
            max = Math.max(max, Number(match[1]));
    });
    return `Area ${max + 1}`;
}
function normalizeAnnotationRegion(region) {
    var _a, _b, _c, _d, _e;
    const seed = (_a = region === null || region === void 0 ? void 0 : region.seed_cell) !== null && _a !== void 0 ? _a : {};
    const x = Number((_b = seed.x) !== null && _b !== void 0 ? _b : NaN);
    const y = Number((_c = seed.y) !== null && _c !== void 0 ? _c : NaN);
    if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0)
        return null;
    const normalized = {
        seed_cell: { x: Math.floor(x), y: Math.floor(y) },
        cell_count: Number.isFinite(Number(region === null || region === void 0 ? void 0 : region.cell_count)) ? Math.max(0, Math.floor(Number(region.cell_count))) : null,
        width: Number.isFinite(Number((_d = region === null || region === void 0 ? void 0 : region.width) !== null && _d !== void 0 ? _d : NaN)) ? Math.max(0, Math.floor(Number(region.width))) : null,
        height: Number.isFinite(Number((_e = region === null || region === void 0 ? void 0 : region.height) !== null && _e !== void 0 ? _e : NaN)) ? Math.max(0, Math.floor(Number(region.height))) : null,
    };
    const bounds = region === null || region === void 0 ? void 0 : region.bounds;
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
                y_max: Math.max(0, Math.max(yMin, yMax)),
            };
        }
    }
    const cells = normalizeRegionCells(region === null || region === void 0 ? void 0 : region.cells, normalized.width, normalized.height);
    if (cells.length) {
        normalized.cells = cells;
        normalized.cell_count = cells.length;
    }
    return normalized;
}
function normalizeAnnotation(annotation) {
    var _a, _b, _c, _d, _e, _f;
    const pose = (_a = annotation === null || annotation === void 0 ? void 0 : annotation.pose) !== null && _a !== void 0 ? _a : {};
    const label = String((_b = annotation === null || annotation === void 0 ? void 0 : annotation.label) !== null && _b !== void 0 ? _b : "Area").trim() || "Area";
    const normalized = {
        id: String((_c = annotation === null || annotation === void 0 ? void 0 : annotation.id) !== null && _c !== void 0 ? _c : makeAnnotationId()),
        label: label.slice(0, 80),
        color: normalizeAnnotationColor(annotation === null || annotation === void 0 ? void 0 : annotation.color),
        pose: {
            frame_id: String((_d = pose.frame_id) !== null && _d !== void 0 ? _d : "map"),
            x: Number((_e = pose.x) !== null && _e !== void 0 ? _e : 0),
            y: Number((_f = pose.y) !== null && _f !== void 0 ? _f : 0),
            yaw: Number(pose.yaw !== null && pose.yaw !== void 0 ? pose.yaw : 0),
        },
    };
    const region = normalizeAnnotationRegion(annotation === null || annotation === void 0 ? void 0 : annotation.region);
    if (region)
        normalized.region = region;
    return normalized;
}
function preferredPgmFile(files, mapName) {
    const exact = files.find((file) => mapNameFromPgmPath(file.path) === mapName);
    return exact || files.find((file) => file.path.includes(mapName));
}
export function useMapEditor({ open, mapName, onMessage, reloadToken = 0, autoSelect = true }) {
    const pixelsRef = useRef(null);
    const annotationsRef = useRef([]);
    const lastPaintPixelRef = useRef(null);
    const areaPaintRef = useRef(null);
    const areaEraseRef = useRef(null);
    const [files, setFiles] = useState([]);
    const [selectedPath, setSelectedPath] = useState("");
    const [image, setImage] = useState(null);
    const [pixels, setPixels] = useState(null);
    const [annotations, setAnnotations] = useState([]);
    const [annotationLabel, setAnnotationLabel] = useState("");
    const [undoStack, setUndoStack] = useState([]);
    const [redoStack, setRedoStack] = useState([]);
    const [dirty, setDirty] = useState(false);
    const [annotationsDirty, setAnnotationsDirty] = useState(false);
    const [tool, setTool] = useState("view");
    const [brushSize, setBrushSize] = useState(DEFAULT_BRUSH_SIZE_CELLS);
    const [busy, setBusy] = useState(false);
    const [selectedAnnotationId, setSelectedAnnotationId] = useState("");
    useEffect(() => {
        if (selectedAnnotationId && !annotations.some((annotation) => annotation.id === selectedAnnotationId)) {
            setSelectedAnnotationId("");
        }
    }, [annotations, selectedAnnotationId]);
    useEffect(() => {
        if (!open) {
            if (!autoSelect)
                setSelectedPath("");
            return;
        }
        let cancelled = false;
        setBusy(true);
        getPgmFiles()
            .then((response) => {
            if (cancelled)
                return;
            setFiles(response.files);
            const preferred = preferredPgmFile(response.files, mapName);
            setSelectedPath((current) => {
                var _a;
                if (!autoSelect) {
                    return response.files.some((file) => file.path === current) ? current : "";
                }
                if (preferred && (reloadToken || mapNameFromPgmPath(current) !== mapName)) {
                    return preferred.path;
                }
                return current || (preferred === null || preferred === void 0 ? void 0 : preferred.path) || ((_a = response.files[0]) === null || _a === void 0 ? void 0 : _a.path) || "";
            });
            if (!response.files.length)
                onMessage("No PGM files found");
        })
            .catch((error) => {
            if (!cancelled)
                onMessage(error instanceof Error ? error.message : "Failed to list PGM files");
        })
            .finally(() => {
            if (!cancelled)
                setBusy(false);
        });
        return () => {
            cancelled = true;
        };
    }, [autoSelect, mapName, onMessage, open, reloadToken]);
    useEffect(() => {
        if (!open || !selectedPath) {
            setImage(null);
            setPixels(null);
            pixelsRef.current = null;
            lastPaintPixelRef.current = null;
            areaPaintRef.current = null;
            areaEraseRef.current = null;
            setUndoStack([]);
            setRedoStack([]);
            setDirty(false);
            setAnnotationsDirty(false);
            return;
        }
        let cancelled = false;
        setBusy(true);
        getPgmImage(selectedPath)
            .then((response) => {
            if (cancelled)
                return;
            setImage(response);
            const decodedPixels = decodePgmPixels(response);
            pixelsRef.current = decodedPixels;
            lastPaintPixelRef.current = null;
            areaPaintRef.current = null;
            areaEraseRef.current = null;
            setPixels(decodedPixels);
            setUndoStack([]);
            setRedoStack([]);
            setDirty(false);
            setAnnotationsDirty(false);
            onMessage(`Loaded ${response.path}`);
        })
            .catch((error) => {
            if (!cancelled)
                onMessage(error instanceof Error ? error.message : "Failed to load PGM file");
        })
            .finally(() => {
            if (!cancelled)
                setBusy(false);
        });
        return () => {
            cancelled = true;
        };
    }, [onMessage, open, selectedPath]);
    useEffect(() => {
        annotationsRef.current = annotations;
    }, [annotations]);
    useEffect(() => {
        if (!open || !selectedPath) {
            annotationsRef.current = [];
            setAnnotations([]);
            setAnnotationsDirty(false);
            return;
        }
        let cancelled = false;
        getMapAnnotations(selectedPath)
            .then((response) => {
            if (cancelled)
                return;
            const next = Array.isArray(response === null || response === void 0 ? void 0 : response.annotations)
                ? response.annotations.map(normalizeAnnotation)
                : [];
            annotationsRef.current = next;
            setAnnotations(next);
            setAnnotationsDirty(false);
        })
            .catch((error) => {
            if (cancelled)
                return;
            annotationsRef.current = [];
            setAnnotations([]);
            setAnnotationsDirty(false);
            onMessage(error instanceof Error ? error.message : "Failed to load map areas");
        });
        return () => {
            cancelled = true;
        };
    }, [onMessage, open, selectedPath]);
    const map = useMemo(() => {
        if (!image || !pixels)
            return null;
        return pgmPixelsToGrid(image, pixels);
    }, [image, pixels]);
    useEffect(() => {
        lastPaintPixelRef.current = null;
        areaPaintRef.current = null;
        areaEraseRef.current = null;
    }, [brushSize, open, selectedPath, tool]);
    const editAtMapPoint = useCallback((x, y, phase = "paint") => {
        if (phase === "end") {
            lastPaintPixelRef.current = null;
            return;
        }
        if (!open || !image || busy || !PIXEL_TOOL_IDS.has(tool))
            return;
        const pixel = mapPointToPgmPixel(image, x, y);
        if (!pixel) {
            lastPaintPixelRef.current = null;
            return;
        }
        const { pixelX, pixelY } = pixel;
        const currentPixels = pixelsRef.current;
        if (!currentPixels)
            return;
        const previousPixel = phase === "move" ? lastPaintPixelRef.current : null;
        const nextPixels = previousPixel
            ? paintPgmPixelSegment(currentPixels, image.width, image.height, previousPixel, pixel, tool, brushSize)
            : paintPgmPixels(currentPixels, image.width, image.height, pixelX, pixelY, tool, brushSize);
        let editedPixels = 0;
        for (let index = 0; index < currentPixels.length; index += 1) {
            if (currentPixels[index] !== nextPixels[index])
                editedPixels += 1;
        }
        if (editedPixels === 0) {
            lastPaintPixelRef.current = pixel;
            onMessage("No pixels changed");
            return;
        }
        lastPaintPixelRef.current = pixel;
        pixelsRef.current = nextPixels;
        setUndoStack((stack) => [...stack, { type: "pixels", pixels: currentPixels }]);
        setRedoStack([]);
        setPixels(nextPixels);
        setDirty(true);
        const action = tool === "erase_black" ? "Removed" : tool === "draw_unknown" ? "Marked" : "Added";
        onMessage(`${action} ${editedPixels} pixels locally`);
    }, [brushSize, busy, image, onMessage, open, tool]);
    const applyAnnotationsSnapshot = useCallback((nextAnnotations, successMessage) => {
        // Callers pass already-normalized annotations (loads/saves normalize at the
        // boundary). Skipping re-normalization here keeps unchanged entries
        // reference-identical — which lets the viewer reuse their rasterized
        // textures — and avoids re-sorting every region's cells per brush move.
        annotationsRef.current = nextAnnotations;
        setAnnotations(nextAnnotations);
        setAnnotationsDirty(true);
        onMessage(successMessage);
    }, [onMessage]);
    const persistAnnotations = useCallback((nextAnnotations, successMessage) => {
        if (!selectedPath || busy)
            return;
        const currentAnnotations = annotationsRef.current;
        setUndoStack((stack) => [...stack, { type: "annotations", annotations: currentAnnotations }]);
        setRedoStack([]);
        applyAnnotationsSnapshot(nextAnnotations, successMessage);
    }, [applyAnnotationsSnapshot, busy, selectedPath]);
    const recordAnnotationUndo = useCallback((session) => {
        if (!session || session.undoPushed)
            return;
        const currentAnnotations = annotationsRef.current;
        setUndoStack((stack) => [...stack, { type: "annotations", annotations: currentAnnotations }]);
        setRedoStack([]);
        session.undoPushed = true;
    }, []);
    const extendSelectedArea = useCallback((pixel, phase) => {
        const currentPixels = pixelsRef.current;
        if (!open || !image || busy || !currentPixels)
            return;
        const target = annotationsRef.current.find((annotation) => annotation.id === selectedAnnotationId);
        if (!target) {
            onMessage("Select an area to extend");
            return;
        }
        let session = areaPaintRef.current;
        if (phase === "start" || !session || session.id !== target.id) {
            // Heavy prep runs ONCE per stroke: the covered-by-others set and the
            // target's visible cells (not raw bounds, so extending a rect-created
            // area can never resurrect cells hidden under earlier areas).
            const covered = coveredAnnotationCellSet(annotationsRef.current, image, currentPixels, target.id);
            const targetSnapshot = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels)
                .find((snapshot) => snapshot.annotation.id === target.id);
            const cells = new Map();
            ((targetSnapshot === null || targetSnapshot === void 0 ? void 0 : targetSnapshot.cells) || []).forEach((cell) => cells.set(cellKey(cell), cell));
            session = { id: target.id, lastPixel: pixel, undoPushed: false, covered, cells };
            areaPaintRef.current = session;
        }
        const startPixel = phase === "move" && session.lastPixel ? session.lastPixel : pixel;
        const newCells = brushCellsForSegment(currentPixels, image, startPixel, pixel, brushSize, session.covered);
        session.lastPixel = pixel;
        if (!newCells.length) {
            onMessage("Paint over unmarked white free-space");
            return;
        }
        let added = 0;
        newCells.forEach((cell) => {
            const key = cellKey(cell);
            if (session.cells.has(key))
                return;
            session.cells.set(key, cell);
            added += 1;
        });
        if (!added)
            return;
        const geometry = regionFromCells(Array.from(session.cells.values()), image);
        if (!geometry)
            return;
        recordAnnotationUndo(session);
        const nextAnnotation = {
            ...target,
            pose: geometry.pose,
            region: geometry.region,
        };
        applyAnnotationsSnapshot(annotationsRef.current.map((annotation) => (annotation.id === target.id ? nextAnnotation : annotation)), `Extended area ${nextAnnotation.label}`);
    }, [applyAnnotationsSnapshot, brushSize, busy, image, onMessage, open, recordAnnotationUndo, selectedAnnotationId]);
    const applyAreaCellErase = useCallback((startPixel, endPixel, session) => {
        const currentPixels = pixelsRef.current;
        if (!open || !image || busy || !currentPixels)
            return false;
        // Snapshot every annotation's visible cells ONCE per stroke; moves then
        // only delete brush keys from these Maps instead of rebuilding the world.
        if (!session.entries) {
            session.entries = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels)
                .map(({ annotation, cells }) => ({
                    annotation,
                    cells: new Map(cells.map((cell) => [cellKey(cell), cell])),
                    changed: false,
                }));
        }
        const eraseCells = brushCellsForSegment(currentPixels, image, startPixel, endPixel, brushSize);
        if (!eraseCells.length)
            return false;
        let changed = false;
        eraseCells.forEach((cell) => {
            const key = cellKey(cell);
            session.entries.forEach((entry) => {
                if (entry.cells.delete(key)) {
                    entry.changed = true;
                    changed = true;
                }
            });
        });
        if (!changed)
            return false;
        recordAnnotationUndo(session);
        session.entries = session.entries.filter((entry) => entry.cells.size > 0);
        const nextAnnotations = session.entries.map((entry) => {
            if (!entry.changed)
                return entry.annotation;
            const geometry = regionFromCells(Array.from(entry.cells.values()), image);
            if (!geometry)
                return entry.annotation;
            const nextAnnotation = {
                ...entry.annotation,
                pose: geometry.pose,
                region: geometry.region,
            };
            entry.annotation = nextAnnotation;
            entry.changed = false;
            return nextAnnotation;
        });
        applyAnnotationsSnapshot(nextAnnotations, "Erased area pixels");
        return true;
    }, [applyAnnotationsSnapshot, brushSize, busy, image, open, recordAnnotationUndo]);
    const editAreaAtMapPoint = useCallback((x, y, phase = "paint") => {
        if (!open || !image || busy || (tool !== ANNOTATION_EXTEND_TOOL.id && tool !== ANNOTATION_ERASE_TOOL.id))
            return;
        if (phase === "end") {
            areaPaintRef.current = null;
            areaEraseRef.current = null;
            return;
        }
        const pixel = mapPointToPgmPixel(image, x, y);
        if (!pixel)
            return;
        if (tool === ANNOTATION_EXTEND_TOOL.id) {
            extendSelectedArea(pixel, phase);
            return;
        }
        let session = areaEraseRef.current;
        if (phase === "start" || !session) {
            session = { lastPixel: pixel, undoPushed: false };
            areaEraseRef.current = session;
            // Erase under the cursor immediately so a plain press has an effect.
            applyAreaCellErase(pixel, pixel, session);
            return;
        }
        if (applyAreaCellErase(session.lastPixel || pixel, pixel, session)) {
            session.lastPixel = pixel;
        }
    }, [applyAreaCellErase, busy, extendSelectedArea, image, open, tool]);
    const placeAnnotationAtMapArea = useCallback((startX, startY, endX = startX, endY = startY) => {
        if (!open || !image || busy || tool !== ANNOTATION_TOOL.id)
            return;
        const startPixel = mapPointToPgmPixel(image, startX, startY);
        const endPixel = mapPointToPgmPixel(image, endX, endY);
        if (!startPixel || !endPixel) {
            onMessage("Select an area inside the map");
            return;
        }
        const currentPixels = pixelsRef.current;
        if (!currentPixels)
            return;
        const region = selectedFreePgmRegion(currentPixels, image, startPixel, endPixel, annotationsRef.current);
        if (!region || region.count === 0) {
            onMessage("Drag over an unmarked white free-space area");
            return;
        }
        const label = annotationLabel.trim() || nextAutoAreaLabel(annotationsRef.current);
        const centroid = pgmPixelToMapPoint(image, region.centroidPixelX, region.centroidPixelY);
        const nextAnnotation = normalizeAnnotation({
            id: makeAnnotationId(),
            label,
            color: chooseAnnotationColor(annotationsRef.current),
            pose: { frame_id: "map", x: centroid.x, y: centroid.y, yaw: 0 },
            region: {
                seed_cell: {
                    x: region.seedCell.x,
                    y: region.seedCell.y,
                },
                bounds: region.bounds,
                cell_count: region.count,
                width: image.width,
                height: image.height,
            },
        });
        persistAnnotations([...annotationsRef.current, nextAnnotation], `Marked area ${nextAnnotation.label}`);
        setSelectedAnnotationId(nextAnnotation.id);
    }, [annotationLabel, busy, image, onMessage, open, persistAnnotations, tool]);
    const deleteAnnotationById = useCallback((id) => {
        if (!open || !image || busy)
            return;
        const currentPixels = pixelsRef.current;
        if (!currentPixels)
            return;
        const target = annotationsRef.current.find((annotation) => annotation.id === id);
        if (!target)
            return;
        const nextAnnotations = visibleAnnotationCellSnapshots(annotationsRef.current, image, currentPixels)
            .filter((snapshot) => snapshot.annotation.id !== id)
            .map((snapshot) => annotationWithCells(snapshot.annotation, snapshot.cells, image))
            .filter(Boolean);
        persistAnnotations(nextAnnotations, `Removed area ${target.label || "Area"}`);
    }, [busy, image, open, persistAnnotations]);
    const renameAnnotation = useCallback((id, label) => {
        const trimmed = String(label || "").trim().slice(0, 80);
        if (!trimmed)
            return;
        const target = annotationsRef.current.find((annotation) => annotation.id === id);
        if (!target || target.label === trimmed)
            return;
        persistAnnotations(annotationsRef.current.map((annotation) => (annotation.id === id ? { ...annotation, label: trimmed } : annotation)), `Renamed area ${trimmed}`);
    }, [persistAnnotations]);
    const clearAnnotations = useCallback(() => {
        if (!annotationsRef.current.length)
            return;
        persistAnnotations([], "Cleared map areas");
    }, [persistAnnotations]);
    const undo = useCallback(() => {
        const currentPixels = pixelsRef.current;
        setUndoStack((stack) => {
            const previous = stack[stack.length - 1];
            if (!previous)
                return stack;
            if (previous.type === "pixels") {
                if (!currentPixels)
                    return stack;
                setRedoStack((redo) => [...redo, { type: "pixels", pixels: currentPixels }]);
                pixelsRef.current = previous.pixels;
                setPixels(previous.pixels);
                setDirty(stack.slice(0, -1).some((item) => item.type === "pixels"));
                onMessage("Undid last map edit");
            }
            else if (previous.type === "annotations") {
                const currentAnnotations = annotationsRef.current;
                setRedoStack((redo) => [...redo, { type: "annotations", annotations: currentAnnotations }]);
                applyAnnotationsSnapshot(previous.annotations, "Undid last area edit");
            }
            return stack.slice(0, -1);
        });
    }, [applyAnnotationsSnapshot, onMessage]);
    const redo = useCallback(() => {
        const currentPixels = pixelsRef.current;
        setRedoStack((stack) => {
            const next = stack[stack.length - 1];
            if (!next)
                return stack;
            if (next.type === "pixels") {
                if (!currentPixels)
                    return stack;
                setUndoStack((undoHistory) => [...undoHistory, { type: "pixels", pixels: currentPixels }]);
                pixelsRef.current = next.pixels;
                setPixels(next.pixels);
                setDirty(true);
                onMessage("Redid last map edit");
            }
            else if (next.type === "annotations") {
                const currentAnnotations = annotationsRef.current;
                setUndoStack((undoHistory) => [...undoHistory, { type: "annotations", annotations: currentAnnotations }]);
                applyAnnotationsSnapshot(next.annotations, "Redid last area edit");
            }
            return stack.slice(0, -1);
        });
    }, [applyAnnotationsSnapshot, onMessage]);
    const save = useCallback(() => {
        if (!image || !pixels || busy || (!dirty && !annotationsDirty))
            return;
        setBusy(true);
        Promise.resolve()
            .then(async () => {
            const savedParts = [];
            let savedPath = image.path;
            if (dirty) {
                const response = await savePgmImage(image.path, image.width, image.height, image.maxval, encodePgmPixels(pixels));
                savedPath = (response === null || response === void 0 ? void 0 : response.path) || savedPath;
                setDirty(false);
                savedParts.push("map");
            }
            if (annotationsDirty) {
                const response = await saveMapAnnotations(image.path, annotationsRef.current);
                const saved = Array.isArray(response === null || response === void 0 ? void 0 : response.annotations)
                    ? response.annotations.map(normalizeAnnotation)
                    : annotationsRef.current;
                annotationsRef.current = saved;
                setAnnotations(saved);
                setAnnotationsDirty(false);
                savedParts.push("areas");
            }
            setUndoStack([]);
            setRedoStack([]);
            onMessage(savedParts.length > 1 ? `Saved map and areas ${savedPath}` : `Saved ${savedParts[0] === "areas" ? "map areas" : savedPath}`);
        })
            .catch((error) => {
            onMessage(error instanceof Error ? error.message : "Failed to save map");
        })
            .finally(() => setBusy(false));
    }, [annotationsDirty, busy, dirty, image, onMessage, pixels]);
    const hasUnsavedChanges = dirty || annotationsDirty;
    return {
        files,
        selectedPath,
        setSelectedPath,
        image,
        map,
        annotations,
        annotationLabel,
        setAnnotationLabel,
        tool,
        setTool,
        brushSize,
        setBrushSize: (value) => setBrushSize(normalizeBrushSize(value)),
        busy,
        dirty: hasUnsavedChanges,
        pixelsDirty: dirty,
        annotationsDirty,
        canUndo: undoStack.length > 0,
        canRedo: redoStack.length > 0,
        undo,
        redo,
        save,
        editAtMapPoint,
        editAreaAtMapPoint,
        placeAnnotationAtMapArea,
        clearAnnotations,
        selectedAnnotationId,
        setSelectedAnnotationId,
        deleteAnnotationById,
        renameAnnotation,
    };
}
// Warm segmented map-editor controls (see design handoff, turn 8).
export const TOOL_ICONS = {
    view: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6z" /><circle cx="12" cy="12" r="2.6" />
        </svg>
    ),
    erase_black: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="m7 21-4-4 11-11 4 4-11 11H7z" /><path d="m14 6 4 4" />
        </svg>
    ),
    draw_black: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19c-4 1.5-8-1-8-5 0-6 8-11 8-11s8 5 8 11c0 4-4 6.5-8 5z" />
        </svg>
    ),
    draw_unknown: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="9" />
            <path d="M9.6 9.2a2.5 2.5 0 1 1 3.5 2.4c-.8.4-1.1 1-1.1 1.9" />
            <path d="M12 16.8h.01" />
        </svg>
    ),
    label_marker: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M20.6 13.2 13.2 20.6a2 2 0 0 1-2.8 0l-7-7A2 2 0 0 1 2.8 12V4.4A1.6 1.6 0 0 1 4.4 2.8H12a2 2 0 0 1 1.4.6l7.2 7a2 2 0 0 1 0 2.8z" />
            <circle cx="7.5" cy="7.5" r="1.2" />
        </svg>
    ),
    extend_area: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 19c-4 1.5-8-1-8-5 0-4.5 5-8.5 7-10" />
            <path d="M17 3v8" /><path d="M13 7h8" />
        </svg>
    ),
    erase_area: (
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M4 5h16" /><path d="M9 5V3h6v2" /><path d="M7 9l1 11h8l1-11" /><path d="M10 12v5" /><path d="M14 12v5" />
        </svg>
    ),
};

export function SegGroup({ children, ariaLabel }) {
    return (
        <div
            role="group"
            aria-label={ariaLabel}
            className="flex items-center gap-0.5 p-[3px]"
            style={{ borderRadius: 12, border: "1px solid var(--mc-border)", backgroundColor: "var(--mc-surface-hover)" }}
        >
            {children}
        </div>
    );
}

export function SegButton({ selected, disabled, onClick, title, ariaLabel, children, narrow = false }) {
    return (
        <button
            type="button"
            disabled={disabled}
            aria-pressed={selected}
            onClick={onClick}
            title={title}
            aria-label={ariaLabel}
            className={`h-[30px] ${narrow ? "min-w-[42px] px-2 justify-center" : "px-3"} inline-flex items-center gap-1.5 whitespace-nowrap text-[12.5px] font-semibold transition-colors disabled:opacity-50`}
            style={{
                borderRadius: 9,
                border: "none",
                color: selected ? "var(--mc-bg)" : "var(--mc-text-muted)",
                backgroundColor: selected ? "var(--mc-text)" : "transparent",
            }}
        >
            {children}
        </button>
    );
}

function EditorSection({ label, children }) {
    return (
        <section
            className="inline-flex flex-wrap items-center gap-2 px-2 py-1"
            style={{ borderRadius: 12, border: "1px solid var(--mc-border)", backgroundColor: "var(--mc-surface)" }}
        >
            <span className="text-[9.5px] font-mono tracking-[0.1em]" style={{ color: "var(--mc-text-subtle)" }}>
                {label}
            </span>
            {children}
        </section>
    );
}

export function MapEditorControls({ files, selectedPath, setSelectedPath, tool, setTool, brushSize, setBrushSize, busy, image, dirty, canUndo = false, canRedo = false, undo, redo, save, enableAnnotations = false, annotations = [], annotationLabel = "Area", setAnnotationLabel = () => { }, selectedAnnotationId = "", setSelectedAnnotationId = () => { }, deleteAnnotationById = () => { }, renameAnnotation = () => { } }) {
    const [renamingId, setRenamingId] = useState("");
    const [renameDraft, setRenameDraft] = useState("");
    const [confirmDeleteId, setConfirmDeleteId] = useState("");
    const confirmTimerRef = useRef(null);
    const disarmConfirms = useCallback(() => {
        setConfirmDeleteId("");
    }, []);
    const armConfirmTimer = useCallback(() => {
        if (confirmTimerRef.current)
            clearTimeout(confirmTimerRef.current);
        confirmTimerRef.current = setTimeout(disarmConfirms, 4000);
    }, [disarmConfirms]);
    useEffect(() => () => {
        if (confirmTimerRef.current)
            clearTimeout(confirmTimerRef.current);
    }, []);
    const iconButtonStyle = (enabled) => ({
        borderRadius: 9,
        border: "1px solid var(--mc-border-strong)",
        backgroundColor: "var(--mc-surface)",
        color: enabled ? "var(--mc-text)" : "var(--mc-text-subtle)",
    });
    const areaTools = [ANNOTATION_TOOL, ANNOTATION_EXTEND_TOOL, ANNOTATION_ERASE_TOOL];
    const commitRename = (annotation) => {
        renameAnnotation(annotation.id, renameDraft);
        setRenamingId("");
        setRenameDraft("");
    };

    const areasChips = enableAnnotations && annotations.length > 0 && (
        <div role="group" aria-label="Map areas" className="flex max-h-24 w-full flex-wrap items-center gap-1.5 overflow-y-auto">
            {annotations.map((annotation) => {
                const selected = annotation.id === selectedAnnotationId;
                const confirming = confirmDeleteId === annotation.id;
                if (renamingId === annotation.id) {
                    return (
                        <input
                            key={annotation.id}
                            autoFocus
                            aria-label={`Rename area ${annotation.label}`}
                            value={renameDraft}
                            disabled={busy}
                            onChange={(event) => setRenameDraft(event.currentTarget.value)}
                            onBlur={() => commitRename(annotation)}
                            onKeyDown={(event) => {
                                if (event.key === "Enter") {
                                    event.preventDefault();
                                    commitRename(annotation);
                                }
                                if (event.key === "Escape") {
                                    event.preventDefault();
                                    setRenamingId("");
                                    setRenameDraft("");
                                }
                            }}
                            className="h-8 w-32 px-3 text-[12px] font-semibold"
                            style={{ borderRadius: 9, color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)" }}
                        />
                    );
                }
                return (
                    <span
                        key={annotation.id}
                        className="inline-flex h-8 items-center gap-1 pl-2.5 pr-1"
                        style={{
                            borderRadius: 999,
                            border: `1px solid ${selected ? "var(--mc-text)" : "var(--mc-border-strong)"}`,
                            backgroundColor: selected ? "var(--mc-surface-hover)" : "var(--mc-surface)",
                        }}
                    >
                        <button
                            type="button"
                            aria-pressed={selected}
                            disabled={busy}
                            onClick={() => setSelectedAnnotationId(annotation.id)}
                            onDoubleClick={() => {
                                setRenamingId(annotation.id);
                                setRenameDraft(annotation.label);
                            }}
                            title={`${annotation.label} — double-click to rename`}
                            className="inline-flex items-center gap-1.5 text-[12px] font-semibold disabled:opacity-50"
                            style={{ border: "none", backgroundColor: "transparent", color: "var(--mc-text)" }}
                        >
                            <span aria-hidden="true" className="shrink-0" style={{ width: 8, height: 8, borderRadius: 999, backgroundColor: annotation.color }} />
                            {annotation.label}
                        </button>
                        <button
                            type="button"
                            disabled={busy}
                            aria-label={confirming ? `Confirm delete area ${annotation.label}` : `Delete area ${annotation.label}`}
                            onClick={() => {
                                if (confirming) {
                                    deleteAnnotationById(annotation.id);
                                    disarmConfirms();
                                    return;
                                }
                                setConfirmDeleteId(annotation.id);
                                armConfirmTimer();
                            }}
                            className="inline-flex h-6 w-6 items-center justify-center text-[13px] leading-none disabled:opacity-50"
                            style={{
                                borderRadius: 999,
                                border: "none",
                                backgroundColor: confirming ? "var(--mc-danger)" : "transparent",
                                color: confirming ? "var(--mc-accent-fg)" : "var(--mc-danger)",
                            }}
                        >
                            ×
                        </button>
                    </span>
                );
            })}
        </div>
    );


    return (
        <div className="flex w-full flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
                <select
                    aria-label="PGM map"
                    value={selectedPath}
                    disabled={busy || files.length === 0}
                    onChange={(event) => setSelectedPath(event.currentTarget.value)}
                    className="h-9 min-w-[220px] px-3 text-[12px] font-mono"
                    style={{ borderRadius: 10, color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)" }}
                >
                    <option value="">Select a PGM</option>
                    {files.map((file) => (
                        <option key={file.path} value={file.path}>{file.path}</option>
                    ))}
                </select>

                {/* Standalone (no SegGroup pill around it), so it carries its own
                    surface + border to read as a button rather than plain text. */}
                <button
                    type="button"
                    disabled={busy}
                    aria-pressed={tool === "view"}
                    onClick={() => setTool("view")}
                    title="View"
                    aria-label="View"
                    className="h-9 px-3 inline-flex items-center gap-1.5 text-[12.5px] font-semibold transition-colors disabled:opacity-50"
                    style={{
                        borderRadius: 9,
                        border: "1px solid var(--mc-border-strong)",
                        backgroundColor: tool === "view" ? "var(--mc-text)" : "var(--mc-surface)",
                        color: tool === "view" ? "var(--mc-bg)" : "var(--mc-text)",
                    }}
                >
                    {TOOL_ICONS.view}View
                </button>

                <EditorSection label="MAP EDIT">
                    <SegGroup ariaLabel="Map edit tools">
                        {EDIT_TOOLS.map((editTool) => (
                            <SegButton
                                key={editTool.id}
                                selected={tool === editTool.id}
                                disabled={busy}
                                onClick={() => setTool(editTool.id)}
                                title={editTool.label}
                                ariaLabel={editTool.label}
                            >
                                {TOOL_ICONS[editTool.id]}{editTool.label}
                            </SegButton>
                        ))}
                    </SegGroup>
                </EditorSection>

                {enableAnnotations && (
                    <EditorSection label="AREA EDIT">
                        <SegGroup ariaLabel="Area edit tools">
                            {areaTools.map((editTool) => (
                                <SegButton
                                    key={editTool.id}
                                    selected={tool === editTool.id}
                                    disabled={busy}
                                    onClick={() => setTool(editTool.id)}
                                    title={editTool.label}
                                    ariaLabel={editTool.label}
                                >
                                    {TOOL_ICONS[editTool.id]}{editTool.label}
                                </SegButton>
                            ))}
                        </SegGroup>
                        <input
                            aria-label="Area name"
                            value={annotationLabel}
                            placeholder={nextAutoAreaLabel(annotations)}
                            disabled={busy}
                            onChange={(event) => setAnnotationLabel(event.currentTarget.value)}
                            className="h-8 w-28 px-3 text-[12px] font-semibold"
                            style={{ borderRadius: 9, color: "var(--mc-text)", backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border-strong)" }}
                        />
                    </EditorSection>
                )}

                <EditorSection label="BRUSH">
                    <SegGroup ariaLabel="Brush size">
                        {BRUSH_SIZE_OPTIONS.map((option) => (
                            <SegButton
                                key={option.value}
                                narrow
                                selected={brushSize === option.value}
                                disabled={busy}
                                onClick={() => setBrushSize(option.value)}
                                title={`Brush ${option.label}: ${option.value}px`}
                                ariaLabel={`Brush size ${option.label}`}
                            >
                                {option.label === "Small" ? "S" : option.label === "Medium" ? "M" : option.label === "Large" ? "L" : "XL"}
                            </SegButton>
                        ))}
                    </SegGroup>
                </EditorSection>

                <button type="button" disabled={busy || !canUndo} onClick={undo} title="Undo" aria-label="Undo"
                    className="h-9 w-9 inline-flex items-center justify-center transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0"
                    style={iconButtonStyle(canUndo)}>
                    <MdUndo size={17} />
                </button>
                <button type="button" disabled={busy || !canRedo} onClick={redo} title="Redo" aria-label="Redo"
                    className="h-9 w-9 inline-flex items-center justify-center transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0"
                    style={iconButtonStyle(canRedo)}>
                    <MdRedo size={17} />
                </button>
                <button type="button" disabled={busy || !dirty} onClick={save}
                    className="h-9 px-4 text-[12.5px] font-semibold transition-all active:translate-y-px disabled:opacity-50 disabled:active:translate-y-0"
                    style={{ borderRadius: 9, border: "none", backgroundColor: "var(--mc-accent)", color: "var(--mc-accent-fg)", boxShadow: dirty ? "var(--mc-shadow)" : "none" }}>
                    Save
                </button>

                <span className="text-[11px] font-mono" style={{ color: "var(--mc-text-subtle)" }}>
                    {image ? (
                        <>
                            {image.width} × {image.height}
                            {dirty && <span style={{ color: "var(--mc-accent)" }}> · unsaved</span>}
                        </>
                    ) : "Select a PGM"}
                </span>
            </div>

            {areasChips}
        </div>
    );
}
