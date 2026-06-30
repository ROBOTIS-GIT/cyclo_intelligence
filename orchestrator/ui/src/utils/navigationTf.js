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
// Author: Howon Kim
export function yawFromPose(pose) {
    var _a, _b, _c, _d;
    const q = pose === null || pose === void 0 ? void 0 : pose.orientation;
    if (!q)
        return 0;
    const x = (_a = q.x) !== null && _a !== void 0 ? _a : 0;
    const y = (_b = q.y) !== null && _b !== void 0 ? _b : 0;
    const z = (_c = q.z) !== null && _c !== void 0 ? _c : 0;
    const w = (_d = q.w) !== null && _d !== void 0 ? _d : 1;
    return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}
export function orientationFromYaw(yaw) {
    return { x: 0, y: 0, z: Math.sin(yaw / 2), w: Math.cos(yaw / 2) };
}
export function poseFromBaseLinkTf(tf) {
    var _a, _b;
    return (_b = (_a = buildTfFramePoses(tf, "map").find(({ frame }) => frame === "base_link")) === null || _a === void 0 ? void 0 : _a.pose) !== null && _b !== void 0 ? _b : null;
}
export function normalizeFrameId(frameId) {
    return (frameId !== null && frameId !== void 0 ? frameId : "").replace(/^\//, "");
}
function poseFromTransform(transform) {
    var _a, _b;
    return {
        position: (_a = transform.transform) === null || _a === void 0 ? void 0 : _a.translation,
        orientation: (_b = transform.transform) === null || _b === void 0 ? void 0 : _b.rotation,
    };
}
export function buildTfFramePoses(tf, rootFrame = "map") {
    var _a;
    const transforms = (_a = tf === null || tf === void 0 ? void 0 : tf.transforms) !== null && _a !== void 0 ? _a : [];
    const edges = new Map();
    const parentFrames = new Set();
    transforms.forEach((transform) => {
        var _a;
        const child = normalizeFrameId(transform.child_frame_id);
        const parent = normalizeFrameId((_a = transform.header) === null || _a === void 0 ? void 0 : _a.frame_id);
        if (!child || !parent || !transform.transform)
            return;
        edges.set(child, transform);
        parentFrames.add(parent);
    });
    if (!parentFrames.has(rootFrame) && !edges.has(rootFrame))
        return [];
    const resolved = new Map([
        [rootFrame, { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }],
    ]);
    const resolveFrame = (frame, visiting = new Set()) => {
        var _a, _b, _c, _d, _e, _f, _g, _h, _j, _k, _l;
        const existing = resolved.get(frame);
        if (existing)
            return existing;
        if (visiting.has(frame))
            return null;
        const edge = edges.get(frame);
        if (!edge)
            return null;
        const parent = normalizeFrameId((_a = edge.header) === null || _a === void 0 ? void 0 : _a.frame_id);
        visiting.add(frame);
        const parentPose = resolveFrame(parent, visiting);
        visiting.delete(frame);
        if (!parentPose)
            return null;
        const localPose = poseFromTransform(edge);
        const parentYaw = yawFromPose(parentPose);
        const localYaw = yawFromPose(localPose);
        const tx = Number((_c = (_b = localPose.position) === null || _b === void 0 ? void 0 : _b.x) !== null && _c !== void 0 ? _c : 0);
        const ty = Number((_e = (_d = localPose.position) === null || _d === void 0 ? void 0 : _d.y) !== null && _e !== void 0 ? _e : 0);
        const x = Number((_g = (_f = parentPose.position) === null || _f === void 0 ? void 0 : _f.x) !== null && _g !== void 0 ? _g : 0) + Math.cos(parentYaw) * tx - Math.sin(parentYaw) * ty;
        const y = Number((_j = (_h = parentPose.position) === null || _h === void 0 ? void 0 : _h.y) !== null && _j !== void 0 ? _j : 0) + Math.sin(parentYaw) * tx + Math.cos(parentYaw) * ty;
        const pose = {
            position: { x, y, z: Number((_l = (_k = localPose.position) === null || _k === void 0 ? void 0 : _k.z) !== null && _l !== void 0 ? _l : 0) },
            orientation: {
                x: 0,
                y: 0,
                z: Math.sin((parentYaw + localYaw) / 2),
                w: Math.cos((parentYaw + localYaw) / 2),
            },
        };
        resolved.set(frame, pose);
        return pose;
    };
    for (const frame of Array.from(edges.keys())) {
        resolveFrame(frame);
    }
    return Array.from(resolved.entries())
        .filter(([frame]) => frame !== rootFrame)
        .map(([frame, pose]) => ({ frame, pose }));
}
export function mergeTfMessages(...messages) {
    const transforms = messages.flatMap((message) => { var _a; return (_a = message === null || message === void 0 ? void 0 : message.transforms) !== null && _a !== void 0 ? _a : []; });
    return transforms.length > 0 ? { transforms } : null;
}
export function updateTfBuffer(buffer, message) {
    var _a, _b, _c;
    let updated = false;
    for (const transform of (_a = message === null || message === void 0 ? void 0 : message.transforms) !== null && _a !== void 0 ? _a : []) {
        const child = normalizeFrameId(transform.child_frame_id);
        const parent = normalizeFrameId((_b = transform.header) === null || _b === void 0 ? void 0 : _b.frame_id);
        if (!child || !parent || !transform.transform)
            continue;
        const existing = buffer.get(child);
        if (existing &&
            normalizeFrameId((_c = existing.header) === null || _c === void 0 ? void 0 : _c.frame_id) === parent &&
            JSON.stringify(existing.transform) === JSON.stringify(transform.transform)) {
            continue;
        }
        buffer.set(child, transform);
        updated = true;
    }
    return updated;
}
export function tfMessageFromBuffer(buffer) {
    const transforms = Array.from(buffer.values());
    return transforms.length > 0 ? { transforms } : null;
}

