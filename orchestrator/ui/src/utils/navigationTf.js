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
export function poseForScanFrame(frameId, framePosesByName, robotPose) {
    const frame = normalizeFrameId(frameId);
    if (frame === "base_link" && robotPose)
        return robotPose;
    const framePose = framePosesByName.get(frame);
    return framePose !== null && framePose !== void 0 ? framePose : robotPose;
}
export function poseForTfAxesFrame(frameId, tfPose, robotPose) {
    return normalizeFrameId(frameId) === "base_link" && robotPose
        ? robotPose
        : tfPose;
}
function rosStampParts(stamp) {
    if (!stamp || typeof stamp !== "object")
        return null;
    const sec = Number(stamp.sec ?? stamp.secs ?? 0);
    const nanosec = Number(stamp.nanosec ?? stamp.nsec ?? stamp.nsecs ?? 0);
    if (!Number.isFinite(sec) || !Number.isFinite(nanosec) || (sec === 0 && nanosec === 0))
        return null;
    return { sec, nanosec };
}
function stampDifferenceSeconds(left, right) {
    const a = rosStampParts(left);
    const b = rosStampParts(right);
    if (!a || !b)
        return null;
    return (a.sec - b.sec) + (a.nanosec - b.nanosec) / 1e9;
}
function stampKey(stamp) {
    const value = rosStampParts(stamp);
    return value ? `${value.sec}:${value.nanosec}` : "";
}
function poseFromStampedEstimate(message) {
    return message?.pose?.pose ?? null;
}
function poseFromOdometry(message) {
    return message?.pose?.pose ?? null;
}
function composePlanarPoses(parentPose, childPose) {
    if (!parentPose?.position || !childPose?.position)
        return null;
    const parentYaw = yawFromPose(parentPose);
    const childYaw = yawFromPose(childPose);
    const cos = Math.cos(parentYaw);
    const sin = Math.sin(parentYaw);
    const childX = Number(childPose.position.x ?? 0);
    const childY = Number(childPose.position.y ?? 0);
    return {
        position: {
            x: Number(parentPose.position.x ?? 0) + cos * childX - sin * childY,
            y: Number(parentPose.position.y ?? 0) + sin * childX + cos * childY,
            z: Number(parentPose.position.z ?? 0) + Number(childPose.position.z ?? 0),
        },
        orientation: orientationFromYaw(parentYaw + childYaw),
    };
}

function relativePlanarPose(parentPose, childPose) {
    if (!parentPose?.position || !childPose?.position)
        return null;
    const parentYaw = yawFromPose(parentPose);
    const childYaw = yawFromPose(childPose);
    const cos = Math.cos(parentYaw);
    const sin = Math.sin(parentYaw);
    const dx = Number(childPose.position.x ?? 0) - Number(parentPose.position.x ?? 0);
    const dy = Number(childPose.position.y ?? 0) - Number(parentPose.position.y ?? 0);
    return {
        position: {
            x: cos * dx + sin * dy,
            y: -sin * dx + cos * dy,
            z: Number(childPose.position.z ?? 0) - Number(parentPose.position.z ?? 0),
        },
        orientation: orientationFromYaw(childYaw - parentYaw),
    };
}

/**
 * Apply the latest base_link -> sensor offset to a synchronized base pose.
 * Laser frames are normally static, so this keeps an offset sensor aligned
 * without relying on a stale map -> odom transform from the browser TF buffer.
 */
export function poseForScanFrameAtBasePose(frameId, framePosesByName, basePose) {
    const frame = normalizeFrameId(frameId) || "base_link";
    if (!basePose?.position)
        return poseForScanFrame(frame, framePosesByName, basePose);
    if (frame === "base_link")
        return basePose;
    const framePose = framePosesByName.get(frame) ?? null;
    const bufferedBasePose = framePosesByName.get("base_link") ?? null;
    if (!framePose?.position || !bufferedBasePose?.position)
        return framePose ?? basePose;
    const baseToFramePose = relativePlanarPose(bufferedBasePose, framePose);
    return composePlanarPoses(basePose, baseToFramePose) ?? framePose;
}

/** Derive map -> odom from a scan-matched map -> base and odom -> base pair. */
export function mapToOdomPoseFromAnchor(mapBasePose, odomBasePose) {
    if (!mapBasePose?.position || !odomBasePose?.position)
        return null;
    const yaw = yawFromPose(mapBasePose) - yawFromPose(odomBasePose);
    const cos = Math.cos(yaw);
    const sin = Math.sin(yaw);
    const odomX = Number(odomBasePose.position.x ?? 0);
    const odomY = Number(odomBasePose.position.y ?? 0);
    return {
        position: {
            x: Number(mapBasePose.position.x ?? 0) - (cos * odomX - sin * odomY),
            y: Number(mapBasePose.position.y ?? 0) - (sin * odomX + cos * odomY),
            z: Number(mapBasePose.position.z ?? 0) - Number(odomBasePose.position.z ?? 0),
        },
        orientation: orientationFromYaw(yaw),
    };
}

/** Propagate a SLAM map-frame anchor with a newer odometry pose. */
export function mappingPoseFromOdom(mapBasePose, anchorOdomPose, targetOdomPose) {
    const mapOdomPose = mapToOdomPoseFromAnchor(mapBasePose, anchorOdomPose);
    return mapOdomPose ? composePlanarPoses(mapOdomPose, targetOdomPose) : null;
}

/** Replace one TF edge without disturbing the other buffered transforms. */
export function replaceTfFramePose(tf, parentFrame, childFrame, pose, stamp = null) {
    if (!pose?.position)
        return tf;
    const normalizedParent = normalizeFrameId(parentFrame);
    const normalizedChild = normalizeFrameId(childFrame);
    const transforms = (tf?.transforms ?? []).filter((transform) => (
        normalizeFrameId(transform?.header?.frame_id) !== normalizedParent
        || normalizeFrameId(transform?.child_frame_id) !== normalizedChild
    ));
    transforms.push({
        header: {
            frame_id: normalizedParent,
            ...(stamp ? { stamp } : {}),
        },
        child_frame_id: normalizedChild,
        transform: {
            translation: {
                x: Number(pose.position.x ?? 0),
                y: Number(pose.position.y ?? 0),
                z: Number(pose.position.z ?? 0),
            },
            rotation: pose.orientation ?? orientationFromYaw(0),
        },
    });
    return { transforms };
}

/**
 * Make the browser TF tree use the same map-frame localization anchor and
 * odometry pose as the robot marker. This keeps odom-frame overlays such as
 * published_footprint aligned even when rosbridge misses map -> odom updates.
 */
export function applyPoseSyncToTf(tf, poseSync) {
    if (!poseSync?.mapToOdomPose)
        return tf;
    const mapCorrectedTf = replaceTfFramePose(
        tf,
        "map",
        "odom",
        poseSync.mapToOdomPose,
        poseSync.anchorStamp,
    );
    return poseSync.odomPose
        ? replaceTfFramePose(
            mapCorrectedTf,
            "odom",
            "base_link",
            poseSync.odomPose,
            poseSync.odomStamp,
        )
        : mapCorrectedTf;
}

/**
 * Keep a short odometry history so slam_toolbox /pose can be paired with the
 * odom sample at the same scan time. The resulting map -> odom anchor is then
 * applied to current odometry for a smooth, map-correct Mapping pose.
 */
export function createMappingPoseSynchronizer({ maxSamples = 240, maxSkewSeconds = 0.25 } = {}) {
    let odometrySamples = [];
    let pendingSlamPose = null;
    let anchor = null;
    let latestOdometryKey = "";
    let latestSlamKey = "";

    const nearestOdometry = (stamp) => {
        let nearest = null;
        let nearestDelta = Infinity;
        odometrySamples.forEach((sample) => {
            const delta = stampDifferenceSeconds(sample.stamp, stamp);
            if (delta === null)
                return;
            const distance = Math.abs(delta);
            if (distance < nearestDelta) {
                nearest = sample;
                nearestDelta = distance;
            }
        });
        return nearest && nearestDelta <= maxSkewSeconds ? nearest : null;
    };

    const tryUpdateAnchor = () => {
        if (!pendingSlamPose)
            return false;
        const latestOdometry = odometrySamples[odometrySamples.length - 1] ?? null;
        const latestDelta = latestOdometry
            ? stampDifferenceSeconds(latestOdometry.stamp, pendingSlamPose.header?.stamp)
            : null;
        // /pose and /odom arrive over independent rosbridge subscriptions.
        // Do not permanently bind a new SLAM pose to an older-but-allowed
        // odometry sample; wait until odometry has reached that timestamp and
        // then select the nearest sample from the bracketed history.
        if (latestDelta === null || latestDelta < 0)
            return false;
        const odometry = nearestOdometry(pendingSlamPose.header?.stamp);
        if (!odometry)
            return false;
        anchor = {
            mapBasePose: poseFromStampedEstimate(pendingSlamPose),
            odomBasePose: odometry.pose,
            stamp: pendingSlamPose.header?.stamp ?? null,
        };
        pendingSlamPose = null;
        return !!anchor.mapBasePose;
    };

    return {
        clear({ preserveSeenKeys = false } = {}) {
            const changed = odometrySamples.length > 0 || !!pendingSlamPose || !!anchor;
            odometrySamples = [];
            pendingSlamPose = null;
            anchor = null;
            if (!preserveSeenKeys) {
                latestOdometryKey = "";
                latestSlamKey = "";
            }
            return changed;
        },
        addOdometry(message) {
            const key = stampKey(message?.header?.stamp);
            const pose = poseFromOdometry(message);
            const parentFrame = normalizeFrameId(message?.header?.frame_id);
            const childFrame = normalizeFrameId(message?.child_frame_id);
            if (
                !key ||
                !pose?.position ||
                key === latestOdometryKey ||
                (parentFrame && parentFrame !== "odom") ||
                (childFrame && childFrame !== "base_link")
            )
                return false;
            latestOdometryKey = key;
            odometrySamples.push({ stamp: message.header.stamp, pose });
            odometrySamples.sort((left, right) => (
                stampDifferenceSeconds(left.stamp, right.stamp) ?? 0
            ));
            if (odometrySamples.length > maxSamples) {
                odometrySamples = odometrySamples.slice(-maxSamples);
            }
            tryUpdateAnchor();
            return true;
        },
        addSlamPose(message) {
            const key = stampKey(message?.header?.stamp);
            const pose = poseFromStampedEstimate(message);
            const frame = normalizeFrameId(message?.header?.frame_id);
            if (!key || !pose?.position || key === latestSlamKey || (frame && frame !== "map"))
                return false;
            latestSlamKey = key;
            pendingSlamPose = message;
            tryUpdateAnchor();
            return true;
        },
        snapshot(scanStamp = null) {
            if (!anchor || odometrySamples.length === 0) {
                return {
                    pose: null,
                scanPose: null,
                mapToOdomPose: null,
                odomPose: null,
                odomStamp: null,
                anchorStamp: null,
            };
            }
            const latest = odometrySamples[odometrySamples.length - 1];
            const scanOdometry = scanStamp ? nearestOdometry(scanStamp) : null;
            return {
                pose: mappingPoseFromOdom(anchor.mapBasePose, anchor.odomBasePose, latest.pose),
                scanPose: scanOdometry
                    ? mappingPoseFromOdom(anchor.mapBasePose, anchor.odomBasePose, scanOdometry.pose)
                    : null,
                mapToOdomPose: mapToOdomPoseFromAnchor(anchor.mapBasePose, anchor.odomBasePose),
                odomPose: latest.pose,
                odomStamp: latest.stamp,
                anchorStamp: anchor.stamp,
            };
        },
    };
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
