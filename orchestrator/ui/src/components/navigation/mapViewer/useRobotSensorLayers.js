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

import { useEffect } from 'react';
import { createSg2TopDownMarker, updateSg2TopDownMarker } from '../robotMapMarker';
import { disposeObject, makeGridPlane } from './sceneObjects';
import { gridMeta } from './gridGeometry';
import { buildTfFramePoses, normalizeFrameId, poseForScanFrameAtBasePose, poseForScanFrame, yawFromPose, poseForTfAxesFrame } from '../../../utils/navigationTf';
import { scanCellsForGrid, forEachVisualizedScanRange } from './scanGeometry';
import * as THREE from 'three';
import { makeTfAxes, makeFootprintMarker } from './robotLayers';
import { makePoseMarker } from './waypointMarkers';
export function useRobotSensorLayers({
  latestFootprintRef,
  footprint,
  tfSyncedFootprintRef,
  tf,
  sceneRef,
  showSg2TopDown,
  sg2MarkerRef,
  robotMarkerRef,
  liveLayerRef,
  map,
  pose,
  showLocalCostmap,
  localCostmap,
  scan,
  scanPose,
  showScan,
  scanProjectionRef,
  scanLayerRef,
  showTf,
  showFootprint
}) {
  useEffect(() => {
    latestFootprintRef.current = footprint;
  }, [footprint, latestFootprintRef]);
  useEffect(() => {
    tfSyncedFootprintRef.current = latestFootprintRef.current;
  }, [latestFootprintRef, tf, tfSyncedFootprintRef]);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || !showSg2TopDown) return undefined;
    const marker = createSg2TopDownMarker();
    sg2MarkerRef.current = marker;
    scene.add(marker);
    return () => {
      if (robotMarkerRef.current === marker) robotMarkerRef.current = null;
      sg2MarkerRef.current = null;
      marker.removeFromParent();
      disposeObject(marker);
    };
  }, [showSg2TopDown, robotMarkerRef, sceneRef, sg2MarkerRef]);
  useEffect(() => {
    var _a, _b, _c, _d, _e, _f, _g, _j, _k, _l, _o, _p, _q, _r, _s, _t, _u;
    robotMarkerRef.current = null;
    const group = liveLayerRef.current;
    if (!group) return;
    disposeObject(group);
    group.clear();
    const meta = gridMeta(map);
    const mapKey = meta ? `${meta.width}:${meta.height}:${meta.resolution}:${meta.originX}:${meta.originY}` : null;
    const tfSyncedFootprint = tfSyncedFootprintRef.current;
    const tfFramePoses = buildTfFramePoses(tf, "map");
    const tfFramePoseByName = new Map(tfFramePoses.map(({
      frame,
      pose: framePose
    }) => [frame, framePose]));
    const robotX = Number(((_b = (_a = pose === null || pose === void 0 ? void 0 : pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null) && (_b !== void 0) ? _b : 0);
    const robotY = Number(((_d = (_c = pose === null || pose === void 0 ? void 0 : pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null) && (_d !== void 0) ? _d : 0);
    if (showLocalCostmap && localCostmap) {
      const localFrame = normalizeFrameId(((_e = localCostmap.header) === null) || (_e === void 0) ? void 0 : _e.frame_id);
      const localFramePose = localFrame && (localFrame !== "map") ? ((_f = tfFramePoseByName.get(localFrame)) !== null) && (_f !== void 0) ? _f : null : null;
      const scanFrame = normalizeFrameId(((_g = scan === null || scan === void 0 ? void 0 : scan.header) === null) || (_g === void 0) ? void 0 : _g.frame_id) || "base_link";
      const resolvedScanPose = scanPose ? poseForScanFrameAtBasePose(scanFrame, tfFramePoseByName, scanPose) : poseForScanFrame(scanFrame, tfFramePoseByName, pose);
      const scanCells = scanCellsForGrid(localCostmap, scan, resolvedScanPose, localFramePose);
      const plane = makeGridPlane(localCostmap, "localCostmap", 0.1, localFramePose, scanCells);
      if (plane) group.add(plane);
    }
    if (showScan && (((_j = scan === null || scan === void 0 ? void 0 : scan.ranges) === null) || (_j === void 0) ? void 0 : _j.length)) {
      const scanFrame = normalizeFrameId(((_l = scan.header) === null) || (_l === void 0) ? void 0 : _l.frame_id) || "base_link";
      const resolvedScanPose = scanPose ? poseForScanFrameAtBasePose(scanFrame, tfFramePoseByName, scanPose) : poseForScanFrame(scanFrame, tfFramePoseByName, pose);
      // Keep the legacy Run projection frozen for the lifetime of one
      // LaserScan. Mapping supplies an explicit scan-time pose; include
      // only that pose in the cache key so a late matching odometry
      // sample can correct the same scan without making live Run scans
      // drift with subsequent robot-pose updates.
      const projectionPoseKey = scanPose?.position && resolvedScanPose?.position ? `${Number(resolvedScanPose.position.x ?? 0)}:${Number(resolvedScanPose.position.y ?? 0)}:${yawFromPose(resolvedScanPose)}` : "legacy";
      let points = (((_k = scanProjectionRef.current) === null || _k === void 0 ? void 0 : _k.scan) === scan) && (scanProjectionRef.current.mapKey === mapKey) && (scanProjectionRef.current.projectionPoseKey === projectionPoseKey) ? scanProjectionRef.current.points : null;
      if (!points) {
        const scanX = Number(((_p = (_o = resolvedScanPose === null || resolvedScanPose === void 0 ? void 0 : resolvedScanPose.position) === null || _o === void 0 ? void 0 : _o.x) !== null) && (_p !== void 0) ? _p : robotX);
        const scanY = Number(((_r = (_q = resolvedScanPose === null || resolvedScanPose === void 0 ? void 0 : resolvedScanPose.position) === null || _q === void 0 ? void 0 : _q.y) !== null) && (_r !== void 0) ? _r : robotY);
        const scanYaw = yawFromPose(resolvedScanPose);
        const min = Number(((_s = scan.range_min) !== null) && (_s !== void 0) ? _s : 0.02);
        const max = Number(((_t = scan.range_max) !== null) && (_t !== void 0) ? _t : 20);
        const angleMin = Number(((_u = scan.angle_min) !== null) && (_u !== void 0) ? _u : 0);
        const inc = Number(scan.angle_increment ?? 0);
        points = [];
        forEachVisualizedScanRange(scan.ranges, (range, index) => {
          const r = Number(range);
          if (!Number.isFinite(r) || (r < min) || (r > max)) return;
          const angle = scanYaw + angleMin + (inc * index);
          points.push(scanX + (Math.cos(angle) * r), scanY + (Math.sin(angle) * r), 0.11);
        });
        scanProjectionRef.current = {
          scan,
          mapKey,
          projectionPoseKey,
          points
        };
      }
      const scanLayer = scanLayerRef.current;
      if (scanLayer && (scanLayer.userData.projectedPoints !== points)) {
        disposeObject(scanLayer);
        scanLayer.clear();
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
        const material = new THREE.PointsMaterial({
          color: 0x22c55e,
          size: 0.1,
          sizeAttenuation: true
        });
        scanLayer.add(new THREE.Points(geometry, material));
        scanLayer.userData.projectedPoints = points;
      }
    } else if (scanLayerRef.current?.children.length) {
      disposeObject(scanLayerRef.current);
      scanLayerRef.current.clear();
      scanLayerRef.current.userData.projectedPoints = null;
    }
    if (showTf && ((tf === null) || (tf === void 0) ? void 0 : tf.transforms?.length)) {
      const framePoses = tfFramePoses.slice(0, 80);
      if (framePoses.length > 0) {
        framePoses.forEach(({
          frame,
          pose: framePose
        }) => {
          group.add(makeTfAxes(poseForTfAxesFrame(frame, framePose, pose), frame));
        });
      } else {
        group.add(makeTfAxes({
          position: {
            x: robotX,
            y: robotY,
            z: 0
          },
          orientation: (pose === null) || (pose === void 0) ? void 0 : pose.orientation
        }, "base_link"));
      }
    }
    const footprintPoints = (tfSyncedFootprint === null) || (tfSyncedFootprint === void 0) ? void 0 : tfSyncedFootprint.polygon?.points;
    // The SG2 display silhouette replaces the circular visual in Autonomy
    // only. The ROS/Nav2 footprint message and collision configuration stay intact.
    if (!showSg2TopDown && showFootprint && ((footprintPoints === null) || (footprintPoints === void 0) ? void 0 : footprintPoints.length)) {
      const footprintFrame = normalizeFrameId(tfSyncedFootprint.header?.frame_id);
      const tfFootprintFramePose = footprintFrame && (footprintFrame !== "map") ? tfFramePoseByName.get(footprintFrame) ?? null : null;
      const footprintFramePose = poseForTfAxesFrame(footprintFrame, tfFootprintFramePose, pose);
      const footprintMarker = makeFootprintMarker(tfSyncedFootprint, footprintFramePose);
      if (footprintMarker) group.add(footprintMarker);
    }
    if (showSg2TopDown && sg2MarkerRef.current) {
      const marker = sg2MarkerRef.current;
      updateSg2TopDownMarker(marker, pose);
      robotMarkerRef.current = map && marker.visible ? marker : null;
    } else if ((pose === null) || (pose === void 0) ? void 0 : pose.position) {
      const marker = makePoseMarker(pose, showFootprint && ((footprintPoints === null) || (footprintPoints === void 0) ? void 0 : footprintPoints.length) ? 0x60a5fa : 0x007acc, 0.16);
      group.add(marker);
      robotMarkerRef.current = map ? marker : null;
    }
  }, [tf, pose, scan, scanPose, localCostmap, showLocalCostmap, showScan, showTf, showFootprint, showSg2TopDown, map, robotMarkerRef, liveLayerRef, tfSyncedFootprintRef, scanLayerRef, sg2MarkerRef, scanProjectionRef]);
}
