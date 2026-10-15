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

import { gridMeta } from './gridGeometry';
import * as THREE from 'three';
import { mapPointToAreaGridCell } from '../../../utils/mapAreaGeometry';
import { waypointHitAreas, nearestWaypointHit } from '../waypointHitTest';
import { CLICK_DRAG_THRESHOLD_PX } from './constants';
import { yawFromPose, orientationFromYaw } from '../../../utils/navigationTf';
export function createMapHitTesting({
  map,
  renderer,
  pointerRef,
  raycasterRef,
  camera,
  layersRef,
  onSpotClick,
  onBehaviorNodeClick,
  spots,
  onSpotPoseChange,
  behaviorNodes,
  selectedBehaviorNodeId,
  onBehaviorNodePoseChange,
  pose
}) {
  const mapPointFromEvent = event => {
    const meta = gridMeta(map);
    if (!meta) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    if ((rect.width <= 0) || (rect.height <= 0)) return null;
    pointerRef.current.x = ((event.clientX - rect.left) / rect.width * 2) - 1;
    pointerRef.current.y = -(((event.clientY - rect.top) / rect.height * 2) - 1);
    raycasterRef.current.setFromCamera(pointerRef.current, camera);
    const point = new THREE.Vector3();
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
    if (!raycasterRef.current.ray.intersectPlane(plane, point)) return null;
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) return null;
    // Validate in the map's local grid frame. Axis-aligned world bounds
    // reject valid points (and accept invalid ones) when origin yaw is
    // non-zero.
    if (!mapPointToAreaGridCell(map, point.x, point.y)) return null;
    return point;
  };
  const targetFromEvent = (event, forBackgroundTap = false) => {
    const layers = layersRef.current;
    if (!layers || (!forBackgroundTap && typeof onSpotClick !== "function" && typeof onBehaviorNodeClick !== "function")) return null;
    const rect = renderer.domElement.getBoundingClientRect();
    if ((rect.width <= 0) || (rect.height <= 0)) return null;
    pointerRef.current.x = ((event.clientX - rect.left) / rect.width * 2) - 1;
    pointerRef.current.y = -(((event.clientY - rect.top) / rect.height * 2) - 1);
    raycasterRef.current.setFromCamera(pointerRef.current, camera);
    const hits = raycasterRef.current.intersectObjects(layers.children, true);
    for (const hit of hits) {
      let object = hit.object;
      while (object) {
        if (object.userData && object.userData.behaviorNodeId && (forBackgroundTap || typeof onBehaviorNodeClick === "function")) return {
          type: "behaviorNode",
          id: object.userData.behaviorNodeId,
          dragAction: "move"
        };
        if (object.userData && object.userData.spotId && (forBackgroundTap || typeof onSpotClick === "function")) return {
          type: "spot",
          id: object.userData.spotId,
          dragAction: object.userData.dragAction || "move"
        };
        object = object.parent;
      }
    }
    // Use screen-space rectangles for waypoint bodies and complete name pills.
    // Exact geometry hits above retain priority (including rotation handles).
    if (forBackgroundTap || (typeof onSpotClick === "function")) {
      const candidates = waypointHitAreas(layers, camera, rect);
      const id = nearestWaypointHit(candidates, event.clientX, event.clientY);
      if (id !== null) return {
        type: "spot",
        id,
        dragAction: "move"
      };
    }
    return null;
  };
  const draggableTargetPose = target => {
    var _a;
    if (((target === null) || (target === void 0) ? void 0 : target.type) === "spot") {
      const spot = spots.find(item => item.id === target.id);
      if (!spot || (spot.poseEditable === false) || (typeof onSpotPoseChange !== "function")) return null;
      const spotPose = spot.pose;
      if (!spotPose) return null;
      return {
        x: Number(((_a = spotPose.x) !== null) && (_a !== void 0) ? _a : 0),
        y: Number((spotPose.y !== null) && (spotPose.y !== void 0) ? spotPose.y : 0),
        yaw: Number((spotPose.yaw !== null) && (spotPose.yaw !== void 0) ? spotPose.yaw : 0)
      };
    }
    if (((target === null) || (target === void 0) ? void 0 : target.type) === "behaviorNode") {
      const node = behaviorNodes.find(item => item.id === target.id);
      if (!node || (node.id !== selectedBehaviorNodeId) || (typeof onBehaviorNodePoseChange !== "function")) return null;
      const nodePose = node.pose;
      if (!nodePose) return null;
      return {
        x: Number((nodePose.x !== null) && (nodePose.x !== void 0) ? nodePose.x : 0),
        y: Number((nodePose.y !== null) && (nodePose.y !== void 0) ? nodePose.y : 0),
        yaw: Number((nodePose.yaw !== null) && (nodePose.yaw !== void 0) ? nodePose.yaw : 0)
      };
    }
    return null;
  };
  const previewPoseFromDrag = (start, point, clientX, clientY) => {
    const moved = Math.hypot(clientX - start.clientX, clientY - start.clientY);
    const yaw = moved > CLICK_DRAG_THRESHOLD_PX ? Math.atan2(point.y - start.mapY, point.x - start.mapX) : yawFromPose(pose);
    return {
      position: {
        x: start.mapX,
        y: start.mapY,
        z: 0
      },
      orientation: orientationFromYaw(yaw)
    };
  };
  return {
    mapPointFromEvent,
    targetFromEvent,
    draggableTargetPose,
    previewPoseFromDrag
  };
}
