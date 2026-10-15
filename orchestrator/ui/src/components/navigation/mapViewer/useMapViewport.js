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
import { gridMeta } from './gridGeometry';
import { focusCameraToWaypoint } from './cameraGeometry';
import { yawFromPose } from '../../../utils/navigationTf';
export function useMapViewport({
  rendererRef,
  controlsRef,
  interactionDisabled,
  editorActive,
  editorAreaSelection,
  interactionMode,
  mapDragActive,
  cameraRef,
  map,
  btSpotPose,
  btFocusKeyRef,
  viewKey,
  btSpotId,
  viewRollRef,
  followRobotRef,
  missionFollowRobot,
  renderActiveUntilRef,
  pose,
  followPoseRef,
  lastMotionPoseRef,
  dragPreviewPose,
  nodeDragPreview,
  editorAreaPreview,
  renderInteractionActiveRef
}) {
  useEffect(() => {
    const renderer = rendererRef.current;
    const controls = controlsRef.current;
    if (!renderer) return;
    const cursor = interactionDisabled ? "cursor-wait" : editorActive ? editorAreaSelection ? "cursor-crosshair" : "cursor-cell" : interactionMode === "view" ? mapDragActive ? "cursor-grabbing" : "cursor-grab" : "cursor-crosshair";
    renderer.domElement.className = `block w-full h-full ${cursor}`;
    if (controls) {
      controls.enabled = !interactionDisabled && !editorActive && (interactionMode === "view");
    }
  }, [controlsRef, editorActive, editorAreaSelection, interactionDisabled, interactionMode, mapDragActive, rendererRef]);
  useEffect(() => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const meta = gridMeta(map);
    if (!btSpotPose) {
      btFocusKeyRef.current = null;
      return;
    }
    if (!camera || !controls || !meta) return;
    // Status polling replaces pose/map objects. Only a changed focus target
    // should reposition the camera, not a new copy of the same status.
    const focusKey = JSON.stringify([viewKey, btSpotId, btSpotPose.x, btSpotPose.y, meta.width, meta.height, meta.resolution]);
    if (btFocusKeyRef.current !== focusKey) {
      focusCameraToWaypoint(camera, controls, meta, btSpotPose, viewRollRef.current);
      btFocusKeyRef.current = focusKey;
    }
  }, [btFocusKeyRef, btSpotId, btSpotPose, cameraRef, controlsRef, map, viewKey, viewRollRef]);
  useEffect(() => {
    followRobotRef.current = missionFollowRobot && !btSpotPose;
    if (followRobotRef.current) renderActiveUntilRef.current = performance.now() + 400;
  }, [missionFollowRobot, btSpotPose, followRobotRef, renderActiveUntilRef]);
  useEffect(() => {
    const nextPosition = ((pose === null) || (pose === void 0) ? void 0 : pose.position) ? {
      x: Number(pose.position.x),
      y: Number(pose.position.y)
    } : null;
    followPoseRef.current = nextPosition;
    const nextMotionPose = nextPosition ? {
      ...nextPosition,
      yaw: yawFromPose(pose)
    } : null;
    const previous = lastMotionPoseRef.current;
    if (nextMotionPose && (!previous || Math.hypot(nextMotionPose.x - previous.x, nextMotionPose.y - previous.y) >= 0.01 || Math.abs(Math.atan2(Math.sin(nextMotionPose.yaw - previous.yaw), Math.cos(nextMotionPose.yaw - previous.yaw))) >= 0.01)) {
      renderActiveUntilRef.current = performance.now() + 350;
    }
    lastMotionPoseRef.current = nextMotionPose;
  }, [followPoseRef, lastMotionPoseRef, pose, renderActiveUntilRef]);
  useEffect(() => {
    const interactionActive = Boolean(mapDragActive || dragPreviewPose || nodeDragPreview || editorAreaPreview);
    renderInteractionActiveRef.current = interactionActive;
    if (!interactionActive) renderActiveUntilRef.current = performance.now() + 400;
  }, [dragPreviewPose, editorAreaPreview, mapDragActive, nodeDragPreview, renderActiveUntilRef, renderInteractionActiveRef]);
}
