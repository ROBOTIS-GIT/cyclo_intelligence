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

import { CLICK_DRAG_THRESHOLD_PX } from './constants';
import { yawFromPose } from '../../../utils/navigationTf';
export function createMapPointerCompletion({
  backgroundTapRef,
  interactionDisabled,
  editorActive,
  interactionMode,
  onMapTap,
  targetFromEvent,
  setMapDragActive,
  editorAreaDragRef,
  mapPointFromEvent,
  setEditorAreaPreview,
  renderer,
  onEditorMapArea,
  editorPaintPointerRef,
  flushEditorMove,
  onEditorMapPoint,
  nodeDragRef,
  setNodeDragPreview,
  onSpotPoseChange,
  onBehaviorNodePoseChange,
  viewRotateDragRef,
  renderInteractionActiveRef,
  renderActiveUntilRef,
  pointerDownRef,
  setDragPreviewPose,
  pose,
  onMapPose
}) {
  const handlePointerUp = event => {
    const backgroundTap = backgroundTapRef.current;
    if (backgroundTap && (backgroundTap.pointerId === event.pointerId)) {
      backgroundTapRef.current = null;
      if (!interactionDisabled && !editorActive && (interactionMode === "view") && (event.button === 0) && (event.isPrimary !== false) && !backgroundTap.dragged && (Math.hypot(event.clientX - backgroundTap.clientX, event.clientY - backgroundTap.clientY) <= CLICK_DRAG_THRESHOLD_PX) && (typeof onMapTap === "function") && !targetFromEvent(event, true)) onMapTap();
    }
    setMapDragActive(false);
    const editorAreaDrag = editorAreaDragRef.current;
    if (editorAreaDrag && (editorAreaDrag.pointerId === event.pointerId)) {
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
      if (typeof onEditorMapPoint === "function") onEditorMapPoint(0, 0, "end");
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
        if ((nodeDrag.type === "spot") && (typeof onSpotPoseChange === "function")) {
          onSpotPoseChange(nodeDrag.id, nodeDrag.x, nodeDrag.y, nodeDrag.yaw);
        } else if ((nodeDrag.type === "behaviorNode") && (typeof onBehaviorNodePoseChange === "function")) {
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
    if (interactionDisabled || (interactionMode === "view") || (event.button !== 0)) return;
    const pointerDown = pointerDownRef.current;
    pointerDownRef.current = null;
    setDragPreviewPose(null);
    if (renderer.domElement.hasPointerCapture(event.pointerId)) {
      renderer.domElement.releasePointerCapture(event.pointerId);
    }
    if (!pointerDown) return;
    const point = mapPointFromEvent(event);
    if (!point) return;
    const moved = Math.hypot(event.clientX - pointerDown.clientX, event.clientY - pointerDown.clientY);
    const yaw = moved > CLICK_DRAG_THRESHOLD_PX ? Math.atan2(point.y - pointerDown.mapY, point.x - pointerDown.mapX) : yawFromPose(pose);
    onMapPose(pointerDown.mapX, pointerDown.mapY, yaw);
  };
  const handlePointerCancel = event => {
    backgroundTapRef.current = null;
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
      if (typeof onEditorMapPoint === "function") onEditorMapPoint(0, 0, "end");
    }
    if (renderer.domElement.hasPointerCapture(event.pointerId)) {
      renderer.domElement.releasePointerCapture(event.pointerId);
    }
  };
  const handlePointerLeave = () => {
    backgroundTapRef.current = null;
    setMapDragActive(false);
  };
  const handleContextMenu = event => {
    event.preventDefault();
  };
  return {
    handlePointerUp,
    handlePointerCancel,
    handlePointerLeave,
    handleContextMenu
  };
}
