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
import { applyTopViewRoll } from './cameraGeometry';
export function createMapPointerMove({
  backgroundTapRef,
  editorAreaDragRef,
  mapPointFromEvent,
  setEditorAreaPreview,
  editorPaintPointerRef,
  queueEditorMove,
  nodeDragRef,
  setNodeDragPreview,
  viewRotateDragRef,
  viewRollRef,
  camera,
  controls,
  interactionDisabled,
  interactionMode,
  pointerDownRef,
  setDragPreviewPose,
  previewPoseFromDrag
}) {
  const handlePointerMove = event => {
    const backgroundTap = backgroundTapRef.current;
    if (backgroundTap && (backgroundTap.pointerId === event.pointerId)) {
      backgroundTap.dragged ||= Math.hypot(event.clientX - backgroundTap.clientX, event.clientY - backgroundTap.clientY) > CLICK_DRAG_THRESHOLD_PX;
    }
    const editorAreaDrag = editorAreaDragRef.current;
    if (editorAreaDrag && (editorAreaDrag.pointerId === event.pointerId)) {
      const point = mapPointFromEvent(event);
      if (!point) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      editorAreaDrag.endX = point.x;
      editorAreaDrag.endY = point.y;
      setEditorAreaPreview({
        startX: editorAreaDrag.startX,
        startY: editorAreaDrag.startY,
        endX: point.x,
        endY: point.y
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
      if (!point) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      const moved = Math.hypot(event.clientX - nodeDrag.clientX, event.clientY - nodeDrag.clientY);
      const nextX = nodeDrag.dragAction === "rotate" ? nodeDrag.x : point.x - nodeDrag.offsetX;
      const nextY = nodeDrag.dragAction === "rotate" ? nodeDrag.y : point.y - nodeDrag.offsetY;
      const nextYaw = nodeDrag.dragAction === "rotate" ? Math.atan2(point.y - nodeDrag.y, point.x - nodeDrag.x) : nodeDrag.yaw;
      nodeDrag.x = nextX;
      nodeDrag.y = nextY;
      nodeDrag.yaw = nextYaw;
      nodeDrag.dragging = nodeDrag.dragging || (moved > CLICK_DRAG_THRESHOLD_PX);
      if (nodeDrag.dragging) {
        setNodeDragPreview({
          type: nodeDrag.type,
          id: nodeDrag.id,
          x: nextX,
          y: nextY,
          yaw: nextYaw
        });
      }
      return;
    }
    const viewRotateDrag = viewRotateDragRef.current;
    if (viewRotateDrag) {
      event.preventDefault();
      event.stopImmediatePropagation();
      const nextRoll = viewRotateDrag.roll - ((event.clientX - viewRotateDrag.clientX) * 0.01);
      viewRollRef.current = nextRoll;
      applyTopViewRoll(camera, controls, nextRoll);
      return;
    }
    if (interactionDisabled || (interactionMode === "view")) return;
    const pointerDown = pointerDownRef.current;
    if (!pointerDown) return;
    const point = mapPointFromEvent(event);
    if (!point) return;
    setDragPreviewPose(previewPoseFromDrag(pointerDown, point, event.clientX, event.clientY));
  };
  return {
    handlePointerMove
  };
}
