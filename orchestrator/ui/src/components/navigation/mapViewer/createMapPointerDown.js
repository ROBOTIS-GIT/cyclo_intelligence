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

export function createMapPointerDown({
  backgroundTapRef,
  interactionDisabled,
  editorActive,
  interactionMode,
  viewRotateDragRef,
  viewRollRef,
  renderInteractionActiveRef,
  renderer,
  pointerDownRef,
  setDragPreviewPose,
  editorAreaSelection,
  onEditorMapArea,
  mapPointFromEvent,
  editorAreaDragRef,
  setEditorAreaPreview,
  paintEditorPoint,
  editorPaintOnDrag,
  editorPaintPointerRef,
  targetFromEvent,
  missionRouteMode,
  onMissionRouteSpotClick,
  onBehaviorNodeClick,
  onSpotClick,
  draggableTargetPose,
  nodeDragRef,
  onMissionRouteMapClick,
  onMapClick,
  onMapTap,
  setMapDragActive,
  previewPoseFromDrag
}) {
  const handlePointerDown = event => {
    backgroundTapRef.current = null;
    if (event.button === 2) {
      if (interactionDisabled || editorActive || (interactionMode !== "view")) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      viewRotateDragRef.current = {
        clientX: event.clientX,
        roll: viewRollRef.current
      };
      // The custom roll drag bypasses OrbitControls, so raise the
      // interaction flag ourselves — otherwise the render loop stays
      // at the idle frame rate and the rotation looks choppy.
      renderInteractionActiveRef.current = true;
      renderer.domElement.setPointerCapture(event.pointerId);
      return;
    }
    if (event.button !== 0) return;
    if (editorActive) {
      if (interactionDisabled) {
        pointerDownRef.current = null;
        setDragPreviewPose(null);
        return;
      }
      if (editorAreaSelection && (typeof onEditorMapArea === "function")) {
        const point = mapPointFromEvent(event);
        if (!point) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        editorAreaDragRef.current = {
          pointerId: event.pointerId,
          clientX: event.clientX,
          clientY: event.clientY,
          startX: point.x,
          startY: point.y,
          endX: point.x,
          endY: point.y
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
    if (!interactionDisabled && (interactionMode === "view")) {
      const target = targetFromEvent(event);
      if (target) {
        event.preventDefault();
        event.stopImmediatePropagation();
        if (missionRouteMode && (target.type === "spot") && (typeof onMissionRouteSpotClick === "function")) {
          onMissionRouteSpotClick(target.id);
        } else if (target.type === "behaviorNode") {
          onBehaviorNodeClick(target.id);
        } else {
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
            dragging: false
          };
          renderer.domElement.setPointerCapture(event.pointerId);
        } else {
          nodeDragRef.current = null;
        }
        pointerDownRef.current = null;
        setDragPreviewPose(null);
        return;
      }
    }
    if (interactionDisabled || (interactionMode === "view")) {
      if (!interactionDisabled && (interactionMode === "view") && missionRouteMode && (typeof onMissionRouteMapClick === "function")) {
        onMissionRouteMapClick();
      } else if (!interactionDisabled && (interactionMode === "view") && (typeof onMapClick === "function")) {
        const point = mapPointFromEvent(event);
        if (point) {
          onMapClick(point.x, point.y);
        }
      }
      if (!interactionDisabled && (interactionMode === "view")) {
        if ((event.isPrimary !== false) && (typeof onMapTap === "function") && !targetFromEvent(event, true)) {
          backgroundTapRef.current = {
            pointerId: event.pointerId,
            clientX: event.clientX,
            clientY: event.clientY,
            dragged: false
          };
        }
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
      mapY: point.y
    };
    pointerDownRef.current = pointerDown;
    setDragPreviewPose(previewPoseFromDrag(pointerDown, point, event.clientX, event.clientY));
    renderer.domElement.setPointerCapture(event.pointerId);
  };
  return {
    handlePointerDown
  };
}
