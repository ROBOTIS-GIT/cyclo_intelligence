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
import { createMapHitTesting } from './createMapHitTesting';
import { createEditorPaintGesture } from './createEditorPaintGesture';
import { createMapPointerDown } from './createMapPointerDown';
import { createMapPointerMove } from './createMapPointerMove';
import { createMapPointerCompletion } from './createMapPointerCompletion';
export function useMapPointerInteractions({
  rendererRef,
  cameraRef,
  controlsRef,
  map,
  pointerRef,
  raycasterRef,
  layersRef,
  onSpotClick,
  onBehaviorNodeClick,
  spots,
  onSpotPoseChange,
  behaviorNodes,
  selectedBehaviorNodeId,
  onBehaviorNodePoseChange,
  pose,
  onEditorMapPoint,
  editorMoveRafRef,
  editorMovePendingRef,
  backgroundTapRef,
  interactionDisabled,
  editorActive,
  interactionMode,
  viewRotateDragRef,
  viewRollRef,
  renderInteractionActiveRef,
  pointerDownRef,
  setDragPreviewPose,
  editorAreaSelection,
  onEditorMapArea,
  editorAreaDragRef,
  setEditorAreaPreview,
  editorPaintOnDrag,
  editorPaintPointerRef,
  missionRouteMode,
  onMissionRouteSpotClick,
  nodeDragRef,
  onMissionRouteMapClick,
  onMapClick,
  onMapTap,
  setMapDragActive,
  setNodeDragPreview,
  renderActiveUntilRef,
  onMapPose,
  selectedSpotId
}) {
  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!renderer || !camera || !controls) return;
    const {
      mapPointFromEvent,
      targetFromEvent,
      draggableTargetPose,
      previewPoseFromDrag
    } = createMapHitTesting({
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
    });
    const {
      paintEditorPoint,
      flushEditorMove,
      queueEditorMove
    } = createEditorPaintGesture({
      onEditorMapPoint,
      mapPointFromEvent,
      editorMoveRafRef,
      editorMovePendingRef
    });
    // Coalesce paint moves to one commit per animation frame — pointermove can
    // fire far faster than strokes need, and each commit re-rasterizes the
    // stroked annotation.

    const {
      handlePointerDown
    } = createMapPointerDown({
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
    });
    const {
      handlePointerMove
    } = createMapPointerMove({
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
    });
    const {
      handlePointerUp,
      handlePointerCancel,
      handlePointerLeave,
      handleContextMenu
    } = createMapPointerCompletion({
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
    });
    renderer.domElement.addEventListener("pointerdown", handlePointerDown, {
      capture: true
    });
    renderer.domElement.addEventListener("pointermove", handlePointerMove);
    renderer.domElement.addEventListener("pointerup", handlePointerUp);
    renderer.domElement.addEventListener("pointercancel", handlePointerCancel);
    renderer.domElement.addEventListener("pointerleave", handlePointerLeave);
    renderer.domElement.addEventListener("contextmenu", handleContextMenu);
    return () => {
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown, {
        capture: true
      });
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
  }, [behaviorNodes, editorActive, editorAreaSelection, editorPaintOnDrag, interactionDisabled, interactionMode, map, missionRouteMode, onBehaviorNodeClick, onBehaviorNodePoseChange, onEditorMapArea, onEditorMapPoint, onMapPose, onMapClick, onMapTap, onMissionRouteMapClick, onMissionRouteSpotClick, onSpotClick, onSpotPoseChange, pose, selectedBehaviorNodeId, selectedSpotId, spots, rendererRef, cameraRef, controlsRef, pointerRef, raycasterRef, layersRef, editorMoveRafRef, editorMovePendingRef, backgroundTapRef, viewRotateDragRef, viewRollRef, renderInteractionActiveRef, pointerDownRef, setDragPreviewPose, editorAreaDragRef, setEditorAreaPreview, editorPaintPointerRef, nodeDragRef, setMapDragActive, setNodeDragPreview, renderActiveUntilRef]);
}
