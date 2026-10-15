// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { useCallback, useRef } from "react";
import * as THREE from "three";

// CSS pixels relative to the map viewport, independent of canvas DPR or its
// page offset. The marker is also the source of live waypoint drag previews.
export function projectSelectedSpotScreenPosition(marker, camera, width, height) {
  if (!marker || !camera || !(width > 0) || !(height > 0)) return null;
  for (let parent = marker; parent; parent = parent.parent) {
    if (!parent.visible) return null;
  }
  camera.updateWorldMatrix(true, false);
  const projected = marker.getWorldPosition(new THREE.Vector3()).project(camera);
  if (
    !Number.isFinite(projected.x) ||
    !Number.isFinite(projected.y) ||
    !Number.isFinite(projected.z) ||
    Math.abs(projected.x) > 1 ||
    Math.abs(projected.y) > 1 ||
    Math.abs(projected.z) > 1
  ) return null;
  // Robot silhouettes provide local bounds so the thought tail remains above
  // the entire icon at any map zoom/yaw. Waypoint editors keep center anchors.
  let screenY = ((1 - projected.y) * height) / 2;
  let robotBounds;
  const bounds = marker.userData?.screenAnchorBounds;
  if (bounds && [bounds.minX, bounds.maxX, bounds.minY, bounds.maxY].every(Number.isFinite)) {
    const corner = new THREE.Vector3();
    robotBounds = { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity };
    for (const x of [bounds.minX, bounds.maxX]) {
      for (const y of [bounds.minY, bounds.maxY]) {
        corner.set(x, y, 0).applyMatrix4(marker.matrixWorld).project(camera);
        if (Number.isFinite(corner.x) && Number.isFinite(corner.y)) {
          const x = ((corner.x + 1) * width) / 2, y = ((1 - corner.y) * height) / 2;
          screenY = Math.min(screenY, y);
          robotBounds.left = Math.min(robotBounds.left, x); robotBounds.right = Math.max(robotBounds.right, x);
          robotBounds.top = Math.min(robotBounds.top, y); robotBounds.bottom = Math.max(robotBounds.bottom, y);
        }
      }
    }
  }
  return {
    x: Math.round(((projected.x + 1) * width) / 2),
    y: Math.round(screenY),
    width,
    height,
    ...(robotBounds ? { robotBounds: Object.fromEntries(Object.entries(robotBounds).map(([key, value]) => [key, Math.round(value)])) } : {}),
  };
}

export function useSelectedSpotScreenPosition({ cameraRef, containerRef, onChange }) {
  const selectedSpotMarkerRef = useRef(null);
  const callbackRef = useRef(onChange);
  callbackRef.current = onChange;
  const lastPositionRef = useRef(undefined);
  const publishSelectedSpotScreenPosition = useCallback(() => {
    if (typeof callbackRef.current !== "function") return;
    const container = containerRef.current;
    const next = projectSelectedSpotScreenPosition(
      selectedSpotMarkerRef.current,
      cameraRef.current,
      container?.clientWidth,
      container?.clientHeight,
    );
    const previous = lastPositionRef.current;
    if (
      previous === next ||
      (previous && next && previous.x === next.x && previous.y === next.y &&
        previous.width === next.width && previous.height === next.height &&
        ["left", "right", "top", "bottom"].every(key => previous.robotBounds?.[key] === next.robotBounds?.[key]))
    ) return;
    lastPositionRef.current = next;
    callbackRef.current(next);
  }, [cameraRef, containerRef]);
  return { selectedSpotMarkerRef, publishSelectedSpotScreenPosition };
}
