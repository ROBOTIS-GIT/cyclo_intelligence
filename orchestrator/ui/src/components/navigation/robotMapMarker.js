// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import * as THREE from "three";
import { yawFromPose } from "../../utils/navigationTf";
import { SG2_TOP_DOWN_BOUNDS, SG2_TOP_DOWN_PARTS } from "./sg2TopDownGeometry";

import { createSg2Surfaces } from "./sg2MarkerSurface";

export function usesSg2TopDownMarker({ renderProfile, robotType, showFootprint }) {
  return renderProfile === "autonomy" && Boolean(showFootprint)
    && /^ffw_sg2(?:_rev\d+)?(?:_follower)?$/i.test(String(robotType || "").trim());
}

// Two retained meshes: shaded body and soft contact shadows. The simplified
// top-view geometry preserves the source footprint and +X heading.
export function createSg2TopDownMarker() {
  const { surface: geometry, shadow: shadowGeometry } = createSg2Surfaces(SG2_TOP_DOWN_PARTS);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  }));
  mesh.name = "sg2-top-down-silhouette";
  const marker = new THREE.Group();
  marker.name = "sg2-top-down-marker";
  marker.userData.screenAnchorBounds = SG2_TOP_DOWN_BOUNDS;
  marker.visible = false;
  const shadow = new THREE.Mesh(shadowGeometry, new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    forceSinglePass: true,
    depthWrite: false,
  }));
  shadow.name = "sg2-contact-shadows";
  marker.add(mesh, shadow);
  return marker;
}

export function updateSg2TopDownMarker(marker, pose) {
  const x = Number(pose?.position?.x);
  const y = Number(pose?.position?.y);
  marker.visible = Boolean(pose?.position) && Number.isFinite(x) && Number.isFinite(y);
  if (!marker.visible) return;
  marker.position.set(x, y, 0.19);
  marker.rotation.z = yawFromPose(pose);
}
