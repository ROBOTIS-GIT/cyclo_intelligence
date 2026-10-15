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
import { disposeObject } from './sceneObjects';
import { gridMeta } from './gridGeometry';
import * as THREE from 'three';
export function useEditorBrushLayer({
  editorBrushLayerRef,
  map,
  editorBrush,
  rendererRef,
  cameraRef,
  pointerRef,
  raycasterRef
}) {
  useEffect(() => {
    const brushLayer = editorBrushLayerRef.current;
    if (!brushLayer) return;
    disposeObject(brushLayer);
    brushLayer.clear();
    const meta = gridMeta(map);
    if (!editorBrush || !meta) return;
    const radius = (Math.max(1, Number(editorBrush.sizeCells) || 1) * meta.resolution) / 2;
    const color = new THREE.Color(editorBrush.color || "#5B8266");
    const group = new THREE.Group();
    // Ink/cream contrast underlay so a paper-colored ring stays visible on the map.
    const underlay = new THREE.Mesh(new THREE.RingGeometry(radius * 0.78, radius * 1.08, 40), new THREE.MeshBasicMaterial({
      color: 0x1c1a17,
      transparent: true,
      opacity: 0.32,
      depthWrite: false,
      side: THREE.DoubleSide
    }));
    group.add(underlay);
    const ring = new THREE.Mesh(new THREE.RingGeometry(radius * 0.84, radius, 40), new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.9,
      depthWrite: false,
      side: THREE.DoubleSide
    }));
    group.add(ring);
    const fill = new THREE.Mesh(new THREE.CircleGeometry(radius * 0.84, 40), new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.12,
      depthWrite: false
    }));
    group.add(fill);
    group.position.set(0, 0, 0.9);
    group.visible = false;
    brushLayer.add(group);
  }, [editorBrush, editorBrushLayerRef, map]);
  useEffect(() => {
    const renderer = rendererRef.current;
    const camera = cameraRef.current;
    if (!renderer || !camera || !editorBrush) return undefined;
    const meta = gridMeta(map);
    if (!meta) return undefined;
    const brushGroup = () => {
      var _a;
      return (((_a = editorBrushLayerRef.current) === null) || (_a === void 0) ? void 0 : _a.children[0]) || null;
    };
    const handleMove = event => {
      const group = brushGroup();
      if (!group) return;
      const rect = renderer.domElement.getBoundingClientRect();
      if ((rect.width <= 0) || (rect.height <= 0)) return;
      pointerRef.current.x = ((event.clientX - rect.left) / rect.width * 2) - 1;
      pointerRef.current.y = -(((event.clientY - rect.top) / rect.height * 2) - 1);
      raycasterRef.current.setFromCamera(pointerRef.current, camera);
      const point = new THREE.Vector3();
      const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
      if (!raycasterRef.current.ray.intersectPlane(plane, point)) {
        group.visible = false;
        return;
      }
      const width = meta.width * meta.resolution;
      const height = meta.height * meta.resolution;
      const inside = (point.x >= meta.originX) && (point.x <= meta.originX + width) && (point.y >= meta.originY) && (point.y <= meta.originY + height);
      group.visible = inside;
      if (inside) group.position.set(point.x, point.y, 0.9);
    };
    const handleLeave = () => {
      const group = brushGroup();
      if (group) group.visible = false;
    };
    renderer.domElement.addEventListener("pointermove", handleMove);
    renderer.domElement.addEventListener("pointerleave", handleLeave);
    return () => {
      renderer.domElement.removeEventListener("pointermove", handleMove);
      renderer.domElement.removeEventListener("pointerleave", handleLeave);
      handleLeave();
    };
  }, [cameraRef, editorBrush, editorBrushLayerRef, map, pointerRef, raycasterRef, rendererRef]);
}
