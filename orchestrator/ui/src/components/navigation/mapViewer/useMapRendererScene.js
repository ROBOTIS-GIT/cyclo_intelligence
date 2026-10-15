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
import * as THREE from 'three';
import { SCENE_BG, CAMERA_NEAR, CAMERA_FAR } from './constants';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { mapRenderIntervalMs } from './cameraGeometry';
import { disposeObject } from './sceneObjects';
export function useMapRendererScene({
  containerRef,
  rendererRef,
  sceneRef,
  cameraRef,
  renderProfileRef,
  mapLayerRef,
  layersRef,
  editorBrushLayerRef,
  costmapLayerRef,
  navPathLayerRef,
  liveLayerRef,
  scanLayerRef,
  controlsRef,
  renderInteractionActiveRef,
  renderActiveUntilRef,
  renderInteractionUntilRef,
  animationFrameRef,
  followPoseRef,
  followRobotRef,
  pulseActiveRef,
  publishSelectedSpotScreenPosition,
  publishRobotScreenPosition,
  setViewerError,
  selectedSpotMarkerRef,
  robotMarkerRef,
  globalCostmapTextureRef,
  renderProfile
}) {
  useEffect(() => {
    const containerEl = containerRef.current;
    if (!containerEl || rendererRef.current) return;
    let scene = null;
    let camera = null;
    let renderer = null;
    let mapLayer = null;
    let layers = null;
    let controls = null;
    let resizeObserver = null;
    let resize = null;
    let handleControlsStart = null;
    let handleControlsEnd = null;
    try {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(SCENE_BG);
      sceneRef.current = scene;
      camera = new THREE.PerspectiveCamera(48, 1, CAMERA_NEAR, CAMERA_FAR);
      camera.up.set(0, 1, 0);
      camera.position.set(0, 0, 10);
      cameraRef.current = camera;
      renderer = new THREE.WebGLRenderer({
        antialias: true
      });
      // Cap the pixel ratio: full DPR on hi-DPI displays nearly doubles
      // the pixels rendered per frame for no visible gain on a map view,
      // and the full-width editor canvas made that cost noticeable.
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, renderProfileRef.current === "autonomy" ? 1 : 1.5));
      renderer.setClearColor(SCENE_BG, 1);
      renderer.domElement.className = "block w-full h-full cursor-grab";
      containerEl.appendChild(renderer.domElement);
      rendererRef.current = renderer;
      mapLayer = new THREE.Group();
      scene.add(mapLayer);
      mapLayerRef.current = mapLayer;
      layers = new THREE.Group();
      scene.add(layers);
      layersRef.current = layers;
      // Dedicated group for the pointer-following editor brush ring — kept
      // outside `layers` so hover tracking never rebuilds the heavy layers.
      const brushLayer = new THREE.Group();
      scene.add(brushLayer);
      editorBrushLayerRef.current = brushLayer;
      const costmapLayer = new THREE.Group();
      scene.add(costmapLayer);
      costmapLayerRef.current = costmapLayer;
      const navPathLayer = new THREE.Group();
      scene.add(navPathLayer);
      navPathLayerRef.current = navPathLayer;
      const liveLayer = new THREE.Group();
      scene.add(liveLayer);
      liveLayerRef.current = liveLayer;
      const scanLayer = new THREE.Group();
      scene.add(scanLayer);
      scanLayerRef.current = scanLayer;
      controls = new OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
      controls.enableRotate = false;
      controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN
      };
      controls.screenSpacePanning = true;
      controlsRef.current = controls;
      handleControlsStart = () => {
        renderInteractionActiveRef.current = true;
      };
      handleControlsEnd = () => {
        renderInteractionActiveRef.current = false;
        renderActiveUntilRef.current = performance.now() + 400;
        renderInteractionUntilRef.current = renderActiveUntilRef.current;
      };
      controls.addEventListener?.("start", handleControlsStart);
      controls.addEventListener?.("end", handleControlsEnd);
      resize = () => {
        const width = containerEl.clientWidth || 1;
        const height = containerEl.clientHeight || 1;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        renderActiveUntilRef.current = performance.now() + 250;
      };
      resize();
      if (typeof ResizeObserver !== "undefined") {
        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(containerEl);
      } else {
        window.addEventListener("resize", resize);
      }
      const pulseStart = performance.now();
      // Adapt rendering to actual activity. Static maps render at 10fps,
      // autonomy motion/pulses use 30fps, gestures use 60fps, and hidden tabs use 2fps.
      let lastFrameAt = 0;
      const animate = () => {
        animationFrameRef.current = requestAnimationFrame(animate);
        const now = performance.now();
        const followTarget = followPoseRef.current;
        const followCameraMoving = Boolean(followRobotRef.current && followTarget && (Math.hypot(followTarget.x - controls.target.x, followTarget.y - controls.target.y) > 0.002));
        const renderActive = renderInteractionActiveRef.current || followCameraMoving || pulseActiveRef.current || (now < renderActiveUntilRef.current);
        const frameInterval = mapRenderIntervalMs({
          hidden: document.visibilityState === "hidden",
          active: renderActive,
          interactive: renderInteractionActiveRef.current || (now < renderInteractionUntilRef.current),
          renderProfile: renderProfileRef.current
        });
        if ((now - lastFrameAt) < frameInterval) return;
        lastFrameAt = now;
        const elapsed = (now - pulseStart) / 1000;
        // Pulse the active-waypoint halo (userData.pulse), and smoothly pan
        // the camera to keep the robot centered while it navigates.
        const wave = 0.5 + (0.5 * Math.sin(elapsed * 3.2));
        const activeLayers = layersRef.current;
        if (activeLayers && pulseActiveRef.current) {
          activeLayers.traverse(object => {
            if (object.userData && object.userData.pulse && object.material) {
              object.material.opacity = object.userData.pulseBase * (0.35 + 0.65 * wave);
              object.scale.setScalar(1 + (0.16 * wave));
            }
          });
        }
        if (followRobotRef.current && followPoseRef.current) {
          const target = followPoseRef.current;
          const dx = (target.x - controls.target.x) * 0.08;
          const dy = (target.y - controls.target.y) * 0.08;
          if ((Math.abs(dx) > 1e-4) || (Math.abs(dy) > 1e-4)) {
            controls.target.x += dx;
            controls.target.y += dy;
            camera.position.x += dx;
            camera.position.y += dy;
          }
        }
        controls.update();
        renderer.render(scene, camera);
        publishSelectedSpotScreenPosition();
        publishRobotScreenPosition();
      };
      animate();
      setViewerError(null);
    } catch (error) {
      console.error("Navigation map viewer failed to initialize:", error);
      setViewerError(error instanceof Error ? error.message : "Map viewer failed to initialize");
      (controls === null) || (controls === void 0) ? void 0 : controls.dispose();
      if (mapLayer) disposeObject(mapLayer);
      if (layers) disposeObject(layers);
      (renderer === null) || (renderer === void 0) ? void 0 : renderer.dispose();
      if (((renderer === null) || (renderer === void 0) ? void 0 : renderer.domElement.parentNode) === containerEl) {
        containerEl.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
      rendererRef.current = null;
      cameraRef.current = null;
      selectedSpotMarkerRef.current = null;
      publishSelectedSpotScreenPosition();
      robotMarkerRef.current = null;
      publishRobotScreenPosition();
      controlsRef.current = null;
      mapLayerRef.current = null;
      layersRef.current = null;
      editorBrushLayerRef.current = null;
      return undefined;
    }
    return () => {
      if (resizeObserver) {
        resizeObserver.disconnect();
      } else if (resize) {
        window.removeEventListener("resize", resize);
      }
      if (animationFrameRef.current != null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (controls && handleControlsStart) controls.removeEventListener?.("start", handleControlsStart);
      if (controls && handleControlsEnd) controls.removeEventListener?.("end", handleControlsEnd);
      (controls === null) || (controls === void 0) ? void 0 : controls.dispose();
      if (mapLayer) disposeObject(mapLayer);
      if (layers) disposeObject(layers);
      if (editorBrushLayerRef.current) disposeObject(editorBrushLayerRef.current);
      if (costmapLayerRef.current) disposeObject(costmapLayerRef.current);
      if (navPathLayerRef.current) disposeObject(navPathLayerRef.current);
      if (liveLayerRef.current) disposeObject(liveLayerRef.current);
      if (scanLayerRef.current) disposeObject(scanLayerRef.current);
      (renderer === null) || (renderer === void 0) ? void 0 : renderer.dispose();
      if (((renderer === null) || (renderer === void 0) ? void 0 : renderer.domElement.parentNode) === containerEl) {
        containerEl.removeChild(renderer.domElement);
      }
      sceneRef.current = null;
      rendererRef.current = null;
      cameraRef.current = null;
      selectedSpotMarkerRef.current = null;
      publishSelectedSpotScreenPosition();
      robotMarkerRef.current = null;
      publishRobotScreenPosition();
      controlsRef.current = null;
      mapLayerRef.current = null;
      layersRef.current = null;
      editorBrushLayerRef.current = null;
      costmapLayerRef.current = null;
      globalCostmapTextureRef.current = null;
      navPathLayerRef.current = null;
      liveLayerRef.current = null;
      scanLayerRef.current = null;
    };
  }, [publishSelectedSpotScreenPosition, selectedSpotMarkerRef, publishRobotScreenPosition, robotMarkerRef, containerRef, rendererRef, sceneRef, cameraRef, renderProfileRef, mapLayerRef, layersRef, editorBrushLayerRef, costmapLayerRef, navPathLayerRef, liveLayerRef, scanLayerRef, controlsRef, setViewerError, renderInteractionActiveRef, renderActiveUntilRef, renderInteractionUntilRef, animationFrameRef, followPoseRef, followRobotRef, pulseActiveRef, globalCostmapTextureRef]);
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, renderProfile === "autonomy" ? 1 : 1.5));
    renderActiveUntilRef.current = performance.now() + 250;
  }, [renderActiveUntilRef, renderProfile, rendererRef]);
}
