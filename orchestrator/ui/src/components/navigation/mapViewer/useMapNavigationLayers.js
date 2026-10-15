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
import { disposeObject, makeGridPlane, makeLine } from './sceneObjects';
import { gridMeta } from './gridGeometry';
import { updateGlobalCostmapTexture } from './occupancyTexture';
import * as THREE from 'three';
import { makePoseMarker } from './waypointMarkers';
export function useMapNavigationLayers({
  mapLayerRef,
  showMap,
  map,
  mapRefined,
  costmapLayerRef,
  showGlobalCostmap,
  globalCostmap,
  globalCostmapTextureRef,
  navPathLayerRef,
  showGlobalPlan,
  plan,
  showGoalPose,
  goalPose
}) {
  useEffect(() => {
    const mapLayer = mapLayerRef.current;
    if (!mapLayer) return;
    disposeObject(mapLayer);
    mapLayer.clear();
    if (!showMap || !map) return;
    const mapPlane = makeGridPlane(map, "map", 0, null, null, mapRefined);
    if (mapPlane) mapLayer.add(mapPlane);
  }, [map, showMap, mapRefined, mapLayerRef]);
  useEffect(() => {
    const group = costmapLayerRef.current;
    if (!group) return;
    if (!showGlobalCostmap || !globalCostmap) {
      disposeObject(group);
      group.clear();
      globalCostmapTextureRef.current = null;
      return;
    }
    const meta = gridMeta(globalCostmap);
    const geometryKey = meta ? `${meta.width}:${meta.height}:${meta.resolution}:${meta.originX}:${meta.originY}:${meta.originYaw}` : null;
    const current = globalCostmapTextureRef.current;
    if (current && geometryKey && (current.geometryKey === geometryKey) && globalCostmap.updateRegion && updateGlobalCostmapTexture(current.texture, globalCostmap, globalCostmap.updateRegion)) {
      return;
    }
    disposeObject(group);
    group.clear();
    globalCostmapTextureRef.current = null;
    const plane = makeGridPlane(globalCostmap, "globalCostmap", 0.03, null, null);
    if (plane) {
      group.add(plane);
      globalCostmapTextureRef.current = {
        geometryKey,
        texture: plane.userData.mapTexture
      };
    }
  }, [costmapLayerRef, globalCostmap, globalCostmapTextureRef, showGlobalCostmap]);
  useEffect(() => {
    var _a, _b;
    const group = navPathLayerRef.current;
    if (!group) return;
    disposeObject(group);
    group.clear();
    if (showGlobalPlan && (((_a = plan === null || plan === void 0 ? void 0 : plan.poses) === null) || (_a === void 0) ? void 0 : _a.length)) {
      const points = plan.poses.map(p => {
        var _c;
        return ((_c = p.pose) === null) || (_c === void 0) ? void 0 : _c.position;
      }).filter(p => !!p).map(p => {
        var _c, _d;
        return new THREE.Vector3(Number(((_c = p.x) !== null) && (_c !== void 0) ? _c : 0), Number(((_d = p.y) !== null) && (_d !== void 0) ? _d : 0), 0.09);
      });
      const planLine = makeLine(points, 0x0e7fd1, 3);
      if (planLine) group.add(planLine);
    }
    if (showGoalPose && (((_b = goalPose === null || goalPose === void 0 ? void 0 : goalPose.pose) === null) || (_b === void 0) ? void 0 : _b.position)) {
      group.add(makePoseMarker(goalPose.pose, 0xc96442, 0.14));
    }
  }, [plan, showGlobalPlan, goalPose, showGoalPose, navPathLayerRef]);
}
