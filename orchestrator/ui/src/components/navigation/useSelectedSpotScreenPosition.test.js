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
// Author: Seongwoo Kim

import { renderHook } from '@testing-library/react';
import * as THREE from 'three';
import {
  projectSelectedSpotScreenPosition,
  useSelectedSpotScreenPosition,
} from './useSelectedSpotScreenPosition';

function scene() {
  const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
  camera.position.z = 10;
  const layers = new THREE.Group();
  const marker = new THREE.Object3D();
  layers.add(marker);
  return { camera, layers, marker };
}

test('projects the current marker world position into viewport-local CSS pixels', () => {
  const { camera, layers, marker } = scene();
  layers.position.set(1, 2, 0);
  marker.position.set(1, 2, 0);
  expect(projectSelectedSpotScreenPosition(marker, camera, 400, 200)).toEqual({
    x: 240, y: 60, width: 400, height: 200,
  });
  marker.position.x = 3;
  expect(projectSelectedSpotScreenPosition(marker, camera, 400, 200).x).toBe(280);
  camera.position.x = 2;
  expect(projectSelectedSpotScreenPosition(marker, camera, 400, 200).x).toBe(240);
  camera.zoom = 2;
  camera.updateProjectionMatrix();
  expect(projectSelectedSpotScreenPosition(marker, camera, 400, 200).x).toBe(280);
  expect(projectSelectedSpotScreenPosition(marker, camera, 800, 400)).toEqual({
    x: 560, y: 40, width: 800, height: 400,
  });
});

test('hides anchors for missing, hidden, clipped, and invalid markers or viewports', () => {
  const { camera, layers, marker } = scene();
  const project = () => projectSelectedSpotScreenPosition(marker, camera, 400, 200);
  expect(projectSelectedSpotScreenPosition(null, camera, 400, 200)).toBeNull();
  expect(projectSelectedSpotScreenPosition(marker, camera, 0, 200)).toBeNull();
  layers.visible = false;
  expect(project()).toBeNull();
  layers.visible = true;
  marker.position.x = 20;
  expect(project()).toBeNull();
  marker.position.set(0, 0, 20);
  expect(project()).toBeNull();
  marker.position.set(NaN, 0, 0);
  expect(project()).toBeNull();
});

test('publishes changed pixels and viewport size only, including disappearance', () => {
  const { camera, marker } = scene();
  const onChange = jest.fn();
  const cameraRef = { current: camera };
  const containerRef = { current: { clientWidth: 400, clientHeight: 200 } };
  const { result } = renderHook(() => useSelectedSpotScreenPosition({
    cameraRef, containerRef, onChange,
  }));
  result.current.selectedSpotMarkerRef.current = marker;
  result.current.publishSelectedSpotScreenPosition();
  result.current.publishSelectedSpotScreenPosition();
  marker.position.x = 0.001;
  result.current.publishSelectedSpotScreenPosition();
  expect(onChange).toHaveBeenCalledTimes(1);
  marker.position.x = 1;
  result.current.publishSelectedSpotScreenPosition();
  expect(onChange).toHaveBeenLastCalledWith({ x: 220, y: 100, width: 400, height: 200 });
  containerRef.current.clientHeight = 300;
  result.current.publishSelectedSpotScreenPosition();
  expect(onChange).toHaveBeenLastCalledWith({ x: 220, y: 150, width: 400, height: 300 });
  result.current.selectedSpotMarkerRef.current = null;
  result.current.publishSelectedSpotScreenPosition();
  result.current.publishSelectedSpotScreenPosition();
  expect(onChange).toHaveBeenLastCalledWith(null);
  expect(onChange).toHaveBeenCalledTimes(4);
});

test('uses the latest callback without recreating the publisher or republishing unchanged pixels', () => {
  const { camera, marker } = scene();
  const cameraRef = { current: camera };
  const containerRef = { current: { clientWidth: 400, clientHeight: 200 } };
  const first = jest.fn();
  const second = jest.fn();
  const { result, rerender } = renderHook(({ onChange }) => useSelectedSpotScreenPosition({
    cameraRef, containerRef, onChange,
  }), { initialProps: { onChange: first } });
  result.current.selectedSpotMarkerRef.current = marker;
  const publish = result.current.publishSelectedSpotScreenPosition;
  publish();
  rerender({ onChange: second });
  expect(result.current.publishSelectedSpotScreenPosition).toBe(publish);
  publish();
  expect(second).not.toHaveBeenCalled();
  marker.position.x = 1;
  publish();
  expect(second).toHaveBeenCalledTimes(1);
  expect(first).toHaveBeenCalledTimes(1);
});

test('robot thought anchor follows silhouette top at different zooms and headings', () => {
 const {camera, marker}=scene();
 marker.userData.screenAnchorBounds={minX:-0.4,maxX:0.33,minY:-0.3,maxY:0.3};
 const project=()=>projectSelectedSpotScreenPosition(marker,camera,400,200);
 expect(project()).toMatchObject({x:200,y:97,robotBounds:{left:192,right:207,top:97,bottom:103}});
 camera.zoom=4;camera.updateProjectionMatrix();
 expect(project()).toMatchObject({x:200,y:88});
 marker.rotation.z=Math.PI/2;
 expect(project()).toMatchObject({x:200,y:87});
 marker.rotation.z=-Math.PI/2;
 expect(project()).toMatchObject({x:200,y:84});
 delete marker.userData.screenAnchorBounds;
 expect(project()).toMatchObject({x:200,y:100});
});


test('republishes robot bounds when its silhouette changes without moving the top anchor', () => {
 const {camera,marker}=scene();
 marker.userData.screenAnchorBounds={minX:-.4,maxX:.3,minY:-.3,maxY:.3};
 const onChange=jest.fn();
 const {result}=renderHook(()=>useSelectedSpotScreenPosition({cameraRef:{current:camera},
  containerRef:{current:{clientWidth:400,clientHeight:200}},onChange}));
 result.current.selectedSpotMarkerRef.current=marker;
 result.current.publishSelectedSpotScreenPosition();
 marker.userData.screenAnchorBounds.minX=-1;
 result.current.publishSelectedSpotScreenPosition();
 expect(onChange).toHaveBeenCalledTimes(2);
 expect(onChange.mock.calls[0][0].x).toBe(onChange.mock.calls[1][0].x);
 expect(onChange.mock.calls[1][0].robotBounds.left).toBe(180);
});
