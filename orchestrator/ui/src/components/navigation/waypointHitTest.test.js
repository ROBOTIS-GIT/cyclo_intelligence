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

import { nearestWaypointHit } from './waypointHitTest';

test('uses a fixed rectangular area including its corners', () => {
  const points = [{id:'wp', x:100, y:100}];
  expect(nearestWaypointHit(points, 135, 135)).toBe('wp');
  expect(nearestWaypointHit(points, 137, 100)).toBeNull();
});
test('chooses the closest waypoint where hit areas overlap', () => {
  expect(nearestWaypointHit([{id:'a',x:100,y:100},{id:'b',x:120,y:100}],114,100)).toBe('b');
});
test('ignores invalid projected positions', () => {
  expect(nearestWaypointHit([{id:'a',x:NaN,y:100}],100,100)).toBeNull();
});

test('accepts the entire name pill including its edges', () => {
  const label = [{id:'wp',x:200,y:80,width:160,height:40}];
  expect(nearestWaypointHit(label, 278, 98)).toBe('wp');
  expect(nearestWaypointHit(label, 282, 80)).toBeNull();
});

test('projected marker areas stay clickable after zooming out', () => {
  const THREE = require('three');
  const {waypointHitAreas} = require('./waypointHitTest');
  const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
  camera.position.z = 10;
  camera.updateMatrixWorld();
  const layers = new THREE.Group();
  const marker = new THREE.Object3D();
  marker.userData = {spotId:'dock', waypointHitAnchor:true};
  layers.add(marker);
  const rect = {left:50, top:20, width:400, height:400};
  for (const zoom of [1, 0.1]) {
    camera.zoom = zoom;
    camera.updateProjectionMatrix();
    expect(nearestWaypointHit(waypointHitAreas(layers, camera, rect), 284, 254)).toBe('dock');
  }
  layers.visible = false;
  expect(waypointHitAreas(layers, camera, rect)).toEqual([]);
});

test('projected name pill edges can select the waypoint beyond the marker area', () => {
  const THREE = require('three');
  const {waypointHitAreas} = require('./waypointHitTest');
  const camera = new THREE.OrthographicCamera(-10, 10, 10, -10, 0.1, 100);
  camera.position.z = 10;
  camera.updateMatrixWorld();
  const layers = new THREE.Group();
  const label = new THREE.Sprite();
  label.userData = {spotId:'dock', waypointHitLabel:true};
  label.scale.set(8, 2, 1);
  label.center.set(0.5, -0.5);
  layers.add(label);
  const hits = waypointHitAreas(layers, camera, {left:0, top:0, width:400, height:400});
  expect(nearestWaypointHit(hits, 278, 158)).toBe('dock');
  label.position.x = 100;
  expect(waypointHitAreas(layers, camera, {left:0, top:0, width:400, height:400})).toEqual([]);
});
