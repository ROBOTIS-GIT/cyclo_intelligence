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

import { markerPalette, TF_AXIS_LENGTH } from './constants';
import * as THREE from 'three';
import { makeLine } from './sceneObjects';
import { makeTfLabelSprite } from './labelSprites';
import { yawFromPose } from '../../../utils/navigationTf';
export function behaviorNodeColor(category, selected = false) {
  const pal = markerPalette;
  if (selected) return pal.selected;
  switch (category) {
    case "control":
      return pal.control;
    case "decorator":
      return pal.decorator;
    case "action":
    default:
      return pal.action;
  }
}
export function makeBehaviorNodeMarker(node, selected = false) {
  var _a, _b, _c, _d, _e;
  const pose = (node === null) || (node === void 0) ? void 0 : node.pose;
  if (!pose) return null;
  const x = Number(((_a = pose.x) !== null) && (_a !== void 0) ? _a : 0);
  const y = Number(((_b = pose.y) !== null) && (_b !== void 0) ? _b : 0);
  const yaw = Number(((_c = pose.yaw) !== null) && (_c !== void 0) ? _c : 0);
  const color = behaviorNodeColor(node.category, selected);
  const group = new THREE.Group();
  group.position.set(x, y, 0.28);
  group.rotation.z = yaw;
  group.userData.behaviorNodeId = node.id;
  const body = new THREE.Mesh(new THREE.PlaneGeometry(0.46, 0.28), new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: selected ? 0.95 : 0.82,
    side: THREE.DoubleSide
  }));
  body.userData.behaviorNodeId = node.id;
  group.add(body);
  const outline = makeLine([new THREE.Vector3(-0.26, -0.16, 0.02), new THREE.Vector3(0.26, -0.16, 0.02), new THREE.Vector3(0.26, 0.16, 0.02), new THREE.Vector3(-0.26, 0.16, 0.02), new THREE.Vector3(-0.26, -0.16, 0.02)], 0xf3f1ea, 2);
  if (outline) {
    outline.userData.behaviorNodeId = node.id;
    group.add(outline);
  }
  const port = new THREE.Mesh(new THREE.CircleGeometry(0.045, 16), new THREE.MeshBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.92
  }));
  port.position.set(-0.16, 0, 0.03);
  port.userData.behaviorNodeId = node.id;
  group.add(port);
  const label = String(((_e = (_d = node.label) !== null && _d !== void 0 ? _d : node.tag) !== null) && (_e !== void 0) ? _e : node.id);
  if (label) {
    const sprite = makeTfLabelSprite(label);
    sprite.position.set(0, 0.38, 0.06);
    sprite.scale.set(0.56, 0.14, 1);
    sprite.userData.behaviorNodeId = node.id;
    group.add(sprite);
  }
  return group;
}
export function makeTfAxes(pose, label) {
  var _a, _b, _c, _d, _e, _f;
  const group = new THREE.Group();
  group.position.set(Number(((_b = (_a = pose.position) === null || _a === void 0 ? void 0 : _a.x) !== null) && (_b !== void 0) ? _b : 0), Number(((_d = (_c = pose.position) === null || _c === void 0 ? void 0 : _c.y) !== null) && (_d !== void 0) ? _d : 0), Number(((_f = (_e = pose.position) === null || _e === void 0 ? void 0 : _e.z) !== null) && (_f !== void 0) ? _f : 0) + 0.08);
  group.rotation.z = yawFromPose(pose);
  const xAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(TF_AXIS_LENGTH, 0, 0)], 0xef4444);
  const yAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, TF_AXIS_LENGTH, 0)], 0x22c55e);
  const zAxis = makeLine([new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, TF_AXIS_LENGTH)], 0x3b82f6);
  if (xAxis) group.add(xAxis);
  if (yAxis) group.add(yAxis);
  if (zAxis) group.add(zAxis);
  const sprite = makeTfLabelSprite(label);
  sprite.position.set(0, -TF_AXIS_LENGTH * 0.85, 0.012);
  group.add(sprite);
  return group;
}
export function makeFootprintMarker(footprint, framePose) {
  var _a, _b, _c, _d, _e, _f, _g, _h;
  const sourcePoints = ((_b = (_a = footprint.polygon) === null || _a === void 0 ? void 0 : _a.points) !== null) && (_b !== void 0) ? _b : [];
  const polygonPoints = sourcePoints.map(point => {
    var _a, _b, _c;
    return {
      x: Number(((_a = point.x) !== null) && (_a !== void 0) ? _a : 0),
      y: Number(((_b = point.y) !== null) && (_b !== void 0) ? _b : 0),
      z: Number(((_c = point.z) !== null) && (_c !== void 0) ? _c : 0)
    };
  }).filter(point => Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z));
  if (polygonPoints.length < 3) return null;
  const group = new THREE.Group();
  const yaw = framePose ? yawFromPose(framePose) : 0;
  const cos = Math.cos(yaw);
  const sin = Math.sin(yaw);
  const frameX = Number(((_d = (_c = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _c === void 0 ? void 0 : _c.x) !== null) && (_d !== void 0) ? _d : 0);
  const frameY = Number(((_f = (_e = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _e === void 0 ? void 0 : _e.y) !== null) && (_f !== void 0) ? _f : 0);
  const frameZ = Number(((_h = (_g = framePose === null || framePose === void 0 ? void 0 : framePose.position) === null || _g === void 0 ? void 0 : _g.z) !== null) && (_h !== void 0) ? _h : 0);
  const transformPoint = point => new THREE.Vector3((frameX + cos * point.x) - (sin * point.y), frameY + (sin * point.x) + (cos * point.y), frameZ + point.z + 0.18);
  const points = polygonPoints.map(transformPoint);
  points.push(points[0].clone());
  const line = makeLine(points, 0x38bdf8, 3);
  if (line) group.add(line);
  const shape = new THREE.Shape();
  polygonPoints.forEach((point, index) => {
    if (index === 0) shape.moveTo(point.x, point.y);else shape.lineTo(point.x, point.y);
  });
  shape.closePath();
  const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape), new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.18,
    depthWrite: false,
    side: THREE.DoubleSide
  }));
  fill.position.set(frameX, frameY, frameZ + 0.17);
  fill.rotation.z = yaw;
  group.add(fill);
  return group;
}
