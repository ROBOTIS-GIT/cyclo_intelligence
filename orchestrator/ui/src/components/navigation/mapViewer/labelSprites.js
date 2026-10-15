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

import { markerPalette, WAYPOINT_LABEL_SELECTED_BG_ALPHA, WAYPOINT_LABEL_BG_ALPHA, TF_AXIS_LENGTH } from './constants';
import * as THREE from 'three';
export function rgbaArrayFromHex(color, alpha = 150) {
  const value = hexColorString(color, "#6D1F2A");
  return [Number.parseInt(value.slice(1, 3), 16), Number.parseInt(value.slice(3, 5), 16), Number.parseInt(value.slice(5, 7), 16), alpha];
}
export function makeTextSprite(text, {
  width = 256,
  height = 64,
  fontSize = 22,
  backgroundAlpha = 0.92
} = {}) {
  const pal = markerPalette;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const r = height / 2; // pill radius
    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    ctx.roundRect(2, 2, width - 4, height - 4, r);
    ctx.fillStyle = pal.labelBg.replace("0.92", String(backgroundAlpha));
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = pal.labelBorder;
    ctx.stroke();
    ctx.fillStyle = pal.labelText;
    ctx.font = `600 ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, width / 2, (height / 2) + 1);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true
  }));
}
export function hexColorString(color, fallback = "#C96442") {
  const match = String(color || "").trim().match(/^#?([0-9A-Fa-f]{6})$/);
  return match ? `#${match[1].toUpperCase()}` : fallback;
}
export function pastelizeHexColor(hex, mix = 0.45) {
  const num = parseInt(hex.slice(1), 16);
  const lift = channel => Math.round(channel + ((255 - channel) * mix));
  const r = lift((num >> 16) & 255);
  const g = lift((num >> 8) & 255);
  const b = lift(num & 255);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0").toUpperCase()}`;
}
export function makeAnnotationLabelSprite(text, color) {
  const label = String(text || "Area");
  const fontSize = 21;
  const font = `700 ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = font;
  const textW = Math.ceil(measure.measureText(label).width);
  const height = 44;
  const padX = 10;
  const width = Math.max(72, textW + (padX * 2));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#000000";
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, width / 2, (height / 2) + 1);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true
  }));
  return {
    sprite,
    aspect: width / height
  };
}
export function makeWaypointLabelSprite(text, {
  fontSize = 56,
  selected = false
} = {}) {
  const pal = markerPalette;
  const font = `${selected ? 700 : 600} ${fontSize}px "Hanken Grotesk", "Pretendard Variable", sans-serif`;
  const measure = document.createElement("canvas").getContext("2d");
  measure.font = font;
  const textW = Math.ceil(measure.measureText(String(text)).width);
  const pad = Math.round(fontSize * 0.7);
  const pillH = Math.round(fontSize * 1.9);
  const tailH = Math.round(fontSize * 0.38);
  const width = Math.max(textW + (pad * 2), pillH);
  const height = pillH + tailH;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (ctx) {
    const bgAlpha = selected ? WAYPOINT_LABEL_SELECTED_BG_ALPHA : WAYPOINT_LABEL_BG_ALPHA;
    const bg = (selected ? pal.labelSelBg : pal.labelBg).replace(/,\s*[\d.]+\s*\)$/, `,${bgAlpha})`);
    const border = selected ? pal.labelSelBorder : pal.labelBorder;
    ctx.clearRect(0, 0, width, height);
    ctx.beginPath();
    ctx.roundRect(3, 3, width - 6, pillH - 6, pillH / 2);
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = border;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo((width / 2) - tailH, pillH - 4);
    ctx.lineTo(width / 2, (pillH + tailH) - 4);
    ctx.lineTo((width / 2) + tailH, pillH - 4);
    ctx.closePath();
    ctx.fillStyle = bg;
    ctx.fill();
    ctx.strokeStyle = border;
    ctx.stroke();
    ctx.fillStyle = bg;
    ctx.fillRect((width / 2 - tailH) + 2, pillH - 7, (tailH * 2) - 4, 6);
    ctx.fillStyle = selected ? pal.labelSelText : pal.labelText;
    ctx.font = font;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(String(text), width / 2, (pillH / 2) + 2);
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false
  }));
  return {
    sprite,
    aspect: width / height
  };
}
export function makeTfLabelSprite(text) {
  const sprite = makeTextSprite(text);
  sprite.scale.set(TF_AXIS_LENGTH * 1.8, TF_AXIS_LENGTH * 0.45, 1);
  return sprite;
}
