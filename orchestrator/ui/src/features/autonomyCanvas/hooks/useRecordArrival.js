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

import { useEffect, useRef, useState } from "react";

export function observationPose(pose) {
  const p = pose?.position || pose?.pose?.position;
  const q = pose?.orientation || pose?.pose?.orientation;
  if (!p || !q || ![p.x, p.y].every(Number.isFinite)) return null;
  return { x: p.x, y: p.y, yaw: Math.atan2(2 * ((q.w || 0) * (q.z || 0) + (q.x || 0) * (q.y || 0)),
    1 - 2 * ((q.y || 0) ** 2 + (q.z || 0) ** 2)) };
}
const angleDistance = (a, b) => Math.abs(Math.atan2(Math.sin(a - b), Math.cos(a - b)));
export function nearObservation(pose, spot) {
  return !!pose && !!spot?.pose && Math.hypot(pose.x - spot.pose.x, pose.y - spot.pose.y) <= 0.25
    && angleDistance(pose.yaw, spot.pose.yaw) <= 0.25;
}
// Reaching a marker is not enough: wait until both position and heading have settled.
export default function useRecordArrival({ enabled, localized, moving, pose, spot }) {
  const [settled, setSettled] = useState(false);
  const anchor = useRef(null), timer = useRef(null);
  useEffect(() => {
    const current = observationPose(pose);
    const eligible = enabled && localized && !moving && nearObservation(current, spot);
    const stable = eligible && anchor.current?.spotId === spot.id
      && Math.hypot(current.x - anchor.current.x, current.y - anchor.current.y) <= 0.03
      && angleDistance(current.yaw, anchor.current.yaw) <= 0.06;
    if (!stable) {
      clearTimeout(timer.current); timer.current = null; setSettled(false);
      anchor.current = eligible ? { ...current, spotId: spot.id } : null;
      if (eligible) timer.current = setTimeout(() => { timer.current = null; setSettled(true); }, 750);
    }
  }, [enabled, localized, moving, pose, spot]);
  useEffect(() => () => clearTimeout(timer.current), []);
  return settled && enabled && localized && !moving && nearObservation(observationPose(pose), spot);
}
