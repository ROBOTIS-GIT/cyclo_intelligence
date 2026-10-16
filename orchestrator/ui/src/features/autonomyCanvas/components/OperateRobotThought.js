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

import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import useRobotThoughtResult from "./useRobotThoughtResult";
import AnimatedReason from "./AnimatedReason";
import usePaintedVisibility from "./usePaintedVisibility";
import LiveHeadCamera from "./LiveHeadCamera";
import OperateThoughtAction from "./OperateThoughtAction";
import { thoughtRunning, thoughtPlacement } from "./robotThought";
import "./OperateExperience.css";

const FADE_MS = 700;
const noSubscribe = () => () => {};
const noAnchor = () => null;

export default function OperateRobotThought({ observation = {}, anchor: directAnchor, anchorStore }) {
  const storedAnchor = useSyncExternalStore(anchorStore?.subscribe || noSubscribe, anchorStore?.getSnapshot || noAnchor, noAnchor);
  const anchor = directAnchor ?? storedAnchor;
  const result = useRobotThoughtResult(observation);
  const running = thoughtRunning(observation);
  const error = observation.error || "";
  const reason = result?.decision?.reason || result?.text || "";
  const lastThought = useRef({ reason: "", result: null });
  const ref = useRef(null);
  const [cameraMounted, setCameraMounted] = useState(running);
  const [cameraReady, setCameraReady] = useState(false);
  const [cameraTopic, setCameraTopic] = useState(observation.result?.camera_topic || undefined);
  useEffect(() => {
    if (observation.result?.camera_topic) setCameraTopic(observation.result.camera_topic);
  }, [observation.result?.camera_topic]);
  const [aspect, setAspect] = useState(16 / 9);
  const [naturalHeight, setNaturalHeight] = useState(480);
  const placementSide = useRef(null);
  useEffect(() => {
    if (running) {
      setCameraMounted(true);
      return;
    }
    // Keep the stream alive just long enough to fade its last visible image out.
    const timer = setTimeout(() => { setCameraMounted(false); setCameraReady(false); }, FADE_MS);
    return () => clearTimeout(timer);
  }, [running]);
  useEffect(() => { if (running) lastThought.current = { reason, result }; }, [running, reason, result]);
  useLayoutEffect(() => {
    if (!ref.current || typeof ResizeObserver === "undefined") return;
    const measure = () => {
      const content = ref.current.querySelector(".operate-thought-content");
      const style = getComputedStyle(ref.current);
      const height = (content?.scrollHeight || 0) + parseFloat(style.paddingTop || 0) + parseFloat(style.paddingBottom || 0);
      if (height > 0) setNaturalHeight(previous => Math.abs(previous - height) < 1 ? previous : height);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(ref.current);
    observer.observe(ref.current.querySelector(".operate-thought-content"));
    measure();
    return () => observer.disconnect();
  }, [reason, aspect, cameraMounted]);
  const placement = thoughtPlacement(anchor, anchor?.width <= 700 ? 390 : 480, naturalHeight, placementSide.current);
  useLayoutEffect(() => { placementSide.current = placement?.side; }, [placement?.side]);
  const visible = running || !!error;
  const paintedVisible = usePaintedVisibility(visible);
  const paintedCamera = usePaintedVisibility(running && cameraReady);
  const shownReason = error || (running ? reason : lastThought.current.reason);
  return <div className={`operate-robot-thought${paintedVisible ? " is-visible" : ""}`}
    aria-label="Robot thought and camera" aria-hidden={!visible}
    style={{ "--thought-fade-duration": `${FADE_MS}ms` }}>
    {placement && <svg className="operate-thought-connector" aria-hidden="true">
      {placement.circles.map((circle, index) => <circle key={index} cx="0" cy="0" r={circle.r}
        style={{ transform: `translate(${circle.x}px, ${circle.y}px)` }}/>) }
    </svg>}
    <article ref={ref} className={`operate-thought-bubble${error ? " tone-danger" : ""}${placement ? "" : " is-docked"}`}
      data-placement={placement?.side || "docked"}
      style={placement ? { left: placement.x, top: placement.y, width: placement.width,
        maxWidth: placement.width, maxHeight: placement.height,
        "--thought-pad-x": `${placement.width * .15}px`, "--thought-pad-top": `${placement.width * .15}px`,
        "--thought-pad-bottom": `${placement.width * .20}px` } : undefined}>
      <svg className="operate-thought-cloud" viewBox="0 0 400 360" preserveAspectRatio="none" aria-hidden="true">
        <path d="M 66 28 C 80 7 117 4 141 20 C 166 1 210 1 234 20 C 260 5 300 9 318 32 C 350 22 383 44 379 76 C 402 91 407 126 387 147 C 404 171 403 207 384 224 C 397 253 377 282 348 284 C 338 313 299 326 272 312 C 249 339 206 345 176 326 C 150 345 108 335 92 312 C 59 321 27 301 27 272 C 3 258 0 225 17 203 C 0 181 0 151 17 130 C 0 102 17 75 43 69 C 36 48 46 33 66 28 Z"/>
      </svg>
      <div className="operate-thought-content">
      {cameraMounted && <div className={`operate-thought-camera${paintedCamera ? " is-visible" : ""}`}
        style={{ aspectRatio: aspect }} aria-hidden={!running}>
        <LiveHeadCamera topic={cameraTopic} onAspectRatio={setAspect}
          onReady={setCameraReady} showStatus={false}/>
      </div>}
      <OperateThoughtAction observation={observation}/>
      <AnimatedReason key={observation.session_id || "current"} text={shownReason} result={error ? null : running ? result : lastThought.current.result}
        immediate={!!error} active={running}/>
      </div>
    </article>
  </div>;
}
