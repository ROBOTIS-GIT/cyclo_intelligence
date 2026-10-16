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
import { imagePoint } from "../lib/maskImageCoordinates";

export default function useMaskViewport({ width, height, onError }) {
  const [view, setView] = useState({ zoom: 1, x: 0, y: 0 });
  const [expanded, setExpanded] = useState(false);
  const editor = useRef(null);
  const canvasRef = useRef(null);
  const gesture = useRef(null);
  const suppressClick = useRef(false);
  useEffect(() => {
    const update = () => setExpanded(document.fullscreenElement === editor.current);
    document.addEventListener("fullscreenchange", update);
    return () => document.removeEventListener("fullscreenchange", update);
  }, []);

  function clampView(next) {
    return { ...next, x: Math.max(0, Math.min(width - width / next.zoom, next.x)),
      y: Math.max(0, Math.min(height - height / next.zoom, next.y)) };
  }
  function zoomBy(factor) {
    setView(previous => {
      const zoom = Math.max(1, Math.min(8, previous.zoom * factor));
      return clampView({ zoom, x: previous.x + width / previous.zoom / 2 - width / zoom / 2,
        y: previous.y + height / previous.zoom / 2 - height / zoom / 2 });
    });
  }
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    function wheel(event) {
      if (!event.ctrlKey) return;
      event.preventDefault();
      event.stopPropagation();
      if (!event.deltaY) return;
      const rect = canvas.getBoundingClientRect();
      setView(previous => {
        const point = imagePoint(event.clientX, event.clientY, rect, width, height, previous);
        if (!point) return previous;
        const zoom = Math.max(1, Math.min(8, previous.zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
        // Keep the image point under the cursor fixed while zooming.
        return {zoom,
          x: Math.max(0, Math.min(width - width / zoom, point.x - (point.x - previous.x) * previous.zoom / zoom)),
          y: Math.max(0, Math.min(height - height / zoom, point.y - (point.y - previous.y) * previous.zoom / zoom))};
      });
    }
    // React's delegated wheel listener can be passive; use a local listener
    // so Ctrl+wheel does not also zoom the browser page.
    canvas.addEventListener("wheel", wheel, {passive:false});
    return () => canvas.removeEventListener("wheel", wheel);
  }, [width, height]);
  async function toggleExpanded() {
    try {
      if (document.fullscreenElement === editor.current) await document.exitFullscreen();
      else await editor.current.requestFullscreen();
    } catch (_) { onError("Expanded view is unavailable in this browser."); }
  }
  function pointerDown(event) {
    if (event.button !== 0) return;
    suppressClick.current = false;
    gesture.current = { id: event.pointerId, x: event.clientX, y: event.clientY, view };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  }
  function pointerMove(event) {
    const start = gesture.current;
    if (!start || start.id !== event.pointerId) return;
    const dx = event.clientX - start.x, dy = event.clientY - start.y;
    if (Math.hypot(dx, dy) < 4 && !suppressClick.current) return;
    suppressClick.current = true;
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = Math.min(rect.width / width, rect.height / height) * start.view.zoom;
    if (scale > 0) setView(clampView({ ...start.view, x: start.view.x - dx / scale, y: start.view.y - dy / scale }));
  }
  function pointerEnd(event) {
    if (gesture.current?.id !== event.pointerId) return;
    gesture.current = null;
    if (event.type === "pointercancel") suppressClick.current = true;
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }

  function pointForClick(event) {
    if (suppressClick.current) { suppressClick.current = false; return null; }
    return imagePoint(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), width, height, view);
  }
  return { view, setView, expanded, editor, canvasRef, zoomBy, toggleExpanded,
    pointerDown, pointerMove, pointerEnd, pointForClick,
    losePointer: () => { gesture.current = null; } };
}
