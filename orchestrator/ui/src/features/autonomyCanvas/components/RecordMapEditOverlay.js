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

import { useLayoutEffect, useRef, useState } from "react";
import { MdNearMe, MdClose } from "react-icons/md";
import RecordObservationEditor from "./RecordObservationEditor";
import RecordWaypointEditor from "./RecordWaypointEditor";
import { RecordToolButton } from "./RecordAuthoringControls";
import "./RecordMapOverlay.css";

// The anchor comes from the map camera, so an open editor follows pan/zoom.
export default function RecordMapEditOverlay({manager, anchor, disabled, selectionDisabled, goDisabled, onGoToObservation}) {
  const popup = useRef(null), bounds = useRef(null);
  const [viewport, setViewport] = useState({width:0,height:0});
  const [size, setSize] = useState({width:280, height:340});
  const selected = manager?.spots?.find(spot => spot.id === manager.selectedId);
  const editing = !!manager?.route?.routeMode;
  const hasDraft = !!manager?.draft, hasManager = !!manager;
  const visible = !!manager?.draft || (!!selected && !selectionDisabled && !manager?.route?.orderMode && (!!anchor || editing));
  useLayoutEffect(() => {
    const node = popup.current;
    if (!node) return undefined;
    const update = () => { const rect=node.getBoundingClientRect(), height=Math.max(rect.height,node.scrollHeight); setSize(old => old.width===rect.width && old.height===height ? old : {width:rect.width,height}); };
    update();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer = new ResizeObserver(update); observer.observe(node);
    return () => observer.disconnect();
  }, [visible, editing, manager?.selectedId, hasDraft]);
  useLayoutEffect(() => {
    const node=bounds.current;
    if (!node) return undefined;
    const update=()=>setViewport(old=>old.width===node.clientWidth && old.height===node.clientHeight ? old : {width:node.clientWidth,height:node.clientHeight});
    update();
    if (typeof ResizeObserver === "undefined") return undefined;
    const observer=new ResizeObserver(update); observer.observe(node);
    return ()=>observer.disconnect();
  }, [hasManager]);
  if (!manager) return null;
  const width=viewport.width || anchor?.width || 960, height=viewport.height || anchor?.height || 720;
  const availableRight=width-340;
  const candidate=anchor && anchor.x+size.width+24 < availableRight ? anchor.x+20 : (anchor?.x || 32)-size.width-20;
  const safeRight = width > 780 ? width - 340 : width;
  const left=Math.max(12,Math.min(safeRight-size.width-12,candidate));
  const below=(anchor?.y || 100)+18;
  const candidateTop=below+size.height > height-12 ? (anchor?.y || 100)-size.height-18 : below;
  const top=Math.max(editing ? (width <= 410 ? 240 : width <= 780 ? 200 : 156) : 80,Math.min(height-size.height-12,candidateTop));
  const error = manager.error || manager.routeEditor?.error;
  return <>
    <div className="record-overlay-bounds" ref={bounds} aria-hidden="true"/>
    {manager.draft ? <div className="record-waypoint-create-position"><RecordWaypointEditor manager={manager} disabled={disabled}/></div> :
      visible && <div ref={popup} className={`record-waypoint-popover${editing ? " is-editing" : ""}`}
        style={{left,top,maxHeight:Math.max(120,height-top-12)}}>
        {editing ? <RecordObservationEditor manager={manager} disabled={disabled}/> :
          <div className="record-waypoint-go" role="region" aria-label="Selected observation">
            <strong title={selected.label}>{selected.label}</strong>
            <RecordToolButton label="Go to selected observation" disabled={goDisabled} onClick={() => onGoToObservation?.(selected.id)}><MdNearMe size={17}/></RecordToolButton>
            <RecordToolButton label="Close observation selection" onClick={() => manager.select("")}><MdClose size={17}/></RecordToolButton>
          </div>}
      </div>}
    {error && <p className="record-map-edit-error" role="alert">{error}</p>}
  </>;
}
