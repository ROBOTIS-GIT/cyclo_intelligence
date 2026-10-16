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

import { MdZoomOut, MdZoomIn, MdFitScreen, MdFullscreen, MdFullscreenExit, MdUndo, MdRestartAlt, MdAdd, MdSave } from "react-icons/md";
import { forwardRef, useId, useImperativeHandle } from "react";
import ObjectLabelPicker from "./ObjectLabelPicker";
import useMaskAnnotationSession from "../hooks/useMaskAnnotationSession";
import useMaskViewport from "../hooks/useMaskViewport";
export { imagePoint } from "../lib/maskImageCoordinates";

const Sam2Annotation = forwardRef(function Sam2Annotation({ episode, frame, missionId = "", onMissionChange, onFrameLoaded, onBusyChange, onMaskSaved, disabled = false, standaloneSave = false, reviewMode = false, onDirtyChange, renderLayout }, ref) {
  const clipId = useId();
  const mask = useMaskAnnotationSession({ episode, frame, missionId, onMissionChange,
    onBusyChange, onMaskSaved, disabled, reviewMode, onDirtyChange });
  const { points, setPoints, mode, setMode, preview, overlayLoaded, setOverlayLoaded,
    imageLoaded, setImageLoaded, working, busy, error, setError, receipt,
    pending, regions, editing, label, setLabel, savedOverlay, dirty, hasCurrentChanges,
    overlayUrl, currentRegions, visibleRegions, regionOverlay, markOverlay,
    canSaveEmpty, reviewReady, hasDrafts, predict, invalidate, selectRegion, removeInstance, save } = mask;
  const width = frame.image?.width || 640, height = frame.image?.height || 480;
  const imageUrl = `/api/collection/episodes/${episode.id}/images/${frame.frame_id}`;
  const viewport = useMaskViewport({ width, height, onError: setError });
  const { view, setView, expanded, editor, canvasRef, zoomBy, toggleExpanded,
    pointerDown, pointerMove, pointerEnd } = viewport;
  function click(event) {
    const point = viewport.pointForClick(event);
    if (point && !busy && imageLoaded && points.length < 64) predict([...points, {...point, label: mode}]);
  }
  useImperativeHandle(ref, () => ({ save }));
  const saveButton = standaloneSave && <button type="button" className="sam2-save-action" aria-label={reviewMode ? "Save review" : "Save"} title={reviewMode ? "Save review" : "Save"} disabled={busy || (!dirty && !hasDrafts && !canSaveEmpty) || (reviewMode && !reviewReady) || (hasCurrentChanges && (!label.trim() || !imageLoaded || (!preview && !savedOverlay) || ((preview || savedOverlay) && !overlayLoaded)))} onClick={() => save(true)}><MdSave size={18}/></button>;
  const instanceControls = <div className="sam2-instance-controls">
    <div className="sam2-instance-list" role="list" aria-label="Frame instances">
      {currentRegions.map((item,index)=><div role="listitem" key={item.object_id}>
        <button type="button" disabled={busy} aria-pressed={item.object_id===editing.object_id} onClick={()=>selectRegion(item.object_id)}>{item.object_label} #{item.instance_number || index+1}{item.review_status === "pending" ? " · Unsaved" : ""}</button>
        <button type="button" disabled={busy} aria-label={`Remove instance ${item.object_label} ${index+1}`} onClick={()=>removeInstance(item.object_id)}>×</button>
      </div>)}
    </div>
  </div>;
  const objectPicker = <ObjectLabelPicker compact episodeId={episode.id} regions={[...regions,...pending.map(r=>({...r,review_status:"approved"}))]} value={label} onChange={setLabel} disabled={busy} />;
  const objectStatus = <>
    {working && <p role="status">Preparing SAM and creating mask…</p>}
    {receipt && <p role="status" className="operate-sidebar-note">{receipt}</p>}
    {error && <p role="alert" className="operate-qwen-error">{error}</p>}
  </>;
  const imageEditor = (
    <div ref={editor} className="sam2-image-editor">
    <div className="sam2-image-tools">
    <div className="sam2-toolbar">
      <button type="button" aria-label="Include point" title="Include point" aria-pressed={mode === 1} disabled={busy} onClick={() => setMode(1)}><span aria-hidden="true" style={{color:"#16a34a"}}>●</span></button>
      <button type="button" aria-label="Exclude point" title="Exclude point" aria-pressed={mode === 0} disabled={busy} onClick={() => setMode(0)}><span aria-hidden="true" style={{color:"#dc2626"}}>●</span></button>
      <button type="button" aria-label="Undo" title="Undo point" disabled={busy || !points.length} onClick={() => predict(points.slice(0, -1))}><MdUndo size={18}/></button>
      <button type="button" aria-label="Reset" title="Reset points" disabled={busy || !points.length} onClick={() => {setPoints([]);invalidate();}}><MdRestartAlt size={18}/></button>
      <span className="sam2-tool-divider" aria-hidden="true" />
      <div className="sam2-object-actions" role="group" aria-label="Object actions">
        {objectPicker}
        <button type="button" aria-label="New instance" title="New instance" disabled={busy} onClick={() => selectRegion("")}><MdAdd size={18}/></button>
        {saveButton}
      </div>
      {error && points.some(p => p.label) && <button type="button" disabled={busy} onClick={() => predict(points)}>Retry mask</button>}
    </div>
    <div className="sam2-toolbar sam2-view-controls" role="group" aria-label="Image view controls">
      <button type="button" aria-label="Zoom out" title="Zoom out" onClick={() => zoomBy(1 / 1.5)} disabled={view.zoom <= 1}><MdZoomOut size={18}/></button>
      <output aria-label="Image zoom">{Math.round(view.zoom * 100)}%</output>
      <button type="button" aria-label="Zoom in" title="Zoom in" onClick={() => zoomBy(1.5)} disabled={view.zoom >= 8}><MdZoomIn size={18}/></button>
      <button type="button" aria-label="Fit image" title="Fit image" onClick={() => setView({zoom: 1, x: 0, y: 0})}><MdFitScreen size={18}/></button>
      <button type="button" aria-label={expanded ? "Exit expanded view" : "Expand view"} title={expanded ? "Exit expanded view" : "Expand view"} onClick={toggleExpanded}>{expanded ? <MdFullscreenExit size={18}/> : <MdFullscreen size={18}/>}</button>
    </div>
    </div>
    <svg ref={canvasRef} className="sam2-canvas" role="img" aria-label="Saved frame for SAM2 annotation" viewBox={`${view.x} ${view.y} ${width / view.zoom} ${height / view.zoom}`} preserveAspectRatio="xMidYMid meet" onClick={click} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerEnd} onPointerCancel={pointerEnd} onLostPointerCapture={() => {viewport.losePointer();}} style={{cursor:busy ? "wait" : "crosshair"}}>
      <defs><clipPath id={clipId}><rect x={view.x} y={view.y} width={width / view.zoom} height={height / view.zoom} /></clipPath></defs>
      <g clipPath={`url(#${clipId})`}>
      <image href={imageUrl} width={width} height={height} onLoad={() => {setImageLoaded(true);onFrameLoaded?.(frame.frame_id);}} onError={() => {setImageLoaded(false);setError("Saved image could not be loaded.");}} />
      {visibleRegions.map((item,index)=><g key={item.object_id} pointerEvents="none">
        <image href={regionOverlay(item)} width={width} height={height} style={{filter:`hue-rotate(${(index+1)*75}deg)`}}
          onLoad={() => markOverlay(regionOverlay(item), true)} onError={() => {markOverlay(regionOverlay(item), false);setError("Mask overlay could not be loaded.");}}/>
        <text x={item.bbox_xywh?.[0] || 4} y={Math.max(14,item.bbox_xywh?.[1] || 14)} fontSize={Math.max(width,height)/35} fill="white" stroke="black" strokeWidth=".5">{item.object_label}</text>
      </g>)}
      {overlayUrl && <image key={preview?.preview_id || overlayUrl} href={overlayUrl} width={width} height={height} opacity={1} pointerEvents="none" onLoad={() => {setOverlayLoaded(true);markOverlay(overlayUrl, true);}} onError={() => {setOverlayLoaded(false);markOverlay(overlayUrl, false);setError("Mask overlay could not be loaded.");}} />}
      {overlayUrl && label && <text pointerEvents="none" x={preview?.bbox_xywh?.[0] || editing.bbox_xywh?.[0] || 4} y={Math.max(14,preview?.bbox_xywh?.[1] || editing.bbox_xywh?.[1] || 14)} fontSize={Math.max(width,height)/35} fill="white" stroke="black" strokeWidth=".5">{label}</text>}
      {points.map((point, i) => <circle key={i} cx={point.x} cy={point.y} r={Math.max(width,height)/110/view.zoom} fill={point.label ? "#22c55e" : "#ef4444"} stroke="white" strokeWidth={Math.max(width,height)/400/view.zoom} pointerEvents="none" />)}
      </g>
    </svg>
    <div className="sam2-image-status">{objectStatus}</div>
    </div>
  );
  return <section className={`sam2-annotation${renderLayout ? " sam2-layout-host" : ""}`} aria-label="SAM2 segmentation">
    {renderLayout ? renderLayout({ imageEditor, instanceControls })
      : <>{imageEditor}{instanceControls}</>}
  </section>;
});

export default Sam2Annotation;
