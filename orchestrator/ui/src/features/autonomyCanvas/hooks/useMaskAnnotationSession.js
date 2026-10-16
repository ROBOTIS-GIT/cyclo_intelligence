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

import { useCallback, useEffect, useRef, useState } from "react";
import request from "./maskAnnotationApi";

const objectId = () => `object-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

// One frame owns its previews, staged instances, revisions and save/review transaction.
export default function useMaskAnnotationSession({ episode, frame, missionId,
  onMissionChange, onBusyChange, onMaskSaved, disabled, reviewMode, onDirtyChange }) {
  const [points, setPoints] = useState([]);
  const [mode, setMode] = useState(1);
  const [preview, setPreview] = useState(null);
  const [overlayLoaded, setOverlayLoaded] = useState(false);
  const [loadedOverlays, setLoadedOverlays] = useState(() => new Set());
  const [imageLoaded, setImageLoaded] = useState(false);
  const [working, setBusy] = useState(false);
  const busy = working || disabled;
  const [error, setError] = useState("");
  const [receipt, setReceipt] = useState("");
  const [pending, setPending] = useState([]);
  const [removed, setRemoved] = useState([]);
  const [regions, setRegions] = useState([]);
  const [regionsLoaded, setRegionsLoaded] = useState(false);
  const [editing, setEditing] = useState({ object_id: objectId(), revision: 0 });
  const [label, setLabel] = useState("");
  const [savedOverlay, setSavedOverlay] = useState(null);
  const session = useRef({ alive: true, controller: null, busy: false });

  const invalidate = useCallback(() => {
    setPreview(null); setSavedOverlay(null); setOverlayLoaded(false); setReceipt("");
  }, []);
  const openInstance = useCallback(region => {
    invalidate();
    setEditing(region || {object_id:objectId(),revision:0});
    setPoints(region?.points || []); setLabel(region?.object_label || "");
    setPreview(region?.preview || null);
    if (region) onMissionChange?.(region.mission_id || "");
    setSavedOverlay(region ? region.overlay || `/api/collection/episodes/${episode.id}/regions/${region.object_id}/overlay?revision=${region.revision}` : null);
  }, [episode.id, onMissionChange, invalidate]);

  useEffect(() => {
    const current = session.current;
    current.alive = true;
    const controller = new AbortController();
    request(`episodes/${episode.id}/regions`, undefined, controller.signal)
      .then(data => { if (current.alive) {
        setRegions(data.regions || []);
        setRegionsLoaded(true);
        if (reviewMode) {
          const first = (data.regions || []).find(r => r.frame_id === frame.frame_id && r.review_status === "draft");
          if (first) openInstance(first);
        }
      } })
      .catch(err => { if (current.alive && err.name !== "AbortError") setError(err.message); });
    return () => { current.alive = false; current.controller?.abort(); controller.abort(); };
  }, [episode.id, frame.frame_id, reviewMode, openInstance]);

  useEffect(() => {
    if (!reviewMode) return undefined;
    const controller = new AbortController();
    const timer = setInterval(() => {
      request(`episodes/${episode.id}/regions`, undefined, controller.signal)
        .then(data => setRegions(data.regions || [])).catch(() => {});
    }, 3000);
    return () => { clearInterval(timer); controller.abort(); };
  }, [reviewMode, episode.id]);

  async function operation(fn) {
    const current = session.current;
    if (current.busy) return false;
    current.busy = true; setBusy(true); onBusyChange?.(true); setError(""); setReceipt("");
    current.controller = new AbortController();
    try { await fn(current.controller.signal); return current.alive; }
    catch (err) { if (current.alive && err.name !== "AbortError") setError(err.message); return false; }
    finally { current.busy = false; if (current.alive) {setBusy(false);onBusyChange?.(false);} }
  }
  async function predict(nextPoints) {
    if (session.current.busy) return;
    setPoints(nextPoints); invalidate();
    if (!nextPoints.some(p => p.label === 1)) return;
    await operation(async signal => {
      const data = await request(`episodes/${episode.id}/segmentation/preview`, {frame_id: frame.frame_id, points: nextPoints}, signal);
      if (session.current.alive && data.frame_id === frame.frame_id) setPreview(data);
    });
  }
  const hasCurrentChanges = !!preview || (!editing.revision && (points.length > 0 || !!label.trim())) ||
    (!!editing.revision && label.trim() !== editing.object_label);
  const dirty = pending.length > 0 || removed.length > 0 || hasCurrentChanges;
  // Automatic regeneration replaces object IDs. Do not retain the old selected overlay.
  useEffect(() => {
    if (!reviewMode || !regionsLoaded || dirty || working || !editing.revision) return;
    const latest = regions.find(item => item.object_id === editing.object_id);
    if (!latest || latest.revision !== editing.revision) {
      openInstance(latest || regions.find(item => item.frame_id === frame.frame_id && item.review_status === "draft") || null);
    }
  }, [regions, regionsLoaded, reviewMode, dirty, working, editing.object_id, editing.revision, frame.frame_id, openInstance]);
  useEffect(() => { onDirtyChange?.(dirty); }, [dirty, onDirtyChange]);
  useEffect(() => () => onDirtyChange?.(false), [onDirtyChange]);
  useEffect(() => {
    if (!dirty) return undefined;
    const warn = event => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload",warn);
    return () => window.removeEventListener("beforeunload",warn);
  }, [dirty]);

  function collectPending() {
    if (!hasCurrentChanges) return pending;
    if (!label.trim() || !imageLoaded || (!preview && !savedOverlay) || ((preview || savedOverlay) && !overlayLoaded)) {
      setError("Finish this object's mask and name before adding another object or saving.");
      return null;
    }
    const item = {...editing, object_label:label.trim(), points, preview,
      overlay:preview ? `data:image/png;base64,${preview.overlay_base64}` : savedOverlay,
      bbox_xywh:preview?.bbox_xywh || editing.bbox_xywh,
      mission_id:missionId || null, review_status:"pending"};
    return [...pending.filter(r=>r.object_id !== editing.object_id), item];
  }
  function selectRegion(id) {
    if (busy || id === editing.object_id) return;
    const staged = collectPending();
    if (!staged) return;
    if (!id && staged.length >= 64) { setError("Save this group before adding more than 64 instances."); return; }
    setPending(staged);
    openInstance(staged.find(r=>r.object_id === id) || regions.find(r=>r.object_id === id));
  }
  function removeInstance(id) {
    if (busy) return;
    setPending(previous=>previous.filter(r=>r.object_id !== id));
    const saved = regions.find(r=>r.object_id === id);
    if (saved) setRemoved(previous=>[...previous.filter(r=>r.object_id !== id),
      {object_id:id,expected_revision:saved.revision}]);
    if (editing.object_id === id) openInstance(null);
  }
  async function save(advance = false) {
    let staged = collectPending();
    if (!staged) return false;
    if (reviewMode) {
      if (!reviewReady) { setError("Wait for the frame and all mask overlays to load before saving the review."); return false; }
      // Keep edits and exclusions, then approve every untouched draft in this frame.
      staged = [...staged, ...currentRegions.filter(item => item.review_status === "draft" &&
        !staged.some(edit => edit.object_id === item.object_id))];
    }
    if (!staged.length && !removed.length && !canSaveEmpty) return true;
    const ok = await operation(async signal => {
      const result = await request(`episodes/${episode.id}/regions/batch`, {
        frame_id:frame.frame_id,
        ...(canSaveEmpty ? {review_empty:true} : {}),
        items:staged.map(item=>({preview_id:item.preview?.preview_id || null,
          frame_id:frame.frame_id,object_id:item.object_id,expected_revision:item.revision || 0,
          object_label:item.object_label,relevance:item.relevance || "target",
          mission_id:item.mission_id,annotator:"collect-ui",reviewed:true})), removed,
      },signal);
      if (!session.current.alive) return;
      const saved = result.regions;
      setRegions(previous=>[...previous.filter(r=>!saved.some(item=>item.object_id===r.object_id)),...saved]);
      setPending([]); setRemoved([]);
      openInstance(saved.find(r=>r.object_id===editing.object_id && r.review_status !== "rejected") || null);
      setReceipt(reviewMode ? "Frame review saved." : `Masks saved · ${staged.length} instances`);
    });
    if (ok) {
      try { await onMaskSaved?.({advance}); }
      catch (err) { setError(`Saved, but the frame list could not refresh: ${err.message}`); }
    }
    return ok;
  }
  const overlayUrl = preview ? `data:image/png;base64,${preview.overlay_base64}` : savedOverlay;
  const currentRegions = [...regions.filter(r => r.frame_id === frame.frame_id && (reviewMode ? r.review_status !== "rejected" : r.review_status === "approved") && !pending.some(p=>p.object_id===r.object_id)),...pending]
    .filter(r=>!removed.some(item=>item.object_id===r.object_id));
  const visibleRegions = currentRegions.filter(r=>r.object_id!==editing.object_id);
  const regionOverlay = item => item.overlay || `/api/collection/episodes/${episode.id}/regions/${item.object_id}/overlay?revision=${item.revision}`;
  const markOverlay = (url, loaded) => setLoadedOverlays(previous => {
    const next = new Set(previous);
    if (loaded) next.add(url); else next.delete(url);
    return next;
  });
  const canSaveEmpty = reviewMode && currentRegions.length === 0;
  const reviewReady = regionsLoaded && imageLoaded && currentRegions.every(item => {
    const url = item.object_id === editing.object_id ? overlayUrl : regionOverlay(item);
    return !!url && loadedOverlays.has(url);
  });
  const hasDrafts = reviewMode && currentRegions.some(item => item.review_status === "draft");
  return { points, setPoints, mode, setMode, preview, overlayLoaded, setOverlayLoaded,
    imageLoaded, setImageLoaded, working, busy, error, setError, receipt,
    pending, regions, editing, label, setLabel, savedOverlay, dirty, hasCurrentChanges,
    overlayUrl, currentRegions, visibleRegions, regionOverlay, markOverlay,
    canSaveEmpty, reviewReady, hasDrafts, predict, invalidate, selectRegion, removeInstance, save };
}
