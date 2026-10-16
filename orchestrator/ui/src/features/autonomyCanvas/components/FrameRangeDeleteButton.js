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
import { createPortal } from "react-dom";
import { MdClose, MdDeleteOutline } from "react-icons/md";
import "./FrameRangeDeleteButton.css";

export function frameIndexAt(clientX, rect, count) {
  if (!count || !rect.width) return 0;
  return Math.max(0, Math.min(count - 1, Math.floor((clientX - rect.left) / rect.width * count)));
}

// Uses the saved episode order, never the editor's sampled/review-only subset.
export default function FrameRangeDeleteButton({ episode, disabled = false, onDelete }) {
  const [open, setOpen] = useState(false);
  const [anchor, setAnchor] = useState(null);
  const [end, setEnd] = useState(null);
  const [confirming, setConfirming] = useState(false);
  const [draggingRange, setDraggingRange] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef(null), track = useRef(null);
  const dragging = useRef(null), pending = useRef(false), revision = useRef(0);
  const frames = episode?.observations || [];
  const count = frames.length;
  const selected = anchor !== null && end !== null && count > 0;
  const first = selected ? Math.max(0, Math.min(anchor, end, count - 1)) : 0;
  const last = selected ? Math.min(count - 1, Math.max(anchor, end)) : 0;
  const size = selected ? last - first + 1 : 0;
  const locked = disabled || deleting;
  const name = episode?.storage_path?.split("/").filter(Boolean).pop() || episode?.scenario || "Episode";

  useEffect(() => {
    revision.current += 1;
    dragging.current = null; setDraggingRange(false);
    setOpen(false); setAnchor(null); setEnd(null); setConfirming(false); setError("");
  }, [episode?.id]);
  useEffect(() => {
    if (!open) return;
    const previous = document.activeElement;
    dialog.current?.querySelector("button")?.focus();
    return () => { if (previous?.isConnected) previous.focus(); };
  }, [open]);
  useEffect(() => { setConfirming(false); }, [count]);

  function close() { if (!pending.current) { setOpen(false); setConfirming(false); setError(""); } }
  function choose(a, b) {
    if (locked || confirming || !count || !Number.isFinite(a) || !Number.isFinite(b)) return;
    setAnchor(Math.max(0, Math.min(count - 1, Math.trunc(a))));
    setEnd(Math.max(0, Math.min(count - 1, Math.trunc(b))));
    setError("");
  }
  function pointerDown(event) {
    if (locked || confirming || event.button !== 0 || !count) return;
    const index = frameIndexAt(event.clientX, event.currentTarget.getBoundingClientRect(), count);
    dragging.current = { pointerId: event.pointerId, anchor: index };
    setDraggingRange(true);
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.currentTarget.focus(); event.preventDefault();
    choose(index, index);
  }
  function pointerMove(event) {
    const drag = dragging.current;
    if (!drag || drag.pointerId !== event.pointerId || locked || confirming) return;
    choose(drag.anchor, frameIndexAt(event.clientX, track.current.getBoundingClientRect(), count));
  }
  function pointerEnd(event) {
    if (dragging.current?.pointerId !== event.pointerId) return;
    pointerMove(event); dragging.current = null; setDraggingRange(false);
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  }
  function keyDown(event) {
    if (event.key === "Escape") { event.preventDefault(); close(); return; }
    if (event.key !== "Tab") return;
    const items = [...dialog.current.querySelectorAll('button:not(:disabled), input:not(:disabled), [tabindex="0"]')];
    if (!items.length) { event.preventDefault(); return; }
    if (event.shiftKey && document.activeElement === items[0]) { event.preventDefault(); items[items.length - 1].focus(); }
    if (!event.shiftKey && document.activeElement === items[items.length - 1]) { event.preventDefault(); items[0].focus(); }
  }
  async function remove() {
    if (locked || pending.current || !size) return;
    const currentRevision = revision.current;
    const ids = frames.slice(first, last + 1).map(frame => frame.frame_id);
    pending.current = true; setDeleting(true); setError("");
    try {
      if (await onDelete(ids) === false) throw new Error("Could not delete the selected frames. Please try again.");
      if (currentRevision === revision.current) { setOpen(false); setAnchor(null); setEnd(null); setConfirming(false); }
    } catch (err) {
      if (currentRevision === revision.current) { setError(err.message || "Could not delete frames."); setConfirming(false); }
    } finally { pending.current = false; setDeleting(false); }
  }
  const preview = (index, label) => <figure>
    <img src={`/api/collection/episodes/${encodeURIComponent(episode.id)}/images/${encodeURIComponent(frames[index].frame_id)}`} alt={`${label} selected frame ${index + 1}`} draggable="false" />
    <figcaption>{label} · {index + 1}</figcaption>
  </figure>;

  return <>
    <button type="button" className="frame-range-delete-button" title="Delete frame range" aria-label="Delete frame range"
      aria-haspopup="dialog" aria-expanded={open} disabled={locked || !episode?.id || !episode?.ended_at || !count} onClick={() => { setOpen(true); setAnchor(null); setEnd(null); setError(""); setConfirming(false); }}>
      <MdDeleteOutline size={18} aria-hidden="true" />
    </button>
    {open && createPortal(<div className="frame-range-overlay" onClick={event => { if (event.target === event.currentTarget) close(); }}>
      <section ref={dialog} className="frame-range-dialog" role="dialog" aria-modal="true" aria-label="Delete frame range" aria-busy={deleting} onKeyDown={keyDown}>
        <header><div><strong>Delete frame range</strong><span>{name} · {count} frames</span></div>
          <button type="button" aria-label="Close frame selection" title="Close" disabled={deleting} onClick={close}><MdClose size={20}/></button>
        </header>
        <div className="frame-range-content">
          <div className="frame-range-previews">{selected && !draggingRange ? <>{preview(first, "From")}{preview(last, "To")}</> : <span>{draggingRange ? `Frames ${first + 1} – ${last + 1}` : "Drag across the strip to select frames."}</span>}</div>
          <div ref={track} className="frame-range-track" role="slider" tabIndex={locked || confirming ? -1 : 0} aria-label="Frame range"
            aria-valuemin={1} aria-valuemax={Math.max(1, count)} aria-valuenow={selected ? end + 1 : 1}
            aria-valuetext={selected ? `Frames ${first + 1} to ${last + 1}; ${size} selected` : "No frames selected"} aria-disabled={locked || confirming}
            onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerEnd} onPointerCancel={() => { dragging.current = null; setDraggingRange(false); }}
            onKeyDown={event => {
              if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
              event.preventDefault();
              const next = event.key === "Home" ? 0 : event.key === "End" ? count - 1 : (end ?? 0) + (event.key === "ArrowLeft" ? -1 : 1);
              choose(anchor ?? 0, next);
            }}>
            {selected && <span className="frame-range-highlight" style={{left: `${first / count * 100}%`, width: `${size / count * 100}%`}}/>}
            <div className="frame-range-ticks" aria-hidden="true">{Array.from({length: Math.min(count, 80)}, (_, i) => <i key={i}/>)}</div>
          </div>
          <div className="frame-range-scale" aria-hidden="true"><span>1</span><span>{count}</span></div>
          <div className="frame-range-inputs">
            <label>From<input type="number" min="1" max={count} value={selected ? first + 1 : ""} disabled={locked || confirming}
              onChange={event => { if (event.target.value) { const value = Number(event.target.value) - 1; choose(value, Math.max(last, value)); } }}/></label>
            <label>To<input type="number" min="1" max={count} value={selected ? last + 1 : ""} disabled={locked || confirming}
              onChange={event => { if (event.target.value) { const value = Number(event.target.value) - 1; choose(Math.min(first, value), value); } }}/></label>
            <strong aria-live="polite">{size} selected</strong>
          </div>
          {confirming && <p className="frame-range-warning">Delete {size} {size === 1 ? "frame" : "frames"} and their images, labels, and masks? This cannot be undone.</p>}
          {error && <p role="alert">{error}</p>}
        </div>
        <footer>
          <button type="button" disabled={deleting} onClick={() => confirming ? setConfirming(false) : close()}>{confirming ? "Keep frames" : "Cancel"}</button>
          <button type="button" className="frame-range-remove" disabled={locked || !size} onClick={() => confirming ? remove() : setConfirming(true)}>
            <MdDeleteOutline size={17} aria-hidden="true"/>{deleting ? "Deleting…" : confirming ? "Delete permanently" : `Delete ${size} ${size === 1 ? "frame" : "frames"}`}
          </button>
        </footer>
      </section>
    </div>, document.body)}
  </>;
}
