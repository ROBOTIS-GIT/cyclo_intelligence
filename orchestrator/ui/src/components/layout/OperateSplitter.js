import React, { useLayoutEffect, useRef, useState } from 'react';

const MIN = 25;
const MAX = 75;
const clamp = value => Math.min(MAX, Math.max(MIN, Math.round(value)));

// Only the separator re-renders during a drag; the live panels stay mounted.
export default function OperateSplitter({ orientation, label, controls, initialPercent }) {
  const vertical = orientation === 'vertical';
  const storageKey = `cyclo_intelligence.operate.split.${orientation}`;
  const handle = useRef(null);
  const drag = useRef(null);
  const [percent, setPercent] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(storageKey));
      if (Number.isFinite(saved)) return clamp(saved);
    } catch { /* Keep the default when storage is unavailable. */ }
    return initialPercent;
  });
  useLayoutEffect(() => {
    const style = handle.current.parentElement.style;
    style.setProperty('--operate-first', `${percent}fr`);
    style.setProperty('--operate-second', `${100 - percent}fr`);
  }, [percent]);

  function commit(value) {
    const next = clamp(value);
    setPercent(next);
    try { localStorage.setItem(storageKey, JSON.stringify(next)); } catch { /* Session only. */ }
  }

  const coordinate = event => vertical ? event.clientX : event.clientY;
  function onPointerDown(event) {
    if (event.button !== 0 || drag.current) return;
    const el = event.currentTarget;
    const axis = vertical ? 'width' : 'height';
    const first = el.previousElementSibling.getBoundingClientRect()[axis];
    const second = el.nextElementSibling.getBoundingClientRect()[axis];
    if (!(first + second > 0)) return;
    el.setPointerCapture(event.pointerId);
    drag.current = { id: event.pointerId, start: coordinate(event), available: first + second, percent };
  }
  function update(event, finish) {
    const d = drag.current;
    if (!d || d.id !== event.pointerId) return;
    const next = clamp(d.percent + 100 * (coordinate(event) - d.start) / d.available);
    if (finish) {
      drag.current = null;
      commit(next);
    } else setPercent(next);
  }
  function cancel(event) {
    if (drag.current?.id !== event.pointerId) return;
    setPercent(drag.current.percent);
    drag.current = null;
  }
  function onKeyDown(event) {
    const keys = { Home: MIN, End: MAX, [vertical ? 'ArrowLeft' : 'ArrowUp']: percent - 5,
      [vertical ? 'ArrowRight' : 'ArrowDown']: percent + 5 };
    if (!Object.hasOwn(keys, event.key)) return;
    event.preventDefault();
    event.stopPropagation();
    commit(keys[event.key]);
  }

  return (
    <div ref={handle} role="separator" tabIndex={0} aria-label={label} aria-orientation={orientation}
      aria-controls={controls} aria-valuemin={MIN} aria-valuemax={MAX} aria-valuenow={percent}
      aria-valuetext={`${percent}%`} title={`${label} (double-click to reset)`}
      className="operate-splitter" onPointerDown={onPointerDown} onPointerMove={e => update(e, false)}
      onPointerUp={e => update(e, true)} onPointerCancel={cancel} onLostPointerCapture={cancel}
      onDoubleClick={() => commit(initialPercent)} onKeyDown={onKeyDown} />
  );
}
