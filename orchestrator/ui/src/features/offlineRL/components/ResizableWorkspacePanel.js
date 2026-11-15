import React, { useRef, useState } from 'react';

const storageKey = (side) => `playground.panel-width.${side}`;
const readWidth = (side) => {
  try {
    const value = Number(window.localStorage.getItem(storageKey(side)));
    return Number.isFinite(value) && value >= 360 ? value : null;
  } catch {
    return null;
  }
};

// Only presentation state is persisted; opening/closing never remounts the content.
export default function ResizableWorkspacePanel({
  side, label, active, onActivate, children, className = '', ...props
}) {
  const [width, setWidth] = useState(() => readWidth(side));
  const panelRef = useRef(null);
  const dragRef = useRef(null);
  const direction = side === 'left' ? 1 : -1;
  const clampWidth = (value) => {
    const available = Math.max(0, (panelRef.current?.parentElement?.clientWidth || window.innerWidth) - 32);
    return Math.round(Math.min(available, Math.max(Math.min(360, available), value)));
  };
  const persist = (value) => {
    try {
      if (value == null) window.localStorage.removeItem(storageKey(side));
      else window.localStorage.setItem(storageKey(side), String(value));
    } catch { /* Resizing still works when browser storage is unavailable. */ }
  };
  const finishDrag = (event, cancelled = false) => {
    const drag = dragRef.current;
    if (!drag) return;
    dragRef.current = null;
    if (cancelled) setWidth(drag.previous);
    else persist(drag.current);
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <aside
      {...props}
      ref={panelRef}
      data-resize-side={side}
      className={`pg-panel ${className}`}
      style={{ '--pg-panel-width': width == null ? undefined : `${width}px`, zIndex: active ? 2 : 1 }}
      onPointerDownCapture={onActivate}
      onFocusCapture={onActivate}
    >
      {children}
      <div
        role="separator"
        tabIndex={0}
        aria-label={`Resize ${label} panel`}
        aria-orientation="vertical"
        aria-valuetext={width == null ? 'Automatic width' : `${width} pixels`}
        title="Drag to resize · arrow keys to adjust · double-click to reset"
        className={`pg-panel-resize pg-panel-resize-${side}`}
        onDoubleClick={() => { setWidth(null); persist(null); }}
        onPointerDown={(event) => {
          if (event.button !== 0) return;
          event.preventDefault();
          const current = panelRef.current.getBoundingClientRect().width;
          dragRef.current = { x: event.clientX, start: current, current, previous: width };
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.focus();
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag) return;
          drag.current = clampWidth(drag.start + direction * (event.clientX - drag.x));
          setWidth(drag.current);
        }}
        onPointerUp={finishDrag}
        onPointerCancel={(event) => finishDrag(event, true)}
        onLostPointerCapture={(event) => finishDrag(event, true)}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && dragRef.current) {
            finishDrag(event, true);
          } else if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
            event.preventDefault();
            const current = panelRef.current.getBoundingClientRect().width;
            const next = clampWidth(current + direction * (event.key === 'ArrowRight' ? 24 : -24));
            setWidth(next);
            persist(next);
          } else if (event.key === 'Home') {
            event.preventDefault();
            setWidth(null);
            persist(null);
          }
        }}
      />
    </aside>
  );
}
