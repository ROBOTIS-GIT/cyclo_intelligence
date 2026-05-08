import { useState, useRef, useCallback, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { swapPanels } from '../features/layout/layoutSlice';

function useGridDrag() {
  const dispatch = useDispatch();
  const panels = useSelector((state) => state.layout.panels);
  const containerRef = useRef(null);

  const [dragPanelId, setDragPanelId] = useState(null);
  const [dropTargetPanel, setDropTargetPanel] = useState(null);
  const dragStartRef = useRef(null);

  // Find which panel element is at a given point
  const getPanelAtPoint = useCallback((x, y) => {
    const container = containerRef.current;
    if (!container) return null;

    const panelElements = container.querySelectorAll('[data-panel-id]');
    for (const el of panelElements) {
      const rect = el.getBoundingClientRect();
      if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
        return el.getAttribute('data-panel-id');
      }
    }
    return null;
  }, []);

  const handleDragStart = useCallback((panelId, e) => {
    // Only left mouse button
    if (e.button !== 0) return;
    e.preventDefault();

    setDragPanelId(panelId);
    dragStartRef.current = { x: e.clientX, y: e.clientY, panelId };
  }, []);

  useEffect(() => {
    if (!dragPanelId) return;

    const handleMouseMove = (e) => {
      const start = dragStartRef.current;
      if (!start) return;

      // Only start visual drag after 5px threshold
      const dx = Math.abs(e.clientX - start.x);
      const dy = Math.abs(e.clientY - start.y);
      if (dx < 5 && dy < 5) return;

      const targetId = getPanelAtPoint(e.clientX, e.clientY);
      if (targetId && targetId !== dragPanelId) {
        const targetPanel = panels[targetId];
        if (targetPanel) {
          setDropTargetPanel(targetPanel);
        }
      } else {
        setDropTargetPanel(null);
      }
    };

    const handleMouseUp = (e) => {
      const targetId = getPanelAtPoint(e.clientX, e.clientY);
      if (targetId && targetId !== dragPanelId) {
        dispatch(swapPanels({ sourceId: dragPanelId, targetId }));
      }

      setDragPanelId(null);
      setDropTargetPanel(null);
      dragStartRef.current = null;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [dragPanelId, panels, dispatch, getPanelAtPoint]);

  return {
    dragPanelId,
    dropTargetPanel,
    handleDragStart,
    containerRef,
  };
}

export default useGridDrag;
