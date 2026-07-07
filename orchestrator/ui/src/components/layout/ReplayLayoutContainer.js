import React, { useCallback, useState } from 'react';
import { useSelector } from 'react-redux';
import { PANEL_IDS } from '../../features/layout/layoutSlice';
import DraggablePanel from './DraggablePanel';
import DropPreview from './DropPreview';
import useGridDrag from '../../hooks/useGridDrag';

function ReplayLayoutContainer({
  cameraPanelContent,
  viewer3DPanelContent,
  jointDataPanelContent,
  sidebarPanelContent,
  sidebarMode = 'grid',
}) {
  const panels = useSelector((state) => state.layout.panels);
  const { dragPanelId, dropTargetPanel, handleDragStart, containerRef } = useGridDrag();
  const drawerSidebar = sidebarMode === 'drawer';
  const [topPanePercent, setTopPanePercent] = useState(66);

  const panelContentMap = {
    [PANEL_IDS.CAMERAS]: cameraPanelContent,
    [PANEL_IDS.VIEWER_3D]: viewer3DPanelContent,
    [PANEL_IDS.JOINT_DATA]: jointDataPanelContent,
    [PANEL_IDS.SIDEBAR]: drawerSidebar ? null : sidebarPanelContent,
  };

  const layoutOverrides = drawerSidebar ? {
    [PANEL_IDS.CAMERAS]: {
      gridColumn: '1 / 13',
      gridRow: '1 / 2',
      expandedGridColumn: '1 / 13',
      expandedGridRow: '1 / 4',
    },
    [PANEL_IDS.VIEWER_3D]: {
      gridColumn: '1 / 4',
      gridRow: '3 / 4',
      expandedGridColumn: '1 / 13',
      expandedGridRow: '1 / 4',
    },
    [PANEL_IDS.JOINT_DATA]: {
      gridColumn: '4 / 13',
      gridRow: '3 / 4',
      expandedGridColumn: '1 / 13',
      expandedGridRow: '1 / 4',
    },
  } : {};

  const visiblePanels = Object.values(panels).filter(
    (p) => p.visible && panelContentMap[p.id]
  );
  const hasExpandedPanel = visiblePanels.some(
    (panel) => panel.expandable !== false && panel.expanded
  );

  const handleSplitterPointerDown = useCallback((event) => {
    if (!drawerSidebar || !containerRef.current) return;

    event.preventDefault();
    const containerRect = containerRef.current.getBoundingClientRect();
    const minPercent = 42;
    const maxPercent = 80;

    const updateSplit = (clientY) => {
      const nextPercent = ((clientY - containerRect.top) / containerRect.height) * 100;
      setTopPanePercent(Math.max(minPercent, Math.min(maxPercent, nextPercent)));
    };

    const handlePointerMove = (moveEvent) => {
      updateSplit(moveEvent.clientY);
    };

    const stopDrag = () => {
      document.removeEventListener('pointermove', handlePointerMove);
      document.removeEventListener('pointerup', stopDrag);
      document.removeEventListener('pointercancel', stopDrag);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    updateSplit(event.clientY);
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('pointermove', handlePointerMove);
    document.addEventListener('pointerup', stopDrag);
    document.addEventListener('pointercancel', stopDrag);
  }, [containerRef, drawerSidebar]);

  return (
    <div
      ref={containerRef}
      className="grid h-full overflow-hidden"
      style={{
        gridTemplateColumns: 'repeat(12, 1fr)',
        gridTemplateRows: drawerSidebar
          ? `minmax(0, ${topPanePercent}%) 8px minmax(0, 1fr)`
          : 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 0.6fr)',
        columnGap: '8px',
        rowGap: drawerSidebar ? '4px' : '8px',
      }}
    >
      {visiblePanels.map((panel) => (
        <DraggablePanel
          key={panel.id}
          panelId={panel.id}
          onDragStart={handleDragStart}
          isDragging={dragPanelId === panel.id}
          layoutOverride={layoutOverrides[panel.id]}
        >
          {panelContentMap[panel.id]}
        </DraggablePanel>
      ))}

      {drawerSidebar && !hasExpandedPanel && (
        <div
          className="z-20 flex cursor-row-resize items-center justify-center"
          style={{ gridColumn: '1 / 13', gridRow: '2 / 3' }}
          onPointerDown={handleSplitterPointerDown}
          title="Resize video and data panels"
          role="separator"
          aria-orientation="horizontal"
        >
          <div className="h-1.5 w-24 rounded-full bg-gray-300 transition-colors hover:bg-blue-500" />
        </div>
      )}

      {/* Drop preview overlay */}
      {dragPanelId && dropTargetPanel && (
        <DropPreview targetPanel={dropTargetPanel} />
      )}
    </div>
  );
}

export default React.memo(ReplayLayoutContainer);
