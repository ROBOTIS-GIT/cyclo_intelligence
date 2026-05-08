import React from 'react';
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
}) {
  const panels = useSelector((state) => state.layout.panels);
  const { dragPanelId, dropTargetPanel, handleDragStart, containerRef } = useGridDrag();

  const panelContentMap = {
    [PANEL_IDS.CAMERAS]: cameraPanelContent,
    [PANEL_IDS.VIEWER_3D]: viewer3DPanelContent,
    [PANEL_IDS.JOINT_DATA]: jointDataPanelContent,
    [PANEL_IDS.SIDEBAR]: sidebarPanelContent,
  };

  const visiblePanels = Object.values(panels).filter((p) => p.visible);

  return (
    <div
      ref={containerRef}
      className="grid h-full overflow-hidden"
      style={{
        gridTemplateColumns: 'repeat(12, 1fr)',
        gridTemplateRows: 'minmax(0, 1fr) minmax(0, 1fr) minmax(0, 0.6fr)',
        gap: '8px',
      }}
    >
      {visiblePanels.map((panel) => (
        <DraggablePanel
          key={panel.id}
          panelId={panel.id}
          onDragStart={handleDragStart}
          isDragging={dragPanelId === panel.id}
        >
          {panelContentMap[panel.id]}
        </DraggablePanel>
      ))}

      {/* Drop preview overlay */}
      {dragPanelId && dropTargetPanel && (
        <DropPreview targetPanel={dropTargetPanel} />
      )}
    </div>
  );
}

export default React.memo(ReplayLayoutContainer);
