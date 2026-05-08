import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { togglePanelVisibility, togglePanelExpanded } from '../../features/layout/layoutSlice';
import PanelTitleBar from './PanelTitleBar';

function DraggablePanel({ panelId, children, onDragStart, isDragging }) {
  const dispatch = useDispatch();
  const panel = useSelector((state) => state.layout.panels[panelId]);

  if (!panel || !panel.visible) return null;

  const gridColumn = panel.expanded ? panel.expandedGridColumn : panel.gridColumn;
  const gridRow = panel.expanded ? panel.expandedGridRow : panel.gridRow;

  const handleMouseDown = (e) => {
    if (onDragStart) {
      onDragStart(panelId, e);
    }
  };

  return (
    <div
      data-panel-id={panelId}
      className={`bg-white rounded-xl shadow-sm overflow-hidden flex flex-col transition-shadow ${
        isDragging ? 'ring-2 ring-blue-400 shadow-lg z-20 opacity-75' : ''
      } ${panel.expanded ? 'z-10' : ''}`}
      style={{
        gridColumn,
        gridRow,
        minHeight: 0,
        minWidth: 0,
      }}
    >
      <PanelTitleBar
        title={panel.title}
        expanded={panel.expanded}
        onToggleExpand={() => dispatch(togglePanelExpanded(panelId))}
        onClose={() => dispatch(togglePanelVisibility(panelId))}
        onMouseDown={handleMouseDown}
        isDragging={isDragging}
      />
      <div className="flex-1 min-h-0 overflow-hidden">
        {children}
      </div>
    </div>
  );
}

export default React.memo(DraggablePanel);
