import React from 'react';
import { MdDragIndicator, MdOpenInFull, MdCloseFullscreen, MdClose } from 'react-icons/md';

function PanelTitleBar({
  title,
  expanded,
  onToggleExpand,
  onClose,
  onMouseDown,
  isDragging,
}) {
  return (
    <div
      className={`flex items-center justify-between px-2 py-1 bg-gray-50 border-b select-none flex-shrink-0 ${
        isDragging ? 'cursor-grabbing' : 'cursor-grab'
      }`}
      onMouseDown={onMouseDown}
    >
      <div className="flex items-center gap-1.5">
        <MdDragIndicator size={14} className="text-gray-400" />
        <span className="text-xs font-semibold text-gray-700">{title}</span>
      </div>
      <div className="flex items-center gap-0.5">
        <button
          onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
          className="p-0.5 rounded hover:bg-gray-200 text-gray-500 hover:text-gray-700 transition-colors"
          title={expanded ? 'Collapse' : 'Expand'}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {expanded ? <MdCloseFullscreen size={13} /> : <MdOpenInFull size={13} />}
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onClose(); }}
          className="p-0.5 rounded hover:bg-red-100 text-gray-500 hover:text-red-600 transition-colors"
          title="Close panel"
          onMouseDown={(e) => e.stopPropagation()}
        >
          <MdClose size={13} />
        </button>
      </div>
    </div>
  );
}

export default React.memo(PanelTitleBar);
