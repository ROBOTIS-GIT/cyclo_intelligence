import React from 'react';
import { MdDragIndicator, MdOpenInFull, MdCloseFullscreen, MdClose } from 'react-icons/md';

function PanelTitleBar({
  title,
  expanded,
  onToggleExpand,
  onClose,
  onMouseDown,
  isDragging,
  canExpand = true,
}) {
  const expandButtonLabel = expanded ? `Collapse ${title}` : `Expand ${title}`;
  const expandButtonClass = expanded
    ? 'p-1 rounded border border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100 hover:text-blue-800 transition-colors'
    : 'p-1 rounded hover:bg-gray-200 text-gray-500 hover:text-gray-700 transition-colors';

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
        {canExpand && (
          <button
            onClick={(e) => { e.stopPropagation(); onToggleExpand(); }}
            className={expandButtonClass}
            title={expandButtonLabel}
            aria-label={expandButtonLabel}
            onMouseDown={(e) => e.stopPropagation()}
          >
            {expanded ? <MdCloseFullscreen size={16} /> : <MdOpenInFull size={15} />}
          </button>
        )}
        <button
          onClick={(e) => { e.stopPropagation(); onClose(); }}
          className="p-1 rounded hover:bg-red-100 text-gray-500 hover:text-red-600 transition-colors"
          title="Close panel"
          aria-label={`Close ${title}`}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <MdClose size={15} />
        </button>
      </div>
    </div>
  );
}

export default React.memo(PanelTitleBar);
