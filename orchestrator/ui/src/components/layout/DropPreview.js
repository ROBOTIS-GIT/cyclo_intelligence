import React from 'react';

function DropPreview({ targetPanel }) {
  if (!targetPanel) return null;

  return (
    <div
      className="rounded-xl border-2 border-blue-400 border-dashed bg-blue-50 bg-opacity-50 pointer-events-none transition-all duration-150"
      style={{
        gridColumn: targetPanel.gridColumn,
        gridRow: targetPanel.gridRow,
      }}
    />
  );
}

export default React.memo(DropPreview);
