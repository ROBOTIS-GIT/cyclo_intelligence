import React from 'react';
import { MdUnfoldMore, MdUnfoldLess } from 'react-icons/md';
import { JointChart } from './index';

const STATE_COLOR = '#dc2626';
const ACTION_COLOR = '#2563eb';

function JointDataPanel({
  allJointNames,
  stateChartData,
  actionChartData,
  currentTime,
  duration,
  expandedJoints,
  toggleJoint,
  expandAllJoints,
  collapseAllJoints,
  hasActionData,
  actionNames,
  handleChartSeek,
}) {
  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Header with legend and expand/collapse all buttons */}
      <div className="flex items-center justify-between mb-2 flex-shrink-0">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-semibold text-gray-800">Joint Data</h2>
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
              <div className="w-3 h-1 rounded" style={{ backgroundColor: STATE_COLOR }} />
              <span className="text-gray-600">State</span>
            </div>
            {hasActionData && (
              <div className="flex items-center gap-1">
                <div className="w-3 h-1 rounded" style={{ backgroundColor: ACTION_COLOR }} />
                <span className="text-gray-600">Action</span>
              </div>
            )}
          </div>
        </div>
        {allJointNames.length > 0 && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-gray-500">
              {expandedJoints.size} / {allJointNames.length}
            </span>
            <button
              onClick={expandAllJoints}
              className="flex items-center gap-1 px-1.5 py-0.5 text-xs text-gray-600 hover:bg-gray-100 rounded transition-colors"
              title="Expand all"
            >
              <MdUnfoldMore size={14} />
            </button>
            <button
              onClick={collapseAllJoints}
              className="flex items-center gap-1 px-1.5 py-0.5 text-xs text-gray-600 hover:bg-gray-100 rounded transition-colors"
              title="Collapse all"
            >
              <MdUnfoldLess size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Joint charts grid */}
      {(stateChartData.length > 0 || actionChartData.length > 0) && allJointNames.length > 0 ? (
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
            {allJointNames.map((name) => (
              <JointChart
                key={name}
                name={name}
                stateData={stateChartData}
                actionData={actionChartData}
                currentTime={currentTime}
                duration={duration}
                isExpanded={expandedJoints.has(name)}
                onToggle={() => toggleJoint(name)}
                hasAction={hasActionData && actionNames.includes(name)}
                onSeek={handleChartSeek}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-gray-400 text-sm">
          No joint data available
        </div>
      )}
    </div>
  );
}

export default React.memo(JointDataPanel);
