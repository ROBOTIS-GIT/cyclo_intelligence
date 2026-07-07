import React, { useMemo } from 'react';
import RobotViewer3D from '../RobotViewer3D';

function Viewer3DPanel({
  robotType,
  jointTimestamps,
  jointNames,
  jointPositions,
  actionTimestamps,
  actionNames,
  actionValues,
  currentTime,
  isPlaying,
  playbackSpeed,
}) {
  const jointData = useMemo(() => ({
    timestamps: jointTimestamps,
    names: jointNames,
    positions: jointPositions,
  }), [jointTimestamps, jointNames, jointPositions]);

  const actionData = useMemo(() => ({
    timestamps: actionTimestamps,
    names: actionNames,
    values: actionValues,
  }), [actionTimestamps, actionNames, actionValues]);

  return (
    <div className="w-full h-full bg-gray-900 rounded-lg overflow-hidden relative" style={{ minHeight: 0, minWidth: 0 }}>
      <div className="absolute top-2 left-2 z-10 px-2 py-1 bg-black bg-opacity-60 text-white text-xs rounded font-medium">
        3D Robot View
      </div>
      <RobotViewer3D
        mode="replay"
        robotTypeOverride={robotType}
        jointData={jointData}
        actionData={actionData}
        currentTime={currentTime}
        isPlaying={isPlaying}
        playbackSpeed={playbackSpeed}
        className="w-full h-full"
      />
    </div>
  );
}

export default React.memo(Viewer3DPanel);
