import React from 'react';
import clsx from 'clsx';
import { MdClose, MdKeyboardArrowUp, MdKeyboardArrowDown } from 'react-icons/md';
import { formatTime, formatFileSize, formatDateTime } from '../../utils/chartUtils';

function SidebarPanel({
  // Metadata
  recordingDate,
  robotType,
  fileSizeBytes,
  duration,
  frameCounts,
  // Task Markers
  taskMarkers,
  currentActiveTask,
  currentActiveTaskIndex,
  trimStart,
  trimEnd,
  excludeRegions,
  pendingExcludeStart,
  isSavingMarkers,
  // Instruction Palette
  instructionPalette,
  newPaletteInput,
  setNewPaletteInput,
  addToPalette,
  removeFromPalette,
  importPaletteFromJson,
  exportPaletteToJson,
  // ROSbag list
  rosbagList,
  currentBagIndex,
  isDownloading,
  navigateRosbag,
  handleSelectBag,
  // Actions
  dispatch,
  setTaskMarkers,
  setTrimStart,
  setLoopStart,
  setLoopEnd,
  seekAndPlay,
  seekAndPause,
  saveMarkers,
  deleteExcludeRegion,
  currentTime,
}) {
  return (
    <div className="flex flex-col gap-2 h-full overflow-y-auto">
      {/* Metadata panel */}
      <div className="bg-white rounded-lg shadow-sm overflow-hidden flex-shrink-0">
        <div className="p-2 border-b bg-gray-50">
          <h3 className="text-xs font-semibold text-gray-700">Metadata</h3>
        </div>
        <div className="p-2 space-y-1.5 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-500">Recorded</span>
            <span className="text-gray-700 font-medium text-right">
              {formatDateTime(recordingDate)}
            </span>
          </div>
          {robotType && (
            <div className="flex justify-between">
              <span className="text-gray-500">Robot</span>
              <span className="text-gray-700 font-medium">{robotType}</span>
            </div>
          )}
          <div className="flex justify-between">
            <span className="text-gray-500">Size</span>
            <span className="text-gray-700 font-medium">
              {formatFileSize(fileSizeBytes)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-500">Duration</span>
            <span className="text-gray-700 font-medium">
              {formatTime(duration)}
            </span>
          </div>
          {frameCounts && Object.keys(frameCounts).length > 0 && (
            <div className="pt-1.5 border-t mt-1.5">
              <div className="text-gray-500 mb-1">Frames</div>
              {Object.entries(frameCounts).map(([camera, count]) => (
                <div key={camera} className="flex justify-between pl-2">
                  <span className="text-gray-400 truncate">{camera}</span>
                  <span className="text-gray-600">{count}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Task Markers panel */}
      <div className="bg-white rounded-lg shadow-sm overflow-hidden flex-shrink-0">
        <div className="p-2 border-b bg-gray-50">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-gray-700">Task Markers</h3>
            <span className="text-[10px] text-gray-500">{taskMarkers.length} markers</span>
          </div>
        </div>
        <div className="p-1.5 max-h-40 overflow-y-auto">
          {(trimStart || taskMarkers.length > 0) ? (
            <div className="space-y-0.5">
              {/* Start point */}
              {trimStart && (
                <div
                  className={clsx(
                    'flex items-center gap-1.5 p-1 rounded cursor-pointer group transition-colors text-xs',
                    !currentActiveTask && currentTime >= trimStart.time && (taskMarkers.length === 0 || currentTime < taskMarkers[0].time)
                      ? 'bg-green-100 border-l-2 border-green-500'
                      : 'hover:bg-gray-50'
                  )}
                  onClick={() => {
                    seekAndPause(trimStart.time);
                  }}
                  title="Click to seek to Start"
                >
                  <span className="w-4 h-4 flex items-center justify-center text-[10px] font-bold rounded bg-green-500 text-white">S</span>
                  <span className="w-10 text-green-700 font-medium">{formatTime(trimStart.time)}</span>
                  <span className="flex-1 truncate text-green-800" title={trimStart.instruction}>{trimStart.instruction}</span>
                  <button
                    onClick={(e) => { e.stopPropagation(); setTrimStart(null); }}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
                  >
                    <MdClose size={12} />
                  </button>
                </div>
              )}
              {/* Task markers */}
              {taskMarkers.map((marker, idx) => (
                <div
                  key={`marker-item-${idx}`}
                  className={clsx(
                    'flex items-center gap-1.5 p-1 rounded cursor-pointer group transition-colors text-xs',
                    idx === currentActiveTaskIndex
                      ? 'bg-purple-100 border-l-2 border-purple-500'
                      : 'hover:bg-gray-50'
                  )}
                  onClick={() => {
                    const segmentStart = marker.time;
                    const nextMarkerIdx = idx + 1;
                    const segmentEnd = nextMarkerIdx < taskMarkers.length
                      ? taskMarkers[nextMarkerIdx].time
                      : (trimEnd?.time || duration);
                    setLoopStart(segmentStart);
                    setLoopEnd(segmentEnd);
                    seekAndPlay(segmentStart);
                  }}
                  title="Click to loop this task segment"
                >
                  <span className={clsx(
                    'w-4 h-4 flex items-center justify-center text-[10px] font-bold rounded',
                    idx === currentActiveTaskIndex ? 'bg-purple-500 text-white' : 'bg-purple-100 text-purple-700'
                  )}>
                    {idx + 1}
                  </span>
                  <span className={clsx('w-10', idx === currentActiveTaskIndex ? 'text-purple-700 font-medium' : 'text-gray-500')}>
                    {formatTime(marker.time)}
                  </span>
                  <span className={clsx(
                    'flex-1 truncate',
                    idx === currentActiveTaskIndex ? 'text-purple-900 font-medium' : 'text-gray-700'
                  )} title={marker.instruction}>
                    {marker.instruction}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const updated = taskMarkers.filter((_, i) => i !== idx);
                      dispatch(setTaskMarkers(updated));
                    }}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
                  >
                    <MdClose size={12} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[10px] text-gray-400 text-center py-2">
              Press 'S' for Start, 'm' to add marker
            </div>
          )}
        </div>
        {/* Trim range info */}
        {(trimStart || trimEnd) && (
          <div className="px-1.5 py-1 bg-blue-50 border-t text-[10px]">
            <div className="flex justify-between items-center text-blue-700">
              <span className="font-medium">Range:</span>
              <span>{formatTime(trimStart?.time || 0)} → {formatTime(trimEnd?.time || duration)}</span>
            </div>
          </div>
        )}
        {/* Exclude Regions */}
        {(excludeRegions.length > 0 || pendingExcludeStart) && (
          <div className="px-1.5 py-1 bg-red-50 border-t text-[10px]">
            <div className="flex justify-between items-center text-red-700 mb-0.5">
              <span className="font-medium">Exclude:</span>
              <span>{excludeRegions.length} region{excludeRegions.length !== 1 ? 's' : ''}</span>
            </div>
            {pendingExcludeStart && (
              <div className="text-red-500 italic mb-0.5 animate-pulse">
                From {formatTime(pendingExcludeStart.time)}... (X to end)
              </div>
            )}
            {excludeRegions.length > 0 && (
              <div className="space-y-0.5 max-h-16 overflow-y-auto">
                {excludeRegions.map((region, idx) => (
                  <div key={`exclude-item-${idx}`} className="flex items-center justify-between text-red-600 group">
                    <span>{formatTime(region.start.time)} → {formatTime(region.end.time)}</span>
                    <button
                      onClick={() => deleteExcludeRegion(idx)}
                      className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-600 transition-opacity"
                    >
                      <MdClose size={10} />
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {/* Save button */}
        {(taskMarkers.length > 0 || trimStart || trimEnd || excludeRegions.length > 0) && (
          <div className="p-1.5 border-t bg-gray-50">
            <button
              onClick={saveMarkers}
              disabled={isSavingMarkers}
              className={clsx(
                'w-full py-1 text-xs rounded transition-colors',
                isSavingMarkers
                  ? 'bg-gray-200 text-gray-400 cursor-wait'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              )}
            >
              {isSavingMarkers ? 'Saving...' : 'Save to File'}
            </button>
          </div>
        )}
      </div>

      {/* Instruction Palette panel */}
      <div className="bg-white rounded-lg shadow-sm overflow-hidden flex-shrink-0">
        <div className="p-2 border-b bg-gray-50">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-gray-700">Instruction Palette</h3>
            <div className="flex items-center gap-1.5">
              <label className="text-[10px] text-blue-600 hover:text-blue-800 cursor-pointer">
                <input type="file" accept=".json" onChange={importPaletteFromJson} className="hidden" />
                Import
              </label>
              <span className="text-gray-300">|</span>
              <button onClick={exportPaletteToJson} className="text-[10px] text-blue-600 hover:text-blue-800">Export</button>
            </div>
          </div>
        </div>
        <div className="p-1.5 max-h-28 overflow-y-auto">
          {instructionPalette.length > 0 ? (
            <div className="space-y-0.5">
              {instructionPalette.map((instruction, idx) => (
                <div key={`palette-${idx}`} className="flex items-center gap-1.5 p-0.5 hover:bg-gray-50 rounded group text-xs">
                  <span className="w-3 text-gray-400 font-medium">{idx < 9 ? idx + 1 : ''}</span>
                  <span className="flex-1 text-gray-600 truncate" title={instruction}>{instruction}</span>
                  <button
                    onClick={() => removeFromPalette(idx)}
                    className="opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition-opacity"
                  >
                    <MdClose size={10} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[10px] text-gray-400 text-center py-2">No instructions</div>
          )}
        </div>
        <div className="p-1.5 border-t bg-gray-50">
          <div className="flex gap-1">
            <input
              type="text"
              value={newPaletteInput}
              onChange={(e) => setNewPaletteInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && newPaletteInput.trim()) {
                  e.preventDefault();
                  addToPalette(newPaletteInput);
                  setNewPaletteInput('');
                }
              }}
              placeholder="New instruction..."
              className="flex-1 px-1.5 py-1 text-xs border rounded focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
            <button
              onClick={() => { if (newPaletteInput.trim()) { addToPalette(newPaletteInput); setNewPaletteInput(''); } }}
              disabled={!newPaletteInput.trim()}
              className={clsx(
                'px-2 py-1 text-xs rounded transition-colors',
                newPaletteInput.trim() ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-200 text-gray-400 cursor-not-allowed'
              )}
            >
              Add
            </button>
          </div>
        </div>
      </div>

      {/* ROSbag list panel */}
      {rosbagList.length > 1 && (
        <div className="flex-1 flex flex-col bg-white rounded-lg shadow-sm overflow-hidden min-h-0">
          <div className="p-2 border-b bg-gray-50 flex-shrink-0">
            <div className="flex items-center justify-between mb-1.5">
              <h3 className="text-xs font-semibold text-gray-700">ROSbag List</h3>
              <span className="text-[10px] text-gray-500">{currentBagIndex + 1} / {rosbagList.length}</span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => navigateRosbag('prev')}
                disabled={currentBagIndex <= 0 || isDownloading}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                  currentBagIndex <= 0 || isDownloading
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                )}
              >
                <MdKeyboardArrowUp size={16} />
                Prev
              </button>
              <button
                onClick={() => navigateRosbag('next')}
                disabled={currentBagIndex >= rosbagList.length - 1 || isDownloading}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                  currentBagIndex >= rosbagList.length - 1 || isDownloading
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                )}
              >
                Next
                <MdKeyboardArrowDown size={16} />
              </button>
            </div>
          </div>
          <div className="flex-1 overflow-y-auto min-h-0">
            {rosbagList.map((bag, index) => (
              <button
                key={bag.path}
                onClick={() => handleSelectBag(bag.path)}
                disabled={isDownloading}
                className={clsx(
                  'w-full text-left px-2 py-1.5 border-b border-gray-100 transition-colors text-xs',
                  index === currentBagIndex ? 'bg-blue-50 border-l-4 border-l-blue-500' : 'hover:bg-gray-50',
                  isDownloading && 'cursor-not-allowed opacity-50'
                )}
              >
                <div className="font-medium text-gray-700 truncate">{bag.name}</div>
                {bag.has_videos && <div className="text-[10px] text-green-600">Has videos</div>}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default React.memo(SidebarPanel);
