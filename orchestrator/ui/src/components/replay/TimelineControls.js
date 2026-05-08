import React from 'react';
import clsx from 'clsx';
import { MdPlayArrow, MdPause, MdReplay, MdClose } from 'react-icons/md';
import { formatTime } from '../../utils/chartUtils';

function TimelineControls({
  // State
  currentTime,
  duration,
  isPlaying,
  isVideoLoaded,
  isDirectMcapMode,
  mcapPlayer,
  // Timeline markers
  trimStart,
  trimEnd,
  loopStart,
  loopEnd,
  excludeRegions,
  pendingExcludeStart,
  taskMarkers,
  currentActiveTask,
  currentActiveTaskIndex,
  // Actions
  handleSeek,
  togglePlayPause,
  restartPlayback,
  openMarkerDialog,
  openTrimStartDialog,
  applyTrimEnd,
  clearTrimPoints,
  clearLoop,
  changePlaybackSpeed,
  playbackSpeed,
  PLAYBACK_SPEEDS,
  setShowHelpModal,
  // Video controls
  videoFiles,
  setLoopStart,
  setLoopEnd,
  seekAndPlay,
  seekToTime,
}) {
  return (
    <div className="p-3 bg-white border-t shadow-sm">
      {/* Timeline */}
      <div
        className={clsx(
          'h-2 bg-gray-300 rounded-full mb-3 cursor-pointer relative',
          { 'opacity-50 cursor-not-allowed': !isVideoLoaded }
        )}
        onClick={isVideoLoaded ? handleSeek : undefined}
      >
        {/* Trim region highlight */}
        {(trimStart || trimEnd) && duration > 0 && (
          <div
            className="absolute h-full bg-blue-100 rounded-full"
            style={{
              left: `${((trimStart?.time || 0) / duration) * 100}%`,
              width: `${(((trimEnd?.time || duration) - (trimStart?.time || 0)) / duration) * 100}%`,
            }}
          />
        )}
        {/* A-B Loop region */}
        {loopStart !== null && loopEnd !== null && duration > 0 && (
          <div
            className="absolute h-full bg-yellow-300 opacity-40 rounded-full"
            style={{
              left: `${(loopStart / duration) * 100}%`,
              width: `${((loopEnd - loopStart) / duration) * 100}%`,
            }}
          />
        )}
        {/* Exclude regions */}
        {excludeRegions.map((region, idx) => (
          <div
            key={`exclude-${idx}`}
            className="absolute h-full bg-red-400 opacity-50 rounded-full"
            style={{
              left: `${(region.start.time / duration) * 100}%`,
              width: `${((region.end.time - region.start.time) / duration) * 100}%`,
              backgroundImage: 'repeating-linear-gradient(45deg, transparent, transparent 2px, rgba(0,0,0,0.1) 2px, rgba(0,0,0,0.1) 4px)',
            }}
            title={`Excluded: ${formatTime(region.start.time)} - ${formatTime(region.end.time)}`}
          />
        ))}
        {/* Pending exclude start */}
        {pendingExcludeStart && duration > 0 && (
          <div
            className="absolute top-1/2 -translate-y-1/2 w-1 h-5 bg-red-500 z-20 animate-pulse"
            style={{ left: `${(pendingExcludeStart.time / duration) * 100}%` }}
          />
        )}
        {/* Progress bar */}
        <div
          className="h-full bg-blue-500 rounded-full relative z-10"
          style={{ width: `${(currentTime / duration) * 100}%` }}
        />
        {/* A marker */}
        {loopStart !== null && duration > 0 && (
          <div
            className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-green-500 z-20"
            style={{ left: `${(loopStart / duration) * 100}%` }}
            title={`A: ${formatTime(loopStart)}`}
          />
        )}
        {/* B marker */}
        {loopEnd !== null && duration > 0 && (
          <div
            className="absolute top-1/2 -translate-y-1/2 w-1 h-4 bg-red-500 z-20"
            style={{ left: `${(loopEnd / duration) * 100}%` }}
            title={`B: ${formatTime(loopEnd)}`}
          />
        )}
        {/* Task Markers on timeline */}
        {taskMarkers.map((marker, idx) => (
          <div
            key={`marker-${idx}`}
            className="absolute top-1/2 -translate-y-1/2 z-15 cursor-pointer group"
            style={{ left: `${(marker.time / duration) * 100}%` }}
            onClick={(e) => {
              e.stopPropagation();
              seekToTime(marker.time);
            }}
            title={`#${idx + 1}: ${marker.instruction}`}
          >
            <div className="w-0.5 h-5 bg-purple-600 group-hover:bg-purple-400" />
            <div className="absolute -top-4 left-1/2 -translate-x-1/2 text-xs font-bold text-purple-600 bg-white px-1 rounded shadow-sm opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
              {idx + 1}
            </div>
          </div>
        ))}
        {/* Trim Start marker */}
        {trimStart && duration > 0 && (
          <div
            className="absolute top-1/2 -translate-y-1/2 z-18 cursor-pointer group"
            style={{ left: `${(trimStart.time / duration) * 100}%` }}
            onClick={(e) => {
              e.stopPropagation();
              seekToTime(trimStart.time);
            }}
            title={`Start: ${formatTime(trimStart.time)} - ${trimStart.instruction}`}
          >
            <div className="w-0 h-0 border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent border-l-[8px] border-l-green-600 group-hover:border-l-green-400" />
            <div className="absolute -top-5 left-0 text-xs font-bold text-green-600 bg-white px-1 rounded shadow-sm opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">S</div>
          </div>
        )}
        {/* Trim End marker */}
        {trimEnd && duration > 0 && (
          <div
            className="absolute top-1/2 -translate-y-1/2 z-18 cursor-pointer group"
            style={{ left: `${(trimEnd.time / duration) * 100}%` }}
            onClick={(e) => {
              e.stopPropagation();
              seekToTime(trimEnd.time);
            }}
            title={`End: ${formatTime(trimEnd.time)}`}
          >
            <div className="w-0 h-0 border-t-[6px] border-t-transparent border-b-[6px] border-b-transparent border-r-[8px] border-r-red-600 group-hover:border-r-red-400" />
            <div className="absolute -top-5 right-0 text-xs font-bold text-red-600 bg-white px-1 rounded shadow-sm opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">E</div>
          </div>
        )}
        {/* Playhead */}
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-blue-600 rounded-full shadow z-30"
          style={{ left: `calc(${(currentTime / duration) * 100}% - 8px)` }}
        />
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={restartPlayback}
          disabled={!isVideoLoaded}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            isVideoLoaded ? 'bg-gray-200 text-gray-700 hover:bg-gray-300' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
          title="Restart from beginning"
        >
          <MdReplay size={18} />
        </button>
        <button
          onClick={togglePlayPause}
          disabled={!isVideoLoaded}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            isVideoLoaded ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          )}
        >
          {isPlaying ? <MdPause size={20} /> : <MdPlayArrow size={20} />}
        </button>
        <span className="text-xs text-gray-600">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <span className="text-xs text-gray-400">
          ({videoFiles.length} cam{videoFiles.length > 1 ? 's' : ''})
        </span>
        {/* A-B Loop status */}
        {(loopStart !== null || loopEnd !== null) && (
          <div className="flex items-center gap-1.5 px-2 py-0.5 bg-yellow-100 rounded text-xs">
            <span className="text-green-600 font-medium">A: {loopStart !== null ? formatTime(loopStart) : '--'}</span>
            <span className="text-gray-400">|</span>
            <span className="text-red-600 font-medium">B: {loopEnd !== null ? formatTime(loopEnd) : '--'}</span>
            <button onClick={clearLoop} className="ml-1 text-gray-500 hover:text-gray-700" title="Clear loop"><MdClose size={12} /></button>
          </div>
        )}
        {/* Current active task */}
        {(currentActiveTask || trimStart) && (
          <button
            className="flex items-center gap-1.5 px-2 py-0.5 bg-purple-100 hover:bg-purple-200 rounded text-xs max-w-xs cursor-pointer transition-colors"
            onClick={() => {
              let segmentStart, segmentEnd;
              if (currentActiveTask) {
                segmentStart = currentActiveTask.time;
                const nextMarkerIdx = currentActiveTaskIndex + 1;
                segmentEnd = nextMarkerIdx < taskMarkers.length ? taskMarkers[nextMarkerIdx].time : (trimEnd?.time || duration);
              } else if (trimStart) {
                segmentStart = trimStart.time;
                segmentEnd = taskMarkers.length > 0 ? taskMarkers[0].time : (trimEnd?.time || duration);
              }
              if (segmentStart !== undefined && segmentEnd !== undefined) {
                setLoopStart(segmentStart);
                setLoopEnd(segmentEnd);
                seekAndPlay(segmentStart);
              }
            }}
            title="Click to loop this task segment"
          >
            <span className="text-purple-600 font-bold">Task:</span>
            <span className="text-purple-800 truncate" title={currentActiveTask?.instruction || trimStart?.instruction}>
              {currentActiveTask?.instruction || trimStart?.instruction}
            </span>
          </button>
        )}
        {/* Trim point buttons */}
        <div className="flex items-center gap-1">
          <button
            onClick={openTrimStartDialog}
            disabled={!isVideoLoaded}
            className={clsx(
              'px-1.5 py-0.5 text-xs rounded transition-colors flex items-center gap-0.5',
              trimStart ? 'bg-green-500 text-white' : isVideoLoaded ? 'bg-green-100 text-green-700 hover:bg-green-200' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
            title={trimStart ? `Start: ${formatTime(trimStart.time)} - ${trimStart.instruction}` : 'Set Start Point (S)'}
          >
            S{trimStart && <span className="opacity-80">{formatTime(trimStart.time)}</span>}
          </button>
          <button
            onClick={applyTrimEnd}
            disabled={!isVideoLoaded}
            className={clsx(
              'px-1.5 py-0.5 text-xs rounded transition-colors flex items-center gap-0.5',
              trimEnd ? 'bg-red-500 text-white' : isVideoLoaded ? 'bg-red-100 text-red-700 hover:bg-red-200' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
            )}
            title={trimEnd ? `End: ${formatTime(trimEnd.time)}` : 'Set End Point (E)'}
          >
            E{trimEnd && <span className="opacity-80">{formatTime(trimEnd.time)}</span>}
          </button>
          {(trimStart || trimEnd) && (
            <button onClick={clearTrimPoints} className="text-gray-400 hover:text-red-500 px-0.5" title="Clear trim points">
              <MdClose size={12} />
            </button>
          )}
        </div>
        {/* Add marker button */}
        <button
          onClick={openMarkerDialog}
          disabled={!isVideoLoaded}
          className={clsx(
            'px-1.5 py-0.5 text-xs rounded transition-colors',
            isVideoLoaded ? 'bg-purple-100 text-purple-700 hover:bg-purple-200' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
          title="Add marker (m)"
        >
          + Marker
        </button>
        {/* Speed selector */}
        <div className="flex items-center gap-1 ml-auto">
          {PLAYBACK_SPEEDS.map((speed) => {
            const speedEnabled = isVideoLoaded || (isDirectMcapMode && mcapPlayer.isReady);
            return (
              <button
                key={speed}
                onClick={() => changePlaybackSpeed(speed)}
                disabled={!speedEnabled}
                className={clsx(
                  'px-1.5 py-0.5 text-xs rounded transition-colors',
                  playbackSpeed === speed ? 'bg-blue-600 text-white' : speedEnabled ? 'bg-gray-200 text-gray-700 hover:bg-gray-300' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                )}
              >
                {speed}x
              </button>
            );
          })}
          <button
            onClick={() => setShowHelpModal(true)}
            className="ml-1 w-5 h-5 flex items-center justify-center rounded-full bg-gray-200 text-gray-600 hover:bg-gray-300 text-xs font-bold"
            title="Keyboard shortcuts (?)"
          >
            ?
          </button>
        </div>
      </div>
    </div>
  );
}

export default React.memo(TimelineControls);
