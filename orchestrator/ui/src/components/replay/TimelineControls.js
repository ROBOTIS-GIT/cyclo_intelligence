import React, { useMemo } from 'react';
import clsx from 'clsx';
import {
  MdPause,
  MdPlayArrow,
  MdReplay,
  MdSkipNext,
  MdSkipPrevious,
} from 'react-icons/md';
import { formatTime } from '../../utils/chartUtils';

const SEGMENT_STYLES = [
  'bg-blue-200/80 border-blue-500',
  'bg-emerald-200/80 border-emerald-500',
  'bg-amber-200/80 border-amber-500',
  'bg-purple-200/80 border-purple-500',
  'bg-rose-200/80 border-rose-500',
  'bg-cyan-200/80 border-cyan-500',
];

const normalizeSegments = (segments) => {
  if (!Array.isArray(segments)) return [];

  return segments
    .map((segment, index) => {
      const start = Number(segment.frame_duration?.[0]);
      const end = Number(segment.frame_duration?.[1]);
      if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
      return {
        ...segment,
        start,
        end: Math.max(start, end),
        label: segment.sub_task_instruction || `Segment ${index + 1}`,
        displayIndex: Number.isInteger(segment.sub_task_index)
          ? segment.sub_task_index + 1
          : index + 1,
      };
    })
    .filter(Boolean)
    .sort((a, b) => a.start - b.start);
};

function TimelineControls({
  currentTime,
  duration,
  isPlaying,
  isVideoLoaded,
  isDirectMcapMode,
  mcapPlayer,
  replaySegments,
  handleSeek,
  togglePlayPause,
  restartPlayback,
  changePlaybackSpeed,
  playbackSpeed,
  PLAYBACK_SPEEDS,
  videoFiles,
  seekToTime,
  seekAndPlay,
}) {
  const segments = useMemo(() => normalizeSegments(replaySegments), [replaySegments]);
  const activeSegmentIndex = segments.findIndex((segment, index) => {
    const isLast = index === segments.length - 1;
    return currentTime >= segment.start && (currentTime < segment.end || isLast);
  });
  const activeSegment = activeSegmentIndex >= 0 ? segments[activeSegmentIndex] : null;
  const progressPercent = duration > 0 ? Math.max(0, Math.min(100, (currentTime / duration) * 100)) : 0;
  const controlsEnabled = isVideoLoaded || (isDirectMcapMode && mcapPlayer?.isReady);

  const seekToSegment = (index, autoplay = false) => {
    const target = segments[index];
    if (!target) return;
    if (autoplay) {
      seekAndPlay(target.start);
    } else {
      seekToTime(target.start);
    }
  };

  return (
    <div className="p-3 bg-white border-t shadow-sm">
      <div
        className={clsx(
          'h-4 bg-gray-200 rounded-full mb-3 cursor-pointer relative overflow-hidden',
          { 'opacity-50 cursor-not-allowed': !controlsEnabled }
        )}
        onClick={controlsEnabled ? handleSeek : undefined}
      >
        {segments.map((segment, index) => {
          const left = duration > 0 ? (segment.start / duration) * 100 : 0;
          const width = duration > 0 ? ((segment.end - segment.start) / duration) * 100 : 0;
          return (
            <div
              key={`${segment.start}-${segment.end}-${index}`}
              className={clsx(
                'absolute top-0 h-full border-l pointer-events-none',
                SEGMENT_STYLES[index % SEGMENT_STYLES.length],
                index === activeSegmentIndex ? 'opacity-95' : 'opacity-60'
              )}
              style={{
                left: `${Math.max(0, Math.min(100, left))}%`,
                width: `${Math.max(0.35, Math.min(100, width))}%`,
              }}
              aria-hidden="true"
            />
          );
        })}

        <div
          className="absolute left-0 top-0 h-full bg-blue-600/25 pointer-events-none"
          style={{ width: `${progressPercent}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-blue-600 rounded-full shadow z-20 pointer-events-none"
          style={{ left: `calc(${progressPercent}% - 8px)` }}
        />
      </div>

      <div className="flex items-center gap-3">
        <button
          onClick={restartPlayback}
          disabled={!controlsEnabled}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            controlsEnabled ? 'bg-gray-200 text-gray-700 hover:bg-gray-300' : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
          title="Restart from beginning"
        >
          <MdReplay size={18} />
        </button>
        <button
          onClick={togglePlayPause}
          disabled={!controlsEnabled}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            controlsEnabled ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          )}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? <MdPause size={20} /> : <MdPlayArrow size={20} />}
        </button>
        <button
          onClick={() => seekToSegment(Math.max(0, activeSegmentIndex - 1))}
          disabled={!controlsEnabled || activeSegmentIndex <= 0}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            controlsEnabled && activeSegmentIndex > 0
              ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
          title="Previous segment"
        >
          <MdSkipPrevious size={18} />
        </button>
        <button
          onClick={() => seekToSegment(activeSegmentIndex + 1)}
          disabled={!controlsEnabled || activeSegmentIndex < 0 || activeSegmentIndex >= segments.length - 1}
          className={clsx(
            'p-1.5 rounded-full transition-colors',
            controlsEnabled && activeSegmentIndex >= 0 && activeSegmentIndex < segments.length - 1
              ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              : 'bg-gray-100 text-gray-400 cursor-not-allowed'
          )}
          title="Next segment"
        >
          <MdSkipNext size={18} />
        </button>

        <span className="text-xs text-gray-600 tabular-nums">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        <span className="text-xs text-gray-400">
          ({videoFiles.length} cam{videoFiles.length > 1 ? 's' : ''})
        </span>

        {activeSegment && (
          <button
            onClick={() => seekToSegment(activeSegmentIndex, true)}
            className="min-w-0 max-w-md flex items-center gap-1.5 px-2 py-0.5 bg-blue-50 hover:bg-blue-100 rounded text-xs transition-colors"
            title={activeSegment.label}
          >
            <span className="text-blue-600 font-bold whitespace-nowrap">
              Segment {activeSegment.displayIndex}
            </span>
            <span className="text-blue-900 truncate">{activeSegment.label}</span>
          </button>
        )}

        <div className="flex items-center gap-1 ml-auto">
          {PLAYBACK_SPEEDS.map((speed) => (
            <button
              key={speed}
              onClick={() => changePlaybackSpeed(speed)}
              disabled={!controlsEnabled}
              className={clsx(
                'px-1.5 py-0.5 text-xs rounded transition-colors',
                playbackSpeed === speed
                  ? 'bg-blue-600 text-white'
                  : controlsEnabled
                    ? 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                    : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              )}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default React.memo(TimelineControls);
