import React, { useMemo } from 'react';
import clsx from 'clsx';
import {
  MdClose,
  MdKeyboardArrowDown,
  MdKeyboardArrowUp,
  MdPlayArrow,
} from 'react-icons/md';
import { formatTime, formatFileSize, formatDateTime } from '../../utils/chartUtils';

const SEGMENT_ACCENTS = [
  'border-l-blue-500 bg-blue-50 text-blue-800',
  'border-l-emerald-500 bg-emerald-50 text-emerald-800',
  'border-l-amber-500 bg-amber-50 text-amber-800',
  'border-l-purple-500 bg-purple-50 text-purple-800',
  'border-l-rose-500 bg-rose-50 text-rose-800',
  'border-l-cyan-500 bg-cyan-50 text-cyan-800',
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

const basename = (path) => {
  if (!path) return '';
  return String(path).split('/').filter(Boolean).pop() || path;
};

const formatRosbagDuration = (durationNs) => {
  const seconds = Number(durationNs || 0) / 1e9;
  return seconds > 0 ? formatTime(seconds) : '--';
};

function SidebarPanel({
  taskPath,
  recordingDate,
  robotType,
  fileSizeBytes,
  duration,
  frameCounts,
  replaySegments,
  currentTime,
  videoCount,
  jointCount,
  actionCount,
  rosbagList,
  currentBagIndex,
  isBusy,
  navigateRosbag,
  handleSelectBag,
  seekToTime,
  seekAndPlay,
  onClose,
}) {
  const segments = useMemo(() => normalizeSegments(replaySegments), [replaySegments]);
  const activeSegmentIndex = segments.findIndex((segment, index) => {
    const isLast = index === segments.length - 1;
    return currentTime >= segment.start && (currentTime < segment.end || isLast);
  });
  const activeSegment = activeSegmentIndex >= 0 ? segments[activeSegmentIndex] : null;
  const frameEntries = frameCounts ? Object.entries(frameCounts) : [];

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="flex-shrink-0 px-3 py-2 border-b bg-gray-50">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-gray-800">Episodes</h3>
            <div className="text-[11px] text-gray-500 truncate" title={taskPath}>
              {basename(taskPath) || 'No task selected'}
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="w-7 h-7 flex items-center justify-center rounded hover:bg-gray-200 text-gray-500 transition-colors"
              title="Hide episodes"
            >
              <MdClose size={18} />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2">
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="p-2 border-b bg-gray-50 flex-shrink-0">
            <div className="flex items-center justify-between mb-1.5">
              <h4 className="text-xs font-semibold text-gray-700">Episode List</h4>
              <span className="text-[10px] text-gray-500">
                {rosbagList.length > 0 ? `${currentBagIndex + 1} / ${rosbagList.length}` : '0 / 0'}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => navigateRosbag('prev')}
                disabled={currentBagIndex <= 0 || isBusy}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                  currentBagIndex <= 0 || isBusy
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                )}
              >
                <MdKeyboardArrowUp size={16} />
                Prev
              </button>
              <button
                onClick={() => navigateRosbag('next')}
                disabled={currentBagIndex >= rosbagList.length - 1 || isBusy}
                className={clsx(
                  'flex-1 flex items-center justify-center gap-1 px-2 py-1 rounded text-xs transition-colors',
                  currentBagIndex >= rosbagList.length - 1 || isBusy
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                    : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                )}
              >
                Next
                <MdKeyboardArrowDown size={16} />
              </button>
            </div>
          </div>
          <div className="max-h-80 overflow-y-auto">
            {rosbagList.length > 0 ? (
              rosbagList.map((bag, index) => (
                <button
                  key={bag.path}
                  onClick={() => handleSelectBag(bag.path)}
                  disabled={isBusy}
                  className={clsx(
                    'w-full text-left px-2 py-1.5 border-b border-gray-100 transition-colors text-xs',
                    index === currentBagIndex ? 'bg-blue-50 border-l-4 border-l-blue-500' : 'hover:bg-gray-50',
                    isBusy && 'cursor-not-allowed opacity-50'
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="font-medium text-gray-800 truncate">{bag.name}</div>
                    <div className="text-[10px] text-gray-500 tabular-nums whitespace-nowrap">
                      {formatRosbagDuration(bag.duration_ns)}
                    </div>
                  </div>
                  <div className="mt-0.5 flex items-center gap-1.5 text-[10px]">
                    <span className={bag.has_videos ? 'text-green-600' : 'text-gray-400'}>
                      {bag.has_videos ? 'videos ready' : 'no videos'}
                    </span>
                    {index === currentBagIndex && <span className="text-blue-600 font-medium">selected</span>}
                  </div>
                </button>
              ))
            ) : (
              <div className="text-[10px] text-gray-400 text-center py-4">
                Select a task folder with episodes
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="p-2 border-b bg-gray-50">
            <h4 className="text-xs font-semibold text-gray-700">Episode Summary</h4>
          </div>
          <div className="p-2 space-y-1.5 text-xs">
            <div className="flex justify-between gap-2">
              <span className="text-gray-500">Recorded</span>
              <span className="text-gray-700 font-medium text-right">
                {formatDateTime(recordingDate)}
              </span>
            </div>
            {robotType && (
              <div className="flex justify-between gap-2">
                <span className="text-gray-500">Robot</span>
                <span className="text-gray-700 font-medium truncate">{robotType}</span>
              </div>
            )}
            <div className="flex justify-between gap-2">
              <span className="text-gray-500">Duration</span>
              <span className="text-gray-700 font-medium">{formatTime(duration)}</span>
            </div>
            <div className="flex justify-between gap-2">
              <span className="text-gray-500">Size</span>
              <span className="text-gray-700 font-medium">{formatFileSize(fileSizeBytes)}</span>
            </div>
            <div className="grid grid-cols-3 gap-1 pt-1.5 border-t mt-1.5">
              <div className="rounded bg-gray-50 px-1.5 py-1">
                <div className="text-[10px] text-gray-400">Cameras</div>
                <div className="text-gray-700 font-semibold">{videoCount}</div>
              </div>
              <div className="rounded bg-gray-50 px-1.5 py-1">
                <div className="text-[10px] text-gray-400">Joints</div>
                <div className="text-gray-700 font-semibold">{jointCount}</div>
              </div>
              <div className="rounded bg-gray-50 px-1.5 py-1">
                <div className="text-[10px] text-gray-400">Actions</div>
                <div className="text-gray-700 font-semibold">{actionCount}</div>
              </div>
            </div>
            {frameEntries.length > 0 && (
              <div className="pt-1.5 border-t mt-1.5">
                <div className="text-gray-500 mb-1">Frames</div>
                <div className="space-y-0.5 max-h-24 overflow-y-auto">
                  {frameEntries.map(([camera, count]) => (
                    <div key={camera} className="flex justify-between gap-2 pl-1">
                      <span className="text-gray-500 truncate" title={camera}>{camera}</span>
                      <span className="text-gray-700 tabular-nums">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="p-2 border-b bg-gray-50">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold text-gray-700">Current Segment</h4>
              <span className="text-[10px] text-gray-500">
                {activeSegment ? `${activeSegmentIndex + 1} / ${segments.length}` : `0 / ${segments.length}`}
              </span>
            </div>
          </div>
          <div className="p-2 text-xs">
            {activeSegment ? (
              <button
                onClick={() => seekToTime(activeSegment.start)}
                className="w-full text-left rounded border border-blue-100 bg-blue-50 px-2 py-1.5 hover:bg-blue-100 transition-colors"
                title={activeSegment.label}
              >
                <div className="flex items-center justify-between gap-2 mb-0.5">
                  <span className="font-semibold text-blue-800 truncate">
                    {activeSegment.label}
                  </span>
                  <span className="text-blue-600 tabular-nums whitespace-nowrap">
                    {formatTime(activeSegment.start)}
                  </span>
                </div>
                <div className="text-[10px] text-blue-700">
                  {formatTime(activeSegment.start)} - {formatTime(activeSegment.end)}
                </div>
              </button>
            ) : (
              <div className="text-gray-400 text-center py-2">No segment metadata</div>
            )}
          </div>
        </div>

        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="p-2 border-b bg-gray-50">
            <div className="flex items-center justify-between">
              <h4 className="text-xs font-semibold text-gray-700">Segments</h4>
              <span className="text-[10px] text-gray-500">{segments.length}</span>
            </div>
          </div>
          <div className="p-1.5 max-h-64 overflow-y-auto">
            {segments.length > 0 ? (
              <div className="space-y-1">
                {segments.map((segment, index) => {
                  const active = index === activeSegmentIndex;
                  const accent = SEGMENT_ACCENTS[index % SEGMENT_ACCENTS.length];
                  return (
                    <div
                      key={`${segment.start}-${segment.end}-${index}`}
                      className={clsx(
                        'group rounded border border-gray-100 border-l-4 transition-colors',
                        active ? accent : 'border-l-gray-300 hover:bg-gray-50'
                      )}
                    >
                      <div className="px-2 py-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <button
                            type="button"
                            onClick={() => seekToTime(segment.start)}
                            className={clsx('min-w-0 flex-1 text-left text-xs font-medium truncate', active ? '' : 'text-gray-700')}
                            title={segment.label}
                          >
                            {segment.displayIndex}. {segment.label}
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              seekAndPlay(segment.start);
                            }}
                            className={clsx(
                              'opacity-0 group-hover:opacity-100 flex-shrink-0 w-5 h-5 rounded-full flex items-center justify-center transition-opacity',
                              active ? 'bg-white/80 text-blue-700' : 'bg-gray-100 text-gray-600 hover:bg-blue-100 hover:text-blue-700'
                            )}
                            title="Play from segment start"
                          >
                            <MdPlayArrow size={14} />
                          </button>
                        </div>
                        <div className="mt-0.5 text-[10px] text-gray-500 tabular-nums">
                          {formatTime(segment.start)} - {formatTime(segment.end)}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="text-[10px] text-gray-400 text-center py-3">
                Segment metadata is empty
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default React.memo(SidebarPanel);
