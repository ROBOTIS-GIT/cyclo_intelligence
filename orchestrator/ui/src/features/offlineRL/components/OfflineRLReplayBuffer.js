// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCheckCircle,
  MdErrorOutline,
  MdHourglassEmpty,
  MdInventory2,
  MdRemove,
  MdSearch,
  MdVideocam,
} from 'react-icons/md';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { RecordPhase } from '../../../constants/taskPhases';
import {
  selectInferenceRecordingControl,
  selectInferenceTaskInfo,
} from '../../tasks/taskSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import {
  selectOfflineRLReplayBufferPath,
  setOfflineRLReplayBufferPath,
} from '../offlineRLSlice';
import ReplayBufferVessel from './ReplayBufferVessel';
import OfflineRLEpisodeMediaModal from './OfflineRLEpisodeMediaModal';

const REPLAY_CAPACITY = 200;

const OUTCOME_STYLE = {
  success: {
    label: 'Success',
    icon: MdCheckCircle,
    className: 'border-[#b9cbb9] bg-[#e5eee4] text-[#3e6046]',
  },
  failure: {
    label: 'Fail',
    icon: MdErrorOutline,
    className: 'border-[#dec3bc] bg-[#f2e5e1] text-[#824a43]',
  },
  unlabeled: {
    label: 'Unlabeled',
    icon: MdInventory2,
    className: 'border-[#d9d2c5] bg-[#f0ece4] text-[#5e574d]',
  },
};

const numericIndices = (values) => (
  Array.isArray(values)
    ? values
      .map((value) => Number(value))
      .filter((value) => Number.isInteger(value) && value >= 0)
    : []
);

export function resolveReplayBufferPath(recordingFolder, taskNum) {
  const selected = String(recordingFolder || '').trim();
  if (selected) return selected.replace(/\/+$/, '');

  const sessionId = String(taskNum || '').trim();
  if (!sessionId) return '';
  const folderName = sessionId.startsWith('Task_')
    ? sessionId
    : `Task_${sessionId}_inference_MCAP`;
  return `${DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '')}/${folderName}`;
}

export function normalizeReplayEpisodes(datasetInfo = {}) {
  const episodesByIndex = new Map();
  const append = (values, outcome) => {
    numericIndices(values).forEach((index) => {
      episodesByIndex.set(index, { index, outcome });
    });
  };

  append(datasetInfo.unlabeled_episode_indices, 'unlabeled');
  append(datasetInfo.failure_episode_indices, 'failure');
  append(datasetInfo.success_episode_indices, 'success');

  return Array.from(episodesByIndex.values())
    .sort((left, right) => left.index - right.index);
}

function fallbackEpisodes(count) {
  const safeCount = Math.max(0, Math.min(REPLAY_CAPACITY, Number(count) || 0));
  return Array.from({ length: safeCount }, (_, index) => ({
    index,
    outcome: 'unlabeled',
  }));
}

function replayVideoUrl(episodePath, videoFile) {
  const file = String(videoFile || '').trim();
  if (!file) return '';
  if (/^(?:blob:|data:|https?:\/\/|\/files\/)/i.test(file)) return file;
  const base = String(episodePath || '').trim().replace(/\/+$/, '');
  const relative = file.replace(/^\/+/, '');
  if (relative.split('/').some((segment) => segment === '..')) return '';
  return base ? `/files${base}/${relative}` : '';
}

function fallbackVideoLabel(videoFile) {
  const filename = String(videoFile || '').split('/').pop() || 'Camera';
  return filename.replace(/\.mp4$/i, '').replace(/_/g, ' ');
}

function replayCameraPresentation(name, videoFile) {
  const candidate = `${String(name || '')} ${String(videoFile || '')}`.toLowerCase();
  if (candidate.includes('cam_left_wrist')) return { label: 'Left wrist', order: 0 };
  if (candidate.includes('cam_left_head')) return { label: 'Head', order: 1 };
  if (candidate.includes('cam_right_wrist')) return { label: 'Right wrist', order: 2 };
  return {
    label: String(name || '').trim() || fallbackVideoLabel(videoFile),
    order: 3,
  };
}

/** Convert the existing replay-data MP4 descriptor arrays into modal media. */
export function resolveMcapEpisodeMedia(replayData = {}, episodePath = '') {
  const segments = Array.isArray(replayData.video_segments)
    ? replayData.video_segments.filter((segment) => (
      Array.isArray(segment?.video_files) && segment.video_files.length > 0
    ))
    : [];

  const descriptorGroups = segments.length > 0
    ? segments.map((segment, segmentIndex) => ({
      files: segment.video_files,
      names: Array.isArray(segment.video_names) ? segment.video_names : [],
      fps: Array.isArray(segment.video_fps) ? segment.video_fps : [],
      segmentIndex,
      segmentName: String(segment.name || '').trim(),
      duration: Math.max(
        0,
        Number(segment.replay_end_s || 0) - Number(segment.replay_start_s || 0)
      ),
    }))
    : [{
      files: Array.isArray(replayData.video_files) ? replayData.video_files : [],
      names: Array.isArray(replayData.video_names) ? replayData.video_names : [],
      fps: Array.isArray(replayData.video_fps) ? replayData.video_fps : [],
      segmentIndex: 0,
      segmentName: '',
      duration: Math.max(0, Number(replayData.duration || 0)),
    }];

  const showSegmentName = descriptorGroups.length > 1;
  return descriptorGroups.flatMap((group) => group.files.map((file, cameraIndex) => {
    const url = replayVideoUrl(episodePath, file);
    if (!url) return null;
    const presentation = replayCameraPresentation(group.names[cameraIndex], file);
    const segmentLabel = group.segmentName || `Segment ${group.segmentIndex + 1}`;
    return {
      key: `${group.segmentIndex}-${String(file)}`,
      label: showSegmentName
        ? `${presentation.label} · ${segmentLabel}`
        : presentation.label,
      url,
      order: (group.segmentIndex * 10) + presentation.order,
      fromS: 0,
      ...(group.duration > 0 ? { toS: group.duration } : {}),
      ...(Number(group.fps[cameraIndex]) > 0
        ? { fps: Number(group.fps[cameraIndex]) }
        : {}),
    };
  }).filter(Boolean))
    .sort((left, right) => left.order - right.order)
    .map(({ order, ...item }) => item);
}

export function ReplayBufferStack({
  episodes = [],
  totalCount = 0,
  compositionEpisodes = null,
  compositionTotalCount = null,
  loading = false,
  error = '',
  onDelete = null,
  deletingIndex = null,
  deleteDisabled = false,
  onOpen = null,
  datasetLabel = 'MCAP episodes',
  listLabel = 'Replay Buffer episodes',
  compositionLabel = 'Buffer composition',
  managerLabel = 'Episode manager',
}) {
  const [query, setQuery] = useState('');
  const [outcomeFilter, setOutcomeFilter] = useState('all');
  const managerTotal = Math.max(0, Number(totalCount) || 0);
  const managerSuccessCount = episodes.filter(
    (item) => item.outcome === 'success'
  ).length;
  const managerFailureCount = episodes.filter(
    (item) => item.outcome === 'failure'
  ).length;
  const managerExplicitUnlabeledCount = episodes.filter(
    (item) => item.outcome !== 'success' && item.outcome !== 'failure'
  ).length;
  const managerUnlabeledCount = Math.max(
    managerExplicitUnlabeledCount,
    managerTotal - managerSuccessCount - managerFailureCount
  );
  const compositionRows = Array.isArray(compositionEpisodes)
    ? compositionEpisodes
    : episodes;
  const safeTotal = Math.max(
    0,
    Number(compositionTotalCount == null ? totalCount : compositionTotalCount) || 0
  );
  const successCount = compositionRows.filter(
    (item) => item.outcome === 'success'
  ).length;
  const failureCount = compositionRows.filter(
    (item) => item.outcome === 'failure'
  ).length;
  const explicitUnlabeledCount = compositionRows.filter(
    (item) => item.outcome !== 'success' && item.outcome !== 'failure'
  ).length;
  const unlabeledCount = Math.max(
    explicitUnlabeledCount,
    safeTotal - successCount - failureCount
  );
  const compositionTotal = Math.max(
    safeTotal,
    successCount + failureCount + explicitUnlabeledCount
  );
  const labeledCount = successCount + failureCount;
  const successRate = labeledCount > 0
    ? Math.round((successCount / labeledCount) * 100)
    : null;
  const successComposition = compositionTotal > 0
    ? Math.round((successCount / compositionTotal) * 100)
    : 0;
  const failureComposition = compositionTotal > 0
    ? Math.round((failureCount / compositionTotal) * 100)
    : 0;
  const unlabeledComposition = compositionTotal > 0
    ? Math.max(0, 100 - successComposition - failureComposition)
    : 0;
  const normalizedQuery = query.trim().toLowerCase();
  const filteredEpisodes = episodes.filter((episode) => {
    if (outcomeFilter !== 'all' && episode.outcome !== outcomeFilter) return false;
    if (!normalizedQuery) return true;
    return `episode_${String(episode.index).padStart(3, '0')}`
      .toLowerCase()
      .includes(normalizedQuery);
  });
  const filterOptions = [
    ['all', 'All', managerTotal],
    ['success', 'Success', managerSuccessCount],
    ['failure', 'Fail', managerFailureCount],
    ['unlabeled', 'Unlabeled', managerUnlabeledCount],
  ];
  // Capacity and outcome composition answer different questions. The cylinder
  // is scaled against the fixed buffer capacity; the bar uses stored episodes.
  const occupiedCount = Math.min(REPLAY_CAPACITY, compositionTotal);
  const capacityPercent = Math.round((occupiedCount / REPLAY_CAPACITY) * 100);

  return (
    <div
      className="pg-replay-layout min-w-0 gap-3"
      data-testid="replay-buffer-composition-layout"
    >
      <section
        className="flex min-h-0 max-w-full flex-col overflow-hidden rounded-xl border border-[#e2dbcf] bg-[#f8f5ef] p-2.5"
        aria-label={`${datasetLabel} composition`}
        data-testid="replay-buffer-composition"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="text-[14px] font-semibold text-[#39352e]">
              {compositionLabel}
            </div>
            <div className="mt-0.5 text-[12px] text-[#6b6459]">{datasetLabel}</div>
          </div>
          <span className="rounded-full border border-[#dcd5c9] bg-white px-2.5 py-1 text-[12px] font-semibold tabular-nums text-[#5d574e]">
            {safeTotal} / {REPLAY_CAPACITY}
          </span>
        </div>

        <div className="mt-1 pg-composition-detail min-h-0 min-w-0 flex-1 items-center gap-2">
          <ReplayBufferVessel
            counts={{ success: successCount, failure: failureCount, unlabeled: unlabeledCount }}
            capacity={REPLAY_CAPACITY}
            label={`${datasetLabel} buffer composition`}
          />

          <div className="flex min-w-0 flex-col justify-center gap-1.5 self-stretch py-1">
            <div
              className="grid grid-cols-2 gap-1.5"
              data-testid="replay-composition-stats"
            >
              <div className="px-1.5 py-2 text-center">
                <div className="text-xl font-bold tabular-nums text-[#3f3b35]">{safeTotal}</div>
                <div className="text-[12px] font-medium text-[#5e574d]">Episodes</div>
                <div className="mt-0.5 text-[12px] tabular-nums text-[#6b6459]">
                  {capacityPercent}% used
                </div>
              </div>
              <div className="border-l border-[#ded8cc] px-1.5 py-2 text-center">
                <span className="sr-only">
                  Success rate {successRate === null ? '—' : `${successRate}%`}
                </span>
                <div className="text-xl font-bold tabular-nums text-[#58705d]">
                  {successRate === null ? '—' : `${successRate}%`}
                </div>
                <div className="text-[12px] font-medium text-[#5e574d]">Success</div>
              </div>
            </div>
            <div className="min-w-0" data-testid="replay-outcome-summary">
              <div className="mb-1 flex flex-wrap items-center justify-between gap-x-2 text-[12px] text-[#6b6459]">
                <span>Outcome composition</span>
                <span className="tabular-nums">{compositionTotal} stored</span>
              </div>
              <div
                className="flex h-2.5 overflow-hidden rounded-full bg-[#e8e3da]"
                role="progressbar"
                aria-label={`${datasetLabel} success rate`}
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={successRate === null ? undefined : successRate}
              >
                <div className="h-full bg-[#75957a]" style={{ width: `${successComposition}%` }} />
                <div className="h-full bg-[#c98278]" style={{ width: `${failureComposition}%` }} />
                <div className="h-full bg-[#b7b1a8]" style={{ width: `${unlabeledComposition}%` }} />
              </div>
              <span className="sr-only">Success {successCount} · Fail {failureCount}</span>
              <div
                className="pg-outcome-legend mt-1.5 gap-1.5 text-[12px] text-[#756e63]"
                data-testid="replay-outcome-legend"
              >
                {[
                  ['#75957a', 'Success', successCount, successComposition],
                  ['#c98278', 'Fail', failureCount, failureComposition],
                  ['#b7b1a8', 'Unlabeled', unlabeledCount, unlabeledComposition],
                ].map(([color, label, count, percent]) => (
                  <div key={label} className="flex min-w-0 items-center gap-1">
                    <span className="h-2 w-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />
                    <span className="min-w-0 truncate font-semibold" title={label}>{label}</span>
                    <span className="ml-auto shrink-0 font-semibold tabular-nums text-[#514b42]">{count}</span>
                    <span className="sr-only">{percent}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
        {error && (
          <div className="mt-2 truncate text-[11px] text-[#a06b61]" title={error}>
            Refresh pending · {error}
          </div>
        )}
      </section>

      <section
        className="flex min-h-0 min-w-0 flex-col rounded-xl border border-[#e2dbcf] bg-white p-2.5"
        aria-label={managerLabel}
        data-testid="replay-buffer-episode-manager"
      >
        <div className="flex items-center justify-between gap-2">
          <div className="text-[14px] font-semibold text-[#39352e]">
            {managerLabel}
          </div>
          <span className="text-[12px] tabular-nums text-[#6b6459]">
            {filteredEpisodes.length} shown
          </span>
        </div>
        <label className="relative mt-1.5 block">
          <MdSearch
            size={14}
            className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2 text-[#9a9286]"
            aria-hidden="true"
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            aria-label={`Search ${datasetLabel}`}
            placeholder="Search episodes…"
            className="h-8 w-full rounded-lg border border-[#d5cdbf] bg-[#fbfaf7] pl-8 pr-2.5 text-[13px] text-[#39352e] placeholder:text-[#746d62] outline-none focus:border-[#879b89]"
          />
        </label>
        <div className="mt-1.5 grid grid-cols-4 gap-1" role="group" aria-label={`${datasetLabel} outcome filter`}>
          {filterOptions.map(([value, label, count]) => (
            <button
              key={value}
              type="button"
              onClick={() => setOutcomeFilter(value)}
              aria-pressed={outcomeFilter === value}
              className={clsx(
                'h-8 truncate rounded-lg border px-1.5 text-[12px] font-semibold transition-colors',
                outcomeFilter === value
                  ? 'border-[#7b9780] bg-[#e7efe6] text-[#3e6046]'
                  : 'border-transparent bg-transparent text-[#5e574d] hover:bg-[#f2eee6]'
              )}
            >
              {label} {count}
            </button>
          ))}
        </div>

        <div
          className="mt-1.5 h-[156px] min-h-[156px] max-h-[156px] flex-none space-y-1 overflow-y-auto overscroll-contain border-t border-[#ded8cc] bg-white py-1.5"
          role="list"
          aria-label={listLabel}
        >
          {filteredEpisodes.map((episode) => {
            const style = OUTCOME_STYLE[episode.outcome] || OUTCOME_STYLE.unlabeled;
            const Icon = style.icon;
            return (
              <div
                key={`${episode.index}-${episode.outcome}`}
                role="listitem"
                className={clsx(
                  'flex h-11 items-center gap-2 rounded-md border-b border-[#ece7dd] bg-white px-2 text-[14px] font-medium',
                  onOpen && 'cursor-pointer transition-colors hover:bg-[#f0f4ed] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px] focus-visible:outline-[#72917a]'
                )}
                onClick={onOpen ? () => onOpen(episode) : undefined}
                onKeyDown={onOpen ? (event) => {
                  if (event.target !== event.currentTarget) return;
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault();
                    onOpen(episode);
                  }
                } : undefined}
                tabIndex={onOpen ? 0 : undefined}
                aria-label={onOpen ? `Open episode ${episode.index} video` : undefined}
              >
                <Icon size={14} className="shrink-0 text-[#6b6459]" />
                <span className="min-w-0 flex-1 truncate font-mono text-[#302d27]" title={`episode_${String(episode.index).padStart(3, '0')}`}>
                  episode_{String(episode.index).padStart(3, '0')}
                </span>
                <span className={clsx('shrink-0 rounded-full border px-1.5 py-0.5 text-[12px] font-semibold', style.className)}>
                  {style.label}
                </span>
                {onOpen && (
                  <span
                    className="flex h-6 shrink-0 items-center gap-1 text-[12px] font-medium text-[#5e574d]"
                    aria-hidden="true"
                  >
                    <MdVideocam size={11} aria-hidden="true" /> View
                  </span>
                )}
                {onDelete && (
                  <button
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation();
                      onDelete(episode.index);
                    }}
                    disabled={deleteDisabled || deletingIndex !== null}
                    className="ml-auto grid h-6 w-6 shrink-0 place-items-center rounded-full bg-[#b96862] text-white transition-colors hover:bg-[#a6534e] focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#824a43] disabled:cursor-not-allowed disabled:opacity-45"
                    aria-label={`Delete episode ${episode.index}`}
                    title={`Delete episode_${String(episode.index).padStart(3, '0')}`}
                  >
                    <MdRemove size={11} />
                  </button>
                )}
              </div>
            );
          })}

          {!filteredEpisodes.length && (
            <div className="flex h-full min-h-[120px] flex-col items-center justify-center gap-1 text-center text-[12px] text-[#6b6459]">
              <MdHourglassEmpty size={15} />
              <span>
                {loading
                  ? 'Loading episodes…'
                  : episodes.length
                    ? 'No episodes match this filter'
                    : 'No saved episodes'}
              </span>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

export default function OfflineRLReplayBuffer({ isActive = true }) {
  const dispatch = useDispatch();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const recordingControl = useSelector(
    selectInferenceRecordingControl,
    shallowEqual
  );
  const recordStatus = useSelector((state) => state.tasks.recordStatus, shallowEqual);
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus?.status || 'idle'
  );
  const retainedReplayPath = useSelector(selectOfflineRLReplayBufferPath);
  const {
    getDatasetInfo,
    getReplayData,
    sendEditDatasetCommand,
  } = useRosServiceCaller();
  const [snapshot, setSnapshot] = useState({
    path: '',
    episodes: [],
    totalCount: 0,
    error: '',
    loaded: false,
  });
  const [loading, setLoading] = useState(false);
  const [deletingIndex, setDeletingIndex] = useState(null);
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [episodeMediaState, setEpisodeMediaState] = useState({
    media: [],
    jointData: null,
    loading: false,
    error: '',
  });
  const mediaRequestRef = useRef(0);

  const liveReplayPath = useMemo(() => resolveReplayBufferPath(
    taskInfo.recordingFolder,
    recordStatus.taskNum
  ), [recordStatus.taskNum, taskInfo.recordingFolder]);
  const replayPath = liveReplayPath || retainedReplayPath;

  useEffect(() => {
    if (liveReplayPath && liveReplayPath !== retainedReplayPath) {
      dispatch(setOfflineRLReplayBufferPath(liveReplayPath));
    }
  }, [dispatch, liveReplayPath, retainedReplayPath]);

  const readReplayBuffer = useCallback(async (path) => {
    const result = await getDatasetInfo(path);
    if (!result?.success || !result.dataset_info) {
      throw new Error(result?.message || 'Replay Buffer status is unavailable');
    }
    const datasetInfo = result.dataset_info;
    return {
      path,
      episodes: normalizeReplayEpisodes(datasetInfo),
      totalCount: Number(datasetInfo.episode_count || 0),
      error: '',
      loaded: true,
    };
  }, [getDatasetInfo]);

  useEffect(() => {
    if (
      !isActive ||
      !replayPath ||
      conversionStatus === 'running' ||
      recordStatus.recordPhase !== RecordPhase.READY ||
      recordingControl.lifecycleLocked
    ) {
      return undefined;
    }

    let cancelled = false;
    setLoading(true);
    readReplayBuffer(replayPath)
      .then((nextSnapshot) => {
        if (!cancelled) setSnapshot(nextSnapshot);
      })
      .catch((fetchError) => {
        if (cancelled) return;
        setSnapshot((current) => ({
          ...current,
          path: replayPath,
          error: fetchError?.message || 'Replay Buffer refresh failed',
        }));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [
    isActive,
    recordStatus.recordPhase,
    recordingControl.episodeCount,
    recordingControl.lifecycleLocked,
    conversionStatus,
    readReplayBuffer,
    replayPath,
  ]);

  const handleDelete = useCallback(async (
    episodeIndex,
    { skipConfirm = false } = {}
  ) => {
    if (
      deletingIndex !== null ||
      recordingControl.lifecycleLocked ||
      !replayPath ||
      !snapshot.loaded ||
      snapshot.path !== replayPath
    ) {
      return false;
    }
    const paddedIndex = String(episodeIndex).padStart(3, '0');
    if (!skipConfirm && !window.confirm(
      `Delete episode_${paddedIndex}? This permanently removes the saved episode.`
    )) {
      return false;
    }

    setDeletingIndex(episodeIndex);
    try {
      const result = await sendEditDatasetCommand('delete', {
        deleteTaskDir: replayPath,
        deleteEpisodeNums: [episodeIndex],
        deleteCompact: false,
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Episode deletion failed');
      }
      setSnapshot((current) => ({
        ...current,
        episodes: current.episodes.filter((episode) => episode.index !== episodeIndex),
        totalCount: Math.max(0, current.totalCount - 1),
        error: '',
      }));
      toast.success(`episode_${paddedIndex} deleted`);
      try {
        const nextSnapshot = await readReplayBuffer(replayPath);
        setSnapshot(nextSnapshot);
      } catch (refreshError) {
        setSnapshot((current) => ({
          ...current,
          error: refreshError?.message || 'Replay Buffer refresh failed',
        }));
      }
      return true;
    } catch (deleteError) {
      toast.error(deleteError?.message || 'Episode deletion failed');
      return false;
    } finally {
      setDeletingIndex(null);
    }
  }, [
    deletingIndex,
    recordingControl.lifecycleLocked,
    readReplayBuffer,
    replayPath,
    sendEditDatasetCommand,
    snapshot.loaded,
    snapshot.path,
  ]);

  const closeEpisodeMedia = useCallback(() => {
    mediaRequestRef.current += 1;
    setSelectedEpisode(null);
    setEpisodeMediaState({
      media: [],
      jointData: null,
      loading: false,
      error: '',
    });
  }, []);

  const openEpisodeMedia = useCallback(async (episode) => {
    if (!replayPath || !episode || !Number.isInteger(Number(episode.index))) return;
    const requestId = mediaRequestRef.current + 1;
    mediaRequestRef.current = requestId;
    const normalizedEpisode = {
      ...episode,
      index: Number(episode.index),
    };
    const episodePath = `${replayPath.replace(/\/+$/, '')}/${normalizedEpisode.index}`;
    setSelectedEpisode(normalizedEpisode);
    setEpisodeMediaState({
      media: [],
      jointData: null,
      loading: true,
      error: '',
    });

    try {
      const replayData = await getReplayData(episodePath);
      if (mediaRequestRef.current !== requestId) return;
      if (!replayData?.success) {
        throw new Error(replayData?.message || 'Episode video is unavailable');
      }
      const media = resolveMcapEpisodeMedia(replayData, episodePath);
      setEpisodeMediaState({
        media,
        jointData: replayData,
        loading: false,
        error: media.length ? '' : 'No playable MP4 video was found for this episode.',
      });
    } catch (mediaError) {
      if (mediaRequestRef.current !== requestId) return;
      setEpisodeMediaState({
        media: [],
        jointData: null,
        loading: false,
        error: mediaError?.message || 'Episode video could not be loaded',
      });
    }
  }, [getReplayData, replayPath]);

  const deleteSelectedEpisode = useCallback(async () => {
    if (!selectedEpisode) return false;
    const deleted = await handleDelete(selectedEpisode.index, {
      skipConfirm: true,
    });
    if (deleted) closeEpisodeMedia();
    return deleted;
  }, [closeEpisodeMedia, handleDelete, selectedEpisode]);

  const snapshotMatches = snapshot.path === replayPath;
  const totalCount = snapshotMatches
    ? snapshot.totalCount
    : recordingControl.episodeCount;
  const episodes = snapshotMatches && snapshot.episodes.length
    ? snapshot.episodes
    : fallbackEpisodes(totalCount);

  return (
    <>
      <ReplayBufferStack
        episodes={episodes}
        totalCount={totalCount}
        loading={loading}
        error={snapshotMatches ? snapshot.error : ''}
        onOpen={replayPath && getReplayData ? openEpisodeMedia : null}
        onDelete={snapshotMatches && snapshot.loaded ? handleDelete : null}
        deletingIndex={deletingIndex}
        deleteDisabled={recordingControl.lifecycleLocked}
      />
      <OfflineRLEpisodeMediaModal
        open={Boolean(selectedEpisode)}
        sourceLabel="MCAP Replay Buffer"
        episode={selectedEpisode}
        media={episodeMediaState.media}
        jointData={episodeMediaState.jointData}
        loading={episodeMediaState.loading}
        error={episodeMediaState.error}
        onBack={closeEpisodeMedia}
        onDelete={deleteSelectedEpisode}
        deletePending={deletingIndex !== null}
        deleteDisabled={recordingControl.lifecycleLocked || !snapshot.loaded}
      />
    </>
  );
}
