// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCheckCircle,
  MdErrorOutline,
  MdHourglassEmpty,
  MdInventory2,
  MdRemove,
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

const REPLAY_CAPACITY = 200;

const OUTCOME_STYLE = {
  success: {
    label: 'Success',
    icon: MdCheckCircle,
    className: 'border-[#b9cbb9] bg-[#e5eee4] text-[#5f7664]',
  },
  failure: {
    label: 'Fail',
    icon: MdErrorOutline,
    className: 'border-[#dec3bc] bg-[#f2e5e1] text-[#95635b]',
  },
  unlabeled: {
    label: 'Unlabeled',
    icon: MdInventory2,
    className: 'border-[#d9d2c5] bg-[#f0ece4] text-[#756e63]',
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

export function ReplayBufferStack({
  episodes = [],
  totalCount = 0,
  loading = false,
  error = '',
  onDelete = null,
  deletingIndex = null,
  deleteDisabled = false,
  datasetLabel = 'MCAP episodes',
  listLabel = 'Replay Buffer episodes',
}) {
  const safeTotal = Math.max(0, Number(totalCount) || 0);
  const successCount = episodes.filter((item) => item.outcome === 'success').length;
  const failureCount = episodes.filter((item) => item.outcome === 'failure').length;
  const labeledCount = successCount + failureCount;
  const successRate = labeledCount > 0
    ? Math.round((successCount / labeledCount) * 100)
    : null;
  const failureRate = successRate === null ? 0 : 100 - successRate;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-1.5 flex items-center justify-between gap-2 text-[9px] text-[#8b8479]">
        <span>{datasetLabel}</span>
        <span className="font-semibold tabular-nums text-[#514b42]">
          {safeTotal} / {REPLAY_CAPACITY}
        </span>
      </div>

      <div
        className="min-h-[104px] flex-1 space-y-1 overflow-y-auto rounded-lg border border-[#e4ded3] bg-[#f6f2ea] p-1.5"
        role="list"
        aria-label={listLabel}
      >
        {episodes.map((episode) => {
          const style = OUTCOME_STYLE[episode.outcome] || OUTCOME_STYLE.unlabeled;
          const Icon = style.icon;
          return (
            <div
              key={`${episode.index}-${episode.outcome}`}
              role="listitem"
              className={clsx(
                'flex h-6 items-center gap-1.5 rounded-md border px-2 text-[9px] font-medium shadow-sm',
                style.className
              )}
            >
              <Icon size={11} className="shrink-0" />
              <span className="min-w-0 flex-1 truncate font-mono">
                episode_{String(episode.index).padStart(3, '0')}
              </span>
              <span className="shrink-0 font-semibold">{style.label}</span>
              {onDelete && (
                <button
                  type="button"
                  onClick={() => onDelete(episode.index)}
                  disabled={deleteDisabled || deletingIndex !== null}
                  className="grid h-4 w-4 shrink-0 place-items-center rounded-full bg-[#b96862] text-white transition-colors hover:bg-[#a6534e] disabled:cursor-not-allowed disabled:opacity-45"
                  aria-label={`Delete episode ${episode.index}`}
                  title={`Delete episode_${String(episode.index).padStart(3, '0')}`}
                >
                  <MdRemove size={11} />
                </button>
              )}
            </div>
          );
        })}

        {!episodes.length && (
          <div className="flex h-full flex-col items-center justify-center gap-1 text-center text-[9px] text-[#9a9286]">
            <MdHourglassEmpty size={15} />
            <span>{loading ? 'Loading episodes…' : 'No saved episodes'}</span>
          </div>
        )}
      </div>

      <div className="mt-2 flex items-center justify-between text-[9px] text-[#8b8479]">
        <span>Success {successCount} · Fail {failureCount}</span>
        <span className={error ? 'truncate text-[#a06b61]' : 'font-semibold text-[#5f7664]'} title={error || undefined}>
          {error ? 'Refresh pending' : `Success rate ${successRate === null ? '—' : `${successRate}%`}`}
        </span>
      </div>
      <div
        className="mt-1 flex h-1.5 overflow-hidden rounded-full bg-[#e8e3da]"
        role="progressbar"
        aria-label={`${datasetLabel} success rate`}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-valuenow={successRate === null ? undefined : successRate}
      >
        <div
          className="h-full bg-[#72917a] transition-[width] duration-300"
          style={{ width: `${successRate || 0}%` }}
        />
        <div
          className="h-full bg-[#c78177] transition-[width] duration-300"
          style={{ width: `${failureRate}%` }}
        />
      </div>
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
  const { getDatasetInfo, sendEditDatasetCommand } = useRosServiceCaller();
  const [snapshot, setSnapshot] = useState({
    path: '',
    episodes: [],
    totalCount: 0,
    error: '',
    loaded: false,
  });
  const [loading, setLoading] = useState(false);
  const [deletingIndex, setDeletingIndex] = useState(null);

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

  const handleDelete = useCallback(async (episodeIndex) => {
    if (
      deletingIndex !== null ||
      recordingControl.lifecycleLocked ||
      !replayPath ||
      !snapshot.loaded ||
      snapshot.path !== replayPath
    ) {
      return;
    }
    const paddedIndex = String(episodeIndex).padStart(3, '0');
    if (!window.confirm(
      `Delete episode_${paddedIndex}? This permanently removes the saved episode.`
    )) {
      return;
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
    } catch (deleteError) {
      toast.error(deleteError?.message || 'Episode deletion failed');
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

  const snapshotMatches = snapshot.path === replayPath;
  const totalCount = snapshotMatches
    ? snapshot.totalCount
    : recordingControl.episodeCount;
  const episodes = snapshotMatches && snapshot.episodes.length
    ? snapshot.episodes
    : fallbackEpisodes(totalCount);

  return (
    <ReplayBufferStack
      episodes={episodes}
      totalCount={totalCount}
      loading={loading}
      error={snapshotMatches ? snapshot.error : ''}
      onDelete={snapshotMatches && snapshot.loaded ? handleDelete : null}
      deletingIndex={deletingIndex}
      deleteDisabled={recordingControl.lifecycleLocked}
    />
  );
}
