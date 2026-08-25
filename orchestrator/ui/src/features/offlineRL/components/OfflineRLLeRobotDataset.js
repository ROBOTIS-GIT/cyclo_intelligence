// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { useDispatch, useSelector } from 'react-redux';
import { MdRefresh } from 'react-icons/md';
import {
  deleteOfflineRLDatasetEpisodes,
  getOfflineRLDatasetInfo,
  getOfflineRLDatasets,
  getOfflineRLStatus,
} from '../../../utils/offlineRlApi';
import {
  selectOfflineRLConversionDestinationPath,
  selectOfflineRLConvertedDatasetPaths,
  selectOfflineRLDatasetPath,
  selectOfflineRLDatasetSelections,
  setOfflineRLDatasetPreview,
  setOfflineRLDatasetSelection,
  setOfflineRLDatasetSelections,
} from '../offlineRLSlice';
import { ReplayBufferStack } from './OfflineRLReplayBuffer';

const outcomeFromEpisode = (episode) => {
  const value = String(episode?.outcome || '').trim().toLowerCase();
  if (value === 'success' || value === 'successful') return 'success';
  if (value === 'failure' || value === 'fail' || value === 'failed') return 'failure';
  if (episode?.episode_success === true || episode?.success === true) return 'success';
  if (episode?.episode_success === false || episode?.success === false) return 'failure';
  return 'unlabeled';
};

export function normalizeLeRobotEpisodes(datasetInfo = {}) {
  if (Array.isArray(datasetInfo.episodes)) {
    return datasetInfo.episodes
      .map((episode, fallbackIndex) => ({
        index: Number(episode?.index ?? episode?.episode_index ?? fallbackIndex),
        outcome: outcomeFromEpisode(episode),
      }))
      .filter((episode) => Number.isInteger(episode.index) && episode.index >= 0)
      .sort((left, right) => left.index - right.index);
  }

  const rows = [];
  const append = (indices, outcome) => {
    (Array.isArray(indices) ? indices : []).forEach((index) => {
      const numericIndex = Number(index);
      if (Number.isInteger(numericIndex) && numericIndex >= 0) {
        rows.push({ index: numericIndex, outcome });
      }
    });
  };
  append(datasetInfo.success_episode_indices, 'success');
  append(datasetInfo.failure_episode_indices, 'failure');
  append(datasetInfo.unlabeled_episode_indices, 'unlabeled');
  return rows.sort((left, right) => left.index - right.index);
}

const emptySnapshot = (path = '') => ({
  path,
  episodes: [],
  totalCount: 0,
  version: '',
  fps: 0,
  error: '',
  loaded: false,
});

export function dataEpochFromDataset(dataset = {}) {
  const provenanceEpoch = dataset?.data_epoch_provenance?.data_epoch;
  const rawEpoch = dataset?.data_epoch ?? provenanceEpoch;
  const epoch = rawEpoch == null ? Number.NaN : Number(rawEpoch);
  if (Number.isInteger(epoch) && epoch >= 0) return epoch;
  const path = String(dataset?.dataset_path || dataset?.path || '');
  const match = path.match(/(?:^|\/)data_epoch_(\d+)(?:\/|$)/i);
  return match ? Number(match[1]) : null;
}

export function compareDatasetSelections(left, right) {
  // Existing single-root checkpoints predate Data Epoch provenance. Keep
  // those legacy roots first so adding data_epoch_0000 remains an ordered
  // suffix and the checkpoint's cumulative-replay prefix stays valid.
  const leftEpoch = left?.dataEpoch == null ? -1 : left.dataEpoch;
  const rightEpoch = right?.dataEpoch == null ? -1 : right.dataEpoch;
  if (leftEpoch !== rightEpoch) return leftEpoch - rightEpoch;
  return String(left?.path || '').localeCompare(String(right?.path || ''));
}

const selectionFromDataset = (dataset = {}) => ({
  path: String(dataset.dataset_path || ''),
  version: String(dataset.version || dataset.codebase_version || ''),
  dataEpoch: dataEpochFromDataset(dataset),
  dataEpochProvenance: dataset.data_epoch_provenance || null,
});

export default function OfflineRLLeRobotDataset({ isActive = true }) {
  const dispatch = useDispatch();
  const datasetPath = useSelector(selectOfflineRLDatasetPath);
  const datasetSelections = useSelector(selectOfflineRLDatasetSelections);
  const destinationPath = useSelector(selectOfflineRLConversionDestinationPath);
  const convertedDatasetPaths = useSelector(selectOfflineRLConvertedDatasetPaths);
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus?.status || 'idle'
  );
  const datasetPathRef = useRef(datasetPath);
  const datasetSelectionsRef = useRef(datasetSelections);
  const inventoryRequestSequence = useRef(0);
  const [inventory, setInventory] = useState([]);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [inventoryError, setInventoryError] = useState('');
  const [snapshot, setSnapshot] = useState(() => emptySnapshot());
  const [loading, setLoading] = useState(false);
  const [deletingIndex, setDeletingIndex] = useState(null);

  useEffect(() => {
    datasetPathRef.current = datasetPath;
  }, [datasetPath]);

  useEffect(() => {
    datasetSelectionsRef.current = datasetSelections;
  }, [datasetSelections]);

  const previewDataset = useCallback((dataset) => {
    dispatch(setOfflineRLDatasetPreview(selectionFromDataset(dataset)));
  }, [dispatch]);

  const refreshInventory = useCallback(async () => {
    const requestSequence = inventoryRequestSequence.current + 1;
    inventoryRequestSequence.current = requestSequence;
    setInventoryLoading(true);
    try {
      const result = await getOfflineRLDatasets(destinationPath);
      if (requestSequence !== inventoryRequestSequence.current) return;
      const datasets = Array.isArray(result?.datasets) ? result.datasets : [];
      setInventory(datasets);
      setInventoryError('');
      if (!datasets.length) return;
      const currentSelections = datasetSelectionsRef.current;
      if (currentSelections.length) {
        const hydrated = currentSelections
          .map((selection) => {
            const dataset = datasets.find((item) => (
              item.dataset_path === selection.path
            ));
            return dataset ? selectionFromDataset(dataset) : selection;
          })
          .sort(compareDatasetSelections);
        dispatch(setOfflineRLDatasetSelections(hydrated));
      }
      const currentPath = datasetPathRef.current;
      const selected = datasets.find((dataset) => dataset.dataset_path === currentPath);
      const convertedV30 = datasets.find(
        (dataset) => dataset.dataset_path === convertedDatasetPaths.v30
      );
      const convertedV21 = datasets.find(
        (dataset) => dataset.dataset_path === convertedDatasetPaths.v21
      );
      const nextDataset = selected || convertedV30 || convertedV21 || datasets[0];
      if (nextDataset) {
        if (currentSelections.length) previewDataset(nextDataset);
        else dispatch(setOfflineRLDatasetSelection(selectionFromDataset(nextDataset)));
      }
    } catch (error) {
      if (requestSequence === inventoryRequestSequence.current) {
        setInventoryError(error?.message || 'LeRobot dataset discovery failed');
      }
    } finally {
      if (requestSequence === inventoryRequestSequence.current) {
        setInventoryLoading(false);
      }
    }
  }, [
    convertedDatasetPaths.v21,
    convertedDatasetPaths.v30,
    destinationPath,
    dispatch,
    previewDataset,
  ]);

  useEffect(() => () => {
    inventoryRequestSequence.current += 1;
  }, []);

  const toggleTrainingDataset = useCallback((dataset) => {
    const selection = selectionFromDataset(dataset);
    if (!selection.path || selection.version !== 'v3.0') return;
    const selected = datasetSelectionsRef.current;
    const included = selected.some((item) => item.path === selection.path);
    const next = included
      ? selected.filter((item) => item.path !== selection.path)
      : [...selected, selection];
    const ordered = next.sort(compareDatasetSelections);
    datasetSelectionsRef.current = ordered;
    dispatch(setOfflineRLDatasetSelections(ordered));
    previewDataset(dataset);
  }, [dispatch, previewDataset]);

  useEffect(() => {
    if (!isActive || conversionStatus === 'running') return;
    refreshInventory();
  }, [conversionStatus, isActive, refreshInventory]);

  const readDataset = useCallback(async (path) => {
    const result = await getOfflineRLDatasetInfo(path);
    const episodes = normalizeLeRobotEpisodes(result);
    return {
      path,
      episodes,
      totalCount: Number(result.total_episodes ?? result.episode_count ?? episodes.length) || 0,
      version: String(result.codebase_version || result.version || 'v3.0'),
      fps: Number(result.fps || 0),
      error: '',
      loaded: true,
    };
  }, []);

  useEffect(() => {
    if (!isActive || !datasetPath || conversionStatus === 'running') return undefined;
    let cancelled = false;
    setLoading(true);
    readDataset(datasetPath)
      .then((nextSnapshot) => {
        if (!cancelled) setSnapshot(nextSnapshot);
      })
      .catch((error) => {
        if (!cancelled) {
          setSnapshot({
            ...emptySnapshot(datasetPath),
            error: error?.message || 'LeRobot dataset inspection failed',
          });
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [conversionStatus, datasetPath, isActive, readDataset]);

  const handleDelete = useCallback(async (episodeIndex) => {
    if (
      deletingIndex !== null ||
      !snapshot.loaded ||
      snapshot.path !== datasetPath ||
      conversionStatus === 'running'
    ) return;
    setDeletingIndex(episodeIndex);
    try {
      const trainingStatus = await getOfflineRLStatus();
      const normalizedStatus = String(trainingStatus?.status || '').trim().toLowerCase();
      const lineagePaths = Array.isArray(trainingStatus?.dataset_paths) &&
        trainingStatus.dataset_paths.length
        ? trainingStatus.dataset_paths
        : [trainingStatus?.dataset_path].filter(Boolean);
      if (
        ['starting', 'running', 'complete', 'completed'].includes(normalizedStatus) &&
        lineagePaths.includes(datasetPath)
      ) {
        toast.error(
          normalizedStatus === 'starting' || normalizedStatus === 'running'
            ? 'This Data Epoch is in use by the current training job'
            : 'This Data Epoch is part of a completed checkpoint lineage and must remain immutable'
        );
        return;
      }

      const paddedIndex = String(episodeIndex).padStart(3, '0');
      const deletingFinalEpisode = snapshot.totalCount === 1;
      if (!window.confirm(
        `Delete LeRobot episode_${paddedIndex}?\n\n` +
        (deletingFinalEpisode
          ? 'This is the final episode, so the entire selected LeRobot dataset folder will be permanently removed.'
          : 'This permanently removes the saved episode. The dataset will be rebuilt and the remaining episode indices will be compacted.')
      )) return;

      const result = await deleteOfflineRLDatasetEpisodes(datasetPath, [episodeIndex]);
      if (result?.dataset_deleted) {
        const remainingSelections = datasetSelectionsRef.current.filter(
          (item) => item.path !== datasetPath
        );
        datasetSelectionsRef.current = remainingSelections;
        dispatch(setOfflineRLDatasetSelections(remainingSelections));
        datasetPathRef.current = '';
        dispatch(setOfflineRLDatasetPreview({ path: '', version: '' }));
        setSnapshot(emptySnapshot());
        toast.success(`LeRobot dataset deleted with its final episode_${paddedIndex}`);
        await refreshInventory();
        return;
      }
      const nextSnapshot = await readDataset(datasetPath);
      setSnapshot(nextSnapshot);
      toast.success(`LeRobot episode_${paddedIndex} deleted`);
    } catch (error) {
      toast.error(error?.message || 'LeRobot episode deletion failed');
    } finally {
      setDeletingIndex(null);
    }
  }, [
    conversionStatus,
    datasetPath,
    deletingIndex,
    dispatch,
    readDataset,
    refreshInventory,
    snapshot.loaded,
    snapshot.path,
    snapshot.totalCount,
  ]);

  const snapshotMatches = snapshot.path === datasetPath;
  const episodes = snapshotMatches ? snapshot.episodes : [];
  const totalCount = snapshotMatches ? snapshot.totalCount : 0;
  const datasetEditable = snapshotMatches && ['v2.1', 'v3.0'].includes(snapshot.version);
  const selectedPathSet = new Set(datasetSelections.map((item) => item.path));
  const detail = useMemo(() => {
    if (!datasetPath) return 'Awaiting converted dataset';
    if (loading) return 'Reading LeRobot metadata…';
    if (snapshotMatches && snapshot.error) return snapshot.error;
    if (!snapshotMatches || !snapshot.loaded) return 'Awaiting dataset inspection';
    if (snapshot.version === 'v2.1') {
      return `v2.1 · ${snapshot.fps || '—'} FPS · editable; select v3.0 for training`;
    }
    return `${snapshot.version} · ${snapshot.fps || '—'} FPS · ready for training`;
  }, [datasetPath, loading, snapshot.error, snapshot.fps, snapshot.loaded, snapshot.version, snapshotMatches]);

  return (
    <div
      className="flex h-full min-h-0 flex-col gap-1.5 overflow-hidden"
      data-testid="offline-rl-lerobot-dataset"
    >
      <div className="flex min-w-0 gap-1">
        <select
          aria-label="LeRobot dataset"
          value={inventory.some((dataset) => dataset.dataset_path === datasetPath)
            ? datasetPath
            : ''}
          onChange={(event) => previewDataset(
            inventory.find((dataset) => dataset.dataset_path === event.target.value)
          )}
          disabled={!isActive || inventoryLoading || conversionStatus === 'running'}
          className="h-7 min-w-0 flex-1 rounded-md border border-[#d9d2c5] bg-white px-1.5 text-[8px] text-[#514b42] outline-none focus:border-[#879b89] disabled:bg-[#efebe3]"
        >
          {!inventory.length && <option value="">No converted datasets</option>}
          {inventory.map((dataset) => (
            <option key={dataset.dataset_path} value={dataset.dataset_path}>
              {dataset.name} · {dataset.version}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={refreshInventory}
          disabled={!isActive || inventoryLoading || conversionStatus === 'running'}
          aria-label="Refresh LeRobot datasets"
          title="Refresh converted datasets"
          className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-[#d9d2c5] bg-[#f1ede4] text-[#70695e] hover:bg-[#e9e4da] disabled:opacity-45"
        >
          <MdRefresh size={13} className={inventoryLoading ? 'animate-spin' : ''} />
        </button>
      </div>
      <div className="truncate text-[8px] text-[#8c857a]" title={detail}>
        {inventoryError || detail}
      </div>
      {inventory.length > 0 && (
        <div
          role="group"
          aria-label="Training Data Epochs"
          className="grid max-h-[52px] shrink-0 grid-cols-2 gap-1 overflow-y-auto pr-0.5"
        >
          {[...inventory]
            .sort((left, right) => compareDatasetSelections(
              selectionFromDataset(left),
              selectionFromDataset(right)
            ))
            .map((dataset) => {
              const selection = selectionFromDataset(dataset);
              const included = selectedPathSet.has(selection.path);
              const isV30 = selection.version === 'v3.0';
              const epochLabel = selection.dataEpochProvenance?.epoch_name || (
                selection.dataEpoch == null
                  ? dataset.name
                  : `data_epoch_${String(selection.dataEpoch).padStart(4, '0')}`
              );
              return (
                <label
                  key={selection.path}
                  title={`${selection.path}${isV30 ? '' : ' · v3.0 required for TD3'}`}
                  className={clsx(
                    'flex h-6 min-w-0 items-center gap-1 rounded-md border px-1.5 text-[8px]',
                    included
                      ? 'border-[#8da391] bg-[#e6eee5] text-[#58705d]'
                      : 'border-[#ded7cb] bg-white text-[#756e63]',
                    !isV30 && 'cursor-not-allowed opacity-55'
                  )}
                >
                  <input
                    type="checkbox"
                    aria-label={`Include ${epochLabel} in training`}
                    checked={included}
                    disabled={!isActive || conversionStatus === 'running' || !isV30}
                    onChange={() => toggleTrainingDataset(dataset)}
                    className="h-3 w-3 shrink-0 accent-[#69866f]"
                  />
                  <span className="min-w-0 flex-1 truncate">{epochLabel}</span>
                  <span className="shrink-0 text-[7px] opacity-70">{selection.version}</span>
                </label>
              );
            })}
        </div>
      )}
      <div className="shrink-0 text-[8px] text-[#81796e]">
        {datasetSelections.length} Data Epoch{datasetSelections.length === 1 ? '' : 's'} included
      </div>
      <div
        className="flex h-[160px] min-h-0 shrink-0 overflow-hidden"
        data-testid="offline-rl-lerobot-episode-region"
      >
        <ReplayBufferStack
          episodes={episodes}
          totalCount={totalCount}
          loading={loading}
          error={snapshotMatches ? snapshot.error : ''}
          onDelete={snapshotMatches && snapshot.loaded && datasetEditable ? handleDelete : null}
          deletingIndex={deletingIndex}
          deleteDisabled={conversionStatus === 'running'}
          datasetLabel="LeRobot episodes"
          listLabel="LeRobot Dataset episodes"
        />
      </div>
    </div>
  );
}
