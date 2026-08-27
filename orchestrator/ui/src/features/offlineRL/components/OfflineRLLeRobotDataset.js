// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { useDispatch, useSelector } from 'react-redux';
import { MdRefresh, MdVisibility } from 'react-icons/md';
import {
  deleteOfflineRLDatasetEpisodes,
  getOfflineRLDatasetEpisodeData,
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
import OfflineRLEpisodeMediaModal from './OfflineRLEpisodeMediaModal';

const CAMERA_PRESENTATION = [
  ['cam_left_wrist', 'Left wrist'],
  ['cam_left_head', 'Head'],
  ['cam_right_wrist', 'Right wrist'],
];

const cameraPresentation = (cameraKey) => {
  const normalized = String(cameraKey || '');
  const matchIndex = CAMERA_PRESENTATION.findIndex(([suffix]) => (
    normalized === suffix || normalized.endsWith(`.${suffix}`)
  ));
  return {
    order: matchIndex < 0 ? CAMERA_PRESENTATION.length : matchIndex,
    label: matchIndex < 0
      ? normalized.split('.').pop() || 'Camera'
      : CAMERA_PRESENTATION[matchIndex][1],
  };
};

const normalizeEpisodeMedia = (values) => (
  (Array.isArray(values) ? values : []).flatMap((item) => {
    const cameraKey = String(item?.camera_key || item?.cameraKey || '').trim();
    const relativePath = String(
      item?.relative_path || item?.relativePath || ''
    ).trim();
    if (!cameraKey || !relativePath) return [];
    const fromS = Number(item?.from_s ?? item?.fromS);
    const toS = Number(item?.to_s ?? item?.toS);
    return [{
      cameraKey,
      relativePath,
      ...(Number.isFinite(fromS) && fromS >= 0 ? { fromS } : {}),
      ...(Number.isFinite(toS) && toS >= 0 ? { toS } : {}),
    }];
  })
);

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
      .map((episode, fallbackIndex) => {
        const normalized = {
          index: Number(episode?.index ?? episode?.episode_index ?? fallbackIndex),
          outcome: outcomeFromEpisode(episode),
        };
        const frames = Number(episode?.frames ?? episode?.length);
        const tasks = Array.isArray(episode?.tasks)
          ? episode.tasks.map((task) => String(task)).filter(Boolean)
          : [];
        const media = normalizeEpisodeMedia(episode?.media);
        if (Number.isInteger(frames) && frames > 0) normalized.frames = frames;
        if (tasks.length) normalized.tasks = tasks;
        if (media.length) normalized.media = media;
        return normalized;
      })
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

export function buildLeRobotEpisodeMedia(datasetPath, episode, fps = 0) {
  const basePath = String(datasetPath || '').trim().replace(/\/+$/, '');
  if (
    !basePath.startsWith('/workspace/') ||
    basePath.split('/').some((segment) => segment === '..')
  ) return [];
  return normalizeEpisodeMedia(episode?.media)
    .flatMap((item) => {
      const segments = item.relativePath.split('/');
      if (
        item.relativePath.startsWith('/') ||
        segments.some((segment) => !segment || segment === '.' || segment === '..')
      ) {
        return [];
      }
      const presentation = cameraPresentation(item.cameraKey);
      return [{
        key: item.cameraKey,
        label: presentation.label,
        url: encodeURI(`/files${basePath}/${item.relativePath}`),
        order: presentation.order,
        ...(Number.isFinite(item.fromS) ? { fromS: item.fromS } : {}),
        ...(Number.isFinite(item.toS) ? { toS: item.toS } : {}),
        ...(Number(fps) > 0 ? { fps: Number(fps) } : {}),
      }];
    })
    .sort((left, right) => left.order - right.order)
    .map(({ order, ...item }) => item);
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

/**
 * Aggregate only the Data Epochs checked for training. The single previewed
 * dataset remains independent so its episodes can still be viewed or deleted
 * without silently changing the cumulative training composition.
 */
export function buildSelectedTrainingComposition(
  datasetSelections = [],
  inventory = [],
  snapshot = emptySnapshot()
) {
  const inventoryByPath = new Map(
    (Array.isArray(inventory) ? inventory : []).map((dataset) => [
      String(dataset?.dataset_path || ''),
      dataset,
    ])
  );
  const aggregatedEpisodes = [];
  let totalCount = 0;

  (Array.isArray(datasetSelections) ? datasetSelections : []).forEach((selection) => {
    const path = String(selection?.path || '');
    if (!path) return;
    const source = snapshot?.loaded && snapshot.path === path
      ? snapshot
      : inventoryByPath.get(path);
    if (!source) return;

    const rows = normalizeLeRobotEpisodes(source);
    const sourceTotal = Number(
      source.totalCount ?? source.total_episodes ?? source.episode_count ?? rows.length
    );
    totalCount += Math.max(
      rows.length,
      Number.isFinite(sourceTotal) && sourceTotal >= 0 ? sourceTotal : 0
    );
    rows.forEach((episode) => {
      aggregatedEpisodes.push({
        ...episode,
        index: aggregatedEpisodes.length,
        sourceIndex: episode.index,
        sourcePath: path,
      });
    });
  });

  return { episodes: aggregatedEpisodes, totalCount };
}

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
  const episodeDataRequestSequence = useRef(0);
  const [inventory, setInventory] = useState([]);
  const [inventoryLoading, setInventoryLoading] = useState(false);
  const [inventoryError, setInventoryError] = useState('');
  const [snapshot, setSnapshot] = useState(() => emptySnapshot());
  const [loading, setLoading] = useState(false);
  const [deletingIndex, setDeletingIndex] = useState(null);
  const [selectedEpisode, setSelectedEpisode] = useState(null);
  const [selectedEpisodeData, setSelectedEpisodeData] = useState({
    data: null,
    loading: false,
    error: '',
  });

  useEffect(() => {
    datasetPathRef.current = datasetPath;
  }, [datasetPath]);

  useEffect(() => {
    datasetSelectionsRef.current = datasetSelections;
  }, [datasetSelections]);

  useEffect(() => {
    episodeDataRequestSequence.current += 1;
    setSelectedEpisode(null);
    setSelectedEpisodeData({ data: null, loading: false, error: '' });
  }, [datasetPath]);

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
        if (currentSelections.length || nextDataset.version !== 'v3.0') {
          previewDataset(nextDataset);
        } else {
          dispatch(setOfflineRLDatasetSelection(selectionFromDataset(nextDataset)));
        }
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
    episodeDataRequestSequence.current += 1;
  }, []);

  const closeSelectedEpisode = useCallback(() => {
    episodeDataRequestSequence.current += 1;
    setSelectedEpisode(null);
    setSelectedEpisodeData({ data: null, loading: false, error: '' });
  }, []);

  const openSelectedEpisode = useCallback(async (episode) => {
    const episodeIndex = Number(episode?.index);
    if (!datasetPath || !Number.isInteger(episodeIndex) || episodeIndex < 0) return;
    const requestSequence = episodeDataRequestSequence.current + 1;
    episodeDataRequestSequence.current = requestSequence;
    setSelectedEpisode({ ...episode, index: episodeIndex });
    setSelectedEpisodeData({ data: null, loading: true, error: '' });
    try {
      const data = await getOfflineRLDatasetEpisodeData(datasetPath, episodeIndex);
      if (requestSequence !== episodeDataRequestSequence.current) return;
      setSelectedEpisodeData({ data, loading: false, error: '' });
    } catch (error) {
      if (requestSequence !== episodeDataRequestSequence.current) return;
      setSelectedEpisodeData({
        data: null,
        loading: false,
        error: error?.message || 'LeRobot episode joint data could not be loaded',
      });
    }
  }, [datasetPath]);

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
  }, [dispatch]);

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

  const handleDelete = useCallback(async (episodeIndex, options = {}) => {
    const skipConfirm = options.skipConfirm === true;
    if (
      deletingIndex !== null ||
      !snapshot.loaded ||
      snapshot.path !== datasetPath ||
      conversionStatus === 'running'
    ) return false;
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
        return false;
      }

      const paddedIndex = String(episodeIndex).padStart(3, '0');
      const deletingFinalEpisode = snapshot.totalCount === 1;
      if (!skipConfirm && !window.confirm(
        `Delete LeRobot episode_${paddedIndex}?\n\n` +
        (deletingFinalEpisode
          ? 'This is the final episode, so the entire selected LeRobot dataset folder will be permanently removed.'
          : 'This permanently removes the saved episode. The dataset will be rebuilt and the remaining episode indices will be compacted.')
      )) return false;

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
        return true;
      }
      const nextSnapshot = await readDataset(datasetPath);
      setSnapshot(nextSnapshot);
      toast.success(`LeRobot episode_${paddedIndex} deleted`);
      return true;
    } catch (error) {
      toast.error(error?.message || 'LeRobot episode deletion failed');
      return false;
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
  const selectedMedia = useMemo(() => buildLeRobotEpisodeMedia(
    datasetPath,
    selectedEpisode,
    snapshot.fps
  ), [datasetPath, selectedEpisode, snapshot.fps]);
  const selectedPathSet = new Set(datasetSelections.map((item) => item.path));
  const trainingComposition = useMemo(() => buildSelectedTrainingComposition(
    datasetSelections,
    inventory,
    snapshot
  ), [datasetSelections, inventory, snapshot]);
  return (
    <>
      <div
        className="flex min-h-0 flex-col gap-2"
        data-testid="offline-rl-lerobot-dataset"
      >
        <div
          className="min-h-[232px] shrink-0"
          data-testid="offline-rl-lerobot-episode-region"
        >
          <ReplayBufferStack
            episodes={episodes}
            totalCount={totalCount}
            compositionEpisodes={trainingComposition.episodes}
            compositionTotalCount={trainingComposition.totalCount}
            loading={loading}
            error={snapshotMatches ? snapshot.error : ''}
            onDelete={snapshotMatches && snapshot.loaded && datasetEditable ? handleDelete : null}
            deletingIndex={deletingIndex}
            deleteDisabled={conversionStatus === 'running'}
            onOpen={openSelectedEpisode}
            datasetLabel="LeRobot episodes"
            listLabel="LeRobot Dataset episodes"
            compositionLabel="Training composition"
          />
        </div>

        <section className="rounded-xl border border-[#e2dbcf] bg-[#f8f5ef] p-2.5">
            <div className="mb-1.5 flex items-center justify-between gap-2">
              <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-[#756e63]">
                Training Data Epochs
              </span>
              <div className="flex min-w-0 items-center gap-1.5">
                <span
                  className={clsx(
                    'truncate text-[9px] font-semibold',
                    inventoryError ? 'text-[#b56255]' : 'text-[#58705d]'
                  )}
                  title={inventoryError || undefined}
                >
                  {inventoryError || `${datasetSelections.length} included`}
                </span>
                <button
                  type="button"
                  onClick={refreshInventory}
                  disabled={!isActive || inventoryLoading || conversionStatus === 'running'}
                  aria-label="Refresh LeRobot datasets"
                  title="Refresh converted datasets"
                  className="grid h-6 w-6 shrink-0 place-items-center rounded-md border border-[#d9d2c5] bg-[#f1ede4] text-[#70695e] hover:bg-[#e9e4da] disabled:opacity-45"
                >
                  <MdRefresh size={12} className={inventoryLoading ? 'animate-spin' : ''} />
                </button>
              </div>
            </div>
            <div
              role="group"
              aria-label="Training Data Epochs"
              className="grid max-h-[108px] shrink-0 grid-cols-2 gap-1.5 overflow-y-auto pr-0.5"
            >
              {!inventory.length && (
                <div className="col-span-2 grid h-8 place-items-center rounded-md border border-dashed border-[#d9d2c5] text-[10px] text-[#8c857a]">
                  No converted datasets
                </div>
              )}
              {[...inventory]
                .sort((left, right) => compareDatasetSelections(
                  selectionFromDataset(left),
                  selectionFromDataset(right)
                ))
                .map((dataset) => {
                  const selection = selectionFromDataset(dataset);
                  const included = selectedPathSet.has(selection.path);
                  const previewing = selection.path === datasetPath;
                  const isV30 = selection.version === 'v3.0';
                  const epochLabel = selection.dataEpochProvenance?.epoch_name || (
                    selection.dataEpoch == null
                      ? dataset.name
                      : `data_epoch_${String(selection.dataEpoch).padStart(4, '0')}`
                  );
                  return (
                    <div
                      key={selection.path}
                      title={selection.path}
                      className={clsx(
                        'flex h-8 min-w-0 items-center rounded-md border text-[10px] transition-colors',
                        previewing
                          ? 'border-[#708c76] ring-1 ring-[#adc0b0]'
                          : included
                            ? 'border-[#aebdad]'
                            : 'border-[#ded7cb]',
                        included ? 'bg-[#e6eee5]' : 'bg-white'
                      )}
                    >
                      <span
                        className="grid h-full w-8 shrink-0 place-items-center border-r border-[#ded7cb]"
                        title={isV30 ? 'Include in training' : 'v3.0 is required for training'}
                      >
                        <input
                          type="checkbox"
                          aria-label={`Include ${epochLabel} ${selection.version} in training`}
                          checked={included}
                          disabled={!isActive || conversionStatus === 'running' || !isV30}
                          onChange={() => toggleTrainingDataset(dataset)}
                          className="h-4 w-4 shrink-0 accent-[#69866f] disabled:opacity-40"
                        />
                      </span>
                      <button
                        type="button"
                        aria-label={`Preview ${epochLabel} ${selection.version}`}
                        aria-pressed={previewing}
                        onClick={() => previewDataset(dataset)}
                        disabled={!isActive || inventoryLoading || conversionStatus === 'running'}
                        className={clsx(
                          'flex h-full min-w-0 flex-1 items-center gap-1.5 px-2 text-left',
                          previewing ? 'text-[#4f6955]' : 'text-[#756e63]',
                          'hover:bg-[#eef2ea] disabled:opacity-45'
                        )}
                      >
                        <span className="min-w-0 flex-1 truncate font-medium">{epochLabel}</span>
                        <span className="shrink-0 text-[9px] opacity-70">{selection.version}</span>
                        {previewing && (
                          <span className="flex shrink-0 items-center gap-0.5 text-[8px] font-bold uppercase tracking-[0.06em] text-[#58705d]">
                            <MdVisibility size={11} aria-hidden="true" />
                            Previewing
                          </span>
                        )}
                      </button>
                    </div>
                  );
                })}
            </div>
          </section>
      </div>
      <OfflineRLEpisodeMediaModal
        open={Boolean(selectedEpisode)}
        sourceLabel={`LeRobot ${snapshot.version || ''}`.trim()}
        episode={selectedEpisode}
        media={selectedMedia}
        jointData={selectedEpisodeData.data}
        jointLoading={selectedEpisodeData.loading}
        jointError={selectedEpisodeData.error}
        onBack={closeSelectedEpisode}
        onDelete={datasetEditable ? async (episode) => handleDelete(
          episode.index,
          { skipConfirm: true }
        ) : null}
        deletePending={deletingIndex !== null}
        deleteDisabled={conversionStatus === 'running'}
      />
    </>
  );
}
