// Copyright 2025 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { useSelector } from 'react-redux';
import {
  MdClose,
  MdFolderOpen,
  MdOutlineContentCut,
  MdPlayArrow,
  MdRefresh,
  MdTune,
} from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

const JOB_STORAGE_KEY = 'cyclo-dataset-curation-job-id';
const ROSBAG_ROOT = '/workspace/rosbag2';

const inputClass = clsx(
  'h-9 rounded-md border border-gray-300 bg-white px-2 text-sm',
  'focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200'
);

const basename = (path) => String(path || '').split('/').filter(Boolean).pop() || '';
const defaultOutputPath = (path) => (
  `${ROSBAG_ROOT}/${basename(path)}_intervention_motion_trimmed`
);

const normalizeStatus = (value) => ({
  0: 'idle',
  1: 'running',
  2: 'completed',
  3: 'failed',
  4: 'cancelled',
}[Number(value)] || 'idle');

const decodeResult = (response) => {
  if (!response?.result_json) return null;
  try {
    return JSON.parse(response.result_json);
  } catch {
    return null;
  }
};

const episodeSelections = (info) => {
  const labels = new Map();
  (info?.success_episode_indices || []).forEach((index) => labels.set(Number(index), true));
  (info?.failure_episode_indices || []).forEach((index) => labels.set(Number(index), false));
  (info?.unlabeled_episode_indices || []).forEach((index) => labels.set(Number(index), null));
  return Array.from(labels.entries())
    .sort((left, right) => left[0] - right[0])
    .map(([index, episodeSuccess]) => ({ index, selected: true, episodeSuccess }));
};

export default function DatasetCurationSection({ isEditable = true }) {
  const {
    getDatasetInfo,
    startDatasetCuration,
    getDatasetCurationStatus,
  } = useRosServiceCaller();
  const curationStatus = useSelector((state) => state.editDataset.curationStatus);
  const [showPicker, setShowPicker] = useState(false);
  const [sources, setSources] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [materialization, setMaterialization] = useState(null);
  const [reviewRows, setReviewRows] = useState([]);
  const [jobId, setJobId] = useState(() => sessionStorage.getItem(JOB_STORAGE_KEY) || '');
  const [busy, setBusy] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [previewRow, setPreviewRow] = useState(null);
  const [datasetName, setDatasetName] = useState('curated_merged');
  const [convertV21, setConvertV21] = useState(true);
  const [convertV30, setConvertV30] = useState(false);
  const [settings, setSettings] = useState({
    fps: 15,
    interventionTopic: '/leader/intervention_mode',
    motionThresholdRad: 0.0005,
    pairToleranceMs: 50,
    trailingStationarySeconds: 2,
  });
  const fetchedStatusRef = useRef('');

  const selectedCount = useMemo(
    () => sources.reduce(
      (count, source) => count + source.episodes.filter((episode) => episode.selected).length,
      0
    ),
    [sources]
  );

  const buildAnalysisRequest = useCallback(() => ({
    schema_version: 1,
    sources: sources.map((source) => ({
      task_dir: source.taskDir,
      output_task_dir: source.outputTaskDir,
      episodes: source.episodes
        .filter((episode) => episode.selected)
        .map((episode) => ({
          index: episode.index,
          episode_success: episode.episodeSuccess,
        })),
    })),
    settings: {
      fps: Number(settings.fps),
      intervention_topic: settings.interventionTopic,
      motion_threshold_rad: Number(settings.motionThresholdRad),
      action_pair_tolerance_ms: Number(settings.pairToleranceMs),
      minimum_trailing_stationary_seconds: Number(
        settings.trailingStationarySeconds
      ),
    },
  }), [settings, sources]);

  const loadJob = useCallback(async (id) => {
    if (!id) return;
    const response = await getDatasetCurationStatus(id);
    if (!response?.success) return;
    const status = normalizeStatus(response.status);
    setBusy(status === 'running');
    const result = decodeResult(response);
    if (result?.kind === 'analysis') {
      setAnalysis(result);
      setReviewRows(result.sources.flatMap((source) => (
        source.episodes.map((row) => ({
          ...row,
          outputTaskDir: source.output_task_dir,
          include: row.status === 'ready',
          startOffsetSeconds: Number(row.start_offset_seconds ?? 0),
          endOffsetSeconds: Number(
            row.end_offset_seconds ?? row.original_duration_seconds ?? 0
          ),
        }))
      )));
    } else if (result?.kind === 'materialization') {
      setMaterialization(result);
    }
  }, [getDatasetCurationStatus]);

  useEffect(() => {
    if (!jobId) return;
    loadJob(jobId).catch(() => {});
  }, [jobId, loadJob]);

  useEffect(() => {
    if (!jobId || curationStatus.jobId !== jobId) return;
    setBusy(curationStatus.status === 'running');
    const terminal = ['completed', 'failed', 'cancelled'].includes(curationStatus.status);
    if (!terminal) return;
    const key = `${jobId}:${curationStatus.status}:${curationStatus.stage}`;
    if (fetchedStatusRef.current === key) return;
    fetchedStatusRef.current = key;
    loadJob(jobId).catch((error) => toast.error(error.message));
    if (curationStatus.status === 'completed') {
      toast.success(
        curationStatus.stage === 'awaiting_confirmation'
          ? 'Analysis completed. Review the detected boundaries.'
          : 'Dataset curation completed.'
      );
    } else {
      toast.error(curationStatus.message || 'Dataset curation failed.');
    }
  }, [curationStatus, jobId, loadJob]);

  const addSources = useCallback(async (items) => {
    const selected = Array.isArray(items) ? items : [items];
    const pendingPaths = new Set(sources.map((source) => source.taskDir));
    for (const item of selected) {
      const path = item.full_path || item.path || '';
      if (!path || pendingPaths.has(path)) continue;
      pendingPaths.add(path);
      try {
        const response = await getDatasetInfo(path);
        if (!response?.success || !response.dataset_info) {
          throw new Error(response?.message || 'Could not read dataset information');
        }
        const episodes = episodeSelections(response.dataset_info);
        if (episodes.length === 0) throw new Error('No numeric episodes found');
        setSources((current) => [...current, {
          taskDir: path,
          outputTaskDir: defaultOutputPath(path),
          robotType: response.dataset_info.robot_type || '',
          episodes,
        }]);
        setDatasetName((current) => (
          current === 'curated_merged'
            ? `${basename(path).replace(/_MCAP$/, '')}_curated_merged`
            : current
        ));
      } catch (error) {
        toast.error(`${basename(path)}: ${error.message}`);
      }
    }
    setShowPicker(false);
  }, [getDatasetInfo, sources]);

  const updateSource = useCallback((sourceIndex, update) => {
    setSources((current) => current.map((source, index) => (
      index === sourceIndex ? { ...source, ...update } : source
    )));
    setAnalysis(null);
  }, []);

  const updateEpisode = useCallback((sourceIndex, episodeIndex, update) => {
    setSources((current) => current.map((source, index) => (
      index !== sourceIndex ? source : {
        ...source,
        episodes: source.episodes.map((episode) => (
          episode.index === episodeIndex ? { ...episode, ...update } : episode
        )),
      }
    )));
    setAnalysis(null);
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!sources.length || selectedCount === 0) {
      toast.error('Select at least one episode.');
      return;
    }
    if (sources.some((source) => !source.outputTaskDir.startsWith(`${ROSBAG_ROOT}/`))) {
      toast.error(`Curated outputs must remain under ${ROSBAG_ROOT}.`);
      return;
    }
    setBusy(true);
    setAnalysis(null);
    setMaterialization(null);
    setReviewRows([]);
    try {
      const response = await startDatasetCuration('analyze', '', buildAnalysisRequest());
      if (!response?.success) throw new Error(response?.message || 'Analysis did not start');
      setJobId(response.job_id);
      sessionStorage.setItem(JOB_STORAGE_KEY, response.job_id);
    } catch (error) {
      setBusy(false);
      toast.error(error.message);
    }
  }, [buildAnalysisRequest, selectedCount, sources, startDatasetCuration]);

  const updateReviewRow = useCallback((rowIndex, update) => {
    setReviewRows((rows) => rows.map((row, index) => (
      index === rowIndex ? { ...row, ...update } : row
    )));
  }, []);

  const handleApply = useCallback(async () => {
    const included = reviewRows.filter((row) => row.include);
    if (!included.length) {
      toast.error('Include at least one reviewed episode.');
      return;
    }
    const invalid = included.find((row) => (
      !Number.isFinite(Number(row.startOffsetSeconds))
      || !Number.isFinite(Number(row.endOffsetSeconds))
      || Number(row.endOffsetSeconds) <= Number(row.startOffsetSeconds)
    ));
    if (invalid) {
      toast.error(`Episode ${invalid.source_episode_index} has an invalid time window.`);
      return;
    }
    setBusy(true);
    try {
      const response = await startDatasetCuration('apply', jobId, {
        schema_version: 1,
        sources: sources.map((source) => ({
          task_dir: source.taskDir,
          output_task_dir: source.outputTaskDir,
        })),
        overrides: reviewRows.map((row) => ({
          source_task_dir: row.source_task_dir,
          source_episode_index: row.source_episode_index,
          include: Boolean(row.include),
          start_time_ns: Math.round(
            Number(row.grid_origin_time_ns) + Number(row.startOffsetSeconds) * 1e9
          ),
          end_time_ns: Math.round(
            Number(row.grid_origin_time_ns) + Number(row.endOffsetSeconds) * 1e9
          ),
          episode_success: row.episode_success,
        })),
        conversion: {
          enabled: convertV21 || convertV30,
          dataset_name: datasetName,
          fps: Number(settings.fps),
          convert_v21: convertV21,
          convert_v30: convertV30,
          output_root: '/workspace/lerobot',
        },
      });
      if (!response?.success) throw new Error(response?.message || 'Curation did not start');
    } catch (error) {
      setBusy(false);
      toast.error(error.message);
    }
  }, [
    convertV21,
    convertV30,
    datasetName,
    jobId,
    reviewRows,
    settings.fps,
    sources,
    startDatasetCuration,
  ]);

  const handleCancel = useCallback(async () => {
    if (!jobId) return;
    const response = await startDatasetCuration('cancel', jobId, {});
    if (!response?.success) toast.error(response?.message || 'Cancellation failed');
  }, [jobId, startDatasetCuration]);

  const canApply = Boolean(analysis?.kind === 'analysis') && !busy;
  const progress = curationStatus.jobId === jobId ? curationStatus.progress : 0;

  return (
    <div className="flex w-full flex-col gap-6 rounded-xl bg-gray-100 p-8">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <MdOutlineContentCut className="h-7 w-7 text-blue-600" />
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Curate & Convert</h2>
            <p className="text-sm text-gray-600">
              Remove pre-intervention and stationary edge frames without modifying source MCAP files.
            </p>
          </div>
        </div>
        <button
          type="button"
          onClick={() => setShowPicker(true)}
          disabled={!isEditable || busy}
          className="flex items-center gap-2 rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
        >
          <MdFolderOpen size={19} /> Add dataset
        </button>
      </div>

      <section className="rounded-lg bg-white p-5 shadow-sm">
        <h3 className="mb-4 text-lg font-semibold">1. Source episodes</h3>
        {sources.length === 0 ? (
          <p className="rounded-md border border-dashed border-gray-300 p-8 text-center text-sm text-gray-500">
            Add one or more task folders from {ROSBAG_ROOT}.
          </p>
        ) : sources.map((source, sourceIndex) => (
          <div key={source.taskDir} className="mb-4 rounded-lg border border-gray-200 p-4 last:mb-0">
            <div className="mb-3 flex items-start gap-3">
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold text-gray-900">{source.taskDir}</div>
                <label className="mt-2 block text-xs font-medium text-gray-600">Curated copy</label>
                <input
                  value={source.outputTaskDir}
                  onChange={(event) => updateSource(sourceIndex, { outputTaskDir: event.target.value })}
                  disabled={busy}
                  className={`${inputClass} mt-1 w-full font-mono`}
                />
              </div>
              <button
                type="button"
                title="Remove source"
                onClick={() => setSources((current) => current.filter((_, index) => index !== sourceIndex))}
                disabled={busy}
                className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-red-600"
              >
                <MdClose size={20} />
              </button>
            </div>
            <div className="max-h-52 overflow-auto rounded border border-gray-200">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-gray-50 text-gray-600">
                  <tr><th className="p-2">Use</th><th>Episode</th><th>Outcome</th></tr>
                </thead>
                <tbody>
                  {source.episodes.map((episode) => (
                    <tr key={episode.index} className="border-t border-gray-100">
                      <td className="p-2">
                        <input
                          type="checkbox"
                          checked={episode.selected}
                          onChange={(event) => updateEpisode(sourceIndex, episode.index, { selected: event.target.checked })}
                          disabled={busy}
                        />
                      </td>
                      <td>{episode.index}</td>
                      <td className="py-1">
                        <select
                          value={episode.episodeSuccess === true ? 'success' : episode.episodeSuccess === false ? 'failure' : 'unlabeled'}
                          onChange={(event) => updateEpisode(sourceIndex, episode.index, {
                            episodeSuccess: event.target.value === 'success'
                              ? true : event.target.value === 'failure' ? false : null,
                          })}
                          disabled={busy}
                          className="rounded border border-gray-300 px-2 py-1"
                        >
                          <option value="success">Success</option>
                          <option value="failure">Failure</option>
                          <option value="unlabeled">Unlabeled</option>
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </section>

      <section className="rounded-lg bg-white p-5 shadow-sm">
        <button
          type="button"
          onClick={() => setAdvancedOpen((open) => !open)}
          className="flex w-full items-center justify-between text-left"
        >
          <span className="flex items-center gap-2 text-lg font-semibold"><MdTune /> Detection settings</span>
          <span className="text-sm text-blue-600">{advancedOpen ? 'Hide' : 'Advanced'}</span>
        </button>
        <div className="mt-4 grid grid-cols-2 gap-4 md:grid-cols-5">
          <label className="text-xs text-gray-600">FPS<input type="number" value={settings.fps} onChange={(e) => setSettings({ ...settings, fps: e.target.value })} className={`${inputClass} mt-1 w-full`} /></label>
          <label className="text-xs text-gray-600">Motion threshold (rad)<input type="number" step="0.0001" value={settings.motionThresholdRad} onChange={(e) => setSettings({ ...settings, motionThresholdRad: e.target.value })} className={`${inputClass} mt-1 w-full`} /></label>
          <label className="text-xs text-gray-600">Action pair (ms)<input type="number" value={settings.pairToleranceMs} onChange={(e) => setSettings({ ...settings, pairToleranceMs: e.target.value })} className={`${inputClass} mt-1 w-full`} /></label>
          <label className="text-xs text-gray-600">Trailing idle (s)<input type="number" step="0.1" value={settings.trailingStationarySeconds} onChange={(e) => setSettings({ ...settings, trailingStationarySeconds: e.target.value })} className={`${inputClass} mt-1 w-full`} /></label>
          <label className="text-xs text-gray-600">LeRobot FPS<input disabled value={settings.fps} className={`${inputClass} mt-1 w-full bg-gray-100`} /></label>
        </div>
        {advancedOpen && (
          <label className="mt-4 block text-xs text-gray-600">
            Intervention topic
            <input value={settings.interventionTopic} onChange={(e) => setSettings({ ...settings, interventionTopic: e.target.value })} className={`${inputClass} mt-1 w-full font-mono`} />
          </label>
        )}
        <div className="mt-5 flex items-center justify-between">
          <span className="text-sm text-gray-600">{selectedCount} episode(s) selected</span>
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!isEditable || busy || selectedCount === 0}
            className="flex items-center gap-2 rounded-md bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <MdRefresh /> Analyze boundaries
          </button>
        </div>
      </section>

      {jobId && (busy || curationStatus.jobId === jobId) && (
        <section className="rounded-lg bg-white p-5 shadow-sm">
          <div className="mb-2 flex justify-between text-sm">
            <span>{curationStatus.message || 'Working…'}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded bg-gray-200">
            <div className="h-full bg-blue-600 transition-all" style={{ width: `${progress}%` }} />
          </div>
          {busy && <button type="button" onClick={handleCancel} className="mt-3 text-sm font-medium text-red-600 hover:underline">Cancel job</button>}
        </section>
      )}

      {analysis?.kind === 'analysis' && (
        <section className="rounded-lg bg-white p-5 shadow-sm">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-lg font-semibold">2. Review detected boundaries</h3>
            <span className="text-sm text-gray-600">
              {analysis.summary.ready} ready · {analysis.summary.needs_review} need review
            </span>
          </div>
          <div className="overflow-x-auto rounded border border-gray-200">
            <table className="w-full min-w-[980px] text-left text-xs">
              <thead className="bg-gray-50 text-gray-600">
                <tr>
                  <th className="p-2">Use</th><th>Source / episode</th><th>Status</th>
                  <th>Start (s)</th><th>End (s)</th><th>Removed</th><th>Outcome</th><th>Preview</th>
                </tr>
              </thead>
              <tbody>
                {reviewRows.map((row, index) => (
                  <tr key={`${row.source_task_dir}:${row.source_episode_index}`} className="border-t border-gray-100">
                    <td className="p-2"><input type="checkbox" checked={row.include} onChange={(e) => updateReviewRow(index, { include: e.target.checked })} disabled={busy} /></td>
                    <td className="max-w-56 truncate" title={row.source_task_dir}>{basename(row.source_task_dir)} / {row.source_episode_index}</td>
                    <td>
                      <span className={clsx('rounded px-2 py-1 font-semibold', row.status === 'ready' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-800')} title={row.review_reason || ''}>
                        {row.status === 'ready' ? 'Ready' : 'Needs review'}
                      </span>
                    </td>
                    <td><input type="number" step="0.001" value={row.startOffsetSeconds} onChange={(e) => updateReviewRow(index, { startOffsetSeconds: e.target.value })} className={`${inputClass} w-24`} /></td>
                    <td><input type="number" step="0.001" value={row.endOffsetSeconds} onChange={(e) => updateReviewRow(index, { endOffsetSeconds: e.target.value })} className={`${inputClass} w-24`} /></td>
                    <td>{row.removed_prefix_frames || 0} + {row.removed_suffix_frames || 0} frames</td>
                    <td>{row.episode_success === true ? 'Success' : row.episode_success === false ? 'Failure' : 'Unlabeled'}</td>
                    <td><button type="button" onClick={() => setPreviewRow(row)} className="flex items-center gap-1 text-blue-600 hover:underline"><MdPlayArrow /> Open</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="mt-5 grid gap-4 md:grid-cols-3">
            <label className="text-xs text-gray-600 md:col-span-2">Merged LeRobot dataset name<input value={datasetName} onChange={(e) => setDatasetName(e.target.value)} className={`${inputClass} mt-1 w-full font-mono`} /></label>
            <div className="flex items-end gap-4 pb-2 text-sm">
              <label><input type="checkbox" checked={convertV21} onChange={(e) => setConvertV21(e.target.checked)} className="mr-2" />LeRobot v2.1</label>
              <label><input type="checkbox" checked={convertV30} onChange={(e) => setConvertV30(e.target.checked)} className="mr-2" />v3.0</label>
            </div>
          </div>
          <div className="mt-5 flex justify-end">
            <button type="button" onClick={handleApply} disabled={!canApply} className="rounded-md bg-green-600 px-6 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:opacity-50">
              Create curated copies{convertV21 || convertV30 ? ' & convert' : ''}
            </button>
          </div>
        </section>
      )}

      {materialization?.kind === 'materialization' && (
        <section className="rounded-lg border border-green-200 bg-green-50 p-5 shadow-sm">
          <h3 className="text-lg font-semibold text-green-900">Curation complete</h3>
          <div className="mt-3 space-y-2 text-sm text-green-900">
            {(materialization.sources || []).map((source) => (
              <div key={source.output_task_dir} className="rounded bg-white/70 p-3">
                <div className="break-all font-mono">{source.output_task_dir}</div>
                <div className="mt-1 text-xs">
                  {source.episodes} episode(s): {(source.episode_indices || []).join(', ')}
                </div>
              </div>
            ))}
            {materialization.conversion && (
              <div className="rounded bg-white/70 p-3">
                <div className="font-semibold">LeRobot outputs</div>
                {materialization.conversion.v21 && (
                  <div className="break-all font-mono">{materialization.conversion.v21}</div>
                )}
                {materialization.conversion.v30 && (
                  <div className="break-all font-mono">{materialization.conversion.v30}</div>
                )}
              </div>
            )}
          </div>
        </section>
      )}

      {previewRow && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-6">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-auto rounded-lg bg-white p-5 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <h3 className="font-semibold">Boundary preview · episode {previewRow.source_episode_index}</h3>
                <p className="text-xs text-gray-500">Seek to {Number(previewRow.start_offset_seconds || 0).toFixed(3)}s in Review Episodes to inspect synchronized joint charts.</p>
              </div>
              <button type="button" onClick={() => setPreviewRow(null)} className="rounded p-1 hover:bg-gray-100"><MdClose size={22} /></button>
            </div>
            <div className="grid gap-3 md:grid-cols-3">
              {(previewRow.cameras || []).map((camera) => (
                <div key={camera.camera} className="rounded border border-gray-200 p-2">
                  <div className="mb-2 text-xs font-semibold">{camera.camera}</div>
                  <video
                    controls
                    preload="metadata"
                    className="aspect-video w-full bg-black"
                    src={`/files${previewRow.source_task_dir}/${camera.video}`}
                    onLoadedMetadata={(event) => {
                      event.currentTarget.currentTime = Number(
                        camera.source_video_offset_seconds || 0
                      );
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <FileBrowserModal
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        onFileSelect={addSources}
        title="Select source datasets"
        selectButtonText="Add"
        allowDirectorySelect
        allowFileSelect={false}
        initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
        defaultPath={DEFAULT_PATHS.ROSBAG2_PATH}
        homePath=""
        multiSelect
      />
    </div>
  );
}
