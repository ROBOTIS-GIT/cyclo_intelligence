// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import React, { useCallback, useEffect, useState } from 'react';
import clsx from 'clsx';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import { MdDeleteSweep, MdFolderOpen, MdMovie, MdRefresh } from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import {
  selectRecordTaskInfo,
  setRecordTaskInfo,
} from '../../../features/tasks/taskSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import { DEFAULT_PATHS } from '../../../constants/paths';

const normalizeRosbagTaskPath = (item) => {
  const rawPath = item?.full_path || item?.path || item?.name || '';
  const trimmedPath = rawPath.replace(/\/+$/, '');
  const rosbagRoot = DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '');

  if (trimmedPath === rosbagRoot) {
    return '';
  }

  if (trimmedPath.startsWith(`${rosbagRoot}/`)) {
    return trimmedPath.slice(rosbagRoot.length + 1);
  }

  return trimmedPath;
};

export default function DatasetConvertSection({
  isEditable = true,
  onBusyChange,
}) {
  const dispatch = useDispatch();
  const {
    getDatasetInfo,
    sendEditDatasetCommand,
    sendRecordCommand,
  } = useRosServiceCaller();
  const info = useSelector(selectRecordTaskInfo, shallowEqual);
  const conversionStatus = useSelector(
    (state) => state.editDataset.conversionStatus
  );

  // ----- local state ------------------------------------------------------
  const [singleTaskName, setSingleTaskName] = useState('');
  const [showSingleBrowser, setShowSingleBrowser] = useState(false);

  const [isConverting, setIsConverting] = useState(false);
  const [hasSeenConverting, setHasSeenConverting] = useState(false);
  const [convertError, setConvertError] = useState('');
  const [pendingSingleConvert, setPendingSingleConvert] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState(null);
  const [statusLoading, setStatusLoading] = useState(false);
  const [isPruning, setIsPruning] = useState(false);
  const [pruneSuccessCount, setPruneSuccessCount] = useState(0);
  const [pruneFailureCount, setPruneFailureCount] = useState(0);
  const [dataStatusMessage, setDataStatusMessage] = useState('');

  // Conversion-only knobs. Defaults match cyclo_data's defaults: fps=15
  // (DEFAULT_CONVERSION_FPS in pipeline_worker.py) and both LeRobot
  // formats enabled.
  const [conversionFps, setConversionFps] = useState(15);
  const [convertV21, setConvertV21] = useState(true);
  const [convertV30, setConvertV30] = useState(true);

  // ----- isConverting tracks the backend status -------------------------
  // Driven by /data/status (DataOperationStatus, OP_CONVERSION) routed
  // through editDatasetSlice. Conversion is a Data-Tools-side flow,
  // distinct from the live recording session reflected in recordStatus.
  useEffect(() => {
    if (conversionStatus.status === 'running') {
      setIsConverting(true);
      setHasSeenConverting(true);
    } else if (
      isConverting &&
      hasSeenConverting &&
      (conversionStatus.status === 'completed' ||
        conversionStatus.status === 'failed' ||
        conversionStatus.status === 'cancelled' ||
        conversionStatus.status === 'idle')
    ) {
      setIsConverting(false);
      setHasSeenConverting(false);
      if (conversionStatus.status === 'failed' && conversionStatus.message) {
        setConvertError(conversionStatus.message);
      }
    }
  }, [conversionStatus.status, conversionStatus.message, isConverting, hasSeenConverting]);

  useEffect(() => {
    if (onBusyChange) onBusyChange(isConverting || isPruning);
  }, [isConverting, isPruning, onBusyChange]);

  const rawTaskPath = singleTaskName
    ? `${DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '')}/${singleTaskName}`
    : '';

  const refreshDatasetStatus = useCallback(async () => {
    if (!rawTaskPath) {
      setDatasetStatus(null);
      return null;
    }
    setStatusLoading(true);
    setDataStatusMessage('');
    try {
      const result = await getDatasetInfo(rawTaskPath);
      if (!result?.success || !result.dataset_info) {
        throw new Error(result?.message || 'Failed to read dataset status');
      }
      const info = result.dataset_info;
      const nextStatus = {
        episodeCount: Number(info.episode_count || 0),
        successCount: Number(info.success_count || 0),
        failureCount: Number(info.failure_count || 0),
        unlabeledCount: Number(info.unlabeled_count || 0),
      };
      setDatasetStatus(nextStatus);
      return nextStatus;
    } catch (error) {
      setDatasetStatus(null);
      setDataStatusMessage(error.message || 'Failed to read dataset status');
      return null;
    } finally {
      setStatusLoading(false);
    }
  }, [getDatasetInfo, rawTaskPath]);

  useEffect(() => {
    refreshDatasetStatus();
  }, [refreshDatasetStatus]);

  useEffect(() => {
    if (conversionStatus.status === 'completed') refreshDatasetStatus();
  }, [conversionStatus.status, refreshDatasetStatus]);

  const handlePruneOldest = useCallback(async () => {
    const successCount = Number(pruneSuccessCount);
    const failureCount = Number(pruneFailureCount);
    if (
      !Number.isInteger(successCount) ||
      !Number.isInteger(failureCount) ||
      successCount < 0 ||
      failureCount < 0 ||
      successCount + failureCount < 1
    ) {
      setDataStatusMessage('Enter a non-negative count and delete at least one episode.');
      return;
    }
    if (
      !datasetStatus ||
      successCount > datasetStatus.successCount ||
      failureCount > datasetStatus.failureCount
    ) {
      setDataStatusMessage('Delete count exceeds the available labeled episodes.');
      return;
    }
    if (!window.confirm(
      `Delete the oldest ${successCount} success and ${failureCount} failure ` +
      `episode(s) from\n${rawTaskPath}?\n\nThis deletes MCAP source data.`
    )) return;

    setIsPruning(true);
    setDataStatusMessage('');
    try {
      const result = await sendEditDatasetCommand('prune_oldest', {
        deleteTaskDir: rawTaskPath,
        pruneOldestSuccessCount: successCount,
        pruneOldestFailureCount: failureCount,
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Oldest episode deletion failed');
      }
      setPruneSuccessCount(0);
      setPruneFailureCount(0);
      await refreshDatasetStatus();
      setDataStatusMessage(result.message || 'Oldest episodes deleted.');
    } catch (error) {
      setDataStatusMessage(error.message || 'Oldest episode deletion failed');
    } finally {
      setIsPruning(false);
    }
  }, [
    datasetStatus,
    pruneFailureCount,
    pruneSuccessCount,
    rawTaskPath,
    refreshDatasetStatus,
    sendEditDatasetCommand,
  ]);

  // ----- single convert ---------------------------------------------------
  // Two-step: dispatch taskInfo first, then fire sendRecordCommand from a
  // useEffect once Redux has propagated the update (sendRecordCommand reads
  // taskInfo via closure, so it must see the new value before being invoked).
  const handleConvertMp4 = useCallback(() => {
    if (!singleTaskName) {
      setConvertError('Please pick a task folder to convert');
      return;
    }
    if (!convertV21 && !convertV30) {
      setConvertError('Pick at least one output format (v2.1 or v3.0)');
      return;
    }
    if (!Number.isFinite(conversionFps) || conversionFps <= 0) {
      setConvertError('FPS must be a positive integer');
      return;
    }
    setConvertError('');
    setIsConverting(true);

    dispatch(
      setRecordTaskInfo({
        taskName: singleTaskName,
        taskInstruction: [singleTaskName],
      })
    );
    setPendingSingleConvert(true);
  }, [singleTaskName, dispatch, convertV21, convertV30, conversionFps]);

  useEffect(() => {
    if (!pendingSingleConvert) return;
    if (info.taskName !== singleTaskName) return;
    setPendingSingleConvert(false);

    const fire = async () => {
      try {
        const result = await sendRecordCommand('convert_mp4', {
          conversionFps,
          convertV21,
          convertV30,
          cameraRotations: {},
          imageResize: null,
        });
        if (!result?.success) {
          setConvertError(result?.message || 'Conversion failed');
          setIsConverting(false);
        }
      } catch (error) {
        setConvertError(error.message || 'Failed to start conversion');
        setIsConverting(false);
      }
    };
    fire();
  }, [
    pendingSingleConvert,
    info.taskName,
    singleTaskName,
    sendRecordCommand,
    conversionFps,
    convertV21,
    convertV30,
  ]);

  // ----- file browser callbacks ------------------------------------------
  const handleSingleFolderSelect = useCallback((item) => {
    setSingleTaskName(normalizeRosbagTaskPath(item));
    setShowSingleBrowser(false);
  }, []);

  // ----- derived ----------------------------------------------------------
  const optionsValid =
    (convertV21 || convertV30) &&
    Number.isFinite(conversionFps) &&
    conversionFps > 0;
  const canConvertSingle =
    !isConverting && !isPruning && Boolean(singleTaskName) && isEditable && optionsValid;
  const canPrune =
    isEditable &&
    !isConverting &&
    !isPruning &&
    !statusLoading &&
    Boolean(datasetStatus) &&
    Number(pruneSuccessCount) + Number(pruneFailureCount) > 0;

  // Match cyclo_data's progress-band layout so the label tracks the
  // actual stage. Stage 1 (MP4) always runs; v21 / v30 are conditional.
  // Each enabled stage gets an equal slice of [0, 100].
  const stageLabels = ['Converting to MP4…'];
  if (convertV21) stageLabels.push('Converting to LeRobot v2.1…');
  if (convertV30) stageLabels.push('Converting to LeRobot v3.0…');
  const bandWidth = 100 / stageLabels.length;
  const computeStageLabel = (pct) => {
    const idx = Math.min(
      stageLabels.length - 1,
      Math.floor(Math.max(0, pct) / bandWidth)
    );
    return `[${idx + 1}/${stageLabels.length}] ${stageLabels[idx]}`;
  };

  return (
    <div className="w-full flex flex-col items-center justify-start bg-gray-100 p-10 gap-8 rounded-xl">
      <div className="w-full flex items-center justify-start gap-2">
        <MdMovie className="w-7 h-7 text-blue-500" />
        <span className="text-2xl font-bold">Convert Dataset</span>
      </div>

      <div className="w-full bg-white p-6 rounded-md shadow-md flex flex-col gap-4">
        {/* Task folder picker --------------------------------------------- */}
        <div className="flex flex-col gap-2">
          <span className="text-sm text-gray-600 font-medium">
            Task folder to convert
          </span>
          <div className="flex flex-row items-center gap-2">
            <input
              type="text"
              value={singleTaskName}
              onChange={(e) => setSingleTaskName(e.target.value)}
              placeholder="e.g. Task_1_1_MCAP or Temp/Task_1_1_MCAP"
              disabled={isConverting || !isEditable}
              className={clsx(
                'text-sm flex-1 p-2 border border-gray-300 rounded-md',
                'focus:outline-none focus:ring-2 focus:ring-blue-500',
                (isConverting || !isEditable) && 'bg-gray-100 cursor-not-allowed'
              )}
            />
            <button
              type="button"
              onClick={() => setShowSingleBrowser(true)}
              disabled={isConverting || !isEditable}
              className="flex items-center justify-center w-10 h-10 text-blue-500 bg-gray-200 rounded-md hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Browse for task folder"
            >
              <MdFolderOpen className="w-6 h-6" />
            </button>
          </div>
          <div className="text-xs text-gray-500">
            Picks a folder under <code>/workspace/rosbag2/</code>.
          </div>
        </div>

        {/* Raw MCAP status and outcome-aware cleanup --------------------- */}
        <div className="flex flex-col gap-3 rounded-lg border border-gray-200 bg-gray-50 p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <MdDeleteSweep className="h-5 w-5 text-red-500" />
              <span className="text-sm font-semibold text-gray-700">
                Data Status &amp; Cleanup
              </span>
            </div>
            <button
              type="button"
              onClick={refreshDatasetStatus}
              disabled={!rawTaskPath || statusLoading || isPruning}
              className="flex items-center gap-1 rounded-md bg-gray-200 px-2 py-1 text-xs text-gray-700 hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <MdRefresh />
              {statusLoading ? 'Checking…' : 'Refresh'}
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Total</span><b>{datasetStatus?.episodeCount ?? '—'} / 200</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Success</span><b>{datasetStatus?.successCount ?? '—'}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Fail</span><b>{datasetStatus?.failureCount ?? '—'}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Unlabeled</span><b>{datasetStatus?.unlabeledCount ?? '—'}</b></div>
          </div>

          <div className="flex flex-wrap items-end gap-3">
            <label className="flex flex-col gap-1 text-xs text-gray-600">
              Delete oldest Success
              <input
                aria-label="Delete oldest success count"
                type="number"
                min={0}
                step={1}
                value={pruneSuccessCount}
                onChange={(event) => setPruneSuccessCount(event.target.value)}
                disabled={!isEditable || isConverting || isPruning}
                className="h-9 w-28 rounded-md border border-gray-300 bg-white px-2 text-sm disabled:bg-gray-100"
              />
            </label>
            <label className="flex flex-col gap-1 text-xs text-gray-600">
              Delete oldest Fail
              <input
                aria-label="Delete oldest failure count"
                type="number"
                min={0}
                step={1}
                value={pruneFailureCount}
                onChange={(event) => setPruneFailureCount(event.target.value)}
                disabled={!isEditable || isConverting || isPruning}
                className="h-9 w-28 rounded-md border border-gray-300 bg-white px-2 text-sm disabled:bg-gray-100"
              />
            </label>
            <button
              type="button"
              onClick={handlePruneOldest}
              disabled={!canPrune}
              className="h-9 rounded-md bg-red-500 px-4 text-sm font-semibold text-white hover:bg-red-600 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500"
            >
              {isPruning ? 'Deleting…' : 'Delete Oldest'}
            </button>
          </div>
          <p className="text-xs leading-relaxed text-amber-700">
            Cleanup changes the MCAP source only. Re-convert to a new LeRobot
            dataset and start a new checkpoint lineage; do not reuse an existing parent.
          </p>
          {dataStatusMessage && (
            <div className="text-xs text-gray-600">{dataStatusMessage}</div>
          )}
        </div>

        {/* Conversion options --------------------------------------------- */}
        <div className="flex flex-col gap-3 p-3 bg-gray-50 rounded-lg border border-gray-200">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-sm text-gray-600 font-medium whitespace-nowrap">
              FPS
            </span>
            <input
              type="number"
              min={1}
              max={120}
              step={1}
              value={conversionFps === 0 ? '' : conversionFps}
              onChange={(e) => {
                const raw = e.target.value;
                if (raw === '') {
                  setConversionFps(0);
                  return;
                }
                const next = parseInt(raw, 10);
                if (Number.isFinite(next)) setConversionFps(next);
              }}
              disabled={isConverting || !isEditable}
              className={clsx(
                'text-sm w-24 p-1.5 border border-gray-300 rounded-md',
                'focus:outline-none focus:ring-2 focus:ring-blue-500',
                (isConverting || !isEditable) && 'bg-gray-100 cursor-not-allowed'
              )}
            />
            <span className="text-xs text-gray-500">
              Encode rate for MP4 + the rate written into LeRobot info.json.
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <span className="text-sm text-gray-600 font-medium whitespace-nowrap">
              Output formats
            </span>
            <label className="flex items-center gap-2 cursor-pointer text-sm">
              <input
                type="checkbox"
                checked={convertV21}
                onChange={(e) => setConvertV21(e.target.checked)}
                disabled={isConverting || !isEditable}
                className="rounded"
              />
              <span>LeRobot v2.1</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer text-sm">
              <input
                type="checkbox"
                checked={convertV30}
                onChange={(e) => setConvertV30(e.target.checked)}
                disabled={isConverting || !isEditable}
                className="rounded"
              />
              <span>LeRobot v3.0</span>
            </label>
            {!convertV21 && !convertV30 && (
              <span className="text-xs text-red-500">
                Pick at least one format.
              </span>
            )}
          </div>

          <div className="text-xs text-gray-500">
            Converted datasets are saved under <code>/workspace/lerobot/</code>
            {' '}(folder is created automatically if missing).
          </div>
        </div>

        {/* Convert button -------------------------------------------------- */}
        <button
          type="button"
          onClick={handleConvertMp4}
          disabled={!canConvertSingle}
          className={clsx(
            'mt-2 px-4 py-2 text-sm font-medium rounded-lg transition-colors flex items-center justify-center gap-2',
            canConvertSingle
              ? 'bg-green-500 text-white hover:bg-green-600'
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          )}
        >
          {isConverting ? 'Converting…' : 'Convert Dataset'}
        </button>

        {/* Progress -------------------------------------------------------- */}
        {isConverting && conversionStatus.status === 'running' && (
          <div className="w-full mt-2">
            <div className="flex justify-between text-xs text-gray-600 mb-1">
              <span>{computeStageLabel(conversionStatus.progress || 0)}</span>
              <span>{Math.round(conversionStatus.progress || 0)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${conversionStatus.progress || 0}%` }}
              />
            </div>
          </div>
        )}

        {convertError && (
          <div className="text-xs text-red-500 mt-1">{convertError}</div>
        )}
        <div className="text-xs text-gray-500 mt-1 leading-relaxed">
          Convert to MP4, LeRobot v2.1, and v3.0 formats.
        </div>
      </div>

      {/* File browser ----------------------------------------------------- */}
      <FileBrowserModal
        isOpen={showSingleBrowser}
        onClose={() => setShowSingleBrowser(false)}
        onFileSelect={handleSingleFolderSelect}
        title="Select task folder to convert"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
        defaultPath={DEFAULT_PATHS.ROSBAG2_PATH}
        homePath=""
      />
    </div>
  );
}
