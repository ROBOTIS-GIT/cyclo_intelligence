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
//
// Author: Kiwoong Park

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { useDispatch, useSelector } from 'react-redux';
import toast from 'react-hot-toast';
import { MdDeleteSweep, MdFolderOpen } from 'react-icons/md';
import {
  setDeleteTaskDir,
  setDeleteEpisodeNums,
  setDeleteCompact,
  setDatasetInfo,
} from '../editDatasetSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { DEFAULT_PATHS } from '../../../constants/paths';

// Parse a string like "3, 5, 10-15" into a sorted unique list of integers.
// Reused unchanged from the previous LeRobot-based delete UI.
export const parseEpisodeNumbers = (input) => {
  if (!input || typeof input !== 'string') return [];
  const numbers = new Set();
  const parts = input.split(',').map((part) => part.trim());
  for (const part of parts) {
    if (part.includes('-')) {
      const [start, end] = part.split('-').map((n) => parseInt(n.trim(), 10));
      if (!isNaN(start) && !isNaN(end) && start <= end) {
        for (let i = start; i <= end; i++) numbers.add(i);
      }
    } else {
      const num = parseInt(part.trim(), 10);
      if (!isNaN(num) && num >= 0) numbers.add(num);
    }
  }
  return Array.from(numbers).sort((a, b) => a - b);
};

const STYLES = {
  textInput: clsx(
    'text-sm w-full h-9 p-2 border border-gray-300 rounded-md',
    'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
  ),
};

export default function DeleteSection({ isEditable = true }) {
  const dispatch = useDispatch();
  const { sendEditDatasetCommand, getDatasetInfo } = useRosServiceCaller();

  const deleteTaskDir = useSelector((state) => state.editDataset.deleteTaskDir);
  const deleteEpisodeNums = useSelector(
    (state) => state.editDataset.deleteEpisodeNums
  );
  const deleteCompact = useSelector((state) => state.editDataset.deleteCompact);
  const datasetInfo = useSelector((state) => state.editDataset.datasetInfo);

  const [episodeInput, setEpisodeInput] = useState(
    (deleteEpisodeNums || []).join(', ')
  );
  const [showPicker, setShowPicker] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [infoLoading, setInfoLoading] = useState(false);

  // Refresh task info whenever the picked task dir changes.
  useEffect(() => {
    if (!deleteTaskDir) return;
    let cancelled = false;
    const fetch = async () => {
      setInfoLoading(true);
      try {
        const result = await getDatasetInfo(deleteTaskDir);
        if (cancelled) return;
        if (result?.success && result.dataset_info) {
          dispatch(
            setDatasetInfo({
              robotType: result.dataset_info.robot_type || '',
              taskInstruction: result.dataset_info.task_instruction || '',
              episodeCount: result.dataset_info.episode_count || 0,
              totalDurationS: result.dataset_info.total_duration_s || 0.0,
              fps: result.dataset_info.fps || 0,
            })
          );
        } else {
          toast.error(result?.message || 'Failed to read task info');
        }
      } catch (err) {
        if (!cancelled) toast.error(err.message || 'Failed to read task info');
      } finally {
        if (!cancelled) setInfoLoading(false);
      }
    };
    fetch();
    return () => {
      cancelled = true;
    };
  }, [deleteTaskDir, getDatasetInfo, dispatch]);

  const handlePickTaskDir = useCallback(
    (item) => {
      dispatch(setDeleteTaskDir(item.full_path || item.path || ''));
      setShowPicker(false);
    },
    [dispatch]
  );

  const handleEpisodeInputChange = useCallback(
    (e) => {
      const text = e.target.value;
      setEpisodeInput(text);
      dispatch(setDeleteEpisodeNums(parseEpisodeNumbers(text)));
    },
    [dispatch]
  );

  const handleDelete = useCallback(async () => {
    if (!deleteTaskDir) {
      toast.error('Pick a task folder first.');
      return;
    }
    if (!deleteEpisodeNums || deleteEpisodeNums.length === 0) {
      toast.error('Type at least one episode index (e.g. 3, 5, 10-15).');
      return;
    }
    if (
      !window.confirm(
        `Delete ${deleteEpisodeNums.length} episode(s) from\n${deleteTaskDir}?\n\n` +
          `Indices: ${deleteEpisodeNums.join(', ')}\n` +
          `Compact remaining indices after delete: ${deleteCompact ? 'yes' : 'no'}`
      )
    ) {
      return;
    }

    setSubmitting(true);
    try {
      const result = await sendEditDatasetCommand('delete');
      if (result?.success) {
        toast.success(
          result.message ||
            `Deleted ${result.affected_count || 0} episode(s) successfully.`
        );
        // Refresh info so the user sees the new episode count
        try {
          const info = await getDatasetInfo(deleteTaskDir);
          if (info?.success && info.dataset_info) {
            dispatch(
              setDatasetInfo({
                robotType: info.dataset_info.robot_type || '',
                taskInstruction: info.dataset_info.task_instruction || '',
                episodeCount: info.dataset_info.episode_count || 0,
                totalDurationS: info.dataset_info.total_duration_s || 0.0,
                fps: info.dataset_info.fps || 0,
              })
            );
          }
        } catch (_) {
          /* ignore secondary errors */
        }
        dispatch(setDeleteEpisodeNums([]));
        setEpisodeInput('');
      } else {
        toast.error(result?.message || 'Delete failed.');
      }
    } catch (err) {
      toast.error(err.message || 'Delete request failed.');
    } finally {
      setSubmitting(false);
    }
  }, [
    deleteTaskDir,
    deleteEpisodeNums,
    deleteCompact,
    sendEditDatasetCommand,
    getDatasetInfo,
    dispatch,
  ]);

  const canSubmit =
    isEditable &&
    !submitting &&
    Boolean(deleteTaskDir) &&
    deleteEpisodeNums &&
    deleteEpisodeNums.length > 0;

  const totalDurationStr = useMemo(() => {
    const s = datasetInfo.totalDurationS || 0;
    if (s < 60) return `${s.toFixed(1)} s`;
    const m = Math.floor(s / 60);
    const r = s - m * 60;
    return `${m}m ${r.toFixed(0)}s`;
  }, [datasetInfo.totalDurationS]);

  return (
    <div className="w-full flex flex-col items-center justify-start bg-gray-100 p-10 gap-8 rounded-xl">
      <div className="w-full flex items-center justify-start gap-2">
        <MdDeleteSweep className="w-7 h-7 text-red-500" />
        <span className="text-2xl font-bold">Delete Episodes</span>
      </div>

      <div className="w-full bg-white p-6 rounded-md shadow-md flex flex-col gap-4">
        {/* Task folder picker -------------------------------------------- */}
        <div className="flex flex-col gap-1">
          <span className="text-sm text-gray-600 font-medium">Task folder</span>
          <div className="flex flex-row items-center gap-2">
            <input
              className={STYLES.textInput}
              type="text"
              placeholder="Pick a rosbag2 task folder"
              value={deleteTaskDir || ''}
              onChange={(e) => dispatch(setDeleteTaskDir(e.target.value))}
              disabled={!isEditable || submitting}
            />
            <button
              type="button"
              onClick={() => setShowPicker(true)}
              disabled={!isEditable || submitting}
              className="flex items-center justify-center w-10 h-10 text-blue-500 bg-gray-200 rounded-md hover:text-blue-700 disabled:opacity-50"
              aria-label="Browse task folder"
            >
              <MdFolderOpen className="w-6 h-6" />
            </button>
          </div>
        </div>

        {/* Task summary --------------------------------------------------- */}
        {deleteTaskDir && (
          <div className="bg-gray-50 border border-gray-200 rounded-md p-3 text-xs text-gray-700 flex flex-col gap-1">
            {infoLoading ? (
              <div>Loading task info…</div>
            ) : (
              <>
                <div>
                  <span className="font-semibold">Episodes:</span>{' '}
                  {datasetInfo.episodeCount}
                </div>
                <div>
                  <span className="font-semibold">Robot:</span>{' '}
                  {datasetInfo.robotType || '—'}
                </div>
                <div>
                  <span className="font-semibold">Task:</span>{' '}
                  {datasetInfo.taskInstruction || '—'}
                </div>
                <div>
                  <span className="font-semibold">Total duration:</span>{' '}
                  {totalDurationStr}
                  {datasetInfo.fps ? `  (fps: ${datasetInfo.fps})` : ''}
                </div>
              </>
            )}
          </div>
        )}

        {/* Episode indices ------------------------------------------------ */}
        <div className="flex flex-col gap-1">
          <span className="text-sm text-gray-600 font-medium">
            Episode indices to delete
          </span>
          <input
            className={STYLES.textInput}
            type="text"
            placeholder="e.g. 3, 5, 10-15"
            value={episodeInput}
            onChange={handleEpisodeInputChange}
            disabled={!isEditable || submitting}
          />
          <div className="text-xs text-gray-500">
            Comma-separated indices and ranges. Parsed:{' '}
            <code>{(deleteEpisodeNums || []).join(', ') || '—'}</code>
          </div>
        </div>

        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={deleteCompact !== false}
            onChange={(e) => dispatch(setDeleteCompact(e.target.checked))}
            disabled={!isEditable || submitting}
            className="rounded"
          />
          <span className="text-sm text-gray-700">
            Compact indices after delete (rename surviving episodes back to 0..N-1)
          </span>
        </label>

        <button
          type="button"
          onClick={handleDelete}
          disabled={!canSubmit}
          className={clsx(
            'mt-2 px-6 py-3 text-base font-semibold rounded-xl shadow-md transition-colors self-start',
            canSubmit
              ? 'bg-red-500 text-white hover:bg-red-600'
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          )}
        >
          {submitting ? 'Deleting…' : 'Delete'}
        </button>
      </div>

      <FileBrowserModal
        isOpen={showPicker}
        onClose={() => setShowPicker(false)}
        onFileSelect={handlePickTaskDir}
        title="Select task folder"
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
