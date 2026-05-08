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

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import clsx from 'clsx';
import { useSelector, useDispatch } from 'react-redux';
import toast from 'react-hot-toast';
import { TbArrowMerge } from 'react-icons/tb';
import { MdFolderOpen, MdClose } from 'react-icons/md';
import {
  setMergeSourceTaskDirs,
  setMergeOutputPath,
  setMergeOutputFolderName,
  setMergeMoveSources,
} from '../editDatasetSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { DEFAULT_PATHS } from '../../../constants/paths';

// Output is fixed to /workspace/rosbag2/ — keeps merged datasets next to
// their sources so the rest of the pipeline (delete, convert) finds them
// without an extra path field on the UI. Dispatched into Redux on mount
// so useRosServiceCaller's mergeOutputPath read sees it on submit.
const FIXED_OUTPUT_PARENT = '/workspace/rosbag2';

const STYLES = {
  textInput: clsx(
    'text-sm w-full h-8 p-2 border border-gray-300 rounded-md',
    'focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent'
  ),
};

export default function MergeSection({ isEditable = true }) {
  const dispatch = useDispatch();
  const { sendEditDatasetCommand } = useRosServiceCaller();

  const mergeSourceTaskDirs = useSelector(
    (state) => state.editDataset.mergeSourceTaskDirs
  );
  const mergeOutputPath = useSelector((state) => state.editDataset.mergeOutputPath);
  const mergeOutputFolderName = useSelector(
    (state) => state.editDataset.mergeOutputFolderName
  );
  const mergeMoveSources = useSelector((state) => state.editDataset.mergeMoveSources);

  const [showSourcePicker, setShowSourcePicker] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  // Pin output parent to /workspace/rosbag2 — the UI no longer exposes a
  // picker for it, so make sure Redux holds the canonical value before
  // the user can hit Merge.
  useEffect(() => {
    if (mergeOutputPath !== FIXED_OUTPUT_PARENT) {
      dispatch(setMergeOutputPath(FIXED_OUTPUT_PARENT));
    }
  }, [dispatch, mergeOutputPath]);

  const handleAddSources = useCallback(
    (selectedItems) => {
      const items = Array.isArray(selectedItems) ? selectedItems : [selectedItems];
      const newPaths = items
        .map((it) => it.full_path || it.path || '')
        .filter((p) => p);
      const merged = Array.from(new Set([...(mergeSourceTaskDirs || []), ...newPaths]));
      dispatch(setMergeSourceTaskDirs(merged));
    },
    [dispatch, mergeSourceTaskDirs]
  );

  const removeSource = useCallback(
    (idx) => {
      const next = (mergeSourceTaskDirs || []).filter((_, i) => i !== idx);
      dispatch(setMergeSourceTaskDirs(next));
    },
    [dispatch, mergeSourceTaskDirs]
  );

  const handleMerge = useCallback(async () => {
    if (!mergeSourceTaskDirs || mergeSourceTaskDirs.length < 2) {
      toast.error('Pick at least 2 source datasets to merge.');
      return;
    }
    if (!mergeOutputFolderName) {
      toast.error('Enter a name for the merged dataset.');
      return;
    }

    setSubmitting(true);
    try {
      const result = await sendEditDatasetCommand('merge');
      if (result?.success) {
        toast.success(
          result.message ||
            `Merged ${result.affected_count || 0} episodes successfully.`
        );
        dispatch(setMergeSourceTaskDirs([]));
        dispatch(setMergeOutputFolderName(''));
      } else {
        toast.error(result?.message || 'Merge failed.');
      }
    } catch (err) {
      toast.error(err.message || 'Merge request failed.');
    } finally {
      setSubmitting(false);
    }
  }, [
    mergeSourceTaskDirs,
    mergeOutputFolderName,
    sendEditDatasetCommand,
    dispatch,
  ]);

  const fullOutputPath = useMemo(
    () =>
      mergeOutputFolderName
        ? `${FIXED_OUTPUT_PARENT}/${mergeOutputFolderName}`
        : FIXED_OUTPUT_PARENT,
    [mergeOutputFolderName]
  );

  const canSubmit =
    isEditable &&
    !submitting &&
    mergeSourceTaskDirs &&
    mergeSourceTaskDirs.length >= 2 &&
    mergeOutputFolderName;

  return (
    <div className="w-full flex flex-col items-center justify-start bg-gray-100 p-10 gap-8 rounded-xl">
      <div className="w-full flex items-center justify-start gap-2">
        <TbArrowMerge className="w-7 h-7 text-blue-500 rotate-90" />
        <span className="text-2xl font-bold">Merge Rosbag Datasets</span>
      </div>

      <div className="w-full flex flex-row items-stretch gap-6">
        {/* Source list ----------------------------------------------------- */}
        <div className="flex-1 bg-white p-5 rounded-md shadow-md flex flex-col gap-3 min-w-72">
          <div className="text-xl font-bold">Source datasets</div>
          <div className="text-xs text-gray-500">
            Pick rosbag2 datasets under <code>/workspace/rosbag2/</code> to combine.
          </div>

          {mergeSourceTaskDirs && mergeSourceTaskDirs.length > 0 ? (
            <div className="flex flex-col gap-1">
              {mergeSourceTaskDirs.map((p, idx) => (
                <div
                  key={`${p}-${idx}`}
                  className="flex items-center gap-2 bg-gray-50 border border-gray-200 rounded px-2 py-1 text-xs"
                >
                  <span className="text-gray-500 font-mono w-5">{idx}.</span>
                  <span className="flex-1 truncate text-gray-700" title={p}>
                    {p}
                  </span>
                  <button
                    type="button"
                    onClick={() => removeSource(idx)}
                    disabled={!isEditable || submitting}
                    className="text-red-400 hover:text-red-600 disabled:opacity-50"
                    aria-label={`Remove source ${idx}`}
                  >
                    <MdClose />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-gray-400">No sources picked yet.</div>
          )}

          <button
            type="button"
            onClick={() => setShowSourcePicker(true)}
            disabled={!isEditable || submitting}
            className="self-start mt-1 px-3 py-1.5 text-xs font-medium rounded-md bg-blue-500 text-white hover:bg-blue-600 disabled:opacity-50 flex items-center gap-1"
          >
            <MdFolderOpen size={14} /> Add dataset
          </button>

          <label className="flex items-center gap-2 mt-2 cursor-pointer">
            <input
              type="checkbox"
              checked={Boolean(mergeMoveSources)}
              onChange={(e) => dispatch(setMergeMoveSources(e.target.checked))}
              disabled={!isEditable || submitting}
              className="rounded"
            />
            <span className="text-sm text-gray-700">
              Move sources after merge (frees disk; cannot be undone)
            </span>
          </label>
        </div>

        {/* Output ---------------------------------------------------------- */}
        <div className="flex-1 bg-white p-5 rounded-md shadow-md flex flex-col gap-3 min-w-72">
          <div className="text-xl font-bold">Merged dataset name</div>
          <div className="text-xs text-gray-500">
            Saved under <code>/workspace/rosbag2/</code>. The folder must not
            already exist (or must be empty).
          </div>

          <input
            className={STYLES.textInput}
            type="text"
            placeholder="e.g. recycle_merged"
            value={mergeOutputFolderName || ''}
            onChange={(e) => dispatch(setMergeOutputFolderName(e.target.value))}
            disabled={!isEditable || submitting}
          />

          <div className="flex items-center gap-2 mt-1 text-xs">
            <span className="bg-blue-400 text-white font-bold py-0.5 px-2 rounded-full">
              Output
            </span>
            <span className="text-blue-600 break-all">{fullOutputPath || '—'}</span>
          </div>
        </div>
      </div>

      <button
        type="button"
        onClick={handleMerge}
        disabled={!canSubmit}
        className={clsx(
          'px-6 py-3 text-base font-semibold rounded-xl shadow-md transition-colors',
          canSubmit
            ? 'bg-green-500 text-white hover:bg-green-600'
            : 'bg-gray-300 text-gray-500 cursor-not-allowed'
        )}
      >
        {submitting ? 'Merging…' : 'Merge'}
      </button>

      {/* Source picker ----------------------------------------------------- */}
      <FileBrowserModal
        isOpen={showSourcePicker}
        onClose={() => setShowSourcePicker(false)}
        onFileSelect={(items) => {
          handleAddSources(items);
          setShowSourcePicker(false);
        }}
        title="Select source datasets"
        selectButtonText="Add"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
        defaultPath={DEFAULT_PATHS.ROSBAG2_PATH}
        homePath=""
        multiSelect={true}
      />
    </div>
  );
}
