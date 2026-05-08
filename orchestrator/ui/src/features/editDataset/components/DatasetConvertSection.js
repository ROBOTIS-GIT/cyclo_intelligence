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
import { useDispatch, useSelector } from 'react-redux';
import { MdFolderOpen, MdMovie } from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { setTaskInfo } from '../../../features/tasks/taskSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import { DEFAULT_PATHS } from '../../../constants/paths';

// Convert a task folder name like "ffw_sg2_rev1_test_0406_1" to the
// task_name expected by the backend (strip the "{robotType}_" prefix when
// it matches). The backend rebuilds the full path as
// /workspace/rosbag2/{robot_type}_{task_name}.
const stripRobotPrefix = (folderName, robotType) => {
  if (!robotType) return folderName;
  const prefix = `${robotType}_`;
  return folderName.startsWith(prefix) ? folderName.slice(prefix.length) : folderName;
};

export default function DatasetConvertSection({ isEditable = true }) {
  const dispatch = useDispatch();
  const { sendRecordCommand } = useRosServiceCaller();
  const info = useSelector((state) => state.tasks.taskInfo);
  const conversionStatus = useSelector(
    (state) => state.editDataset.conversionStatus
  );
  const robotType = useSelector((state) => state.tasks.robotType) || '';

  // ----- local state ------------------------------------------------------
  const [singleTaskName, setSingleTaskName] = useState('');
  const [showSingleBrowser, setShowSingleBrowser] = useState(false);

  const [isConverting, setIsConverting] = useState(false);
  const [hasSeenConverting, setHasSeenConverting] = useState(false);
  const [convertError, setConvertError] = useState('');
  const [pendingSingleConvert, setPendingSingleConvert] = useState(false);

  // Conversion-only knobs. Defaults match cyclo_data's defaults: fps=15
  // (DEFAULT_CONVERSION_FPS in pipeline_worker.py) and both LeRobot
  // formats enabled.
  const [conversionFps, setConversionFps] = useState(15);
  const [convertV21, setConvertV21] = useState(true);
  const [convertV30, setConvertV30] = useState(true);

  // Camera rotation + image resize (CONVERT_MP4). Cameras are
  // auto-discovered from the live image-topic list — robot_config
  // already enumerates them, so the user only needs to specify how
  // each one should be rotated. Empty rotation per camera = no
  // rotation; both resize dims 0 = keep native resolution.
  const [showCameraOptions, setShowCameraOptions] = useState(false);
  const [cameraNames, setCameraNames] = useState([]);
  const [cameraRotations, setCameraRotations] = useState({}); // {cam: 0|90|180|270}
  const [resizeHeight, setResizeHeight] = useState(0);
  const [resizeWidth, setResizeWidth] = useState(0);

  const { getImageTopicList } = useRosServiceCaller();

  // Map an image topic to a stable camera name. Mirrors
  // cyclo_data/cyclo_data/reader/video_metadata_extractor.py
  // extract_camera_name_from_topic() so UI and backend agree.
  const extractCameraName = (topic) => {
    const t = String(topic || '');
    const lower = t.toLowerCase();

    // ZED stereo head — split into left/right.
    if (lower.includes('zed')) {
      if (lower.includes('/right/') || lower.includes('_right')) return 'cam_head_right';
      return 'cam_head_left';
    }

    // RealSense wrists.
    if (lower.includes('camera_left')) return 'cam_wrist_left';
    if (lower.includes('camera_right')) return 'cam_wrist_right';

    // Legacy `/robot/camera/<cam_name>/...` pattern.
    const m = t.match(/\/robot\/camera\/([^/]+)\//);
    if (m) return m[1];

    return null;
  };

  // Discover camera names on first expand of the camera-options panel.
  // Robot type is already pinned by the time we reach this UI, so we
  // just trust whatever cameras the orchestrator currently advertises.
  // Seed the rotation dropdowns with the yaml defaults
  // (observation.images.<cam>.rotation_deg) — the user sees the value
  // that will actually be applied if they don't override it.
  useEffect(() => {
    if (!showCameraOptions || cameraNames.length > 0) return;
    let cancelled = false;
    (async () => {
      try {
        const result = await getImageTopicList();
        const topics = result?.image_topic_list || [];
        const yamlRotations = result?.rotation_deg_list || [];
        const names = topics.map(extractCameraName).filter(Boolean);
        if (!cancelled && names.length > 0) {
          setCameraNames(names);
          setCameraRotations((prev) => {
            const next = { ...prev };
            names.forEach((name, idx) => {
              if (!(name in next)) {
                next[name] = Number(yamlRotations[idx] || 0);
              }
            });
            return next;
          });
        }
      } catch (err) {
        console.warn('Failed to fetch camera list:', err);
      }
    })();
    return () => { cancelled = true; };
  }, [showCameraOptions, cameraNames.length, getImageTopicList]);

  const handleRotationChange = (camera, value) => {
    setCameraRotations((prev) => ({ ...prev, [camera]: Number(value) }));
  };

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
      setTaskInfo({
        ...info,
        taskName: singleTaskName,
        taskInstruction: [singleTaskName],
      })
    );
    setPendingSingleConvert(true);
  }, [singleTaskName, info, dispatch, convertV21, convertV30, conversionFps]);

  useEffect(() => {
    if (!pendingSingleConvert) return;
    if (info.taskName !== singleTaskName) return;
    setPendingSingleConvert(false);

    const fire = async () => {
      try {
        // Send every cameraRotations entry — including 0 — so the
        // server can tell "user explicitly picked 0" apart from "user
        // didn't touch this camera". The orchestrator merge does
        // {**yaml_defaults, **ui_rotations}, so a missing key falls
        // back to the yaml's rotation_deg; an explicit 0 wins over it.
        const rotations = { ...cameraRotations };
        const imageResize =
          Number(resizeHeight) > 0 && Number(resizeWidth) > 0
            ? { height: Number(resizeHeight), width: Number(resizeWidth) }
            : null;

        const result = await sendRecordCommand('convert_mp4', {
          conversionFps,
          convertV21,
          convertV30,
          cameraRotations: rotations,
          imageResize,
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
    cameraRotations,
    resizeHeight,
    resizeWidth,
  ]);

  // ----- file browser callbacks ------------------------------------------
  const handleSingleFolderSelect = useCallback(
    (item) => {
      const taskName = stripRobotPrefix(item.name, robotType);
      setSingleTaskName(taskName);
      setShowSingleBrowser(false);
    },
    [robotType]
  );

  // ----- derived ----------------------------------------------------------
  const optionsValid =
    (convertV21 || convertV30) &&
    Number.isFinite(conversionFps) &&
    conversionFps > 0;
  const canConvertSingle =
    !isConverting && Boolean(singleTaskName) && isEditable && optionsValid;

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
              placeholder="e.g. Task_1_1_MCAP"
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

        {/* Camera options (rotation + resize) ------------------------------ */}
        <div className="flex flex-col gap-1">
          <button
            type="button"
            onClick={() => setShowCameraOptions((v) => !v)}
            disabled={isConverting}
            className="self-start text-sm text-blue-600 hover:underline disabled:text-gray-400"
          >
            {showCameraOptions ? '▼' : '▶'} Camera options (rotation + resize)
          </button>

          {showCameraOptions && (
            <div className="flex flex-col gap-3 mt-1 p-3 bg-gray-50 rounded-lg border border-gray-200 text-sm">
              <div className="text-xs text-gray-500">
                Other selection (which cameras / topics / joints) is read
                from <code>robot_config.yaml</code>; you only need to specify
                rotation per camera and an optional resize.
              </div>

              {/* Per-camera rotation dropdowns. Cameras come from the live
                  image-topic list once Camera options is expanded. */}
              <div className="flex flex-col gap-2">
                <span className="text-gray-600 font-medium">Camera rotation</span>
                {cameraNames.length === 0 ? (
                  <span className="text-xs text-gray-500 italic">
                    Loading camera list…
                  </span>
                ) : (
                  cameraNames.map((cam) => (
                    <div key={cam} className="flex items-center gap-2">
                      <code className="text-sm text-gray-700 w-40 shrink-0">{cam}</code>
                      <select
                        value={cameraRotations[cam] ?? 0}
                        onChange={(e) => handleRotationChange(cam, e.target.value)}
                        disabled={isConverting || !isEditable}
                        className={clsx(
                          'p-1.5 border border-gray-300 rounded-md text-sm bg-white',
                          'focus:outline-none focus:ring-2 focus:ring-blue-500',
                          (isConverting || !isEditable) && 'bg-gray-100 cursor-not-allowed'
                        )}
                      >
                        <option value={0}>0°</option>
                        <option value={90}>90° CW</option>
                        <option value={180}>180°</option>
                        <option value={270}>270° (90° CCW)</option>
                      </select>
                    </div>
                  ))
                )}
              </div>

              <label className="flex flex-col gap-1">
                <span className="text-gray-600 font-medium">Output resize (HxW)</span>
                <div className="flex items-center gap-2">
                  <input
                    type="number"
                    min={0}
                    value={resizeHeight === 0 ? '' : resizeHeight}
                    onChange={(e) => setResizeHeight(Number(e.target.value) || 0)}
                    placeholder="height"
                    disabled={isConverting || !isEditable}
                    className={clsx(
                      'w-24 p-2 border border-gray-300 rounded-md text-sm',
                      'focus:outline-none focus:ring-2 focus:ring-blue-500',
                      (isConverting || !isEditable) && 'bg-gray-100 cursor-not-allowed'
                    )}
                  />
                  <span>×</span>
                  <input
                    type="number"
                    min={0}
                    value={resizeWidth === 0 ? '' : resizeWidth}
                    onChange={(e) => setResizeWidth(Number(e.target.value) || 0)}
                    placeholder="width"
                    disabled={isConverting || !isEditable}
                    className={clsx(
                      'w-24 p-2 border border-gray-300 rounded-md text-sm',
                      'focus:outline-none focus:ring-2 focus:ring-blue-500',
                      (isConverting || !isEditable) && 'bg-gray-100 cursor-not-allowed'
                    )}
                  />
                  <span className="text-xs text-gray-500">
                    Both 0 = keep native resolution.
                  </span>
                </div>
              </label>
            </div>
          )}
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
