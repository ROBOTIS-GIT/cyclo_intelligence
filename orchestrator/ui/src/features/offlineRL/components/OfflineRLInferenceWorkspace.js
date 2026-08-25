// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import toast from 'react-hot-toast';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import {
  MdCreateNewFolder,
  MdFolderOpen,
  MdKeyboardDoubleArrowDown,
  MdKeyboardDoubleArrowUp,
} from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import ImageGrid from '../../../components/ImageGrid';
import InferenceControlPanel from '../../../components/InferenceControlPanel';
import InferenceRecordingControls from '../../../components/InferenceRecordingControls';
import InferencePanel from '../../../components/InferencePanel';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { InferencePhase, RecordPhase } from '../../../constants/taskPhases';
import {
  markLocalTaskInfoEdited,
  InferenceRecordingUiPhase,
  selectInferenceRecordingControl,
  selectInferenceTaskInfo,
  selectRecordTaskInfo,
  setInferenceTaskInfo,
  setInferenceRecordingUiPhase,
  setRecordTaskInfo,
  setRecordInferenceMode,
  setSharedTaskInstruction,
} from '../../tasks/taskSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import {
  createInferenceRecordingFolder,
  getInferenceRecordingSessionId,
} from '../../../utils/inferenceRecordingFolder';
import {
  selectOfflineRLReplayBufferPath,
  setOfflineRLReplayBufferPath,
} from '../offlineRLSlice';
import OfflineRLRecordingSubtaskPlan from './OfflineRLRecordingSubtaskPlan';

const CAMERA_LABELS = ['Left wrist', 'Head', 'Right wrist'];
const DEFAULT_EXPANDED_SETTINGS_HEIGHT = 360;

const PATH_CONFIG = {
  model: {
    label: 'Model',
    placeholder: '/workspace/model/.../pretrained_model',
    root: DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH,
    title: 'Select inference model',
    allowDirectorySelect: true,
    allowFileSelect: false,
  },
  dataset: {
    label: 'MCAP Dataset',
    placeholder: '/workspace/rosbag2/Task_*_inference_MCAP',
    root: DEFAULT_PATHS.ROSBAG2_PATH,
    title: 'Select source MCAP dataset',
    allowDirectorySelect: true,
    allowFileSelect: false,
  },
};

function WorkspacePathField({ name, value, onChange, onBrowse, disabled }) {
  const config = PATH_CONFIG[name];
  return (
    <div>
      <label
        htmlFor={`offline-rl-${name}-path`}
        className="mb-1 block text-[10px] font-semibold text-[#6e675c]"
      >
        {config.label}
      </label>
      <div className="flex min-w-0 gap-1.5">
        <input
          id={`offline-rl-${name}-path`}
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={disabled}
          placeholder={config.placeholder}
          className="h-9 min-w-0 flex-1 rounded-lg border border-[#d9d2c5] bg-white px-2.5 text-[10px] text-[#403b34] outline-none focus:border-[#9a9182] focus:ring-1 focus:ring-[#c7beb0] disabled:cursor-not-allowed disabled:bg-[#ece8df]"
        />
        <button
          type="button"
          onClick={onBrowse}
          disabled={disabled}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[#d9d2c5] bg-[#f5f2eb] text-[#625b50] hover:bg-[#ebe6dd] disabled:cursor-not-allowed disabled:opacity-50"
          aria-label={`Browse ${config.label}`}
        >
          <MdFolderOpen size={17} />
        </button>
      </div>
    </div>
  );
}

export default function OfflineRLInferenceWorkspace({
  isActive = true,
  workspaceMode = 'inference',
  policyEpoch = 0,
}) {
  const dispatch = useDispatch();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const recordTaskInfo = useSelector(selectRecordTaskInfo, shallowEqual);
  const recordingControl = useSelector(
    selectInferenceRecordingControl,
    shallowEqual
  );
  const recordStatus = useSelector(
    (state) => state.tasks.recordStatus,
    shallowEqual
  );
  const robotType = useSelector((state) => state.tasks.robotType);
  const inferencePhase = useSelector(
    (state) => state.tasks.inferenceStatus.inferencePhase
  );
  const replayBufferPath = useSelector(selectOfflineRLReplayBufferPath);
  const { sendRecordCommand } = useRosServiceCaller();
  const [browserTarget, setBrowserTarget] = useState(null);
  const [isSettingsCollapsed, setIsSettingsCollapsed] = useState(false);
  const [expandedSettingsHeight, setExpandedSettingsHeight] = useState(
    DEFAULT_EXPANDED_SETTINGS_HEIGHT
  );
  const [subtaskInstructions, setSubtaskInstructions] = useState([]);
  const [activeSubtaskIndex, setActiveSubtaskIndex] = useState(0);
  const [savedSubtaskIndices, setSavedSubtaskIndices] = useState([]);
  const [subtaskAdvancing, setSubtaskAdvancing] = useState(false);
  const recordingFolderRef = useRef(taskInfo.recordingFolder || '');
  const settingsSlotRef = useRef(null);
  const isRecordingWorkspace = workspaceMode === 'recording';
  const settingsTitle = isRecordingWorkspace
    ? 'Recording Settings'
    : 'Inference Settings';
  const settingsKind = isRecordingWorkspace ? 'recording' : 'inference';

  const inferenceMode = taskInfo.inferenceMode || 'simulation';
  const isEditable =
    isActive &&
    inferencePhase === InferencePhase.READY &&
    recordStatus.recordPhase === RecordPhase.READY &&
    !recordingControl.lifecycleLocked;
  const isSubtaskMode = subtaskInstructions.length > 0;
  const subtaskPlanComplete = !isSubtaskMode || subtaskInstructions.every(
    (instruction) => Boolean(String(instruction || '').trim())
  );
  const subtaskPlanLocked = !isEditable || savedSubtaskIndices.length > 0;
  const isSubtaskRecordingActive = isSubtaskMode && recordingControl.active;
  const canSaveAndNext = (
    isSubtaskRecordingActive &&
    !recordingControl.pending &&
    !subtaskAdvancing &&
    activeSubtaskIndex < subtaskInstructions.length - 1
  );

  useEffect(() => {
    recordingFolderRef.current = taskInfo.recordingFolder || '';
  }, [taskInfo.recordingFolder]);

  useEffect(() => {
    if (!isActive || !isRecordingWorkspace) return;
    sendRecordCommand('refresh_topics', {
      taskSource: 'inference',
    }).catch(() => {});
  }, [isActive, isRecordingWorkspace, sendRecordCommand]);

  // Recording Settings is intentionally shorter than Inference Settings.
  // Preserve the expanded Inference footprint across the mode switch so the
  // flex camera row keeps the same height; any unused Recording space remains
  // blank below the settings card. Collapsing settings still releases this
  // footprint and lets the camera grow, as before.
  useLayoutEffect(() => {
    if (isRecordingWorkspace || isSettingsCollapsed) return undefined;
    const slot = settingsSlotRef.current;
    if (!slot) return undefined;

    const rememberHeight = () => {
      const measuredHeight = Math.ceil(slot.getBoundingClientRect().height);
      if (measuredHeight > 0) {
        setExpandedSettingsHeight((current) => (
          current === measuredHeight ? current : measuredHeight
        ));
      }
    };

    rememberHeight();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(rememberHeight);
    observer.observe(slot);
    return () => observer.disconnect();
  }, [isRecordingWorkspace, isSettingsCollapsed]);

  // Offline RL always exposes inference recording without an extra opt-in
  // toggle. The backend contract still requires this flag for command-enabled
  // ("robot") deploys, while 3D Sim deploy intentionally keeps it disabled.
  useEffect(() => {
    if (!isActive || recordingControl.lifecycleLocked) return;
    const shouldEnable = inferenceMode === 'robot';
    if (Boolean(taskInfo.recordInferenceMode) === shouldEnable) return;
    dispatch(setRecordInferenceMode(shouldEnable));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  }, [
    dispatch,
    inferenceMode,
    isActive,
    recordingControl.lifecycleLocked,
    taskInfo.recordInferenceMode,
  ]);

  const pathValues = useMemo(() => ({
    model: taskInfo.policyPath || '',
    dataset: replayBufferPath,
  }), [replayBufferPath, taskInfo.policyPath]);

  const updatePath = useCallback((name, value) => {
    if (name === 'model') {
      dispatch(setInferenceTaskInfo({ policyPath: value }));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
      return;
    }
    if (name === 'dataset') {
      // Workspace Paths owns the raw inference recording. The converted
      // LeRobot training dataset is selected independently in Step 3.
      dispatch(setOfflineRLReplayBufferPath(value));
    }
  }, [dispatch]);

  const setRecordingTarget = useCallback((value) => {
    const normalized = String(value || '').trim().replace(/\/+$/, '');
    recordingFolderRef.current = normalized;
    dispatch(setInferenceTaskInfo({ recordingFolder: normalized }));
    dispatch(setOfflineRLReplayBufferPath(normalized));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  }, [dispatch]);

  const updateRecordingTarget = useCallback((value) => {
    const normalized = String(value || '').trim().replace(/\/+$/, '');
    if (normalized && !getInferenceRecordingSessionId(normalized)) {
      toast.error(
        'Select a Task_*_inference_MCAP folder directly under /workspace/rosbag2'
      );
      return false;
    }
    setRecordingTarget(normalized);
    return true;
  }, [setRecordingTarget]);

  const resetSubtaskProgress = useCallback(() => {
    setActiveSubtaskIndex(0);
    setSavedSubtaskIndices([]);
    setSubtaskAdvancing(false);
  }, []);

  const updateSubtaskCount = useCallback((nextCount) => {
    if (subtaskPlanLocked) return;
    setSubtaskInstructions((current) => current
      .slice(0, nextCount)
      .concat(Array(Math.max(0, nextCount - current.length)).fill('')));
    resetSubtaskProgress();
  }, [resetSubtaskProgress, subtaskPlanLocked]);

  const updateSubtaskInstruction = useCallback((index, value) => {
    if (subtaskPlanLocked) return;
    setSubtaskInstructions((current) => current.map((instruction, itemIndex) => (
      itemIndex === index ? value : instruction
    )));
  }, [subtaskPlanLocked]);

  const resetSubtaskPlan = useCallback(() => {
    if (subtaskPlanLocked) return;
    setSubtaskInstructions([]);
    resetSubtaskProgress();
  }, [resetSubtaskProgress, subtaskPlanLocked]);

  const prepareRecording = useCallback(() => {
    const candidates = [
      recordingFolderRef.current,
      taskInfo.recordingFolder,
      replayBufferPath,
    ];
    let recordingFolder = candidates
      .map((value) => String(value || '').trim().replace(/\/+$/, ''))
      .find(Boolean) || '';
    if (recordingFolder && !getInferenceRecordingSessionId(recordingFolder)) {
      throw new Error(
        'Recording target must be a Task_*_inference_MCAP folder under /workspace/rosbag2'
      );
    }
    if (!recordingFolder) {
      recordingFolder = createInferenceRecordingFolder();
    }
    setRecordingTarget(recordingFolder);

    const taskInstruction = String(taskInfo.taskInstruction?.[0] || '').trim()
      || 'ACT_dataset';
    if (!String(taskInfo.taskInstruction?.[0] || '').trim()) {
      dispatch(setSharedTaskInstruction([taskInstruction]));
    }
    const plannedSubtasks = subtaskInstructions.map(
      (instruction) => String(instruction || '').trim()
    );
    if (plannedSubtasks.some((instruction) => !instruction)) {
      throw new Error('Complete every Subtask instruction before recording');
    }
    return {
      recordingFolder,
      taskInstruction: [taskInstruction],
      includeRobotisLicense: Boolean(recordTaskInfo.includeRobotisLicense),
      ...(plannedSubtasks.length > 0
        ? {
          subtaskInstruction: plannedSubtasks,
          segmentIndex: activeSubtaskIndex,
        }
        : {}),
    };
  }, [
    activeSubtaskIndex,
    dispatch,
    recordTaskInfo.includeRobotisLicense,
    replayBufferPath,
    setRecordingTarget,
    subtaskInstructions,
    taskInfo.recordingFolder,
    taskInfo.taskInstruction,
  ]);

  const handleSaveAndNextSubtask = useCallback(async () => {
    if (!canSaveAndNext) return;
    setSubtaskAdvancing(true);
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.SAVING
    ));
    let stoppedCurrentSegment = false;
    try {
      const commandOptions = prepareRecording();
      const stopResult = await sendRecordCommand('stop_segment', {
        ...commandOptions,
        segmentIndex: activeSubtaskIndex,
        taskSource: 'inference',
      });
      if (!stopResult?.success) {
        throw new Error(stopResult?.message || 'Subtask save failed');
      }
      stoppedCurrentSegment = true;
      setSavedSubtaskIndices((current) => (
        current.includes(activeSubtaskIndex)
          ? current
          : [...current, activeSubtaskIndex]
      ));

      const nextSubtaskIndex = activeSubtaskIndex + 1;
      setActiveSubtaskIndex(nextSubtaskIndex);
      const startResult = await sendRecordCommand('start_segment', {
        ...commandOptions,
        segmentIndex: nextSubtaskIndex,
        taskSource: 'inference',
      });
      if (!startResult?.success) {
        throw new Error(startResult?.message || 'Next subtask start failed');
      }
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
      toast.success(`Subtask ${activeSubtaskIndex + 1} saved`);
    } catch (error) {
      dispatch(setInferenceRecordingUiPhase(
        stoppedCurrentSegment
          ? InferenceRecordingUiPhase.IDLE
          : InferenceRecordingUiPhase.RECORDING
      ));
      toast.error(error?.message || 'Failed to advance subtask');
    } finally {
      setSubtaskAdvancing(false);
    }
  }, [
    activeSubtaskIndex,
    canSaveAndNext,
    dispatch,
    prepareRecording,
    sendRecordCommand,
  ]);

  const handleNewRecordingDataset = useCallback(() => {
    if (!isEditable) return;
    setRecordingTarget('');
    toast.success('A new MCAP dataset will be created when recording starts');
  }, [isEditable, setRecordingTarget]);

  const selectedConfig = browserTarget ? PATH_CONFIG[browserTarget] : null;
  const recordingTarget = taskInfo.recordingFolder || replayBufferPath;
  const workspacePathsPanel = (
    <div
      className="min-w-0 rounded-xl border border-[#e2dcd1] bg-[#fbfaf6] p-3"
      data-testid="offline-rl-workspace-paths"
    >
      <div className="mb-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#8f877b]">
        <MdFolderOpen size={13} /> Workspace paths
      </div>
      <div className="space-y-2.5">
        {Object.keys(PATH_CONFIG).map((name) => (
          <WorkspacePathField
            key={name}
            name={name}
            value={pathValues[name]}
            onChange={(value) => updatePath(name, value)}
            onBrowse={() => setBrowserTarget(name)}
            disabled={!isEditable}
          />
        ))}
      </div>
    </div>
  );

  return (
    <div className="mt-4 flex min-h-0 min-w-0 flex-1 flex-col gap-3">
      <div
        className="flex min-h-[260px] flex-1 overflow-hidden rounded-xl border border-[#ded8cc] bg-[#1f1e1a]"
        data-testid="offline-rl-camera-region"
      >
        <div className="min-h-0 w-full flex-1">
          <ImageGrid
            isActive={isActive}
            labels={CAMERA_LABELS}
            preferConfiguredOrder
            persistAssignment={false}
            readOnly
            fillHeight
            columnWeights={[4, 5, 4]}
            edgeToEdge
            coverCell
          />
        </div>
      </div>

      <div
        className="mx-auto w-full max-w-[1080px] shrink-0"
        data-testid="offline-rl-recording-dock"
      >
        {isRecordingWorkspace ? (
          <InferenceRecordingControls
            variant="workspace"
            mode="recording"
            policyEpoch={policyEpoch}
            isActive={isActive}
            prepareRecording={prepareRecording}
            startBlocked={!subtaskPlanComplete}
            segmentedRecording={isSubtaskMode}
            segmentIndex={activeSubtaskIndex}
            canFinalize={
              !isSubtaskMode ||
              activeSubtaskIndex === subtaskInstructions.length - 1
            }
            discardEpisodeOnCancel={isSubtaskMode}
            onEpisodeSaved={resetSubtaskProgress}
            onEpisodeCancelled={resetSubtaskProgress}
            guideMessage={isSubtaskMode
              ? (
                subtaskPlanComplete
                  ? `${subtaskInstructions.length} subtasks ready`
                  : 'Complete every subtask instruction'
              )
              : ''}
          />
        ) : (
          <InferenceControlPanel
            showRecordingControls
            variant="offlineRL"
            policyEpoch={policyEpoch}
          />
        )}
      </div>

      <div
        ref={settingsSlotRef}
        className="shrink-0"
        style={isRecordingWorkspace && !isSettingsCollapsed
          ? { minHeight: `${expandedSettingsHeight}px` }
          : undefined}
        data-testid="offline-rl-settings-slot"
      >
        <div
          className={`relative rounded-xl border border-[#e2dcd1] bg-[#f6f3ec] ${
            isSettingsCollapsed ? 'min-h-10 px-3 py-2' : 'p-3'
          }`}
          data-testid="offline-rl-inference-settings-panel"
          data-collapsed={String(isSettingsCollapsed)}
        >
          {isSettingsCollapsed && (
            <div className="pr-10 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#81796d]">
              {settingsTitle}
            </div>
          )}
          <button
            type="button"
            onClick={() => setIsSettingsCollapsed((collapsed) => !collapsed)}
            aria-label={isSettingsCollapsed
              ? `Show ${settingsKind} settings`
              : `Hide ${settingsKind} settings`}
            aria-expanded={!isSettingsCollapsed}
            title={isSettingsCollapsed
              ? `Show ${settingsKind} settings`
              : `Hide ${settingsKind} settings`}
            className="absolute right-2 top-1.5 z-10 grid h-7 w-7 place-items-center rounded-full border border-[#d4ccbf] bg-[#fbfaf6] text-[#6f685d] shadow-sm transition-colors hover:bg-[#ebe6dd] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1"
          >
            {isSettingsCollapsed ? (
              <MdKeyboardDoubleArrowUp size={18} aria-hidden="true" />
            ) : (
              <MdKeyboardDoubleArrowDown size={18} aria-hidden="true" />
            )}
          </button>
          <div
            className={isSettingsCollapsed ? 'hidden' : 'block'}
            aria-hidden={isSettingsCollapsed}
            data-testid="offline-rl-inference-settings-content"
          >
            {isRecordingWorkspace ? (
              <>
                <div
                  className="grid gap-3 rounded-xl border border-[#e2dcd1] bg-[#fbfaf6] p-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]"
                  data-testid="offline-rl-recording-settings"
                >
              <div className="min-w-0 space-y-2.5">
                <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#81796d]">
                  Recording Settings
                </div>
                <div className="grid grid-cols-[92px_minmax(0,1fr)] items-center gap-2">
                  <span className="text-[10px] font-semibold text-[#6e675c]">
                    Robot Type
                  </span>
                  <span
                    className="h-9 truncate rounded-lg border border-[#d9d2c5] bg-[#f1ede4] px-2.5 py-2 text-[10px] font-medium text-[#625b50]"
                    title={robotType || 'Not selected'}
                  >
                    {robotType || 'Not selected'}
                  </span>
                </div>
                <label className="block text-[10px] font-semibold text-[#6e675c]">
                  Task Instruction
                  <textarea
                    value={taskInfo.taskInstruction?.[0] || ''}
                    onChange={(event) => {
                      dispatch(setSharedTaskInstruction([event.target.value]));
                      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
                    }}
                    disabled={!isEditable}
                    title={isEditable
                      ? 'Editable before recording starts'
                      : 'Locked while recording or saving an episode'}
                    placeholder="ACT_dataset"
                    className="mt-1 min-h-16 w-full resize-y rounded-lg border border-[#d9d2c5] bg-white px-2.5 py-2 text-[10px] font-medium text-[#403b34] outline-none focus:border-[#9a9182] focus:ring-1 focus:ring-[#c7beb0] disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <label className="flex h-9 items-center justify-between rounded-lg border border-[#d9d2c5] bg-[#f6f3ec] px-2.5 text-[10px] font-semibold text-[#6e675c]">
                  Add ROBOTIS License
                  <input
                    type="checkbox"
                    checked={Boolean(recordTaskInfo.includeRobotisLicense)}
                    onChange={(event) => dispatch(setRecordTaskInfo({
                      includeRobotisLicense: event.target.checked,
                    }))}
                    disabled={!isEditable}
                    className="h-4 w-4 accent-[#69866f]"
                  />
                </label>
              </div>

              <div className="min-w-0 rounded-xl border border-[#e2dcd1] bg-[#f6f3ec] p-3">
                <div className="mb-3 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#8f877b]">
                  <MdFolderOpen size={13} /> Save session
                </div>
                <label className="block text-[10px] font-semibold text-[#6e675c]">
                  MCAP Dataset
                  <div className="mt-1 flex min-w-0 gap-1.5">
                    <input
                      type="text"
                      readOnly
                      value={recordingTarget || ''}
                      placeholder="Automatic new dataset on Record"
                      className="h-9 min-w-0 flex-1 rounded-lg border border-[#d9d2c5] bg-white px-2.5 text-[10px] text-[#403b34] outline-none"
                      aria-label="Recording MCAP Dataset"
                    />
                    <button
                      type="button"
                      onClick={() => setBrowserTarget('dataset')}
                      disabled={!isEditable}
                      className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[#d9d2c5] bg-[#f5f2eb] text-[#625b50] hover:bg-[#ebe6dd] disabled:cursor-not-allowed disabled:opacity-50"
                      aria-label="Select recording MCAP Dataset"
                      title="Append to an existing inference MCAP dataset"
                    >
                      <MdFolderOpen size={17} />
                    </button>
                  </div>
                </label>
                <button
                  type="button"
                  onClick={handleNewRecordingDataset}
                  disabled={!isEditable}
                  className="mt-2 flex h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-[#b8c6b7] bg-[#e4ebe3] px-3 text-[10px] font-semibold text-[#607563] hover:bg-[#d9e4d8] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <MdCreateNewFolder size={16} /> New Dataset
                </button>
                <p className="mt-2 text-[9px] leading-relaxed text-[#8f877b]">
                  Recordings append to this session. New Dataset starts a new
                  folder without deleting saved episodes.
                </p>
              </div>
                </div>
                <div className="mt-3">
                  <OfflineRLRecordingSubtaskPlan
                    count={subtaskInstructions.length}
                    instructions={subtaskInstructions}
                    disabled={subtaskPlanLocked}
                    onCountChange={updateSubtaskCount}
                    onInstructionChange={updateSubtaskInstruction}
                    onReset={resetSubtaskPlan}
                    activeIndex={activeSubtaskIndex}
                    savedIndices={savedSubtaskIndices}
                    recordingActive={isSubtaskRecordingActive}
                    advancing={subtaskAdvancing}
                    onSaveAndNext={handleSaveAndNextSubtask}
                  />
                </div>
              </>
            ) : (
              <InferencePanel
                title="Inference Settings"
                modelLabel="Policy"
                embedded
                showPolicyPath={false}
                showRecordingSettings={false}
                variant="offlineRL"
                settingsAside={workspacePathsPanel}
              />
            )}
          </div>
        </div>
      </div>

      <FileBrowserModal
        key={browserTarget || 'closed'}
        isOpen={Boolean(browserTarget)}
        onClose={() => setBrowserTarget(null)}
        onFileSelect={(item) => {
          if (browserTarget) {
            if (isRecordingWorkspace && browserTarget === 'dataset') {
              updateRecordingTarget(item?.full_path || '');
            } else {
              updatePath(browserTarget, item?.full_path || '');
            }
          }
          setBrowserTarget(null);
        }}
        title={selectedConfig?.title || 'Select folder'}
        selectButtonText="Use Folder"
        allowDirectorySelect={selectedConfig?.allowDirectorySelect ?? true}
        allowFileSelect={selectedConfig?.allowFileSelect ?? false}
        fileFilter={selectedConfig?.fileFilter || null}
        initialPath={selectedConfig?.root || DEFAULT_PATHS.BASE_WORKSPACE}
        defaultPath={selectedConfig?.root || DEFAULT_PATHS.BASE_WORKSPACE}
        homePath={selectedConfig?.root || DEFAULT_PATHS.BASE_WORKSPACE}
      />
    </div>
  );
}
