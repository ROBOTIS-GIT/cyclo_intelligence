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
import toast from 'react-hot-toast';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import {
  MdCreateNewFolder,
  MdFolderOpen,
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
  createNamedRecordingFolder,
  getRecordingDatasetName,
  getRecordingDatasetFolderInfo,
  resolveRecordingDatasetFolder,
} from '../../../utils/inferenceRecordingFolder';
import {
  selectOfflineRLReplayBufferPath,
  setOfflineRLReplayBufferPath,
} from '../offlineRLSlice';
import OfflineRLRecordingSubtaskPlan from './OfflineRLRecordingSubtaskPlan';
import OfflineRLWorkspaceStatusModal from './OfflineRLWorkspaceStatusModal';
import FlowSDEPPOInferenceControls from './FlowSDEPPOInferenceControls';

const CAMERA_LABELS = ['Left wrist', 'Head', 'Right wrist'];

export const getNextDatasetEpisodeNumber = (datasetInfo = {}) => {
  const indices = [
    ...(datasetInfo.success_episode_indices || []),
    ...(datasetInfo.failure_episode_indices || []),
    ...(datasetInfo.unlabeled_episode_indices || []),
  ].map(Number).filter((value) => Number.isInteger(value) && value >= 0);
  if (indices.length > 0) return Math.max(...indices) + 1;
  const episodeCount = Number(datasetInfo.episode_count || 0);
  return Number.isInteger(episodeCount) && episodeCount >= 0 ? episodeCount : 0;
};

const PATH_CONFIG = {
  model: {
    label: 'Model',
    placeholder: '/workspace/model/.../pretrained_model',
    root: DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH,
    title: 'Select inference model',
    allowDirectorySelect: true,
    allowFileSelect: false,
  },
  recording: {
    label: 'Inference Data Folder',
    placeholder: 'Dataset name',
    root: DEFAULT_PATHS.ROSBAG2_PATH,
    title: 'Select inference data folder',
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

function RecordingDatasetNameField({
  value,
  resolvedPath,
  onChange,
  onCommit,
  onBrowse,
  disabled,
  inputRef = null,
}) {
  return (
    <div>
      <label
        htmlFor="offline-rl-recording-dataset-name"
        className="mb-1 block text-[10px] font-semibold text-[#6e675c]"
      >
        Inference Dataset Name
      </label>
      <div className="flex min-w-0 gap-1.5">
        <input
          ref={inputRef}
          id="offline-rl-recording-dataset-name"
          type="text"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onBlur={onCommit}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              onCommit();
              event.currentTarget.blur();
            }
          }}
          disabled={disabled}
          placeholder="예: Drill_into_Screw_Inference"
          className="h-9 min-w-0 flex-1 rounded-lg border border-[#d9d2c5] bg-white px-2.5 text-[10px] text-[#403b34] outline-none focus:border-[#9a9182] focus:ring-1 focus:ring-[#c7beb0] disabled:cursor-not-allowed disabled:bg-[#ece8df]"
          aria-label="Inference Dataset Name"
        />
        <button
          type="button"
          onClick={onBrowse}
          disabled={disabled}
          className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[#d9d2c5] bg-[#f5f2eb] text-[#625b50] hover:bg-[#ebe6dd] disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Browse Inference Data Folder"
          title="기존 데이터셋을 선택하면 다음 episode 번호부터 이어서 저장합니다"
        >
          <MdFolderOpen size={17} />
        </button>
      </div>
      <p
        className="mt-1 truncate font-mono text-[9px] text-[#8f877b]"
        title={resolvedPath || DEFAULT_PATHS.ROSBAG2_PATH}
      >
        {resolvedPath || `${DEFAULT_PATHS.ROSBAG2_PATH}/Task_<name>_MCAP`}
      </p>
    </div>
  );
}

export default function OfflineRLInferenceWorkspace({
  isActive = true,
  workspaceMode = 'inference',
  policyEpoch = 0,
  workspaceStatusOpen = false,
  onCloseWorkspaceStatus = () => {},
  getFlowSDEPPOPolicyRolloutStatus,
  onStartFlowSDEPPOPolicyRollout,
  onStopFlowSDEPPOPolicyRollout,
  onSubmitFlowSDEPPOPolicyRolloutOutcome,
  getFlowSDEPPOValueWarmupStatus,
  onFlowSDEPPOPolicyRolloutBundleChange = () => {},
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
  const { sendRecordCommand, getDatasetInfo } = useRosServiceCaller();
  const [browserTarget, setBrowserTarget] = useState(null);
  const [subtaskInstructions, setSubtaskInstructions] = useState([]);
  const [activeSubtaskIndex, setActiveSubtaskIndex] = useState(0);
  const [savedSubtaskIndices, setSavedSubtaskIndices] = useState([]);
  const [subtaskAdvancing, setSubtaskAdvancing] = useState(false);
  const [datasetPlanLoading, setDatasetPlanLoading] = useState(false);
  const [selectedDatasetEpisodeNumber, setSelectedDatasetEpisodeNumber] = useState(0);
  const [flowSdeJobBusy, setFlowSdeJobBusy] = useState(false);
  const [recordingDatasetName, setRecordingDatasetName] = useState(() => (
    getRecordingDatasetName(taskInfo.recordingFolder || replayBufferPath)
  ));
  const recordingFolderRef = useRef(taskInfo.recordingFolder || '');
  const recordingDatasetInputRef = useRef(null);
  const preparedSessionSignatureRef = useRef('');
  const preparingSessionSignatureRef = useRef('');
  const awaitingDatasetNameRef = useRef(false);
  const isRecordingWorkspace = workspaceMode === 'recording';

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
  const subtaskPlanLocked =
    !isEditable || datasetPlanLoading || savedSubtaskIndices.length > 0;
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
  }), [taskInfo.policyPath]);

  const updatePath = useCallback((name, value) => {
    if (name === 'model') {
      dispatch(setInferenceTaskInfo({ policyPath: value }));
      dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
      return;
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
    if (normalized && !getRecordingDatasetFolderInfo(normalized)) {
      toast.error(
        'Select a Task_*_MCAP folder directly under /workspace/rosbag2'
      );
      return false;
    }
    setRecordingTarget(normalized);
    setRecordingDatasetName(getRecordingDatasetName(normalized));
    return true;
  }, [setRecordingTarget]);

  const updateRecordingDatasetName = useCallback((value) => {
    setRecordingDatasetName(value);
    preparedSessionSignatureRef.current = '';
    const normalizedName = String(value || '').trim();
    if (!normalizedName) {
      setRecordingTarget('');
      return;
    }
    try {
      setRecordingTarget(createNamedRecordingFolder(normalizedName));
    } catch {
      // Keep the text editable while incomplete; blur/Enter shows validation.
      setRecordingTarget('');
    }
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
    if (recordingFolder) {
      recordingFolder = resolveRecordingDatasetFolder(recordingFolder);
    } else if (String(recordingDatasetName || '').trim()) {
      recordingFolder = createNamedRecordingFolder(recordingDatasetName);
    } else {
      throw new Error('Enter an inference dataset name before recording');
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
    recordingDatasetName,
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
    awaitingDatasetNameRef.current = true;
    preparedSessionSignatureRef.current = '';
    setSelectedDatasetEpisodeNumber(0);
    setRecordingDatasetName('');
    setRecordingTarget('');
    window.setTimeout(() => recordingDatasetInputRef.current?.focus(), 0);
    toast.success('Enter a dataset name, then press Enter or leave the field');
  }, [isEditable, setRecordingTarget]);

  const loadExistingDatasetPlan = useCallback(async (recordingFolder) => {
    setSelectedDatasetEpisodeNumber(0);
    if (!getDatasetInfo || !getRecordingDatasetFolderInfo(recordingFolder)) {
      return;
    }
    setDatasetPlanLoading(true);
    try {
      const result = await getDatasetInfo(recordingFolder);
      if (!result?.success) return;
      const info = result.dataset_info || {};
      setSelectedDatasetEpisodeNumber(getNextDatasetEpisodeNumber(info));
      const existingSubtasks = Array.isArray(info.subtask_instruction)
        ? info.subtask_instruction.map((item) => String(item || '').trim())
        : [];
      setSubtaskInstructions(
        existingSubtasks.length > 0
          ? existingSubtasks.map(
            (instruction, index) => instruction || `Subtask ${index + 1}`
          )
          : []
      );
      resetSubtaskProgress();
      const existingInstruction = String(info.task_instruction || '').trim();
      const currentInstruction = String(
        taskInfo.taskInstruction?.[0] || ''
      ).trim();
      if (
        existingInstruction &&
        (!currentInstruction || currentInstruction === 'ACT_dataset')
      ) {
        dispatch(setSharedTaskInstruction([existingInstruction]));
      }
    } catch (error) {
      console.error('Failed to load existing dataset recording plan:', error);
    } finally {
      setDatasetPlanLoading(false);
    }
  }, [
    dispatch,
    getDatasetInfo,
    resetSubtaskProgress,
    taskInfo.taskInstruction,
  ]);

  useEffect(() => {
    const selectedFolder = String(
      taskInfo.recordingFolder || replayBufferPath || ''
    ).trim().replace(/\/+$/, '');
    const serverFolder = String(
      recordStatus.recordingFolder || ''
    ).trim().replace(/\/+$/, '');
    if (!selectedFolder || serverFolder !== selectedFolder) return;
    const serverEpisodeNumber = Number(recordStatus.currentEpisodeNumber);
    if (!Number.isInteger(serverEpisodeNumber) || serverEpisodeNumber < 0) return;
    setSelectedDatasetEpisodeNumber((current) => (
      Math.max(current, serverEpisodeNumber)
    ));
  }, [
    recordStatus.currentEpisodeNumber,
    recordStatus.recordingFolder,
    replayBufferPath,
    taskInfo.recordingFolder,
  ]);

  const handleEpisodeSaved = useCallback(() => {
    resetSubtaskProgress();
    setSelectedDatasetEpisodeNumber((current) => current + 1);
  }, [resetSubtaskProgress]);

  const finalizeRecordingTarget = useCallback(() => {
    const rawValue = String(recordingDatasetName || '').trim();
    try {
      if (!rawValue) {
        throw new Error('Enter an inference dataset name');
      }
      const resolved = createNamedRecordingFolder(rawValue);
      awaitingDatasetNameRef.current = false;
      setRecordingDatasetName(getRecordingDatasetName(resolved));
      setRecordingTarget(resolved);
      loadExistingDatasetPlan(resolved);
      return resolved;
    } catch (error) {
      toast.error(error?.message || 'Invalid recording dataset');
      return '';
    }
  }, [loadExistingDatasetPlan, recordingDatasetName, setRecordingTarget]);

  useEffect(() => {
    if (
      !isActive ||
      !isRecordingWorkspace ||
      !isEditable ||
      datasetPlanLoading
    ) return undefined;
    if (awaitingDatasetNameRef.current) return undefined;

    let target = String(recordingFolderRef.current || '').trim();
    if (!target) return undefined;
    if (!getRecordingDatasetFolderInfo(target)) return undefined;

    const timer = window.setTimeout(async () => {
      try {
        const commandOptions = prepareRecording();
        const signature = JSON.stringify({
          recordingFolder: commandOptions.recordingFolder,
          taskInstruction: commandOptions.taskInstruction,
          subtaskInstruction: commandOptions.subtaskInstruction || [],
          includeRobotisLicense: commandOptions.includeRobotisLicense,
          robotType,
        });
        if (
          signature === preparedSessionSignatureRef.current ||
          signature === preparingSessionSignatureRef.current
        ) {
          return;
        }
        preparingSessionSignatureRef.current = signature;
        const result = await sendRecordCommand('prepare_session', {
          ...commandOptions,
          taskSource: 'inference',
        });
        if (!result?.success) {
          throw new Error(result?.message || 'Recording session preparation failed');
        }
        preparedSessionSignatureRef.current = signature;
      } catch (error) {
        console.error('Failed to prepare Playground recording session:', error);
      } finally {
        preparingSessionSignatureRef.current = '';
      }
    }, 250);
    return () => window.clearTimeout(timer);
  }, [
    isActive,
    datasetPlanLoading,
    isEditable,
    isRecordingWorkspace,
    prepareRecording,
    robotType,
    sendRecordCommand,
    setRecordingTarget,
  ]);

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
        {['model'].map((name) => (
          <WorkspacePathField
            key={name}
            name={name}
            value={pathValues[name]}
            onChange={(value) => updatePath(name, value)}
            onBrowse={() => setBrowserTarget(name)}
            disabled={!isEditable}
          />
        ))}
        <RecordingDatasetNameField
          value={recordingDatasetName}
          resolvedPath={recordingTarget}
          onChange={updateRecordingDatasetName}
          onCommit={finalizeRecordingTarget}
          onBrowse={() => setBrowserTarget('recording')}
          disabled={!isEditable}
        />
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
            columnWeights={[9, 16, 9]}
            edgeToEdge
            imageFit="contain"
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
            startBlocked={!subtaskPlanComplete || !recordingTarget}
            segmentedRecording={isSubtaskMode}
            segmentIndex={activeSubtaskIndex}
            canFinalize={
              !isSubtaskMode ||
              activeSubtaskIndex === subtaskInstructions.length - 1
            }
            discardEpisodeOnCancel={isSubtaskMode}
            onEpisodeSaved={handleEpisodeSaved}
            onEpisodeCancelled={resetSubtaskProgress}
            episodeCountOverride={selectedDatasetEpisodeNumber}
            guideMessage={isSubtaskMode
              ? (
                datasetPlanLoading
                  ? 'Loading the selected dataset plan'
                  : subtaskPlanComplete
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
            startBlockedReason={flowSdeJobBusy
              ? 'Stop the Flow-SDE PPO job before standard VLA inference'
              : ''}
          />
        )}
        {!isRecordingWorkspace && (
          <FlowSDEPPOInferenceControls
            isActive={isActive}
            inferencePhase={inferencePhase}
            getRolloutStatus={getFlowSDEPPOPolicyRolloutStatus}
            onStartRollout={onStartFlowSDEPPOPolicyRollout}
            onStopRollout={onStopFlowSDEPPOPolicyRollout}
            onSubmitOutcome={onSubmitFlowSDEPPOPolicyRolloutOutcome}
            getValueWarmupStatus={getFlowSDEPPOValueWarmupStatus}
            onBusyChange={setFlowSdeJobBusy}
            onRolloutBundleChange={onFlowSDEPPOPolicyRolloutBundleChange}
          />
        )}
      </div>

      <OfflineRLWorkspaceStatusModal
        isOpen={isActive && workspaceStatusOpen}
        onClose={onCloseWorkspaceStatus}
        workspaceMode={workspaceMode}
        settingsContent={(
        <div
          className="rounded-xl border border-[#e2dcd1] bg-[#f6f3ec] p-3"
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
                <RecordingDatasetNameField
                  value={recordingDatasetName}
                  resolvedPath={recordingTarget}
                  onChange={updateRecordingDatasetName}
                  onCommit={finalizeRecordingTarget}
                  onBrowse={() => setBrowserTarget('recording')}
                  disabled={!isEditable}
                  inputRef={recordingDatasetInputRef}
                />
                <button
                  type="button"
                  onClick={handleNewRecordingDataset}
                  disabled={!isEditable}
                  className="mt-2 flex h-9 w-full items-center justify-center gap-1.5 rounded-lg border border-[#b8c6b7] bg-[#e4ebe3] px-3 text-[10px] font-semibold text-[#607563] hover:bg-[#d9e4d8] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <MdCreateNewFolder size={16} /> New Dataset
                </button>
                <p className="mt-2 text-[9px] leading-relaxed text-[#8f877b]">
                  이름만 입력하면 /workspace/rosbag2/Task_&lt;name&gt;_MCAP에
                  저장됩니다. 같은 이름을 사용하면 다음 episode 번호부터
                  이어서 쌓입니다.
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
        )}
      />

      <FileBrowserModal
        key={browserTarget || 'closed'}
        isOpen={Boolean(browserTarget)}
        onClose={() => setBrowserTarget(null)}
        onFileSelect={(item) => {
          if (browserTarget) {
            if (browserTarget === 'recording') {
              const selectedPath = item?.full_path || '';
              if (updateRecordingTarget(selectedPath)) {
                if (isRecordingWorkspace) {
                  loadExistingDatasetPlan(selectedPath);
                }
              }
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
        overlayZClass="z-[80]"
      />
    </div>
  );
}
