// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useRef } from 'react';
import { shallowEqual, useDispatch, useSelector, useStore } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCancel,
  MdCheckCircle,
  MdFiberManualRecord,
  MdHighlightOff,
} from 'react-icons/md';
import { EpisodeOutcome } from '../constants/taskCommand';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import {
  InferenceRecordingUiPhase,
  selectInferenceRecordingControl,
  selectInferenceTaskInfo,
  setInferenceRecordingUiPhase,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { buildRosbagStartWarningMessage } from '../utils/recordingMonitorWarnings';

const phaseFromServer = (state) => {
  const status = state.tasks.recordStatus || {};
  const owned = status.taskType === 'inference' ||
    Boolean(status.recordInferenceMode);
  if (owned && status.recordPhase === RecordPhase.RECORDING) {
    return InferenceRecordingUiPhase.RECORDING;
  }
  if (owned && status.recordPhase === RecordPhase.SAVING) {
    return InferenceRecordingUiPhase.SAVING;
  }
  return InferenceRecordingUiPhase.IDLE;
};

export default function InferenceRecordingControls({
  variant = 'default',
  inferenceActions = null,
  guideMessage = '',
  guideSpinner = '',
  policyEpoch = null,
  mode = 'inference',
  isActive = true,
  prepareRecording = null,
  startBlocked = false,
  segmentedRecording = false,
  segmentIndex = 0,
  canFinalize = true,
  discardEpisodeOnCancel = false,
  onEpisodeSaved = null,
  onEpisodeCancelled = null,
}) {
  const dispatch = useDispatch();
  const store = useStore();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const inferencePhase = useSelector(
    (state) => state.tasks.inferenceStatus.inferencePhase
  );
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const robotType = useSelector((state) => state.tasks.robotType);
  const recordingMonitor = useSelector((state) => state.tasks.recordingMonitor);
  const control = useSelector(selectInferenceRecordingControl, shallowEqual);
  const { sendRecordCommand } = useRosServiceCaller();
  const commandPendingRef = useRef(false);
  const activeCommandOptionsRef = useRef({});
  const isStandaloneRecording = mode === 'recording';
  const parsedPolicyEpoch = Number(policyEpoch);
  const displayedPolicyEpoch = (
    Number.isInteger(parsedPolicyEpoch) && parsedPolicyEpoch >= 0
  ) ? parsedPolicyEpoch : null;

  useEffect(() => {
    if (
      control.uiPhase === InferenceRecordingUiPhase.IDLE ||
      control.uiPhase === InferenceRecordingUiPhase.RECORDING
    ) {
      commandPendingRef.current = false;
    }
  }, [control.uiPhase]);

  const recordingAvailable =
    isStandaloneRecording
      ? isActive && Boolean(String(robotType || '').trim())
      : taskInfo.inferenceMode === 'robot' && taskInfo.recordInferenceMode;
  const canStart =
    recordingAvailable &&
    inferencePhase === (
      isStandaloneRecording
        ? InferencePhase.READY
        : InferencePhase.INFERENCING
    ) &&
    recordStatus.recordPhase === RecordPhase.READY &&
    (!isStandaloneRecording || recordStatus.recordingOperationStatus !== 'running') &&
    control.uiPhase === InferenceRecordingUiPhase.IDLE &&
    !startBlocked;
  const canCancel = control.active && !control.pending && !control.serverSaving;
  const canLabel = canCancel && canFinalize;

  const recoverFromServer = useCallback(() => {
    dispatch(setInferenceRecordingUiPhase(phaseFromServer(store.getState())));
    commandPendingRef.current = false;
  }, [dispatch, store]);

  const handleRecord = useCallback(async () => {
    if (!canStart || commandPendingRef.current) return;
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.STARTING
    ));
    try {
      const commandOptions = isStandaloneRecording
        ? (await prepareRecording?.()) || {}
        : {};
      activeCommandOptionsRef.current = commandOptions;
      if (isStandaloneRecording) {
        const warningMessage = buildRosbagStartWarningMessage(recordingMonitor);
        if (warningMessage) {
          toast.error(warningMessage, { duration: 9000 });
        }
      }
      const startCommand = isStandaloneRecording
        ? (segmentedRecording ? 'start_segment' : 'start_record')
        : 'start_inference_record';
      const result = await sendRecordCommand(startCommand, {
        ...commandOptions,
        ...(segmentedRecording
          ? { segmentIndex: Number(commandOptions.segmentIndex ?? segmentIndex) }
          : {}),
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording start failed');
      }
      const phase = store.getState().tasks.inferenceRecordingUi.phase;
      if (phase === InferenceRecordingUiPhase.STARTING) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.RECORDING
        ));
      }
      commandPendingRef.current = false;
      toast.success(
        isStandaloneRecording
          ? 'Recording started'
          : 'Inference recording started'
      );
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Recording start failed');
    }
  }, [
    canStart,
    dispatch,
    isStandaloneRecording,
    prepareRecording,
    recordingMonitor,
    recoverFromServer,
    segmentIndex,
    segmentedRecording,
    sendRecordCommand,
    store,
  ]);

  const handleOutcome = useCallback(async (episodeOutcome, label) => {
    if (!canLabel || commandPendingRef.current) return;
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.SAVING
    ));
    try {
      const result = await sendRecordCommand('stop_inference_record', {
        ...(isStandaloneRecording ? activeCommandOptionsRef.current : {}),
        episodeOutcome,
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording save failed');
      }
      await onEpisodeSaved?.({
        episodeOutcome,
        commandOptions: activeCommandOptionsRef.current,
      });
      toast.success(`Episode saved as ${label}`);
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Recording save failed');
    }
  }, [
    canLabel,
    dispatch,
    isStandaloneRecording,
    onEpisodeSaved,
    recoverFromServer,
    sendRecordCommand,
  ]);

  const handleCancel = useCallback(async () => {
    if (!canCancel || commandPendingRef.current) return;
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.CANCELLING
    ));
    try {
      const result = await sendRecordCommand('cancel_inference_record', {
        ...(isStandaloneRecording ? activeCommandOptionsRef.current : {}),
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording cancel failed');
      }
      if (isStandaloneRecording && discardEpisodeOnCancel) {
        const discardResult = await sendRecordCommand('discard_episode', {
          ...activeCommandOptionsRef.current,
          segmentIndex: 0,
          taskSource: 'inference',
        });
        if (!discardResult?.success) {
          throw new Error(
            discardResult?.message || 'Saved subtask cleanup failed'
          );
        }
      }
      await onEpisodeCancelled?.({
        commandOptions: activeCommandOptionsRef.current,
      });
      toast.success('Episode discarded');
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Recording cancel failed');
    }
  }, [
    canCancel,
    dispatch,
    discardEpisodeOnCancel,
    isStandaloneRecording,
    onEpisodeCancelled,
    recoverFromServer,
    sendRecordCommand,
  ]);

  const isWorkspace = variant === 'workspace';

  if (!isWorkspace && !recordingAvailable) return null;

  const recordLabel = {
    [InferenceRecordingUiPhase.STARTING]: 'Starting',
    [InferenceRecordingUiPhase.RECORDING]: 'Recording',
    [InferenceRecordingUiPhase.SAVING]: 'Saving',
    [InferenceRecordingUiPhase.CANCELLING]: 'Cancelling',
  }[control.uiPhase] || 'Record';
  const workspaceTitle = isStandaloneRecording
    ? 'Recording'
    : 'Inference Recording';
  const rolloutName = isStandaloneRecording ? 'recording' : 'inference rollout';
  const displayedGuideMessage = guideMessage || (
    isStandaloneRecording
      ? (recordingAvailable ? 'Ready to record' : 'Select a robot type first')
      : ''
  );

  const buttonClass = (enabled, activeClass) => clsx(
    'h-9 min-w-[92px] px-2.5 rounded-lg flex items-center justify-center',
    'gap-1 text-base font-semibold transition-colors',
    enabled
      ? `${activeClass} cursor-pointer`
      : 'bg-gray-100 text-gray-400 cursor-not-allowed opacity-60'
  );

  if (isWorkspace) {
    const workspaceButtonClass = (enabled, activeClass) => clsx(
      'flex h-9 items-center justify-center gap-1 rounded-lg px-2',
      'text-[10px] font-semibold transition-colors',
      enabled
        ? `${activeClass} cursor-pointer`
        : 'cursor-not-allowed bg-[#ece8df] text-[#aaa295] opacity-60'
    );

    return (
      <div
        className="rounded-xl border border-[#ded8cc] bg-[#f6f3ec] p-3 shadow-[0_2px_8px_rgba(69,61,47,0.04)]"
        role="group"
        aria-label={`${workspaceTitle} controls`}
        data-appearance="offline-rl"
      >
        <div className="mb-2 flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <span className="shrink-0 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#81796d]">
              {workspaceTitle}
            </span>
            {displayedPolicyEpoch !== null && (
              <span
                className="shrink-0 rounded-md border border-[#cfd8cd] bg-[#e8eee6] px-1.5 py-0.5 font-mono text-[9px] font-bold text-[#58705d]"
                aria-label={`Current policy RL Epoch ${displayedPolicyEpoch}`}
              >
                RL Epoch E{String(displayedPolicyEpoch).padStart(4, '0')}
              </span>
            )}
            {(displayedGuideMessage || guideSpinner) && (
              <>
                <span className="h-3.5 w-px shrink-0 bg-[#d2cabd]" />
                <span
                  className="min-w-0 truncate text-[10px] font-medium text-[#665f54]"
                  title={displayedGuideMessage}
                  role="status"
                  aria-live="polite"
                >
                  {displayedGuideMessage}
                </span>
                {guideSpinner && (
                  <span className="shrink-0 font-mono text-[10px] text-[#69866f]">
                    {guideSpinner}
                  </span>
                )}
              </>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-1 text-[9px] font-medium text-[#81796d]">
            <span>Episodes</span>
            <span
              className="min-w-7 rounded-md bg-[#e7e1d7] px-1.5 py-0.5 text-center font-bold text-[#5f584e]"
              aria-label={isStandaloneRecording
                ? 'Saved recording episodes'
                : 'Saved inference episodes'}
            >
              {control.episodeCount}
            </span>
          </div>
        </div>

        <div className={clsx('grid gap-2', inferenceActions ? 'grid-cols-4' : 'grid-cols-1')}>
          <button
            type="button"
            onClick={handleRecord}
            disabled={!canStart}
            className={clsx(
              workspaceButtonClass(
                canStart,
                'bg-[#f1e2dd] text-[#9c6158] hover:bg-[#ead5cf]'
              ),
              'w-full'
            )}
            aria-label={isStandaloneRecording
              ? 'Start recording'
              : 'Record inference rollout'}
          >
            <MdFiberManualRecord size={16} />
            {recordLabel}
          </button>
          {inferenceActions}
        </div>

        <div className="mt-2 grid grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => handleOutcome(EpisodeOutcome.SUCCESS, 'Success')}
            disabled={!canLabel}
            className={workspaceButtonClass(
              canLabel,
              'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
            )}
            aria-label={`Save ${rolloutName} as Success`}
          >
            <MdCheckCircle size={16} />
            Success
          </button>
          <button
            type="button"
            onClick={() => handleOutcome(EpisodeOutcome.FAILURE, 'Fail')}
            disabled={!canLabel}
            className={workspaceButtonClass(
              canLabel,
              'bg-[#f1e2df] text-[#995e58] hover:bg-[#ead5d1]'
            )}
            aria-label={`Save ${rolloutName} as Fail`}
          >
            <MdHighlightOff size={16} />
            Fail
          </button>
          <button
            type="button"
            onClick={handleCancel}
            disabled={!canCancel}
            className={workspaceButtonClass(
              canCancel,
              'bg-[#e8e3da] text-[#655e53] hover:bg-[#ddd6ca]'
            )}
            aria-label={`Cancel and discard ${rolloutName}`}
          >
            <MdCancel size={16} />
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex items-center gap-1.5 rounded-full border border-gray-100 bg-white/90 px-3 py-1 shadow-md backdrop-blur-sm"
      role="group"
      aria-label="RL Recording controls"
    >
      <span className="shrink-0 whitespace-nowrap px-1 text-base font-semibold text-gray-500">
        RL Recording
      </span>
      <div className="h-6 w-px shrink-0 bg-gray-300" />
      <div className="flex shrink-0 items-center gap-1 px-1 text-sm font-medium text-gray-500">
        <span>EP</span>
        <span
          className="min-w-7 rounded bg-gray-100 px-1.5 py-0.5 text-center font-bold text-gray-700"
          aria-label="Saved RL episodes"
        >
          {control.episodeCount}
        </span>
      </div>
      <div className="h-6 w-px shrink-0 bg-gray-300" />
      <button
        type="button"
        onClick={handleRecord}
        disabled={!canStart}
        className={buttonClass(canStart, 'bg-red-50 text-red-600 hover:bg-red-100')}
        aria-label="Record inference rollout"
      >
        <MdFiberManualRecord size={17} />
        {recordLabel}
      </button>
      <button
        type="button"
        onClick={() => handleOutcome(EpisodeOutcome.SUCCESS, 'Success')}
        disabled={!canLabel}
        className={buttonClass(
          canLabel,
          'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
        )}
        aria-label="Save inference rollout as Success"
      >
        <MdCheckCircle size={17} />
        Success
      </button>
      <button
        type="button"
        onClick={() => handleOutcome(EpisodeOutcome.FAILURE, 'Failed')}
        disabled={!canLabel}
        className={buttonClass(
          canLabel,
          'bg-rose-50 text-rose-700 hover:bg-rose-100'
        )}
        aria-label="Save inference rollout as Failed"
      >
        <MdHighlightOff size={17} />
        Failed
      </button>
      <button
        type="button"
        onClick={handleCancel}
        disabled={!canLabel}
        className={buttonClass(
          canLabel,
          'bg-gray-100 text-gray-700 hover:bg-gray-200'
        )}
        aria-label="Cancel and discard inference rollout"
      >
        <MdCancel size={17} />
        Cancel
      </button>
    </div>
  );
}
