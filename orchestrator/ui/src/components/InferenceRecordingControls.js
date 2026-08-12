// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useRef } from 'react';
import { shallowEqual, useDispatch, useSelector, useStore } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
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

export default function InferenceRecordingControls() {
  const dispatch = useDispatch();
  const store = useStore();
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const inferencePhase = useSelector(
    (state) => state.tasks.inferenceStatus.inferencePhase
  );
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const control = useSelector(selectInferenceRecordingControl, shallowEqual);
  const { sendRecordCommand } = useRosServiceCaller();
  const commandPendingRef = useRef(false);

  useEffect(() => {
    if (
      control.uiPhase === InferenceRecordingUiPhase.IDLE ||
      control.uiPhase === InferenceRecordingUiPhase.RECORDING
    ) {
      commandPendingRef.current = false;
    }
  }, [control.uiPhase]);

  const visible =
    taskInfo.inferenceMode === 'robot' && taskInfo.recordInferenceMode;
  const canStart =
    inferencePhase === InferencePhase.INFERENCING &&
    recordStatus.recordPhase === RecordPhase.READY &&
    control.uiPhase === InferenceRecordingUiPhase.IDLE;
  const canLabel = control.active && !control.pending && !control.serverSaving;

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
      const result = await sendRecordCommand('start_inference_record');
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
      toast.success('Inference recording started');
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Recording start failed');
    }
  }, [canStart, dispatch, recoverFromServer, sendRecordCommand, store]);

  const handleOutcome = useCallback(async (episodeOutcome, label) => {
    if (!canLabel || commandPendingRef.current) return;
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.SAVING
    ));
    try {
      const result = await sendRecordCommand('stop_inference_record', {
        episodeOutcome,
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording save failed');
      }
      toast.success(`Episode saved as ${label}`);
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Recording save failed');
    }
  }, [canLabel, dispatch, recoverFromServer, sendRecordCommand]);

  if (!visible) return null;

  const recordLabel = {
    [InferenceRecordingUiPhase.STARTING]: 'Starting',
    [InferenceRecordingUiPhase.RECORDING]: 'Recording',
    [InferenceRecordingUiPhase.SAVING]: 'Saving',
  }[control.uiPhase] || 'Record';

  const buttonClass = (enabled, activeClass) => clsx(
    'h-9 min-w-[92px] px-2.5 rounded-lg flex items-center justify-center',
    'gap-1 text-base font-semibold transition-colors',
    enabled
      ? `${activeClass} cursor-pointer`
      : 'bg-gray-100 text-gray-400 cursor-not-allowed opacity-60'
  );

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
    </div>
  );
}
