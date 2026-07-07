// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdArrowDropDown,
  MdArrowDropUp,
  MdDelete,
  MdDeleteSweep,
  MdDone,
  MdFiberManualRecord,
  MdSave,
} from 'react-icons/md';

import { RecordPhase } from '../constants/taskPhases';
import {
  markLocalTaskInfoEdited,
  resetSegmentPlan,
  resetSegmentProgress,
  selectRecordTaskInfo,
  setActiveSlotIndex,
  setPlannedCount,
  setPlannedSubTaskAt,
  setPlannedSubTasks,
  setSlotToServerIdx,
} from '../features/tasks/taskSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { buildRosbagStartWarningMessage } from '../utils/recordingMonitorWarnings';
import InfoPanel from './InfoPanel';
import Tooltip from './Tooltip';

const MAX_PLANNED_SLOTS = 50;

const isInputFocused = () => {
  const el = document.activeElement;
  if (!el) return false;
  const tag = el.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || el.contentEditable === 'true';
};

export default function SegmentPanel() {
  const dispatch = useDispatch();
  const { sendRecordCommand } = useRosServiceCaller();

  const taskInfo = useSelector(selectRecordTaskInfo, shallowEqual);
  const recordStatus = useSelector((state) => state.tasks.recordStatus);
  const plannedCount = useSelector((state) => state.tasks.plannedCount);
  const plannedSubTasks = useSelector((state) => state.tasks.plannedSubTasks);
  const slotToServerIdx = useSelector((state) => state.tasks.slotToServerIdx);
  const activeSlotIndex = useSelector((state) => state.tasks.activeSlotIndex);
  const recordingMonitor = useSelector((state) => state.tasks.recordingMonitor);

  const [optimisticRecording, setOptimisticRecording] = useState(false);
  const [episodeAcquisitionStarted, setEpisodeAcquisitionStarted] = useState(false);
  const [savingInProgress, setSavingInProgress] = useState(false);
  const [serverResetInProgress, setServerResetInProgress] = useState(false);
  const episodeFullIndexRef = useRef(null);
  const lastServerEpisodeRef = useRef(null);
  const commandSequenceInProgressRef = useRef(false);
  const pendingEpisodeResetRef = useRef(null);

  const serverRecording =
    recordStatus.recordPhase === RecordPhase.RECORDING || Boolean(recordStatus.running);
  const isRecording = serverRecording || optimisticRecording;
  const plannedCountNumber = Number(plannedCount) || 0;
  const savedCount = useMemo(
    () => slotToServerIdx.filter((v) => v >= 0).length,
    [slotToServerIdx]
  );
  const serverSubtaskIndex = Number(recordStatus.currentSubtaskIndex || 0);
  const serverSubtaskCount = Number(recordStatus.subtaskCount || 0);
  const serverSavedSubtaskIndices = useMemo(() => (
    Array.isArray(recordStatus.savedSubtaskIndices)
      ? recordStatus.savedSubtaskIndices
          .map((idx) => Number(idx))
          .filter((idx) => Number.isInteger(idx) && idx >= 0)
      : null
  ), [recordStatus.savedSubtaskIndices]);
  const hasServerSavedSubtasks =
    Array.isArray(serverSavedSubtaskIndices)
      ? serverSavedSubtaskIndices.length > 0
      : serverSubtaskCount > 0 && serverSubtaskIndex > 0 && !isRecording;
  const hasLocalSavedSubtasks = savedCount > 0;
  const isPlanMode = plannedCountNumber > 0;
  const isSingleMode = plannedCountNumber === 0;
  const currentFullEpisodeIndex = Number(recordStatus.currentEpisodeNumber || 0);
  const firstPendingSlot = useMemo(
    () => slotToServerIdx.findIndex((v) => v === -1),
    [slotToServerIdx]
  );
  const planComplete = isPlanMode && firstPendingSlot === -1;
  const taskInfoComplete = Boolean(
    String(taskInfo.taskNum ?? '').trim()
  );
  const allSubTasksFilled = useMemo(
    () =>
      isSingleMode ||
      (isPlanMode &&
        plannedSubTasks.length === plannedCountNumber &&
        plannedSubTasks.every((s) => !!(s || '').trim())),
    [isSingleMode, isPlanMode, plannedSubTasks, plannedCountNumber]
  );

  useEffect(() => {
    setOptimisticRecording(serverRecording);
  }, [serverRecording]);

  useEffect(() => {
    if (serverRecording || hasLocalSavedSubtasks || hasServerSavedSubtasks) {
      setEpisodeAcquisitionStarted(true);
      if (episodeFullIndexRef.current === null) {
        episodeFullIndexRef.current = currentFullEpisodeIndex;
      }
    }
  }, [currentFullEpisodeIndex, hasLocalSavedSubtasks, hasServerSavedSubtasks, serverRecording]);

  useEffect(() => {
    if (!recordStatus.topicReceived) return;
    if (commandSequenceInProgressRef.current) return;

    if (pendingEpisodeResetRef.current) {
      const pendingReset = pendingEpisodeResetRef.current;
      const operationFailed = (
        pendingReset.type === 'finish' &&
        recordStatus.recordingOperationStage === 'FINISH_EPISODE' &&
        ['failed', 'cancelled'].includes(recordStatus.recordingOperationStatus)
      );
      const savedStatusKnown = Array.isArray(serverSavedSubtaskIndices);
      const noServerSavedSubtasks = (
        !savedStatusKnown || serverSavedSubtaskIndices.length === 0
      );
      const resetObserved = pendingReset.type === 'finish'
        ? (
          !serverRecording &&
          serverSubtaskIndex === 0 &&
          currentFullEpisodeIndex > pendingReset.fullEpisodeIndex
        )
        : (
          !serverRecording &&
          serverSubtaskIndex === 0 &&
          noServerSavedSubtasks
        );
      if (operationFailed) {
        pendingEpisodeResetRef.current = null;
        setServerResetInProgress(false);
        setEpisodeAcquisitionStarted(true);
        toast.error(
          recordStatus.recordingOperationMessage ||
          'Finish episode failed; saved subtasks are still available.'
        );
      } else if (resetObserved) {
        pendingEpisodeResetRef.current = null;
        setServerResetInProgress(false);
        lastServerEpisodeRef.current = currentFullEpisodeIndex;
        dispatch(resetSegmentProgress());
      } else {
        return;
      }
    }

    const currentEpisode = currentFullEpisodeIndex;
    if (lastServerEpisodeRef.current === null) {
      lastServerEpisodeRef.current = currentEpisode;
    } else if (
      isPlanMode &&
      !serverRecording &&
      currentEpisode > lastServerEpisodeRef.current
    ) {
      lastServerEpisodeRef.current = currentEpisode;
      setOptimisticRecording(false);
      setSavingInProgress(false);
      setEpisodeAcquisitionStarted(false);
      episodeFullIndexRef.current = null;
      dispatch(resetSegmentProgress());
      return;
    } else {
      lastServerEpisodeRef.current = currentEpisode;
    }

    if (!isPlanMode || serverSubtaskCount <= 0 || plannedCountNumber <= 0) {
      return;
    }

    const boundedServerSlot = Math.max(
      0,
      Math.min(serverSubtaskIndex, plannedCountNumber - 1)
    );
    const savedSet = Array.isArray(serverSavedSubtaskIndices)
      ? new Set(serverSavedSubtaskIndices)
      : null;
    const syncedSlotMap = savedSet
      ? slotToServerIdx.map((_, index) => (savedSet.has(index) ? index : -1))
      : slotToServerIdx.map((value, index) => (
        index < boundedServerSlot && value < 0 ? index : value
      ));
    const slotMapChanged = syncedSlotMap.some(
      (value, index) => value !== slotToServerIdx[index]
    );

    if (slotMapChanged) {
      dispatch(setSlotToServerIdx(syncedSlotMap));
    }

    const firstServerPendingSlot = savedSet
      ? syncedSlotMap.findIndex((value) => value < 0)
      : -1;
    const nextActiveSlot = (
      savedSet && firstServerPendingSlot >= 0 && !serverRecording
    )
      ? firstServerPendingSlot
      : boundedServerSlot;
    if (activeSlotIndex !== nextActiveSlot && !planComplete) {
      dispatch(setActiveSlotIndex(nextActiveSlot));
    }

    if (serverRecording) {
      setSavingInProgress(false);
      setEpisodeAcquisitionStarted(true);
    }
  }, [
    activeSlotIndex,
    dispatch,
    isPlanMode,
    planComplete,
    plannedCountNumber,
    currentFullEpisodeIndex,
    recordStatus.recordingOperationMessage,
    recordStatus.recordingOperationStage,
    recordStatus.recordingOperationStatus,
    recordStatus.topicReceived,
    serverRecording,
    serverResetInProgress,
    serverSavedSubtaskIndices,
    serverSubtaskCount,
    serverSubtaskIndex,
    slotToServerIdx,
  ]);

  const minAllowedCount = useMemo(() => {
    let highest = -1;
    for (let i = 0; i < slotToServerIdx.length; i += 1) {
      if (slotToServerIdx[i] >= 0) highest = i;
    }
    return highest + 1;
  }, [slotToServerIdx]);

  const applyPlanCount = useCallback(
    (n) => {
      const bounded = Math.max(0, Math.min(MAX_PLANNED_SLOTS, n));
      if (bounded === plannedCountNumber) return;
      let nextSubTasks;
      let nextSlotMap;
      if (bounded > plannedCountNumber) {
        const add = bounded - plannedCountNumber;
        nextSubTasks = [...plannedSubTasks, ...Array(add).fill('')];
        nextSlotMap = [...slotToServerIdx, ...Array(add).fill(-1)];
      } else {
        nextSubTasks = plannedSubTasks.slice(0, bounded);
        nextSlotMap = slotToServerIdx.slice(0, bounded);
      }
      dispatch(setPlannedCount(bounded));
      dispatch(setPlannedSubTasks(nextSubTasks));
      dispatch(setSlotToServerIdx(nextSlotMap));
      const nextPending = nextSlotMap.findIndex((v) => v === -1);
      dispatch(setActiveSlotIndex(nextPending >= 0 ? nextPending : Math.max(0, bounded - 1)));
      dispatch(markLocalTaskInfoEdited());
    },
    [dispatch, plannedCountNumber, plannedSubTasks, slotToServerIdx]
  );

  const handlePlanCountInput = useCallback(
    (rawValue) => {
      if (isRecording || savingInProgress) return;
      const n = parseInt(rawValue, 10);
      if (!Number.isFinite(n) || n < 0) return;
      if (n < minAllowedCount) {
        toast.error(`Cannot reduce below ${minAllowedCount}; subtasks are already saved`);
        applyPlanCount(minAllowedCount);
        return;
      }
      if (n > MAX_PLANNED_SLOTS) {
        toast.error(`Max ${MAX_PLANNED_SLOTS} subtasks per episode`);
      }
      applyPlanCount(n);
    },
    [applyPlanCount, isRecording, minAllowedCount, savingInProgress]
  );

  const runCommand = useCallback(
    async (label, command, opts = {}) => {
      try {
        const result = await sendRecordCommand(command, {
          subtaskInstruction: plannedSubTasks,
          ...opts,
        });
        if (result && result.success) {
          toast.success(`${label}: ${result.message || 'OK'}`);
        } else {
          toast.error(`${label} failed: ${result?.message || 'Unknown error'}`);
        }
        return result;
      } catch (error) {
        toast.error(`${label} failed: ${error.message || error}`);
        return null;
      }
    },
    [plannedSubTasks, sendRecordCommand]
  );

  const canStartRecord =
    !isRecording &&
    !savingInProgress &&
    !serverResetInProgress &&
    !planComplete &&
    allSubTasksFilled &&
    taskInfoComplete;

  const startRecordingSlot = useCallback(
    async (slotIdx) => {
      const subTask = (plannedSubTasks[slotIdx] || '').trim();
      if (!isSingleMode && !subTask) return null;
      dispatch(setActiveSlotIndex(slotIdx));
      setOptimisticRecording(true);
      setEpisodeAcquisitionStarted(true);
      if (episodeFullIndexRef.current === null) {
        episodeFullIndexRef.current = currentFullEpisodeIndex;
      }
      const result = await runCommand('Record', 'start_segment', {
        segmentIndex: slotIdx,
      });
      if (!result || result.success === false) {
        setOptimisticRecording(false);
        if (!hasLocalSavedSubtasks && !hasServerSavedSubtasks) {
          setEpisodeAcquisitionStarted(false);
        }
      }
      return result;
    },
    [
      currentFullEpisodeIndex,
      dispatch,
      hasLocalSavedSubtasks,
      hasServerSavedSubtasks,
      isSingleMode,
      plannedSubTasks,
      runCommand,
    ]
  );

  const handleRecordStart = useCallback(async () => {
    if (!canStartRecord) return;
    if (commandSequenceInProgressRef.current) return;
    commandSequenceInProgressRef.current = true;
    try {
      const warningMessage = buildRosbagStartWarningMessage(recordingMonitor);
      if (warningMessage) {
        toast.error(warningMessage, { duration: 9000 });
      }
      await startRecordingSlot(activeSlotIndex);
    } finally {
      commandSequenceInProgressRef.current = false;
    }
  }, [activeSlotIndex, canStartRecord, recordingMonitor, startRecordingSlot]);

  const handleSlotSave = useCallback(
    async (slotIdx) => {
      if (!isSingleMode && slotIdx !== activeSlotIndex) return;
      if (!isRecording || savingInProgress) return;
      if (serverResetInProgress) return;
      if (commandSequenceInProgressRef.current) return;
      commandSequenceInProgressRef.current = true;
      setSavingInProgress(true);
      setOptimisticRecording(false);
      const isLastSlot = isSingleMode || slotIdx >= plannedCountNumber - 1;
      try {
        const result = await runCommand('Save', 'stop_segment', {
          segmentIndex: slotIdx,
        });
        if (!result || result.success === false) {
          return;
        }

        const updatedSlotMap = isSingleMode
          ? slotToServerIdx
          : slotToServerIdx.map((v, i) => (i === slotIdx ? slotIdx : v));
        if (!isSingleMode) {
          dispatch(setSlotToServerIdx(updatedSlotMap));
        }

        const nextPending = updatedSlotMap.findIndex((v) => v === -1);
        if (isLastSlot || nextPending < 0) {
          const targetFullEpisodeIndex = (
            episodeFullIndexRef.current === null
              ? currentFullEpisodeIndex
              : episodeFullIndexRef.current
          );
          const finishResult = await runCommand('Finish episode', 'finish_episode');
          if (finishResult && finishResult.success) {
            pendingEpisodeResetRef.current = {
              type: 'finish',
              fullEpisodeIndex: targetFullEpisodeIndex,
            };
            setServerResetInProgress(true);
            setEpisodeAcquisitionStarted(false);
            episodeFullIndexRef.current = null;
            dispatch(resetSegmentProgress());
          }
          return;
        }

        dispatch(setActiveSlotIndex(nextPending));
        await startRecordingSlot(nextPending);
      } finally {
        setSavingInProgress(false);
        commandSequenceInProgressRef.current = false;
      }
    },
    [
      activeSlotIndex,
      currentFullEpisodeIndex,
      dispatch,
      isRecording,
      isSingleMode,
      plannedCountNumber,
      runCommand,
      savingInProgress,
      serverResetInProgress,
      slotToServerIdx,
      startRecordingSlot,
    ]
  );

  const handleSlotTrash = useCallback(
    async (slotIdx) => {
      if (savingInProgress) return;
      if (serverResetInProgress) return;
      if (commandSequenceInProgressRef.current) return;
      commandSequenceInProgressRef.current = true;
      const isActiveRecording = slotIdx === activeSlotIndex && isRecording;
      const serverIdx = slotToServerIdx[slotIdx];
      try {
        if (isActiveRecording) {
          setOptimisticRecording(false);
          await runCommand('Discard', 'cancel_segment', { segmentIndex: slotIdx });
          return;
        }
        if (serverIdx < 0) return;
        if (!window.confirm(`Discard subtask ${slotIdx + 1}?`)) return;
        const result = await runCommand(`Discard #${slotIdx + 1}`, 'discard_segment', {
          segmentIndex: slotIdx,
        });
        if (!result || result.success === false) return;
        const updated = slotToServerIdx.map((v, i) => (i === slotIdx ? -1 : v));
        dispatch(setSlotToServerIdx(updated));
        dispatch(setActiveSlotIndex(slotIdx));
      } finally {
        commandSequenceInProgressRef.current = false;
      }
    },
    [
      activeSlotIndex,
      dispatch,
      isRecording,
      runCommand,
      savingInProgress,
      serverResetInProgress,
      slotToServerIdx,
    ]
  );

  const handleDiscardEpisode = useCallback(async () => {
    if (
      savingInProgress ||
      serverResetInProgress ||
      !episodeAcquisitionStarted
    ) {
      return;
    }
    if (commandSequenceInProgressRef.current) return;
    commandSequenceInProgressRef.current = true;
    const targetFullEpisodeIndex = episodeFullIndexRef.current;
    if (targetFullEpisodeIndex === null) {
      commandSequenceInProgressRef.current = false;
      return;
    }
    try {
      const result = await runCommand('Discard episode', 'discard_episode', {
        segmentIndex: targetFullEpisodeIndex + 1,
      });
      if (result && result.success) {
        pendingEpisodeResetRef.current = {
          type: 'discard',
          fullEpisodeIndex: targetFullEpisodeIndex,
        };
        setServerResetInProgress(true);
        setOptimisticRecording(false);
        setEpisodeAcquisitionStarted(false);
        episodeFullIndexRef.current = null;
        dispatch(resetSegmentProgress());
      }
    } finally {
      commandSequenceInProgressRef.current = false;
    }
  }, [
    dispatch,
    episodeAcquisitionStarted,
    runCommand,
    savingInProgress,
    serverResetInProgress,
  ]);

  const canSaveSingle =
    isSingleMode && isRecording && !savingInProgress && !serverResetInProgress;

  const handlePrimaryRecordButton = useCallback(async () => {
    if (canSaveSingle) {
      await handleSlotSave(0);
      return;
    }
    if (canStartRecord) {
      await handleRecordStart();
    }
  }, [canSaveSingle, canStartRecord, handleRecordStart, handleSlotSave]);

  const handleResetPlan = useCallback(() => {
    setEpisodeAcquisitionStarted(false);
    episodeFullIndexRef.current = null;
    dispatch(resetSegmentPlan());
    dispatch(markLocalTaskInfoEdited());
  }, [dispatch]);

  const handleSubTaskChange = useCallback(
    (index, value) => {
      dispatch(setPlannedSubTaskAt({ index, value }));
      dispatch(markLocalTaskInfoEdited());
    },
    [dispatch]
  );

  useEffect(() => {
    const onKeyUp = (e) => {
      if (isInputFocused()) return;
      if (e.key === ' ' || e.code === 'Space') {
        if (canStartRecord) handleRecordStart();
      } else if ((e.ctrlKey || e.metaKey) && e.shiftKey && (e.key === 'x' || e.key === 'X')) {
        if (isRecording && !savingInProgress && !serverResetInProgress) {
          handleSlotSave(isSingleMode ? 0 : activeSlotIndex);
        }
      } else if (e.key === 'Escape') {
        if (isRecording && !savingInProgress && !serverResetInProgress) {
          handleSlotTrash(activeSlotIndex);
        }
      }
    };
    window.addEventListener('keyup', onKeyUp);
    return () => window.removeEventListener('keyup', onKeyUp);
  }, [
    activeSlotIndex,
    canStartRecord,
    handleRecordStart,
    handleSlotSave,
    handleSlotTrash,
    isRecording,
    isSingleMode,
    savingInProgress,
    serverResetInProgress,
  ]);

  const canResetPlan = (
    isPlanMode &&
    !isRecording &&
    !savingInProgress &&
    !serverResetInProgress &&
    savedCount === 0
  );
  const canDiscardEpisode =
    !savingInProgress && !serverResetInProgress && episodeAcquisitionStarted;

  const secondaryBtn = (enabled, color) =>
    clsx(
      'px-2.5 py-1.5 rounded-md text-sm font-semibold transition-colors',
      'flex items-center justify-center gap-1',
      enabled
        ? color === 'red'
          ? 'bg-red-500 text-white hover:bg-red-600'
          : 'bg-indigo-500 text-white hover:bg-indigo-600'
        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
    );

  const renderSlotRow = (i) => {
    const isSaved = slotToServerIdx[i] >= 0;
    const isActive = i === activeSlotIndex && !planComplete;
    const isCurrentlyRecording = isActive && isRecording;
    const inputDisabled = isSaved || isRecording || savingInProgress;
    const saveEnabled = isCurrentlyRecording && !savingInProgress && !serverResetInProgress;
    const trashEnabled = (
      !savingInProgress &&
      !serverResetInProgress &&
      (isCurrentlyRecording || (isSaved && !isRecording))
    );

    return (
      <div
        key={`slot-${i}`}
        className={clsx('flex items-center gap-2 px-2 py-1.5 rounded-md border', {
          'border-gray-100 opacity-70 bg-gray-50': isSaved,
          'border-red-300 bg-red-50': isCurrentlyRecording,
          'border-blue-200 bg-blue-50': isActive && !isCurrentlyRecording,
          'border-gray-100': !isSaved && !isActive,
        })}
      >
        <span
          className={clsx('text-xs font-mono w-6 shrink-0 text-center rounded flex items-center justify-center', {
            'bg-green-100 text-green-700': isSaved,
            'text-blue-700 font-bold': isActive && !isSaved,
            'text-gray-500': !isSaved && !isActive,
          })}
        >
          {isSaved ? <MdDone size={14} /> : `#${i + 1}`}
        </span>
        <input
          type="text"
          lang="ko"
          className={clsx(
            'flex-1 text-sm p-1 border border-gray-300 rounded-md min-w-0',
            'focus:outline-none focus:ring-2 focus:ring-blue-500',
            { 'bg-gray-100 cursor-not-allowed text-gray-500': inputDisabled }
          )}
          value={plannedSubTasks[i] || ''}
          placeholder="Sub Task Instruction"
          onChange={(e) => handleSubTaskChange(i, e.target.value)}
          disabled={inputDisabled}
        />
        <button
          onClick={() => handleSlotSave(i)}
          disabled={!saveEnabled}
          className={clsx(
            'px-2 py-1 rounded-md text-xs font-semibold flex items-center gap-1',
            saveEnabled
              ? 'bg-green-500 text-white hover:bg-green-600'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          )}
          aria-label={`Save subtask ${i + 1}`}
          title="Save this subtask"
        >
          <MdSave size={14} />
          Save
        </button>
        <button
          onClick={() => handleSlotTrash(i)}
          disabled={!trashEnabled}
          className={clsx(
            'p-1 rounded',
            trashEnabled ? 'hover:bg-red-50 text-red-500' : 'text-gray-300 cursor-not-allowed'
          )}
          aria-label={`Discard subtask ${i + 1}`}
          title={isCurrentlyRecording ? 'Cancel current recording' : 'Discard saved subtask'}
        >
          <MdDelete size={16} />
        </button>
      </div>
    );
  };

  return (
    <div className="bg-white border border-gray-200 rounded-2xl shadow-md p-4 w-full max-w-[350px] mt-3">
      <div className="mb-3 text-lg font-semibold text-gray-800">Rosbag Recorder</div>

      <div className="mb-3">
        <div className="text-sm font-semibold text-gray-700 mb-2">
          Task Information
          <span className="ml-1 text-xs font-normal text-gray-400">(required)</span>
        </div>
        <InfoPanel variant="embedded" />
      </div>

      <Tooltip
        position="top"
        content={
          <div className="text-center">
            <div className="font-semibold">
              {isPlanMode
                ? 'Start recording the active subtask'
                : canSaveSingle
                  ? 'Save the current single-task episode'
                  : 'Start single-task recording'}
            </div>
            {canStartRecord && (
              <div className="text-sm mt-1 text-gray-300">
                <span className="font-mono bg-gray-700 px-1 rounded">Space</span>
              </div>
            )}
          </div>
        }
        className="block w-full"
      >
        <button
          onClick={handlePrimaryRecordButton}
          disabled={!canStartRecord && !canSaveSingle}
          className={clsx(
            'w-full mb-3 px-3 py-2.5 rounded-lg font-semibold text-sm',
            'flex items-center justify-center gap-2 transition-colors',
            canStartRecord || canSaveSingle
              ? canSaveSingle
                ? 'bg-green-500 text-white hover:bg-green-600'
                : 'bg-red-500 text-white hover:bg-red-600'
              : 'bg-gray-200 text-gray-400 cursor-not-allowed'
          )}
        >
          {canSaveSingle ? <MdSave size={18} /> : <MdFiberManualRecord size={18} />}
          {canSaveSingle ? 'Save Episode' : 'Record Start'}
        </button>
      </Tooltip>

      <div className="mb-3">
        <div className="text-sm font-semibold text-gray-700 mb-2">
          Number of SubTasks
          <span className="ml-1 text-xs font-normal text-gray-400">(plan before recording)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex flex-1 items-stretch border border-gray-300 rounded-md overflow-hidden">
            <input
              type="number"
              min={minAllowedCount}
              max={MAX_PLANNED_SLOTS}
              value={plannedCount}
              onChange={(e) => handlePlanCountInput(e.target.value)}
              disabled={isRecording || savingInProgress}
              className="flex-1 text-sm p-1.5 outline-none focus:ring-2 focus:ring-blue-500 [appearance:textfield]"
            />
            <div className="flex flex-col border-l border-gray-300">
              <button
                type="button"
                onClick={() => handlePlanCountInput(plannedCountNumber + 1)}
                disabled={isRecording || savingInProgress || plannedCountNumber >= MAX_PLANNED_SLOTS}
                className="flex-1 px-1 border-b border-gray-300 text-gray-600 hover:bg-gray-100 disabled:text-gray-300"
              >
                <MdArrowDropUp size={18} />
              </button>
              <button
                type="button"
                onClick={() => handlePlanCountInput(plannedCountNumber - 1)}
                disabled={isRecording || savingInProgress || plannedCountNumber <= minAllowedCount}
                className="flex-1 px-1 text-gray-600 hover:bg-gray-100 disabled:text-gray-300"
              >
                <MdArrowDropDown size={18} />
              </button>
            </div>
          </div>
          <button
            onClick={handleResetPlan}
            disabled={!canResetPlan}
            className={clsx(
              'px-3 py-1.5 rounded-md text-sm font-semibold transition-colors',
              canResetPlan
                ? 'bg-gray-300 text-gray-800 hover:bg-gray-400'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            )}
          >
            Reset
          </button>
        </div>
      </div>

      <div className="flex flex-col gap-1.5 mb-3">
        {Array.from({ length: plannedCountNumber }, (_, i) => renderSlotRow(i))}
        {isRecording && activeSlotIndex < plannedCountNumber && (
          <div className="text-xs text-red-600 font-mono px-2">
            Recording slot #{activeSlotIndex + 1}: {plannedSubTasks[activeSlotIndex] || '-'} ({recordStatus.proceedTime}s)
          </div>
        )}
        {savingInProgress && (
          <div className="text-xs text-amber-700 font-mono px-2">Saving / advancing...</div>
        )}
        {!isPlanMode && (
          <div className="text-xs text-gray-400 italic px-2">
            Single-task mode. The whole recording will be saved as one episode.
          </div>
        )}
      </div>

      <button
        onClick={handleDiscardEpisode}
        disabled={!canDiscardEpisode}
        className={secondaryBtn(canDiscardEpisode, 'red')}
      >
        <MdDeleteSweep size={16} />
        Discard Episode
      </button>
    </div>
  );
}
