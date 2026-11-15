// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { shallowEqual, useDispatch, useSelector, useStore } from 'react-redux';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdCancel,
  MdCheckCircle,
  MdFiberManualRecord,
  MdHighlightOff,
  MdLinkOff,
  MdReplay,
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
import {
  getInterventionState,
  sendInterventionAction,
  sendOperatorPose,
} from '../utils/interventionApi';
import { normalizeActionRequestMode } from '../utils/taskInfoSync';

const INTERVENTION_LEADER_KEY = 'cyclo-intervention-leader';
const INTERVENTION_LEADERS = [
  { value: 'mini', label: 'Mini' },
  { value: 'a3', label: 'A3' },
];

const INTERVENTION_RESUME_MAX_ATTEMPTS = 3;
const INTERVENTION_RESUME_RETRY_DELAY_MS = 250;
// Episode video conversion can legitimately take longer than the ROS stop
// acknowledgement. Keep the UI transaction alive long enough to observe the
// authoritative READY state instead of reporting a false save failure.
const RECORDING_RESET_MAX_ATTEMPTS = 160;
const RECORDING_RESET_RETRY_DELAY_MS = 250;
const OPERATOR_POSE_STORAGE_KEY = 'cyclo-operator-initial-pose';
const OPERATOR_POSE_FIELDS = [
  { key: 'left_arm', label: 'Left arm', count: 8 },
  { key: 'right_arm', label: 'Right arm', count: 8 },
  { key: 'head', label: 'Head', count: 2 },
  { key: 'lift', label: 'Lift', count: 1 },
];

const emptyOperatorPose = () => ({
  left_arm: '',
  right_arm: '',
  head: '',
  lift: '',
  duration: '8',
});

export const parseOperatorPoseInputs = (inputs, robotType) => {
  const model = { f2: 'f2', ffw_sg2_rev1: 'sg2' }[robotType];
  if (!model) {
    throw new Error('초기 위치 이동을 지원하는 F2 또는 SG2를 선택하십시오.');
  }
  const result = { model };
  OPERATOR_POSE_FIELDS.forEach(({ key, label, count }) => {
    let values;
    try {
      values = JSON.parse(String(inputs?.[key] || ''));
    } catch {
      throw new Error(`${label} 값을 [숫자, ...] 형식으로 입력하십시오.`);
    }
    if (
      !Array.isArray(values) ||
      values.length !== count ||
      values.some((value) => typeof value !== 'number' || !Number.isFinite(value))
    ) {
      throw new Error(`${label} 값은 숫자 ${count}개의 list여야 합니다.`);
    }
    result[key] = values.map(Number);
  });
  const duration = Number(inputs?.duration);
  if (!Number.isFinite(duration) || duration < 5 || duration > 30) {
    throw new Error('이동 시간은 5초 이상 30초 이하로 입력하십시오.');
  }
  result.duration = duration;
  return result;
};

const waitForInterventionResumeRetry = () => new Promise((resolve) => {
  window.setTimeout(resolve, INTERVENTION_RESUME_RETRY_DELAY_MS);
});

const waitForRecordingResetRetry = () => new Promise((resolve) => {
  window.setTimeout(resolve, RECORDING_RESET_RETRY_DELAY_MS);
});

export const getRecordingScopeFromTags = (tags = []) => {
  const marker = (Array.isArray(tags) ? tags : []).find((tag) => (
    String(tag || '').startsWith('recording_source:')
  ));
  const scope = String(marker || '').slice('recording_source:'.length);
  return ['inference', 'intervention', 'total'].includes(scope) ? scope : '';
};

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

const inferencePhaseFromInterventionState = (phase) => ({
  READY: InferencePhase.READY,
  LOADING: InferencePhase.LOADING,
  INFERENCING: InferencePhase.INFERENCING,
  PAUSED: InferencePhase.PAUSED,
  SYNCING: InferencePhase.SYNCING,
}[String(phase || '').trim().toUpperCase()]);

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
  episodeCountOverride = null,
  partialRecordMode = false,
  mlpActivationSequence = 0,
  showIndependentInterventionRecord = true,
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
  const interventionPendingRef = useRef(false);
  const [interventionState, setInterventionState] = useState({
    available: null,
    active: false,
    current: null,
    error: '',
  });
  const [interventionPending, setInterventionPending] = useState('');
  const [localRecordingScope, setLocalRecordingScope] = useState('');
  const [totalRecordingArmed, setTotalRecordingArmed] = useState(false);
  const [partialRecordingSelected, setPartialRecordingSelected] = useState(false);
  const [autoPartialTriggers, setAutoPartialTriggers] = useState({
    intervention: true,
    mlp: true,
  });
  const partialServerRecordingSeenRef = useRef(false);
  const mlpActivationHandledRef = useRef(false);
  const lastMlpActivationSequenceRef = useRef(0);
  const [operatorPoseInputs, setOperatorPoseInputs] = useState(() => {
    try {
      const saved = JSON.parse(
        window.localStorage.getItem(OPERATOR_POSE_STORAGE_KEY) || 'null'
      );
      return saved && typeof saved === 'object'
        ? { ...emptyOperatorPose(), ...saved }
        : emptyOperatorPose();
    } catch {
      return emptyOperatorPose();
    }
  });
  const [poseMovePending, setPoseMovePending] = useState(false);
  const [selectedLeader, setSelectedLeader] = useState(() => {
    try {
      const stored = window.sessionStorage.getItem(INTERVENTION_LEADER_KEY);
      return INTERVENTION_LEADERS.some(({ value }) => value === stored)
        ? stored
        : '';
    } catch {
      return '';
    }
  });
  const isStandaloneRecording = mode === 'recording';
  const parsedPolicyEpoch = Number(policyEpoch);
  const displayedPolicyEpoch = (
    Number.isInteger(parsedPolicyEpoch) && parsedPolicyEpoch >= 0
  ) ? parsedPolicyEpoch : null;
  const parsedEpisodeCountOverride = Number(episodeCountOverride);
  const displayedEpisodeCount = (
    episodeCountOverride !== null &&
    episodeCountOverride !== '' &&
    Number.isFinite(parsedEpisodeCountOverride) &&
    parsedEpisodeCountOverride >= 0
  ) ? Math.floor(parsedEpisodeCountOverride) : control.episodeCount;
  const recordingAvailable =
    isStandaloneRecording
      ? isActive && Boolean(String(robotType || '').trim())
      : taskInfo.inferenceMode === 'robot' && taskInfo.recordInferenceMode;
  const interventionAvailable =
    !isStandaloneRecording && taskInfo.inferenceMode === 'robot';
  // READY may still echo the just-closed task_info for one status cycle. Only
  // let server tags override the local button choice while a recording is
  // actually active, otherwise a rapid Total -> Intervention click could be
  // mistaken for a standalone inference recording.
  const serverRecordingScope = recordStatus.recordPhase === RecordPhase.READY
    ? ''
    : getRecordingScopeFromTags(recordStatus.tags);
  const activeRecordingScope = control.active
    ? (serverRecordingScope || localRecordingScope)
    : '';
  const totalRecordingActive = activeRecordingScope === 'total';
  const partialRecordingActive = partialRecordMode && (
    partialRecordingSelected || activeRecordingScope === 'inference'
  );
  const continuousRecordingActive =
    totalRecordingActive || partialRecordingActive;
  const totalSessionSelected = totalRecordingArmed || totalRecordingActive;
  const continuousSessionSelected =
    totalSessionSelected || partialRecordingActive;

  useEffect(() => {
    if (serverRecordingScope) {
      setLocalRecordingScope(serverRecordingScope);
      if (
        partialRecordMode &&
        serverRecordingScope === 'inference' &&
        recordStatus.recordPhase === RecordPhase.RECORDING
      ) {
        partialServerRecordingSeenRef.current = true;
        setPartialRecordingSelected(true);
      }
    } else if (
      recordStatus.recordPhase === RecordPhase.READY &&
      control.uiPhase === InferenceRecordingUiPhase.IDLE &&
      !partialRecordingSelected
    ) {
      setLocalRecordingScope('');
    }
    if (
      partialRecordMode &&
      recordStatus.recordPhase === RecordPhase.READY &&
      partialServerRecordingSeenRef.current
    ) {
      partialServerRecordingSeenRef.current = false;
      setPartialRecordingSelected(false);
      setLocalRecordingScope('');
    }
  }, [
    control.uiPhase,
    partialRecordMode,
    partialRecordingSelected,
    recordStatus.recordPhase,
    serverRecordingScope,
  ]);

  const refreshInterventionState = useCallback(async ({ silent = false } = {}) => {
    try {
      const next = await getInterventionState();
      setInterventionState({
        ...next,
        available: true,
        error: '',
      });
      const serverLeader = next?.current?.leader_model || (
        next?.standby?.active === true && next?.standby?.ready === true
          ? next?.standby?.current?.model
          : ''
      );
      if (INTERVENTION_LEADERS.some(({ value }) => value === serverLeader)) {
        const serverOwnsActiveIntervention = Boolean(next?.current?.leader_model);
        setSelectedLeader((previous) => (
          serverOwnsActiveIntervention ? serverLeader : (previous || serverLeader)
        ));
        if (serverOwnsActiveIntervention) {
          try {
            window.sessionStorage.setItem(INTERVENTION_LEADER_KEY, serverLeader);
          } catch {
            // Server state remains authoritative when session storage is unavailable.
          }
        }
      }
      return next;
    } catch (error) {
      setInterventionState((previous) => ({
        ...previous,
        available: false,
        error: error?.message || 'Intervention Controller unavailable',
      }));
      if (!silent) throw error;
      return null;
    }
  }, []);

  useEffect(() => {
    // The workspace keeps the custom-pose controls visible even after the
    // inference task metadata returns to READY. Keep polling in that view so
    // the button does not remain stuck on the temporary/Moving state.
    if (variant !== 'workspace' && !interventionAvailable) return undefined;
    let mounted = true;
    const refresh = async () => {
      if (!mounted) return;
      await refreshInterventionState({ silent: true });
    };
    refresh();
    const timer = window.setInterval(refresh, 1000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, [interventionAvailable, refreshInterventionState, variant]);

  const intervention = interventionState.current;
  const interventionActive = Boolean(interventionState.active || intervention);
  const interventionPhase = String(intervention?.phase || '');
  const interventionInferencePhase = interventionState.inference_status_ready === true
    ? inferencePhaseFromInterventionState(interventionState.inference_phase)
    : inferencePhase;
  const interventionModeOn = interventionActive &&
    ['READY', 'RECORDING'].includes(interventionPhase);
  const showCompletedInterventionReset =
    interventionPhase === 'READY_TO_RESUME' &&
    inferencePhase === InferencePhase.READY &&
    recordStatus.recordPhase === RecordPhase.READY;
  const canResetCompletedIntervention =
    showCompletedInterventionReset &&
    interventionState.command_recent !== true &&
    interventionState.arbiter?.available !== false &&
    interventionState.arbiter?.mode === 'NONE' &&
    !interventionState.arbiter?.fault &&
    !interventionState.arbiter?.publisher_conflict &&
    !interventionPending;

  useEffect(() => {
    if (
      control.uiPhase === InferenceRecordingUiPhase.IDLE ||
      control.uiPhase === InferenceRecordingUiPhase.RECORDING
    ) {
      commandPendingRef.current = false;
    }
  }, [control.uiPhase]);

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
    !startBlocked &&
    !interventionActive &&
    !interventionPending;
  const totalRecordIdle =
    !isStandaloneRecording &&
    interventionAvailable &&
    recordStatus.recordPhase === RecordPhase.READY &&
    control.uiPhase === InferenceRecordingUiPhase.IDLE &&
    !startBlocked &&
    !interventionActive &&
    !interventionPending;
  const canStartTotalNow = totalRecordIdle &&
    inferencePhase === InferencePhase.INFERENCING;
  const canPrearmTotal = totalRecordIdle &&
    inferencePhase === InferencePhase.READY;
  const canCancelTotalPrearm = totalRecordingArmed &&
    !control.active && !interventionActive && !interventionPending;
  const canUseTotalButton = !totalRecordingActive && (
    canStartTotalNow || canPrearmTotal || canCancelTotalPrearm
  );
  const canCancel = control.active && !control.pending && !control.serverSaving &&
    !interventionActive && !interventionPending;
  const canLabel = canCancel && canFinalize;
  const canCloseContinuousIntervention =
    continuousRecordingActive &&
    interventionActive &&
    interventionPhase === 'RECORDING' &&
    !control.pending &&
    !control.serverSaving &&
    !interventionPending;
  const canSaveContinuousIntervention =
    canCloseContinuousIntervention && canFinalize;
  const poseMoveGate = interventionState.pose_move || {
    ok: false,
    failures: ['초기 위치 이동 상태를 확인하는 중입니다.'],
    warnings: [],
  };
  const poseOperationRunning = Boolean(
    interventionState.operation?.running &&
    ['F2 사용자 초기 위치 이동', 'SG2 사용자 초기 위치 이동'].includes(
      interventionState.operation?.kind
    )
  );
  let operatorPosePayload = null;
  let operatorPoseInputError = '';
  try {
    operatorPosePayload = parseOperatorPoseInputs(operatorPoseInputs, robotType);
  } catch (error) {
    operatorPoseInputError = error?.message || '초기 위치 list를 확인하십시오.';
  }
  const canMoveToOperatorPose =
    poseMoveGate.ok === true &&
    !poseMovePending &&
    !poseOperationRunning;

  const recoverFromServer = useCallback(() => {
    dispatch(setInferenceRecordingUiPhase(phaseFromServer(store.getState())));
    commandPendingRef.current = false;
  }, [dispatch, store]);

  const updateOperatorPoseInput = useCallback((key, value) => {
    setOperatorPoseInputs((previous) => {
      const next = { ...previous, [key]: value };
      try {
        window.localStorage.setItem(
          OPERATOR_POSE_STORAGE_KEY,
          JSON.stringify(next)
        );
      } catch {
        // The entered values still remain available for this browser session.
      }
      return next;
    });
  }, []);

  const handleOperatorPoseMove = useCallback(async () => {
    if (!canMoveToOperatorPose) return;
    if (!operatorPosePayload) {
      toast.error(operatorPoseInputError || '초기 위치 list를 확인하십시오.');
      return;
    }
    setPoseMovePending(true);
    try {
      const result = await sendOperatorPose(operatorPosePayload);
      if (!result?.ok) {
        throw new Error(result?.message || 'Initial pose movement failed');
      }
      await refreshInterventionState({ silent: true });
      toast.success('입력한 초기 위치로 이동을 시작했습니다.');
    } catch (error) {
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || '초기 위치 이동을 시작하지 못했습니다.');
    } finally {
      setPoseMovePending(false);
    }
  }, [
    canMoveToOperatorPose,
    operatorPoseInputError,
    operatorPosePayload,
    refreshInterventionState,
  ]);

  const startTotalRecording = useCallback(async () => {
    if (!canStartTotalNow || commandPendingRef.current) return;
    setTotalRecordingArmed(false);
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.STARTING
    ));
    try {
      const result = await sendRecordCommand('start_inference_record', {
        forceInferenceRecording: true,
        recordingSource: 'total',
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Total recording start failed');
      }
      setLocalRecordingScope('total');
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
      commandPendingRef.current = false;
      toast.success('Total recording started');
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'Total recording start failed');
    }
  }, [canStartTotalNow, dispatch, recoverFromServer, sendRecordCommand]);

  const handleTotalRecord = useCallback(async () => {
    if (commandPendingRef.current) return;
    if (totalRecordingArmed) {
      if (canStartTotalNow) {
        await startTotalRecording();
        return;
      }
      if (canCancelTotalPrearm) {
        setTotalRecordingArmed(false);
        toast.success('Total recording pre-arm cancelled');
      }
      return;
    }
    if (canPrearmTotal) {
      setTotalRecordingArmed(true);
      toast.success('Total recording armed; it will start with inference');
      return;
    }
    await startTotalRecording();
  }, [
    canCancelTotalPrearm,
    canPrearmTotal,
    canStartTotalNow,
    startTotalRecording,
    totalRecordingArmed,
  ]);

  useEffect(() => {
    if (!totalRecordingArmed || !canStartTotalNow) return;
    startTotalRecording();
  }, [canStartTotalNow, startTotalRecording, totalRecordingArmed]);

  const selectInterventionLeader = useCallback(async (model) => {
    if (interventionActive || interventionPending) return;
    setSelectedLeader(model);
    try {
      window.sessionStorage.setItem(INTERVENTION_LEADER_KEY, model);
    } catch {
      // Session persistence is optional; the selected button still updates.
    }
    const standby = interventionState.standby || {};
    const standbyCurrent = standby.current || {};
    const miniAlreadyReady = model === 'mini' && standby.ready === true &&
      standbyCurrent.model === 'mini';
    const shouldWarmMini = model === 'mini' && !miniAlreadyReady &&
      [
        InferencePhase.READY,
        InferencePhase.INFERENCING,
        InferencePhase.PAUSED,
      ].includes(inferencePhase);
    const shouldStopMini = model !== 'mini' && Boolean(standby.active);
    if (!shouldWarmMini && !shouldStopMini) return;

    interventionPendingRef.current = true;
    setInterventionPending(shouldWarmMini ? 'warming' : 'stopping-standby');
    try {
      await sendInterventionAction(
        shouldWarmMini ? 'standby' : 'standby_stop',
        { model: 'mini' }
      );
      await refreshInterventionState({ silent: false });
      toast.success(shouldWarmMini
        ? 'Mini Leader standby ready'
        : 'Mini Leader standby stopped');
    } catch (error) {
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Mini Leader standby transition failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    inferencePhase,
    interventionActive,
    interventionPending,
    interventionState.standby,
    refreshInterventionState,
  ]);

  const standby = interventionState.standby || {};
  const standbyCurrent = standby.current || {};
  const miniStandbyReady = standby.ready === true &&
    standbyCurrent.model === 'mini';
  const miniStandbyActive = standby.active === true &&
    standbyCurrent.model === 'mini';

  const disconnectMiniLeader = useCallback(async () => {
    if (
      interventionActive ||
      interventionPendingRef.current ||
      !miniStandbyActive
    ) return;

    interventionPendingRef.current = true;
    setInterventionPending('stopping-standby');
    try {
      await sendInterventionAction('standby_stop', { model: 'mini' });
      await refreshInterventionState({ silent: false });
      toast.success('Mini Leader disconnected');
    } catch (error) {
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Failed to disconnect Mini Leader');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    interventionActive,
    miniStandbyActive,
    refreshInterventionState,
  ]);

  const recordingLifecycleReadyForIntervention =
    [RecordPhase.READY, RecordPhase.RECORDING].includes(recordStatus.recordPhase) &&
    !commandPendingRef.current &&
    !control.serverSaving;

  const canStartIntervention =
    !isStandaloneRecording &&
    interventionAvailable &&
    [InferencePhase.INFERENCING, InferencePhase.PAUSED].includes(inferencePhase) &&
    recordingLifecycleReadyForIntervention &&
    interventionState.available === true &&
    interventionState.preflight?.ok === true &&
    Boolean(selectedLeader) &&
    (selectedLeader !== 'mini' || miniStandbyReady) &&
    !interventionActive &&
    !interventionPending;

  const startInterventionRecording = useCallback(async ({
    continueExisting,
    recordingSource = 'intervention',
  }) => {
    let recordingStarted = false;
    try {
      if (!continueExisting) {
        const startResult = await sendRecordCommand('start_inference_record', {
          forceInferenceRecording: true,
          recordingSource,
          taskSource: 'inference',
        });
        if (!startResult?.success) {
          throw new Error(startResult?.message || 'Intervention recording start failed');
        }
        recordingStarted = true;
        setLocalRecordingScope(recordingSource);
        if (recordingSource === 'inference') {
          setPartialRecordingSelected(true);
        }
      }
      await sendInterventionAction('recording', { model: selectedLeader });
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
    } catch (error) {
      if (recordingStarted) {
        await sendRecordCommand('cancel_inference_record', {
          recordingSource,
          taskSource: 'inference',
        }).catch(() => {});
        setLocalRecordingScope('');
        if (recordingSource === 'inference') {
          setPartialRecordingSelected(false);
        }
      }
      throw error;
    }
  }, [dispatch, selectedLeader, sendRecordCommand]);

  const handleInterventionStart = useCallback(async () => {
    if (!canStartIntervention || interventionPendingRef.current) return;
    interventionPendingRef.current = true;
    setInterventionPending('handoff');
    let handoffStarted = false;
    let armedTotalStarted = false;
    let continueExistingRecording = continuousRecordingActive;
    const startArmedTotal = totalRecordingArmed && !totalRecordingActive;
    const implicitPartialRecording = partialRecordMode &&
      autoPartialTriggers.intervention &&
      !showIndependentInterventionRecord &&
      !continueExistingRecording &&
      !startArmedTotal;
    const closeInferencePart = (
      control.active && activeRecordingScope === 'inference' &&
      !partialRecordingActive
    );
    try {
      if (inferencePhase === InferencePhase.INFERENCING) {
        const pauseResult = await sendRecordCommand('stop_inference', {
          taskSource: 'inference',
        });
        if (!pauseResult?.success) {
          throw new Error(pauseResult?.message || 'Inference pause failed');
        }
      }

      // Only Total Record crosses the ownership boundary.  A standalone
      // inference record is closed as a failed rollout before the human
      // correction starts its own episode.
      if (closeInferencePart) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.SAVING
        ));
        const stopResult = await sendRecordCommand('stop_inference_record', {
          episodeOutcome: EpisodeOutcome.FAILURE,
          recordingSource: 'inference',
          taskSource: 'inference',
        });
        if (!stopResult?.success) {
          throw new Error(
            stopResult?.message || 'Inference recording save failed'
          );
        }
        setLocalRecordingScope('');
      }

      if (startArmedTotal) {
        dispatch(setInferenceRecordingUiPhase(
          InferenceRecordingUiPhase.STARTING
        ));
        const totalStartResult = await sendRecordCommand(
          'start_inference_record',
          {
            forceInferenceRecording: true,
            recordingSource: 'total',
            taskSource: 'inference',
          }
        );
        if (!totalStartResult?.success) {
          throw new Error(
            totalStartResult?.message || 'Total recording start failed'
          );
        }
        armedTotalStarted = true;
        continueExistingRecording = true;
        setTotalRecordingArmed(false);
        setLocalRecordingScope('total');
      }

      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.STARTING
      ));
      await sendInterventionAction('start', { model: selectedLeader });
      handoffStarted = true;
      await startInterventionRecording({
        continueExisting: continueExistingRecording,
        recordingSource: implicitPartialRecording
          ? 'inference'
          : 'intervention',
      });
      await refreshInterventionState({ silent: true });
      toast.success(
        `${selectedLeader === 'mini' ? 'Mini' : 'A3'} Relative intervention recording started`
      );
    } catch (error) {
      if (handoffStarted) {
        await sendInterventionAction('abort', { model: selectedLeader })
          .catch(() => {});
      }
      if (armedTotalStarted) {
        await sendRecordCommand('cancel_inference_record', {
          recordingSource: 'total',
          taskSource: 'inference',
        }).catch(() => {});
        setLocalRecordingScope('');
      }
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(
        `${error?.message || 'Intervention start failed'} · Inference remains paused`
      );
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    canStartIntervention,
    activeRecordingScope,
    autoPartialTriggers.intervention,
    continuousRecordingActive,
    control.active,
    dispatch,
    inferencePhase,
    recoverFromServer,
    refreshInterventionState,
    selectedLeader,
    sendRecordCommand,
    startInterventionRecording,
    partialRecordingActive,
    partialRecordMode,
    showIndependentInterventionRecord,
    totalRecordingArmed,
    totalRecordingActive,
  ]);

  const resumeAndCompleteIntervention = useCallback(async (leaderModel) => {
    let lastError = new Error('Inference resume failed');
    let shouldResume = true;

    for (
      let attempt = 1;
      attempt <= INTERVENTION_RESUME_MAX_ATTEMPTS;
      attempt += 1
    ) {
      let resumeSucceeded = !shouldResume;
      if (shouldResume) {
        try {
          const resumeResult = await sendRecordCommand('resume_inference', {
            taskSource: 'inference',
          });
          if (!resumeResult?.success) {
            throw new Error(
              resumeResult?.message || 'Inference resume failed'
            );
          }
          resumeSucceeded = true;
        } catch (error) {
          lastError = error;
        }
      }

      if (resumeSucceeded) {
        try {
          return await sendInterventionAction('complete', {
            model: leaderModel,
          });
        } catch (error) {
          lastError = error;
        }
      }

      const latest = await refreshInterventionState({ silent: true });
      if (latest?.active === false || !latest?.current) {
        // The server completed the transition but its first response was
        // lost. COMPLETE is idempotent, so the local transaction is done.
        return latest;
      }

      const currentPhase = String(latest.current.phase || '');
      const observedInferencePhase = String(latest.inference_phase || '');
      const arbiter = latest.arbiter || {};
      if (currentPhase !== 'READY_TO_RESUME') {
        throw lastError;
      }

      if (['SYNCING', 'INFERENCING'].includes(observedInferencePhase)) {
        // Resume succeeded and slow-start is active. Retry only COMPLETE;
        // issuing RESUME again during SYNCING would restart the lifecycle.
        shouldResume = false;
      } else {
        const safeToRetryResume =
          observedInferencePhase === 'PAUSED' &&
          arbiter.available !== false &&
          arbiter.mode === 'NONE' &&
          !arbiter.fault &&
          !arbiter.publisher_conflict &&
          arbiter.intervention_enabled !== true;
        if (!safeToRetryResume) {
          throw lastError;
        }
        shouldResume = true;
      }

      if (attempt < INTERVENTION_RESUME_MAX_ATTEMPTS) {
        await waitForInterventionResumeRetry();
      }
    }

    throw lastError;
  }, [refreshInterventionState, sendRecordCommand]);

  const handleInterventionFinish = useCallback(async (
    save,
    resumePolicy = false
  ) => {
    if (
      interventionPendingRef.current ||
      !['READY', 'RECORDING'].includes(interventionPhase)
    ) return;
    interventionPendingRef.current = true;
    setInterventionPending(
      continuousRecordingActive || resumePolicy
        ? 'returning'
        : (save ? 'saving' : 'discarding')
    );
    dispatch(setInferenceRecordingUiPhase(
      continuousRecordingActive
        ? InferenceRecordingUiPhase.RECORDING
        : save
          ? InferenceRecordingUiPhase.SAVING
          : InferenceRecordingUiPhase.CANCELLING
    ));
    try {
      await sendInterventionAction('end', { model: selectedLeader });
      const standaloneRecordingActive =
        !continuousRecordingActive &&
        interventionPhase === 'RECORDING' &&
        recordStatus.recordPhase === RecordPhase.RECORDING;
      if (standaloneRecordingActive) {
        const recordResult = await sendRecordCommand(
          save ? 'stop_inference_record' : 'cancel_inference_record',
          {
            ...(save ? { episodeOutcome: EpisodeOutcome.SUCCESS } : {}),
            recordingSource: 'intervention',
            taskSource: 'inference',
          }
        );
        if (!recordResult?.success) {
          throw new Error(
            recordResult?.message || 'Intervention recording close failed'
          );
        }
        setLocalRecordingScope('');
      }
      if (continuousRecordingActive || resumePolicy) {
        await resumeAndCompleteIntervention(selectedLeader);
      } else {
        // Saving an Intervention episode must not restart policy motion.
        // With recording closed and command routing quiet, RESET clears only
        // the hand-off session and intentionally leaves Inference PAUSED.
        await sendInterventionAction('reset', { model: selectedLeader });
      }
      dispatch(setInferenceRecordingUiPhase(
        continuousRecordingActive
          ? InferenceRecordingUiPhase.RECORDING
          : InferenceRecordingUiPhase.IDLE
      ));
      await refreshInterventionState({ silent: true });
      toast.success(
        continuousRecordingActive
          ? partialRecordingActive
            ? 'Partial recording continues in inference mode'
            : 'Total recording continues in inference mode'
          : resumePolicy
            ? interventionPhase === 'RECORDING'
              ? 'Intervention record saved; inference resumed'
              : 'Intervention ended; inference resumed'
          : save
            ? 'Intervention record saved; robot remains paused'
            : 'Intervention record discarded; robot remains paused'
      );
    } catch (error) {
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Intervention close failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    dispatch,
    continuousRecordingActive,
    interventionPhase,
    partialRecordingActive,
    recoverFromServer,
    refreshInterventionState,
    recordStatus.recordPhase,
    resumeAndCompleteIntervention,
    selectedLeader,
    sendRecordCommand,
  ]);

  const handleContinuousInterventionOutcome = useCallback(async (
    { save, episodeOutcome = '', label = '' }
  ) => {
    if (
      !canCloseContinuousIntervention ||
      (save && !canSaveContinuousIntervention) ||
      interventionPendingRef.current
    ) return;

    interventionPendingRef.current = true;
    setInterventionPending(save ? 'saving-total' : 'discarding-total');
    dispatch(setInferenceRecordingUiPhase(
      save
        ? InferenceRecordingUiPhase.SAVING
        : InferenceRecordingUiPhase.CANCELLING
    ));
    try {
      // Stop human command ownership before finalizing the continuous
      // inference + intervention episode. Policy remains PAUSED after the
      // recorder has acknowledged the save/discard operation.
      await sendInterventionAction('end', { model: selectedLeader });
      const recordResult = await sendRecordCommand(
        save ? 'stop_inference_record' : 'cancel_inference_record',
        {
          ...(save ? { episodeOutcome } : {}),
          recordingSource: partialRecordingActive ? 'inference' : 'total',
          taskSource: 'inference',
        }
      );
      if (!recordResult?.success) {
        throw new Error(
          recordResult?.message || 'Continuous recording close failed'
        );
      }

      if (save) {
        await onEpisodeSaved?.({
          episodeOutcome,
          commandOptions: activeCommandOptionsRef.current,
        });
      } else {
        await onEpisodeCancelled?.({
          commandOptions: activeCommandOptionsRef.current,
        });
      }
      setLocalRecordingScope('');
      setPartialRecordingSelected(false);
      await sendInterventionAction('reset', { model: selectedLeader });
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.IDLE
      ));
      await refreshInterventionState({ silent: true });
      toast.success(
        save
          ? `${partialRecordingActive ? 'Partial' : 'Total'} episode saved as ${label}; robot remains paused`
          : `${partialRecordingActive ? 'Partial' : 'Total'} episode discarded; robot remains paused`
      );
    } catch (error) {
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Continuous recording close failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    canCloseContinuousIntervention,
    canSaveContinuousIntervention,
    dispatch,
    onEpisodeCancelled,
    onEpisodeSaved,
    partialRecordingActive,
    recoverFromServer,
    refreshInterventionState,
    selectedLeader,
    sendRecordCommand,
  ]);

  const handleInterventionAbort = useCallback(async () => {
    if (!interventionActive || interventionPendingRef.current) return;
    interventionPendingRef.current = true;
    setInterventionPending('aborting');
    try {
      if (
        recordStatus.recordPhase === RecordPhase.RECORDING &&
        !continuousRecordingActive
      ) {
        const cancelResult = await sendRecordCommand('cancel_inference_record', {
          recordingSource: 'intervention',
          taskSource: 'inference',
        });
        if (!cancelResult?.success) {
          throw new Error(cancelResult?.message || 'Intervention recording cancel failed');
        }
        setLocalRecordingScope('');
      }
      await sendInterventionAction('abort', { model: selectedLeader || 'a3' });
      dispatch(setInferenceRecordingUiPhase(
        continuousRecordingActive
          ? InferenceRecordingUiPhase.RECORDING
          : InferenceRecordingUiPhase.IDLE
      ));
      await refreshInterventionState({ silent: true });
      toast.success('Intervention stopped safely; inference remains paused');
    } catch (error) {
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Intervention abort failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    dispatch,
    continuousRecordingActive,
    interventionActive,
    recordStatus.recordPhase,
    recoverFromServer,
    refreshInterventionState,
    selectedLeader,
    sendRecordCommand,
  ]);

  const handleInterventionResume = useCallback(async () => {
    if (
      !['PAUSED_SAFE', 'READY_TO_RESUME'].includes(interventionPhase) ||
      interventionInferencePhase !== InferencePhase.PAUSED ||
      interventionPendingRef.current
    ) return;
    interventionPendingRef.current = true;
    const resettingAbortedSession = interventionPhase === 'PAUSED_SAFE';
    const restartContinuousRecording = resettingAbortedSession && (
      continuousRecordingActive || intervention?.recording_continued === true
    );
    setInterventionPending(
      resettingAbortedSession ? 'resetting-session' : 'resuming'
    );
    try {
      if (
        recordStatus.recordPhase === RecordPhase.RECORDING &&
        resettingAbortedSession
      ) {
        const cancelResult = await sendRecordCommand('cancel_inference_record', {
          recordingSource: totalRecordingActive
            ? 'total'
            : partialRecordingActive
              ? 'inference'
              : 'intervention',
          taskSource: 'inference',
        });
        if (!cancelResult?.success) {
          throw new Error(cancelResult?.message || 'Aborted recording cancel failed');
        }
        let recordingReady = false;
        for (let attempt = 0; attempt < RECORDING_RESET_MAX_ATTEMPTS; attempt += 1) {
          const next = await getInterventionState();
          if (next?.record_phase === RecordPhase.READY) {
            recordingReady = true;
            break;
          }
          await waitForRecordingResetRetry();
        }
        if (!recordingReady) {
          throw new Error('Recording did not become READY after abort cancellation');
        }
        setLocalRecordingScope('');
        if (partialRecordMode) {
          setPartialRecordingSelected(false);
        }
      }
      if (resettingAbortedSession) {
        await sendInterventionAction('reset', { model: selectedLeader || 'a3' });
        if ((selectedLeader || 'a3') === 'mini') {
          await sendInterventionAction('standby', { model: 'mini' });
        }
        const resumeResult = await sendRecordCommand('resume_inference', {
          taskSource: 'inference',
        });
        if (!resumeResult?.success) {
          throw new Error(resumeResult?.message || 'Inference resume failed');
        }
      } else {
        await resumeAndCompleteIntervention(selectedLeader || 'a3');
      }
      dispatch(setInferenceRecordingUiPhase(
        restartContinuousRecording
          ? InferenceRecordingUiPhase.IDLE
          : totalRecordingActive
            ? InferenceRecordingUiPhase.RECORDING
            : InferenceRecordingUiPhase.IDLE
      ));
      if (restartContinuousRecording && totalRecordingActive) {
        setTotalRecordingArmed(true);
      }
      await refreshInterventionState({ silent: true });
      toast.success(
        restartContinuousRecording
          ? partialRecordingActive
            ? 'Aborted partial episode discarded; start Partial Record again'
            : 'Aborted episode discarded; Total Record will restart with inference'
          : resettingAbortedSession
            ? 'Aborted intervention reset; inference resumed'
            : 'Inference resumed'
      );
    } catch (error) {
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Inference resume failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    dispatch,
    continuousRecordingActive,
    intervention,
    interventionInferencePhase,
    interventionPhase,
    partialRecordMode,
    recordStatus.recordPhase,
    recoverFromServer,
    refreshInterventionState,
    resumeAndCompleteIntervention,
    selectedLeader,
    sendRecordCommand,
    partialRecordingActive,
    totalRecordingActive,
  ]);

  const handleCompletedInterventionReset = useCallback(async () => {
    if (!canResetCompletedIntervention || interventionPendingRef.current) return;
    interventionPendingRef.current = true;
    setInterventionPending('closing-session');
    try {
      await sendInterventionAction('reset', {
        model: selectedLeader || 'a3',
      });
      setLocalRecordingScope('');
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.IDLE
      ));
      await refreshInterventionState({ silent: true });
      toast.success('Previous intervention session closed; Mini standby remains ready');
    } catch (error) {
      recoverFromServer();
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Intervention session cleanup failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    canResetCompletedIntervention,
    dispatch,
    recoverFromServer,
    refreshInterventionState,
    selectedLeader,
  ]);

  const handleFaultReset = useCallback(async () => {
    if (interventionActive || interventionPendingRef.current) return;
    interventionPendingRef.current = true;
    setInterventionPending('resetting');
    try {
      await sendInterventionAction('reset_fault', {
        model: selectedLeader || 'a3',
      });
      await refreshInterventionState({ silent: true });
      toast.success('Command routing fault cleared; mode remains NONE');
    } catch (error) {
      await refreshInterventionState({ silent: true });
      toast.error(error?.message || 'Command routing fault reset failed');
    } finally {
      interventionPendingRef.current = false;
      setInterventionPending('');
    }
  }, [
    interventionActive,
    refreshInterventionState,
    selectedLeader,
  ]);

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
        ...(!isStandaloneRecording ? { recordingSource: 'inference' } : {}),
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Recording start failed');
      }
      if (!isStandaloneRecording) {
        setLocalRecordingScope('inference');
        if (partialRecordMode) {
          setPartialRecordingSelected(true);
        }
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
    partialRecordMode,
    recordingMonitor,
    recoverFromServer,
    segmentIndex,
    segmentedRecording,
    sendRecordCommand,
    store,
  ]);

  const mlpActionActive =
    normalizeActionRequestMode(taskInfo.actionRequestMode) === 'tt_rtc' &&
    taskInfo.actionPolicyMode === 'rlt';

  const canAutoStartPartial =
    partialRecordMode &&
    !isStandaloneRecording &&
    taskInfo.inferenceMode === 'robot' &&
    inferencePhase === InferencePhase.INFERENCING &&
    recordStatus.recordPhase === RecordPhase.READY &&
    control.uiPhase === InferenceRecordingUiPhase.IDLE &&
    !startBlocked &&
    !interventionActive &&
    !interventionPending;

  const startAutomaticMlpPartialRecord = useCallback(async () => {
    if (!canAutoStartPartial || commandPendingRef.current) return false;
    commandPendingRef.current = true;
    dispatch(setInferenceRecordingUiPhase(
      InferenceRecordingUiPhase.STARTING
    ));
    try {
      const result = await sendRecordCommand('start_inference_record', {
        forceInferenceRecording: true,
        recordingSource: 'inference',
        taskSource: 'inference',
      });
      if (!result?.success) {
        throw new Error(result?.message || 'Automatic Partial Record start failed');
      }
      setLocalRecordingScope('inference');
      setPartialRecordingSelected(true);
      dispatch(setInferenceRecordingUiPhase(
        InferenceRecordingUiPhase.RECORDING
      ));
      commandPendingRef.current = false;
      toast.success('MLP 활성화와 함께 Partial Record를 시작했습니다.');
      return true;
    } catch (error) {
      recoverFromServer();
      toast.error(error?.message || 'MLP Partial Record를 시작하지 못했습니다.');
      return false;
    }
  }, [
    canAutoStartPartial,
    dispatch,
    recoverFromServer,
    sendRecordCommand,
  ]);

  useEffect(() => {
    const sequence = Number(mlpActivationSequence || 0);
    const newActivationConfirmed =
      sequence > lastMlpActivationSequenceRef.current;
    if (!autoPartialTriggers.mlp) {
      lastMlpActivationSequenceRef.current = sequence;
      mlpActivationHandledRef.current = false;
      return;
    }
    if (!mlpActionActive && !newActivationConfirmed) {
      mlpActivationHandledRef.current = false;
      return;
    }
    if (
      !partialRecordMode ||
      (mlpActivationHandledRef.current && !newActivationConfirmed)
    ) return;
    if (continuousSessionSelected || control.active) {
      lastMlpActivationSequenceRef.current = sequence;
      mlpActivationHandledRef.current = true;
      return;
    }
    // The parent sequence changes only after set_action_policy succeeds. The
    // task snapshot may still contain the old route, so accept either the
    // confirmed sequence or the synchronized Redux route as authoritative.
    if (!canAutoStartPartial) return;
    lastMlpActivationSequenceRef.current = sequence;
    mlpActivationHandledRef.current = true;
    void startAutomaticMlpPartialRecord();
  }, [
    autoPartialTriggers.mlp,
    canAutoStartPartial,
    continuousSessionSelected,
    control.active,
    mlpActionActive,
    mlpActivationSequence,
    partialRecordMode,
    startAutomaticMlpPartialRecord,
  ]);

  const toggleAutoPartialTrigger = useCallback((trigger) => {
    setAutoPartialTriggers((previous) => ({
      ...previous,
      [trigger]: !previous[trigger],
    }));
  }, []);

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
      if (!isStandaloneRecording) {
        setLocalRecordingScope('');
        setPartialRecordingSelected(false);
      }
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
      if (!isStandaloneRecording) {
        setLocalRecordingScope('');
        setPartialRecordingSelected(false);
      }
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

  if (!isWorkspace && !recordingAvailable && !interventionAvailable) return null;

  const recordLabel = {
    [InferenceRecordingUiPhase.STARTING]: 'Starting',
    [InferenceRecordingUiPhase.RECORDING]: 'Recording',
    [InferenceRecordingUiPhase.SAVING]: 'Saving',
    [InferenceRecordingUiPhase.CANCELLING]: 'Cancelling',
  }[control.uiPhase] || (
    isStandaloneRecording
      ? 'Record'
      : partialRecordMode
        ? 'Partial Record'
        : 'Inference Record'
  );
  const inferenceRecordLabel = totalRecordingActive
    ? (partialRecordMode ? 'Partial Record' : 'Inference Record')
    : recordLabel;
  const totalRecordLabel = totalRecordingActive
    ? 'Total Recording'
    : totalRecordingArmed
      ? 'Total Armed'
    : 'Total Record';
  const totalRecordAriaLabel = totalRecordingActive
    ? 'Total recording active'
    : totalRecordingArmed
      ? 'Cancel total recording pre-arm'
      : canPrearmTotal
        ? 'Pre-arm total recording'
        : 'Start total recording';
  const totalRecordTitle = totalRecordingArmed
    ? 'Inference가 시작되면 Total Record가 자동으로 시작됩니다'
    : 'Inference와 Intervention을 하나의 에피소드로 연속 기록합니다';
  const interventionLabel = interventionPending === 'handoff'
    ? 'Handing over'
    : interventionPending === 'warming'
      ? 'Connecting Mini'
      : interventionPending === 'stopping-standby'
        ? 'Stopping Mini'
    : interventionPhase === 'RECORDING'
      ? 'Intervention Recording'
      : 'Intervention Record';
  const canCloseIntervention =
    ['READY', 'RECORDING'].includes(interventionPhase) && !interventionPending;
  const canAbortIntervention = interventionActive &&
    !['PAUSED_SAFE', 'READY_TO_RESUME'].includes(interventionPhase) &&
    !interventionPending;
  const showResumeIntervention =
    ['PAUSED_SAFE', 'READY_TO_RESUME'].includes(interventionPhase) &&
    interventionInferencePhase === InferencePhase.PAUSED;
  const canResumeIntervention = showResumeIntervention && !interventionPending;
  const resetAndResumeLabel = (
    totalRecordingActive || intervention?.recording_continued === true
  )
    ? 'Reset & Record Again'
    : 'Reset & Resume';
  const resumeInterventionLabel = interventionPending === 'resetting-session'
    ? 'Resetting & Reconnecting'
    : interventionPhase === 'PAUSED_SAFE'
      ? resetAndResumeLabel
      : 'Resume Inference';
  const resumeInterventionAriaLabel = interventionPhase === 'PAUSED_SAFE'
    ? 'Reset aborted intervention and acquire new data'
    : 'Resume inference after intervention';
  const arbiterFault = String(interventionState.arbiter?.fault || '');
  const faultResetFailures = interventionState.fault_reset?.failures || [];
  const canResetFault = !interventionActive && Boolean(arbiterFault) &&
    interventionState.fault_reset?.ok === true && !interventionPending;
  const preflightFailures = interventionState.preflight?.failures || [];
  const interventionStartTitle = !selectedLeader
    ? 'Select Mini or A3 Leader first'
    : selectedLeader === 'mini' && !miniStandbyReady
      ? 'Wait for isolated Mini Leader standby to become ready'
    : preflightFailures.length
      ? preflightFailures.join(' · ')
      : interventionState.error || 'Pause inference and start Relative leader control';
  const totalModeTransitionPending = ['handoff', 'returning'].includes(
    interventionPending
  );
  const totalInferenceModeActive = !interventionModeOn &&
    interventionPending !== 'handoff';
  const totalInferenceModeLabel = interventionPending === 'returning'
    ? 'Switching to Inference'
    : 'Inference';
  const totalInterventionModeLabel = interventionPending === 'handoff'
    ? 'Switching ON'
    : interventionModeOn
      ? 'Intervention ON'
      : 'Intervention OFF';
  const totalInterventionModeTitle = totalRecordingArmed
    ? 'Inference가 시작되면 Intervention 전환을 사용할 수 있습니다'
    : interventionStartTitle;
  const continuousControlName = totalSessionSelected
    ? 'Total Record'
    : partialRecordingActive
      ? 'Partial Record'
      : 'Inference';
  const continuousControlAriaPrefix = totalSessionSelected
    ? 'total recording'
    : partialRecordingActive
      ? 'partial recording'
      : 'inference';
  const showControlMode = !isStandaloneRecording && interventionAvailable && (
    !showIndependentInterventionRecord || continuousSessionSelected
  );
  const inferenceModeActiveAriaLabel = continuousControlName === 'Inference'
    ? 'Inference mode active'
    : `${continuousControlName} inference mode active`;
  const workspaceTitle = isStandaloneRecording
    ? 'Recording'
    : 'Inference Recording';
  const rolloutName = totalRecordingActive
    ? 'total recording'
    : isStandaloneRecording
      ? 'recording'
      : partialRecordMode
        ? 'partial recording'
        : 'inference rollout';
  const displayedGuideMessage = guideMessage || (
    isStandaloneRecording
      ? (recordingAvailable ? 'Ready to record' : 'Select a robot type first')
      : ''
  );
  const operatorPoseBlockedReason = operatorPoseInputError ||
    (poseMoveGate.failures || []).join(' · ');

  const buttonClass = (enabled, activeClass) => clsx(
    'h-9 min-w-[92px] px-2.5 rounded-lg flex items-center justify-center',
    'gap-1 text-base font-semibold transition-colors',
    enabled
      ? `${activeClass} cursor-pointer`
      : 'bg-gray-100 text-gray-400 cursor-not-allowed opacity-60'
  );

  const renderLeaderSelector = (compact = false) => !isStandaloneRecording && (
    <div
      className={clsx(
        'flex shrink-0 items-center rounded-lg border',
        compact
          ? 'gap-0.5 border-[#d7d0c4] bg-[#ece8df] p-0.5'
          : 'gap-1 border-gray-200 bg-gray-50 p-1'
      )}
      role="group"
      aria-label="Intervention leader"
      title="Select the connected leader before starting Intervention Record"
    >
      {!compact && (
        <span className="px-1 text-xs font-semibold text-gray-500">Leader</span>
      )}
      {INTERVENTION_LEADERS.map(({ value, label }) => {
        const selected = selectedLeader === value;
        return (
          <button
            key={value}
            type="button"
            onClick={() => selectInterventionLeader(value)}
            disabled={interventionActive || Boolean(interventionPending)}
            aria-pressed={selected}
            aria-label={`Use ${label} Leader for intervention`}
            className={clsx(
              'rounded-md font-semibold transition-colors',
              compact ? 'h-7 px-2 text-[10px]' : 'h-7 px-2.5 text-xs',
              selected
                ? 'bg-amber-100 text-amber-800 shadow-sm'
                : compact
                  ? 'text-[#81796d] hover:bg-[#e3ddd2]'
                  : 'text-gray-500 hover:bg-gray-200',
              (interventionActive || interventionPending) && 'cursor-not-allowed opacity-60'
            )}
          >
            {label}
          </button>
        );
      })}
      {selectedLeader === 'mini' && (
        <>
          <span
            role="status"
            aria-label="Mini Leader standby status"
            className={clsx(
              'px-1 text-[9px] font-semibold',
              miniStandbyReady ? 'text-emerald-700' : 'text-amber-700'
            )}
          >
            {interventionPending === 'warming'
              ? 'Connecting'
              : interventionPending === 'stopping-standby'
                ? 'Disconnecting'
                : miniStandbyReady
                  ? 'Standby'
                  : 'Offline'}
          </span>
          {miniStandbyActive && (
            <button
              type="button"
              onClick={disconnectMiniLeader}
              disabled={interventionActive || Boolean(interventionPending)}
              aria-label="Disconnect Mini Leader"
              title={interventionActive
                ? 'End the active intervention before disconnecting Mini Leader'
                : 'Stop Mini Leader standby and disconnect it'}
              className={clsx(
                'flex h-7 items-center gap-1 rounded-md px-2 font-semibold transition-colors',
                compact ? 'text-[10px]' : 'text-xs',
                interventionActive || interventionPending
                  ? 'cursor-not-allowed bg-gray-100 text-gray-400 opacity-60'
                  : 'bg-red-50 text-red-700 hover:bg-red-100'
              )}
            >
              <MdLinkOff size={compact ? 13 : 15} />
              Disconnect
            </button>
          )}
        </>
      )}
    </div>
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
            {renderLeaderSelector(true)}
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
              {displayedEpisodeCount}
            </span>
          </div>
        </div>

        <div className={clsx(
          'grid gap-2',
          inferenceActions
            ? (isStandaloneRecording
              ? 'grid-cols-4'
              : showIndependentInterventionRecord
                ? 'grid-cols-6'
                : 'grid-cols-5')
            : (isStandaloneRecording
              ? 'grid-cols-1'
              : showIndependentInterventionRecord
                ? 'grid-cols-3'
                : 'grid-cols-2')
        )}>
          {!isStandaloneRecording && (
            <button
              type="button"
              onClick={handleTotalRecord}
              disabled={!canUseTotalButton}
              className={workspaceButtonClass(
                canUseTotalButton,
                'bg-[#e5e0f2] text-[#67578d] hover:bg-[#dcd4ec]'
              )}
              aria-label={totalRecordAriaLabel}
              title={totalRecordTitle}
            >
              <MdFiberManualRecord size={16} />
              {totalRecordLabel}
            </button>
          )}
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
              : partialRecordMode
                ? 'Start partial recording'
                : 'Record inference rollout'}
          >
            <MdFiberManualRecord size={16} />
            {inferenceRecordLabel}
          </button>
          {!isStandaloneRecording && showIndependentInterventionRecord &&
          !continuousSessionSelected && (
            <button
              type="button"
              onClick={handleInterventionStart}
              disabled={!canStartIntervention}
              className={workspaceButtonClass(
                canStartIntervention,
                'bg-[#f4e4c7] text-[#8b641f] hover:bg-[#ecd6ad]'
              )}
              aria-label="Start intervention recording"
              title={interventionStartTitle}
            >
              <MdFiberManualRecord size={16} />
              {interventionLabel}
            </button>
          )}
          {inferenceActions}
        </div>

        {partialRecordMode && !isStandaloneRecording && (
          <div
            className="mt-2 grid grid-cols-[auto_1fr_1fr] items-center gap-2 rounded-lg border border-[#d7d0c4] bg-[#ece8df] p-1.5"
            role="group"
            aria-label="Automatic Partial Record triggers"
          >
            <span className="px-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#776f63]">
              Auto Partial Record
            </span>
            <button
              type="button"
              onClick={() => toggleAutoPartialTrigger('intervention')}
              aria-pressed={autoPartialTriggers.intervention}
              aria-label="Automatically start Partial Record on Intervention ON"
              title="Intervention ON 전환이 완료되면 Partial Record를 자동으로 시작합니다"
              className={workspaceButtonClass(
                true,
                autoPartialTriggers.intervention
                  ? 'bg-[#f1d7a8] text-[#805b19] shadow-sm'
                  : 'bg-[#f5f1e9] text-[#8a8276] hover:bg-[#e7e1d7]'
              )}
            >
              Intervention Auto: {autoPartialTriggers.intervention ? 'ON' : 'OFF'}
            </button>
            <button
              type="button"
              onClick={() => toggleAutoPartialTrigger('mlp')}
              aria-pressed={autoPartialTriggers.mlp}
              aria-label="Automatically start Partial Record when MLP becomes active"
              title="MLP 전환이 완료되어 실제 활성 상태가 되면 Partial Record를 자동으로 시작합니다"
              className={workspaceButtonClass(
                true,
                autoPartialTriggers.mlp
                  ? 'bg-[#dce8dc] text-[#536c58] shadow-sm'
                  : 'bg-[#f5f1e9] text-[#8a8276] hover:bg-[#e7e1d7]'
              )}
            >
              MLP Auto: {autoPartialTriggers.mlp ? 'ON' : 'OFF'}
            </button>
          </div>
        )}

        {showControlMode && (
          <div
            className="mt-2 grid grid-cols-[auto_1fr_1fr] items-center gap-2 rounded-lg border border-[#d7d0c4] bg-[#ece8df] p-1.5"
            role="group"
            aria-label={`${continuousControlName} control mode`}
          >
            <span className="px-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#776f63]">
              Control Mode
            </span>
            <button
              type="button"
              onClick={() => handleInterventionFinish(true, true)}
              disabled={!interventionModeOn || !canCloseIntervention || totalModeTransitionPending}
              className={workspaceButtonClass(
                totalInferenceModeActive || (interventionModeOn && canCloseIntervention),
                totalInferenceModeActive
                  ? 'bg-[#dce8dc] text-[#536c58] shadow-sm'
                  : 'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
              )}
              aria-label={interventionModeOn
                ? `Switch ${continuousControlAriaPrefix} to inference mode`
                : inferenceModeActiveAriaLabel}
              aria-pressed={totalInferenceModeActive}
              title={interventionModeOn
                ? `Inference로 돌아가며 ${continuousControlName}는 계속됩니다`
                : `현재 ${continuousControlName}는 Inference 모드입니다`}
            >
              {partialRecordingActive
                ? <MdReplay size={16} />
                : <MdCheckCircle size={16} />}
              {totalInferenceModeLabel}
            </button>
            <button
              type="button"
              onClick={handleInterventionStart}
              disabled={interventionModeOn || !canStartIntervention || totalModeTransitionPending}
              className={workspaceButtonClass(
                interventionModeOn || canStartIntervention,
                interventionModeOn
                  ? 'bg-[#f1d7a8] text-[#805b19] shadow-sm'
                  : 'bg-[#f4e4c7] text-[#8b641f] hover:bg-[#ecd6ad]'
              )}
              aria-label={interventionModeOn
                ? `${continuousControlName} intervention mode active`
                : `Switch ${continuousControlAriaPrefix} to intervention mode`}
              aria-pressed={interventionModeOn}
              title={interventionModeOn
                ? `현재 ${continuousControlName}는 Intervention 모드입니다`
                : totalInterventionModeTitle}
            >
              <MdFiberManualRecord size={16} />
              {totalInterventionModeLabel}
            </button>
          </div>
        )}

        {interventionActive && continuousRecordingActive &&
        interventionPhase === 'RECORDING' ? (
          <div
            className="mt-2 grid grid-cols-4 gap-2"
            role="group"
            aria-label={`${partialRecordingActive ? 'Partial' : 'Total'} recording intervention outcome`}
          >
            <button
              type="button"
              onClick={() => handleContinuousInterventionOutcome({
                save: true,
                episodeOutcome: EpisodeOutcome.SUCCESS,
                label: 'Success',
              })}
              disabled={!canSaveContinuousIntervention}
              className={workspaceButtonClass(
                canSaveContinuousIntervention,
                'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
              )}
              aria-label={`Save ${partialRecordingActive ? 'partial' : 'total'} intervention episode as Success`}
            >
              <MdCheckCircle size={16} />
              Success
            </button>
            <button
              type="button"
              onClick={() => handleContinuousInterventionOutcome({
                save: true,
                episodeOutcome: EpisodeOutcome.FAILURE,
                label: 'Fail',
              })}
              disabled={!canSaveContinuousIntervention}
              className={workspaceButtonClass(
                canSaveContinuousIntervention,
                'bg-[#f1e2df] text-[#995e58] hover:bg-[#ead5d1]'
              )}
              aria-label={`Save ${partialRecordingActive ? 'partial' : 'total'} intervention episode as Fail`}
            >
              <MdHighlightOff size={16} />
              Fail
            </button>
            <button
              type="button"
              onClick={() => handleContinuousInterventionOutcome({ save: false })}
              disabled={!canCloseContinuousIntervention}
              className={workspaceButtonClass(
                canCloseContinuousIntervention,
                'bg-[#e8e3da] text-[#655e53] hover:bg-[#ddd6ca]'
              )}
              aria-label={`Cancel and discard ${partialRecordingActive ? 'partial' : 'total'} intervention episode`}
            >
              <MdCancel size={16} />
              Cancel
            </button>
            <button
              type="button"
              onClick={handleInterventionAbort}
              disabled={!canAbortIntervention}
              className={workspaceButtonClass(
                canAbortIntervention,
                'bg-[#f1e2df] text-[#995e58] hover:bg-[#ead5d1]'
              )}
              aria-label="Abort intervention safely"
            >
              <MdHighlightOff size={16} />
              Abort
            </button>
          </div>
        ) : showCompletedInterventionReset ? (
          <button
            type="button"
            onClick={handleCompletedInterventionReset}
            disabled={!canResetCompletedIntervention}
            className={clsx(
              workspaceButtonClass(
                canResetCompletedIntervention,
                'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
              ),
              'mt-2 w-full'
            )}
            aria-label="Close completed intervention session"
            title={canResetCompletedIntervention
              ? '이전 Intervention 세션만 종료하고 Mini standby는 유지합니다'
              : '명령 출력과 arbiter가 안전한 NONE 상태가 될 때까지 기다립니다'}
          >
            <MdCheckCircle size={16} />
            {interventionPending === 'closing-session'
              ? 'Closing Session'
              : 'Close Previous Session'}
          </button>
        ) : interventionActive && interventionPhase !== 'READY' ? (
          <div
            className={clsx(
              'mt-2 grid gap-2',
              partialRecordingActive && interventionPhase === 'RECORDING'
                ? 'grid-cols-2'
                : 'grid-cols-3'
            )}
            role="group"
            aria-label="Intervention recording outcome"
          >
            {showResumeIntervention ? (
              <button
                type="button"
                onClick={handleInterventionResume}
                disabled={!canResumeIntervention}
                className={workspaceButtonClass(
                  canResumeIntervention,
                  'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
                )}
                aria-label={resumeInterventionAriaLabel}
                title={interventionPhase === 'PAUSED_SAFE'
                  ? '중단된 녹화를 폐기하고 세션을 초기화한 뒤 새 데이터 취득을 준비합니다'
                  : 'Inference를 다시 시작합니다'}
              >
                <MdReplay size={16} />
                {resumeInterventionLabel}
              </button>
            ) : (
              <>
            <button
              type="button"
              onClick={() => handleInterventionFinish(true)}
              disabled={!canCloseIntervention}
              className={workspaceButtonClass(
                canCloseIntervention,
                'bg-[#e4ebe3] text-[#607563] hover:bg-[#d9e4d8]'
              )}
              aria-label={partialRecordingActive
                ? 'Switch partial recording to inference mode'
                : 'Save intervention record'}
            >
              <MdReplay size={16} />
              {partialRecordingActive ? 'Inference' : 'Save'}
            </button>
              </>
            )}
            {!partialRecordingActive && (
            <button
              type="button"
              onClick={showResumeIntervention
                ? handleInterventionAbort
                : () => handleInterventionFinish(false)}
              disabled={showResumeIntervention || !canCloseIntervention}
              className={workspaceButtonClass(
                !showResumeIntervention && canCloseIntervention,
                'bg-[#e8e3da] text-[#655e53] hover:bg-[#ddd6ca]'
              )}
              aria-label="Discard intervention record"
            >
              <MdCancel size={16} />
              Discard
            </button>
            )}
            <button
              type="button"
              onClick={handleInterventionAbort}
              disabled={!canAbortIntervention}
              className={workspaceButtonClass(
                canAbortIntervention,
                'bg-[#f1e2df] text-[#995e58] hover:bg-[#ead5d1]'
              )}
              aria-label="Abort intervention safely"
            >
              <MdHighlightOff size={16} />
              Abort
            </button>
          </div>
        ) : (
        <>
        {arbiterFault && (
          <button
            type="button"
            onClick={handleFaultReset}
            disabled={!canResetFault}
            className={clsx(
              workspaceButtonClass(
                canResetFault,
                'bg-[#f1e2df] text-[#995e58] hover:bg-[#ead5d1]'
              ),
              'mt-2 w-full'
            )}
            aria-label="Reset command routing fault"
            title={faultResetFailures.join(' · ') || arbiterFault}
          >
            <MdReplay size={16} />
            Safety Reset
          </button>
        )}
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
        </>
        )}
        <div
          className="mt-2 rounded-lg border border-[#d7d0c4] bg-[#ece8df] p-2"
          role="group"
          aria-label="Custom initial pose"
        >
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="text-[10px] font-bold uppercase tracking-[0.08em] text-[#776f63]">
              Custom Initial Pose
            </span>
            <span className="text-[9px] text-[#81796d]">
              rad / lift m · 녹화 종료 후 정지 상태에서 이동
            </span>
          </div>
          <div className="grid grid-cols-2 gap-1.5">
            {OPERATOR_POSE_FIELDS.map(({ key, label, count }) => (
              <label key={key} className="flex min-w-0 items-center gap-1.5">
                <span className="w-16 shrink-0 text-[9px] font-semibold text-[#665f54]">
                  {label} [{count}]
                </span>
                <input
                  type="text"
                  value={operatorPoseInputs[key]}
                  onChange={(event) => updateOperatorPoseInput(key, event.target.value)}
                  placeholder={`[${Array(count).fill('0').join(', ')}]`}
                  aria-label={`${label} initial position list`}
                  className="h-7 min-w-0 flex-1 rounded-md border border-[#d2cabd] bg-white px-2 font-mono text-[9px] text-[#514b43] outline-none focus:border-[#8aa08d]"
                />
              </label>
            ))}
          </div>
          <div className="mt-1.5 flex items-center gap-2">
            <label className="flex items-center gap-1 text-[9px] font-semibold text-[#665f54]">
              이동 시간
              <input
                type="number"
                min="5"
                max="30"
                step="1"
                value={operatorPoseInputs.duration}
                onChange={(event) => updateOperatorPoseInput('duration', event.target.value)}
                aria-label="Initial pose movement duration"
                className="h-7 w-16 rounded-md border border-[#d2cabd] bg-white px-2 text-right font-mono text-[9px] outline-none focus:border-[#8aa08d]"
              />
              초
            </label>
            <button
              type="button"
              onClick={handleOperatorPoseMove}
              disabled={!canMoveToOperatorPose}
              className={clsx(
                workspaceButtonClass(
                  canMoveToOperatorPose,
                  'bg-[#dce8dc] text-[#536c58] hover:bg-[#d3e1d3]'
                ),
                'flex-1'
              )}
              aria-label="Move to operator-entered initial pose"
              title={operatorPoseBlockedReason ||
                '입력값과 URDF joint limit을 확인한 뒤 천천히 이동합니다'}
            >
              <MdReplay size={16} />
              {poseMovePending || poseOperationRunning ? 'Moving' : 'Move to Pose'}
            </button>
          </div>
          {(operatorPoseBlockedReason || poseOperationRunning) && (
            <p
              className={clsx(
                'mt-1 text-[9px]',
                poseOperationRunning ? 'text-[#58705d]' : 'text-[#8b641f]'
              )}
              aria-live="polite"
            >
              {poseOperationRunning
                ? '입력한 초기 위치로 이동 중입니다. 안전 상태가 바뀌면 자동 중단됩니다.'
                : operatorPoseBlockedReason}
            </p>
          )}
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
          {displayedEpisodeCount}
        </span>
      </div>
      <div className="h-6 w-px shrink-0 bg-gray-300" />
      {renderLeaderSelector()}
      <div className="h-6 w-px shrink-0 bg-gray-300" />
      {!isStandaloneRecording && (
        <button
          type="button"
          onClick={handleTotalRecord}
          disabled={!canUseTotalButton}
          className={buttonClass(
            canUseTotalButton,
            'bg-violet-50 text-violet-700 hover:bg-violet-100'
          )}
          aria-label={totalRecordAriaLabel}
          title={totalRecordTitle}
        >
          <MdFiberManualRecord size={17} />
          {totalRecordLabel}
        </button>
      )}
      <button
        type="button"
        onClick={handleRecord}
        disabled={!canStart}
        className={buttonClass(canStart, 'bg-red-50 text-red-600 hover:bg-red-100')}
        aria-label="Record inference rollout"
      >
        <MdFiberManualRecord size={17} />
        {inferenceRecordLabel}
      </button>
      {!isStandaloneRecording && showIndependentInterventionRecord &&
      !continuousSessionSelected && (
        <button
          type="button"
          onClick={handleInterventionStart}
          disabled={!canStartIntervention}
          className={buttonClass(
            canStartIntervention,
            'bg-amber-50 text-amber-700 hover:bg-amber-100'
          )}
          aria-label="Start intervention recording"
          title={interventionStartTitle}
        >
          <MdFiberManualRecord size={17} />
          {interventionLabel}
        </button>
      )}
      {showControlMode && (
        <div
          className="flex shrink-0 items-center gap-1 rounded-lg border border-gray-200 bg-gray-50 p-1"
          role="group"
          aria-label={`${continuousControlName} control mode`}
        >
          <span className="px-1 text-xs font-semibold text-gray-500">Mode</span>
        <button
          type="button"
          onClick={() => handleInterventionFinish(true, true)}
            disabled={!interventionModeOn || !canCloseIntervention || totalModeTransitionPending}
            className={buttonClass(
              totalInferenceModeActive || (interventionModeOn && canCloseIntervention),
              totalInferenceModeActive
                ? 'bg-emerald-100 text-emerald-800'
                : 'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
            )}
            aria-label={interventionModeOn
              ? `Switch ${continuousControlAriaPrefix} to inference mode`
              : inferenceModeActiveAriaLabel}
            aria-pressed={totalInferenceModeActive}
          >
            <MdReplay size={17} />
            {totalInferenceModeLabel}
          </button>
          <button
            type="button"
            onClick={handleInterventionStart}
            disabled={interventionModeOn || !canStartIntervention || totalModeTransitionPending}
            className={buttonClass(
              interventionModeOn || canStartIntervention,
              interventionModeOn
                ? 'bg-amber-100 text-amber-800'
                : 'bg-amber-50 text-amber-700 hover:bg-amber-100'
            )}
            aria-label={interventionModeOn
              ? `${continuousControlName} intervention mode active`
              : `Switch ${continuousControlAriaPrefix} to intervention mode`}
            aria-pressed={interventionModeOn}
            title={interventionModeOn
              ? `현재 ${continuousControlName}는 Intervention 모드입니다`
              : totalInterventionModeTitle}
          >
            <MdFiberManualRecord size={17} />
            {totalInterventionModeLabel}
          </button>
        </div>
      )}
      {interventionActive && continuousRecordingActive &&
      interventionPhase === 'RECORDING' ? (
        <>
          <button
            type="button"
            onClick={() => handleContinuousInterventionOutcome({
              save: true,
              episodeOutcome: EpisodeOutcome.SUCCESS,
              label: 'Success',
            })}
            disabled={!canSaveContinuousIntervention}
            className={buttonClass(
              canSaveContinuousIntervention,
              'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
            )}
            aria-label={`Save ${partialRecordingActive ? 'partial' : 'total'} intervention episode as Success`}
          >
            <MdCheckCircle size={17} />
            Success
          </button>
          <button
            type="button"
            onClick={() => handleContinuousInterventionOutcome({
              save: true,
              episodeOutcome: EpisodeOutcome.FAILURE,
              label: 'Failed',
            })}
            disabled={!canSaveContinuousIntervention}
            className={buttonClass(
              canSaveContinuousIntervention,
              'bg-rose-50 text-rose-700 hover:bg-rose-100'
            )}
            aria-label={`Save ${partialRecordingActive ? 'partial' : 'total'} intervention episode as Failed`}
          >
            <MdHighlightOff size={17} />
            Failed
          </button>
          <button
            type="button"
            onClick={() => handleContinuousInterventionOutcome({ save: false })}
            disabled={!canCloseContinuousIntervention}
            className={buttonClass(
              canCloseContinuousIntervention,
              'bg-gray-100 text-gray-700 hover:bg-gray-200'
            )}
            aria-label={`Cancel and discard ${partialRecordingActive ? 'partial' : 'total'} intervention episode`}
          >
            <MdCancel size={17} />
            Cancel
          </button>
          <button
            type="button"
            onClick={handleInterventionAbort}
            disabled={!canAbortIntervention}
            className={buttonClass(
              canAbortIntervention,
              'bg-rose-50 text-rose-700 hover:bg-rose-100'
            )}
            aria-label="Abort intervention safely"
          >
            <MdHighlightOff size={17} />
            Abort
          </button>
        </>
      ) : showCompletedInterventionReset ? (
        <button
          type="button"
          onClick={handleCompletedInterventionReset}
          disabled={!canResetCompletedIntervention}
          className={buttonClass(
            canResetCompletedIntervention,
            'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
          )}
          aria-label="Close completed intervention session"
          title={canResetCompletedIntervention
            ? '이전 Intervention 세션만 종료하고 Mini standby는 유지합니다'
            : '명령 출력과 arbiter가 안전한 NONE 상태가 될 때까지 기다립니다'}
        >
          <MdCheckCircle size={17} />
          {interventionPending === 'closing-session' ? 'Closing' : 'Close Session'}
        </button>
      ) : interventionActive && interventionPhase !== 'READY' ? (
        <>
          {showResumeIntervention && (
            <button
              type="button"
              onClick={handleInterventionResume}
              disabled={!canResumeIntervention}
              className={buttonClass(
                canResumeIntervention,
                'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
              )}
              aria-label={resumeInterventionAriaLabel}
              title={interventionPhase === 'PAUSED_SAFE'
                ? '중단된 녹화를 폐기하고 세션을 초기화한 뒤 새 데이터 취득을 준비합니다'
                : 'Inference를 다시 시작합니다'}
            >
              <MdReplay size={17} />
              {resumeInterventionLabel}
            </button>
          )}
          <button
            type="button"
            onClick={() => handleInterventionFinish(true)}
            disabled={!canCloseIntervention}
            className={buttonClass(
              canCloseIntervention,
              'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
            )}
          aria-label={partialRecordingActive
            ? 'Switch partial recording to inference mode'
            : 'Save intervention record'}
        >
          {partialRecordingActive
            ? <MdReplay size={17} />
            : <MdCheckCircle size={17} />}
          {partialRecordingActive ? 'Inference' : 'Save'}
        </button>
        {!partialRecordingActive && (
        <button
            type="button"
            onClick={() => handleInterventionFinish(false)}
            disabled={!canCloseIntervention}
            className={buttonClass(
              canCloseIntervention,
              'bg-gray-100 text-gray-700 hover:bg-gray-200'
            )}
            aria-label="Discard intervention record"
          >
          <MdCancel size={17} />
          Discard
        </button>
        )}
          <button
            type="button"
            onClick={handleInterventionAbort}
            disabled={!canAbortIntervention}
            className={buttonClass(
              canAbortIntervention,
              'bg-rose-50 text-rose-700 hover:bg-rose-100'
            )}
            aria-label="Abort intervention safely"
          >
            <MdHighlightOff size={17} />
            Abort
          </button>
        </>
      ) : (
      <>
      {arbiterFault && (
        <button
          type="button"
          onClick={handleFaultReset}
          disabled={!canResetFault}
          className={buttonClass(
            canResetFault,
            'bg-rose-50 text-rose-700 hover:bg-rose-100'
          )}
          aria-label="Reset command routing fault"
          title={faultResetFailures.join(' · ') || arbiterFault}
        >
          <MdReplay size={17} />
          Safety Reset
        </button>
      )}
      <button
        type="button"
        onClick={() => handleOutcome(EpisodeOutcome.SUCCESS, 'Success')}
        disabled={!canLabel}
        className={buttonClass(
          canLabel,
          'bg-emerald-50 text-emerald-700 hover:bg-emerald-100'
        )}
        aria-label={`Save ${rolloutName} as Success`}
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
        aria-label={`Save ${rolloutName} as Failed`}
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
        aria-label={`Cancel and discard ${rolloutName}`}
      >
        <MdCancel size={17} />
        Cancel
      </button>
      </>
      )}
    </div>
  );
}
