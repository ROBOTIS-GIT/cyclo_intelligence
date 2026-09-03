// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { shallowEqual, useSelector } from 'react-redux';
import { MdSwapHoriz } from 'react-icons/md';
import { InferencePhase } from '../../../constants/taskPhases';
import { selectInferenceTaskInfo } from '../../tasks/taskSlice';

const POLL_INTERVAL_MS = 2000;
const RUNNING_STATUSES = new Set(['starting', 'running']);
const COMPLETE_STATUSES = new Set(['complete', 'completed']);

const normalizePath = (value) => String(value || '').trim().replace(/\/+$/, '');

const samePolicyPath = (left, right) => {
  const selected = normalizePath(left);
  const reported = normalizePath(right);
  if (!selected || !reported) return false;
  return (
    selected === reported ||
    `${selected}/pretrained_model` === reported ||
    selected === `${reported}/pretrained_model`
  );
};

const firstInstruction = (taskInfo) => {
  const value = taskInfo?.taskInstruction;
  return String(Array.isArray(value) ? value[0] : value || '').trim();
};

export default function FlowSDEPPOInferenceControls({
  isActive = true,
  inferencePhase = InferencePhase.READY,
  getRolloutStatus,
  onStartRollout,
  onStopRollout,
  onSubmitOutcome,
  getValueWarmupStatus,
  onBusyChange = () => {},
  onRolloutBundleChange = () => {},
}) {
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const robotType = useSelector((state) => state.tasks.robotType);
  const [status, setStatus] = useState({ status: 'idle', operation: 'combined' });
  const [warmupStatus, setWarmupStatus] = useState({ status: 'idle' });
  const [statusReady, setStatusReady] = useState(false);
  const [pendingAction, setPendingAction] = useState('');
  const pollInFlight = useRef(false);

  const isSupported = (
    String(taskInfo.serviceType || '').trim().toLowerCase() === 'lerobot' &&
    String(taskInfo.policyType || '').trim().toLowerCase() === 'multi_task_dit'
  );
  const operation = String(status?.operation || 'combined').toLowerCase();
  const normalizedStatus = String(status?.status || 'idle').toLowerCase();
  const collectorRunning = (
    operation === 'collect' && RUNNING_STATUSES.has(normalizedStatus)
  );
  const updateRunning = (
    operation === 'update' && RUNNING_STATUSES.has(normalizedStatus)
  );
  const flowJobRunning = collectorRunning || updateRunning;
  const vlaRunning = [InferencePhase.INFERENCING, InferencePhase.PAUSED]
    .includes(inferencePhase);
  const rolloutBundles = Array.isArray(status?.rollout_bundles)
    ? status.rollout_bundles.map(normalizePath).filter(Boolean)
    : [];
  const sealedRolloutBundle = (
    operation === 'collect' && COMPLETE_STATUSES.has(normalizedStatus)
  ) ? rolloutBundles[rolloutBundles.length - 1] || '' : '';

  const refreshStatus = useCallback(async () => {
    if (!isActive || !isSupported || pollInFlight.current) return null;
    pollInFlight.current = true;
    try {
      const [nextStatus, nextWarmup] = await Promise.all([
        getRolloutStatus(),
        typeof getValueWarmupStatus === 'function'
          ? getValueWarmupStatus().catch(() => ({ status: 'idle' }))
          : Promise.resolve({ status: 'idle' }),
      ]);
      setStatus(nextStatus || { status: 'idle', operation: 'combined' });
      setWarmupStatus(nextWarmup || { status: 'idle' });
      setStatusReady(true);
      return nextStatus;
    } catch {
      setStatusReady(false);
      return null;
    } finally {
      pollInFlight.current = false;
    }
  }, [getRolloutStatus, getValueWarmupStatus, isActive, isSupported]);

  useEffect(() => {
    if (!isActive || !isSupported || typeof getRolloutStatus !== 'function') {
      setStatusReady(false);
      return undefined;
    }
    let cancelled = false;
    let timer = null;
    const poll = async () => {
      if (cancelled) return;
      await refreshStatus();
      if (!cancelled) timer = setTimeout(poll, POLL_INTERVAL_MS);
    };
    poll();
    return () => {
      cancelled = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [getRolloutStatus, isActive, isSupported, refreshStatus]);

  useEffect(() => {
    onBusyChange(flowJobRunning || Boolean(pendingAction));
  }, [flowJobRunning, onBusyChange, pendingAction]);

  useEffect(() => {
    onRolloutBundleChange(sealedRolloutBundle);
  }, [onRolloutBundleChange, sealedRolloutBundle]);

  const rolloutRequest = useMemo(() => {
    const policyCheckpoint = normalizePath(taskInfo.policyPath);
    const taskInstruction = firstInstruction(taskInfo);
    const request = {
      policy_checkpoint: policyCheckpoint,
      policy_type: 'multi_task_dit',
      algorithm: 'flow_sde_ppo',
      robot_type: String(robotType || '').trim(),
      task_instruction: taskInstruction,
      episodes: 1,
    };

    const statusTask = String(status?.task_instruction || '').trim();
    const resumeMatches = (
      ['combined', 'update'].includes(operation) &&
      COMPLETE_STATUSES.has(normalizedStatus) &&
      normalizePath(status?.checkpoint_path) &&
      statusTask === taskInstruction &&
      [
        status?.policy_checkpoint,
        status?.lineage_policy_checkpoint,
        status?.model_path,
      ].some((candidate) => samePolicyPath(policyCheckpoint, candidate))
    );
    if (resumeMatches) {
      request.resume_checkpoint = normalizePath(status.checkpoint_path);
      return request;
    }

    const warmupMatches = (
      COMPLETE_STATUSES.has(String(warmupStatus?.status || '').toLowerCase()) &&
      normalizePath(warmupStatus?.bundle_path) &&
      samePolicyPath(policyCheckpoint, warmupStatus?.policy_checkpoint) &&
      String(warmupStatus?.task_instruction || '').trim() === taskInstruction
    );
    if (warmupMatches) {
      request.value_warmup_bundle = normalizePath(warmupStatus.bundle_path);
    }
    return request;
  }, [normalizedStatus, operation, robotType, status, taskInfo, warmupStatus]);

  const startBlockedReason = useMemo(() => {
    if (!statusReady) return 'Checking PPO rollout status';
    if (inferencePhase !== InferencePhase.READY) {
      return 'Clear standard VLA inference before PPO rollout';
    }
    if (sealedRolloutBundle) {
      return 'Update the sealed rollout in Training before collecting another episode';
    }
    if (flowJobRunning) return 'A Flow-SDE PPO job is already running';
    if (!rolloutRequest.policy_checkpoint) return 'Select a Diffusion Transformer policy';
    if (!rolloutRequest.robot_type) return 'Select a robot type';
    if (!rolloutRequest.task_instruction) return 'Enter a task instruction';
    if (typeof onStartRollout !== 'function') return 'PPO rollout backend is unavailable';
    return '';
  }, [
    flowJobRunning,
    inferencePhase,
    onStartRollout,
    rolloutRequest,
    sealedRolloutBundle,
    statusReady,
  ]);

  const handleStartRollout = useCallback(async () => {
    if (startBlockedReason || pendingAction) return;
    setPendingAction('start');
    try {
      const result = await onStartRollout(rolloutRequest);
      setStatus(result || { status: 'starting', operation: 'collect' });
      toast.success('PPO rollout started');
    } catch (error) {
      toast.error(`PPO rollout start failed: ${error.message}`);
      await refreshStatus();
    } finally {
      setPendingAction('');
    }
  }, [onStartRollout, pendingAction, refreshStatus, rolloutRequest, startBlockedReason]);

  const handleSelectVla = useCallback(async () => {
    const jobId = String(status?.job_id || '').trim();
    if (!collectorRunning || !jobId || typeof onStopRollout !== 'function' || pendingAction) {
      return;
    }
    setPendingAction('stop');
    try {
      const result = await onStopRollout(jobId);
      setStatus(result || status);
      toast.success('PPO rollout stopped. Press Start to use VLA Action.');
    } catch (error) {
      toast.error(`PPO rollout stop failed: ${error.message}`);
    } finally {
      setPendingAction('');
    }
  }, [collectorRunning, onStopRollout, pendingAction, status]);

  const handleOutcome = useCallback(async (outcome) => {
    const jobId = String(status?.job_id || '').trim();
    if (
      !collectorRunning ||
      status?.awaiting_outcome !== true ||
      !jobId ||
      typeof onSubmitOutcome !== 'function' ||
      pendingAction
    ) return;
    setPendingAction(outcome);
    try {
      const result = await onSubmitOutcome(jobId, outcome);
      setStatus(result || status);
      toast.success(`PPO rollout marked ${outcome}`);
    } catch (error) {
      toast.error(`PPO rollout outcome failed: ${error.message}`);
    } finally {
      setPendingAction('');
    }
  }, [collectorRunning, onSubmitOutcome, pendingAction, status]);

  if (!isSupported) return null;

  const outcomeDisabled = !collectorRunning || status?.awaiting_outcome !== true || Boolean(pendingAction);
  const statusMessage = sealedRolloutBundle
    ? 'Rollout sealed · open Training to update the policy'
    : collectorRunning
      ? (status?.awaiting_outcome ? 'Select the episode outcome' : 'Collecting on-policy rollout')
      : updateRunning
        ? 'PPO update is running in Training'
        : (startBlockedReason || 'Ready for PPO rollout');

  return (
    <div
      className="mt-2 rounded-xl border border-[#d9d2c5] bg-[#f6f3ec] p-2"
      data-testid="flow-sde-ppo-inference-controls"
    >
      <div className="flex flex-wrap items-center gap-2">
        <MdSwapHoriz size={18} className="text-[#71806b]" aria-hidden="true" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[#766f64]">
          Action Output
        </span>
        <button
          type="button"
          onClick={handleSelectVla}
          disabled={!collectorRunning || Boolean(pendingAction)}
          aria-pressed={vlaRunning}
          className={clsx(
            'h-8 rounded-lg border px-3 text-[11px] font-semibold transition-colors',
            vlaRunning
              ? 'border-[#69866f] bg-[#69866f] text-white'
              : 'border-[#d2cabd] bg-[#fffefa] text-[#514b42]',
            (!collectorRunning || pendingAction) && !vlaRunning && 'cursor-not-allowed opacity-50'
          )}
        >
          {pendingAction === 'stop' ? 'Stopping…' : 'VLA Action'}
        </button>
        <button
          type="button"
          onClick={handleStartRollout}
          disabled={Boolean(startBlockedReason || pendingAction || collectorRunning)}
          aria-pressed={collectorRunning}
          title={startBlockedReason || 'Collect one on-policy episode'}
          className={clsx(
            'h-8 rounded-lg border px-3 text-[11px] font-semibold transition-colors',
            collectorRunning
              ? 'border-[#69866f] bg-[#69866f] text-white'
              : 'border-[#d2cabd] bg-[#fffefa] text-[#514b42] hover:bg-[#ece7dd]',
            (startBlockedReason || pendingAction) && !collectorRunning && 'cursor-not-allowed opacity-50'
          )}
        >
          {pendingAction === 'start' ? 'Starting…' : 'PPO Rollout'}
        </button>
        <span role="status" className="min-w-0 flex-1 truncate text-right text-[10px] text-[#756e63]">
          {statusMessage}
        </span>
      </div>

      {collectorRunning && (
        <div
          className="mt-2 grid grid-cols-3 gap-1.5 border-t border-[#e2dcd1] pt-2"
          role="group"
          aria-label="PPO rollout episode outcome"
        >
          <button
            type="button"
            onClick={() => handleOutcome('success')}
            disabled={outcomeDisabled}
            className="h-7 rounded-md border border-[#78937d] bg-[#edf4ed] text-[9px] font-semibold text-[#48654e] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Success
          </button>
          <button
            type="button"
            onClick={() => handleOutcome('fail')}
            disabled={outcomeDisabled}
            className="h-7 rounded-md border border-[#bd8177] bg-[#fff4f1] text-[9px] font-semibold text-[#9d584f] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Fail
          </button>
          <button
            type="button"
            onClick={() => handleOutcome('cancel')}
            disabled={outcomeDisabled}
            className="h-7 rounded-md border border-[#c9c0b2] bg-[#f1ede5] text-[9px] font-semibold text-[#6f685d] disabled:cursor-not-allowed disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      )}
    </div>
  );
}
