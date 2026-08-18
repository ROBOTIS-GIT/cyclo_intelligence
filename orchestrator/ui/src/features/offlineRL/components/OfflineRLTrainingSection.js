// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { useSelector } from 'react-redux';
import {
  MdFolderOpen,
  MdModelTraining,
  MdPlayArrow,
} from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import ProgressBar from '../../../components/ProgressBar';
import { DEFAULT_PATHS } from '../../../constants/paths';
import {
  getOfflineRLStatus,
  startOfflineRLTraining,
} from '../../../utils/offlineRlApi';

const POLL_INTERVAL_MS = 2000;
const RUNNING_STATUSES = new Set(['starting', 'running']);
const COMPLETE_STATUSES = new Set(['complete', 'completed']);

const boundedPercentage = (value) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.min(100, Math.max(0, parsed));
};

const statusValue = (status, ...keys) => {
  for (const key of keys) {
    const value = status?.[key];
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
};

const formatCount = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString() : '—';
};

const formatLoss = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toPrecision(5) : '—';
};

const formatEta = (value) => {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return '—';
  const seconds = Math.round(parsed);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainingSeconds = seconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${remainingSeconds}s`;
  return `${remainingSeconds}s`;
};

const PathField = ({
  id,
  label,
  value,
  onChange,
  onBrowse,
  placeholder,
  disabled,
  optional = false,
}) => (
  <div className="flex flex-col gap-1.5">
    <label htmlFor={id} className="text-sm font-medium text-gray-600">
      {label}
      {optional && <span className="ml-1 font-normal text-gray-400">(optional)</span>}
    </label>
    <div className="flex items-start gap-2">
      <textarea
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        placeholder={placeholder}
        rows={2}
        className={clsx(
          'min-h-16 flex-1 resize-y rounded-md border border-gray-300 p-2 text-sm',
          'focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500',
          disabled ? 'cursor-not-allowed bg-gray-100' : 'bg-white'
        )}
      />
      <button
        type="button"
        onClick={onBrowse}
        disabled={disabled}
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-gray-200 text-blue-600 hover:bg-gray-300 disabled:cursor-not-allowed disabled:opacity-50"
        aria-label={`Browse ${label}`}
      >
        <MdFolderOpen size={21} />
      </button>
    </div>
  </div>
);

export default function OfflineRLTrainingSection({
  isActive = true,
  onRunningChange,
}) {
  const robotType = useSelector((state) => state.tasks.robotType);
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus?.status || 'idle'
  );
  const [datasetPath, setDatasetPath] = useState('');
  const [actCheckpoint, setActCheckpoint] = useState('');
  const [parentCheckpoint, setParentCheckpoint] = useState('');
  const [algorithm, setAlgorithm] = useState('td3');
  const [criticEpochs, setCriticEpochs] = useState('10');
  const [actorEquivalentEpochs, setActorEquivalentEpochs] = useState('5');
  const [jobStatus, setJobStatus] = useState({ status: 'idle' });
  const [statusReady, setStatusReady] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [showDatasetBrowser, setShowDatasetBrowser] = useState(false);
  const [showActBrowser, setShowActBrowser] = useState(false);
  const [showParentBrowser, setShowParentBrowser] = useState(false);
  const lastAnnouncedStatus = useRef('idle');
  const statusRequestSequence = useRef(0);
  const activeStatusRequest = useRef(null);
  const isStartingRef = useRef(false);

  const requestStatus = useCallback(async ({ isCancelled = () => false } = {}) => {
    if (isStartingRef.current || activeStatusRequest.current !== null) {
      return null;
    }

    const requestSequence = ++statusRequestSequence.current;
    activeStatusRequest.current = requestSequence;
    try {
      const status = await getOfflineRLStatus();
      if (
        isCancelled() ||
        isStartingRef.current ||
        requestSequence !== statusRequestSequence.current
      ) {
        return null;
      }
      setJobStatus(status || { status: 'idle' });
      setStatusReady(true);
      return status;
    } catch {
      if (
        !isCancelled() &&
        !isStartingRef.current &&
        requestSequence === statusRequestSequence.current
      ) {
        setStatusReady(false);
      }
      return null;
    } finally {
      if (activeStatusRequest.current === requestSequence) {
        activeStatusRequest.current = null;
      }
    }
  }, []);

  useEffect(() => {
    if (!isActive) return undefined;
    let cancelled = false;
    let nextPollTimer = null;

    const scheduleNextPoll = () => {
      if (!cancelled) {
        nextPollTimer = setTimeout(poll, POLL_INTERVAL_MS);
      }
    };

    async function poll() {
      if (cancelled) return;
      if (isStartingRef.current) {
        scheduleNextPoll();
        return;
      }

      try {
        await requestStatus({ isCancelled: () => cancelled });
      } finally {
        scheduleNextPoll();
      }
    }

    poll();
    return () => {
      cancelled = true;
      statusRequestSequence.current += 1;
      activeStatusRequest.current = null;
      if (nextPollTimer !== null) clearTimeout(nextPollTimer);
    };
  }, [isActive, requestStatus]);

  const normalizedStatus = String(jobStatus?.status || 'idle').toLowerCase();
  const isRunning = isStarting || RUNNING_STATUSES.has(normalizedStatus);
  const isComplete = COMPLETE_STATUSES.has(normalizedStatus);
  const isFailed = normalizedStatus === 'failed' || normalizedStatus === 'error';
  const isConversionRunning = conversionStatus === 'running';
  const interactionLocked = !statusReady || isRunning;
  const parsedCriticEpochs = Number(criticEpochs);
  const parsedActorEquivalentEpochs = Number(actorEquivalentEpochs);
  const scheduleValid =
    Number.isInteger(parsedCriticEpochs) &&
    Number.isInteger(parsedActorEquivalentEpochs) &&
    parsedCriticEpochs > 0 &&
    parsedActorEquivalentEpochs > 0 &&
    parsedCriticEpochs === 2 * parsedActorEquivalentEpochs;

  useEffect(() => {
    if (onRunningChange) onRunningChange(interactionLocked);
  }, [interactionLocked, onRunningChange]);

  useEffect(() => {
    if (normalizedStatus === lastAnnouncedStatus.current) return;
    if (isComplete) toast.success('Offline RL training completed');
    if (isFailed) {
      toast.error(jobStatus?.message || 'Offline RL training failed');
    }
    lastAnnouncedStatus.current = normalizedStatus;
  }, [isComplete, isFailed, jobStatus?.message, normalizedStatus]);

  const progress = boundedPercentage(
    statusValue(jobStatus, 'percentage', 'progress_percentage', 'progress')
  );
  const displayProgress = Number(progress.toFixed(1));
  const modelPath = String(
    statusValue(jobStatus, 'model_path', 'pretrained_model_path') || ''
  );
  const checkpointPath = String(
    statusValue(jobStatus, 'checkpoint_path', 'training_state_path') || ''
  );

  const statusLabel = useMemo(() => {
    if (!statusReady) return 'Checking';
    if (isStarting || normalizedStatus === 'starting') return 'Starting';
    if (normalizedStatus === 'running') return 'Training';
    if (isComplete) return 'Complete';
    if (isFailed) return 'Failed';
    if (normalizedStatus === 'stopped') return 'Stopped';
    return 'Ready';
  }, [isComplete, isFailed, isStarting, normalizedStatus, statusReady]);

  const validateRequest = () => {
    if (!statusReady) return 'Wait for Offline RL status to load';
    if (isConversionRunning) return 'Wait for dataset conversion to finish';
    if (!datasetPath.trim()) return 'Select a LeRobot v3 dataset path';
    if (!actCheckpoint.trim()) return 'Select the original ACT checkpoint';
    if (!robotType?.trim()) return 'Select a robot type on the Home page first';
    if (algorithm !== 'td3') return 'Only TD3 is available';
    if (!scheduleValid) {
      return 'TD3 requires Critic epochs = 2 × Actor equivalent epochs';
    }
    return '';
  };

  const handleStart = async () => {
    const validationError = validateRequest();
    if (validationError) {
      toast.error(validationError);
      return;
    }
    isStartingRef.current = true;
    statusRequestSequence.current += 1;
    activeStatusRequest.current = null;
    setIsStarting(true);
    try {
      const result = await startOfflineRLTraining({
        dataset_path: datasetPath.trim(),
        act_checkpoint: actCheckpoint.trim(),
        parent_checkpoint: parentCheckpoint.trim(),
        algorithm,
        robot_type: robotType.trim(),
        critic_epochs: parsedCriticEpochs,
        actor_equivalent_epochs: parsedActorEquivalentEpochs,
      });
      setJobStatus(result || { status: 'starting' });
      toast.success('Offline RL training started');
    } catch (error) {
      toast.error(`Offline RL start failed: ${error.message}`);
      setStatusReady(false);
      isStartingRef.current = false;
      setIsStarting(false);
      await requestStatus();
      return;
    }
    isStartingRef.current = false;
    setIsStarting(false);
  };

  const browserDisabled = interactionLocked || !isActive;

  return (
    <section className="flex w-full flex-col gap-6 rounded-xl bg-gray-100 p-10">
      <div className="flex items-center gap-2">
        <MdModelTraining className="h-7 w-7 text-indigo-500" />
        <h2 className="text-2xl font-bold">Offline RL Training</h2>
      </div>

      <div className="grid w-full gap-6 rounded-md bg-white p-6 shadow-md lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.72fr)]">
        <div className="flex min-w-0 flex-col gap-4">
          <PathField
            id="offline-rl-dataset-path"
            label="LeRobot v3 Dataset Path"
            value={datasetPath}
            onChange={setDatasetPath}
            onBrowse={() => setShowDatasetBrowser(true)}
            placeholder="/workspace/lerobot/Task_*_lerobot_v30"
            disabled={browserDisabled}
          />
          <PathField
            id="offline-rl-act-checkpoint"
            label="Original ACT Checkpoint"
            value={actCheckpoint}
            onChange={setActCheckpoint}
            onBrowse={() => setShowActBrowser(true)}
            placeholder="/workspace/model/lerobot/.../pretrained_model"
            disabled={browserDisabled}
          />
          <p className="-mt-2 text-xs text-gray-500">
            Keep this original ACT checkpoint identical for every cumulative round.
          </p>
          <PathField
            id="offline-rl-parent-checkpoint"
            label="Previous Round Checkpoint"
            value={parentCheckpoint}
            onChange={setParentCheckpoint}
            onBrowse={() => setShowParentBrowser(true)}
            placeholder=".../training_state/act_td3.pt"
            disabled={browserDisabled}
            optional
          />

          <div className="flex flex-col gap-1.5">
            <label htmlFor="offline-rl-algorithm" className="text-sm font-medium text-gray-600">
              Training Algorithm
            </label>
            <select
              id="offline-rl-algorithm"
              value={algorithm}
              onChange={(event) => setAlgorithm(event.target.value)}
              disabled={browserDisabled}
              className="h-10 rounded-md border border-gray-300 bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-100"
            >
              <option value="td3">TD3 (ACT-TD3)</option>
              <option value="sac" disabled>SAC — Coming soon</option>
              <option value="rlt" disabled>RLT — Coming soon</option>
            </select>
          </div>

          <div className="grid grid-cols-2 gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600 sm:grid-cols-4">
            <div><span className="block text-gray-400">Maximum</span><b>200 episodes</b></div>
            <div><span className="block text-gray-400">New this round</span><b>Auto · 1–50</b></div>
            <label className="flex flex-col gap-1">
              <span className="text-gray-400">Critic epochs</span>
              <input
                aria-label="Critic epochs"
                type="number"
                min={2}
                step={2}
                value={criticEpochs}
                onChange={(event) => setCriticEpochs(event.target.value)}
                disabled={browserDisabled}
                className="h-8 rounded-md border border-gray-300 bg-white px-2 text-sm font-semibold disabled:bg-gray-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-gray-400">Actor equivalent epochs</span>
              <input
                aria-label="Actor equivalent epochs"
                type="number"
                min={1}
                step={1}
                value={actorEquivalentEpochs}
                onChange={(event) => setActorEquivalentEpochs(event.target.value)}
                disabled={browserDisabled}
                className="h-8 rounded-md border border-gray-300 bg-white px-2 text-sm font-semibold disabled:bg-gray-100"
              />
            </label>
          </div>
          <p className="-mt-2 text-xs text-gray-500">
            Round size is inferred from dataset growth. TD3 policy delay stays at 2,
            so Critic epochs must be exactly twice Actor equivalent epochs. A resumed
            round keeps its schedule; the next completed-data round may use new values.
          </p>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleStart}
              disabled={interactionLocked || isConversionRunning || !scheduleValid || !isActive}
              className="flex h-11 items-center justify-center gap-2 rounded-lg bg-indigo-600 px-6 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500"
            >
              <MdPlayArrow size={21} />
              {!statusReady ? 'Checking status…' : isRunning ? 'Training…' : 'Start Offline RL'}
            </button>
            <span className="text-sm text-gray-500">
              Robot: <b>{robotType || 'Not selected'}</b>
            </span>
          </div>
          {isConversionRunning && (
            <p className="text-xs font-medium text-amber-600">
              Dataset conversion is running. Training will unlock after it finishes.
            </p>
          )}
        </div>

        <div className="flex min-w-0 flex-col gap-4 rounded-2xl border border-gray-200 bg-gray-50 p-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-lg font-semibold text-gray-800">Training Status</h3>
            <span className={clsx(
              'rounded-full px-3 py-1 text-xs font-semibold',
              isComplete && 'bg-emerald-100 text-emerald-700',
              isFailed && 'bg-red-100 text-red-700',
              isRunning && 'bg-blue-100 text-blue-700',
              !isComplete && !isFailed && !isRunning && 'bg-gray-200 text-gray-600'
            )}>
              {statusLabel}
            </span>
          </div>

          <ProgressBar percent={displayProgress} />

          <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Episodes</span><b>{formatCount(statusValue(jobStatus, 'episode_count'))} / 200</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Round / New</span><b>{formatCount(statusValue(jobStatus, 'round_index'))} / {formatCount(statusValue(jobStatus, 'round_episode_count'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Success / Fail</span><b>{formatCount(statusValue(jobStatus, 'success_count'))} / {formatCount(statusValue(jobStatus, 'failure_count'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Epoch</span><b>{formatCount(statusValue(jobStatus, 'completed_epochs'))} / {formatCount(statusValue(jobStatus, 'total_epochs'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Critic updates</span><b>{formatCount(statusValue(jobStatus, 'completed_critic_updates'))} / {formatCount(statusValue(jobStatus, 'total_critic_updates'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Actor updates</span><b>{formatCount(statusValue(jobStatus, 'completed_actor_updates'))} / {formatCount(statusValue(jobStatus, 'total_actor_updates'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">ETA</span><b>{formatEta(statusValue(jobStatus, 'eta_seconds'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Critic loss</span><b>{formatLoss(statusValue(jobStatus, 'critic_loss'))}</b></div>
            <div className="rounded-md bg-white p-2"><span className="block text-gray-400">Actor loss</span><b>{formatLoss(statusValue(jobStatus, 'actor_loss'))}</b></div>
          </div>

          {jobStatus?.message && (
            <div className={clsx(
              'rounded-md p-2 text-xs',
              isFailed ? 'bg-red-50 text-red-700' : 'bg-white text-gray-600'
            )}>
              {jobStatus.message}
            </div>
          )}

          <div className="mt-auto flex flex-col gap-3 border-t border-gray-200 pt-3">
            <div>
              <div className="mb-1 text-xs font-medium text-gray-500">Final pretrained_model path</div>
              <div className="min-h-10 break-all rounded-md border border-gray-200 bg-white p-2 font-mono text-xs text-gray-700">
                {modelPath || 'Available after training completes'}
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-medium text-gray-500">Full training checkpoint path</div>
              <div className="min-h-10 break-all rounded-md border border-gray-200 bg-white p-2 font-mono text-xs text-gray-700">
                {checkpointPath || 'Available after training starts'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <FileBrowserModal
        isOpen={showDatasetBrowser}
        onClose={() => setShowDatasetBrowser(false)}
        onFileSelect={(item) => {
          setDatasetPath(item?.full_path || '');
          setShowDatasetBrowser(false);
        }}
        title="Select LeRobot v3 dataset"
        selectButtonText="Use Dataset"
        allowDirectorySelect
        allowFileSelect={false}
        targetFolderName="meta"
        targetFileLabel="LeRobot metadata"
        initialPath={DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
        defaultPath={DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
        homePath={DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
      />
      <FileBrowserModal
        isOpen={showActBrowser}
        onClose={() => setShowActBrowser(false)}
        onFileSelect={(item) => {
          setActCheckpoint(item?.full_path || '');
          setShowActBrowser(false);
        }}
        title="Select original ACT checkpoint"
        selectButtonText="Use Checkpoint"
        allowDirectorySelect
        allowFileSelect={false}
        targetFileName="config.json"
        targetFileLabel="ACT config"
        initialPath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
        defaultPath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
        homePath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
      />
      <FileBrowserModal
        isOpen={showParentBrowser}
        onClose={() => setShowParentBrowser(false)}
        onFileSelect={(item) => {
          setParentCheckpoint(item?.full_path || '');
          setShowParentBrowser(false);
        }}
        title="Select previous round checkpoint"
        selectButtonText="Use Checkpoint"
        allowDirectorySelect={false}
        allowFileSelect
        fileFilter={(item) => item?.name === 'act_td3.pt'}
        initialPath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
        defaultPath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
        homePath={DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH}
      />
    </section>
  );
}
