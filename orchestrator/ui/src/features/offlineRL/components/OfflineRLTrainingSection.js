// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import {
  MdDataObject,
  MdFolderOpen,
  MdModelTraining,
  MdPlayArrow,
  MdStop,
} from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import ProgressBar from '../../../components/ProgressBar';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { InferencePhase } from '../../../constants/taskPhases';
import {
  getFlowSDEPPOValueWarmupStatus,
  getImitationLearningStatus,
  getOfflineRLStatus,
  startFlowSDEPPOValueWarmup,
  startImitationLearningTraining,
  startOfflineRLTraining,
  stopFlowSDEPPOValueWarmup,
  stopImitationLearningTraining,
  stopOfflineRLTraining,
} from '../../../utils/offlineRlApi';
import {
  markLocalTaskInfoEdited,
  selectInferenceTaskInfo,
  setInferenceTaskInfo,
} from '../../tasks/taskSlice';
import {
  selectOfflineRLCheckpointPath,
  selectOfflineRLDatasetPath,
  selectOfflineRLDatasetSelections,
  setOfflineRLCheckpointPath,
  setOfflineRLDatasetPath,
} from '../offlineRLSlice';
import ACTArchitectureDiagram, {
  DEFAULT_ACT_TRAINABLE_GROUPS,
} from './ACTArchitectureDiagram';
import FlowSDEPPOArchitectureDiagram from './FlowSDEPPOArchitectureDiagram';
import GrootArchitectureDiagram from './GrootArchitectureDiagram';
import MultiTaskDiTArchitectureDiagram from './MultiTaskDiTArchitectureDiagram';
import PI05ArchitectureDiagram from './PI05ArchitectureDiagram';
import TD3ArchitectureDiagram from './TD3ArchitectureDiagram';

const POLL_INTERVAL_MS = 2000;
const IMITATION_ACTION_CHUNK_SIZES = Object.freeze({
  act: 30,
  multi_task_dit: 16,
});
const RUNNING_STATUSES = new Set(['starting', 'running']);
const COMPLETE_STATUSES = new Set(['complete', 'completed']);
const INFERENCE_PHASE_NAMES = {
  [InferencePhase.READY]: 'READY',
  [InferencePhase.LOADING]: 'LOADING',
  [InferencePhase.INFERENCING]: 'INFERENCING',
  [InferencePhase.PAUSED]: 'PAUSED',
};
const DETERMINISTIC_ACT_GROUPS = new Set([
  'visual_backbone',
  'transformer_encoder',
  'action_decoder',
]);

export const validateActorTrainableGroups = (groups) => {
  if (!Array.isArray(groups) || groups.length === 0) {
    return 'At least one ACT network block must be trainable';
  }
  if (!groups.some((group) => DETERMINISTIC_ACT_GROUPS.has(group))) {
    return 'TD3 requires a trainable deterministic ACT path; CVAE-only is not supported';
  }
  return '';
};

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

const normalizeContractPath = (value) => {
  const normalized = String(value || '').trim();
  if (normalized === '/') return normalized;
  return normalized.replace(/\/+$/, '');
};

const shortWarmupSource = (status, bundlePath) => {
  const jobId = String(status?.job_id || '').trim();
  if (jobId) return jobId.slice(0, 8);
  const normalizedPath = normalizeContractPath(bundlePath);
  return normalizedPath.split('/').filter(Boolean).pop() || 'bundle';
};

const formatCount = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toLocaleString() : '—';
};

const formatPolicyEpoch = (value) => {
  const parsed = Number(value);
  const epoch = Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
  return `E${String(epoch).padStart(4, '0')}`;
};

const formatLoss = (value) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toPrecision(5) : '—';
};

const formatEta = (value) => {
  if (value === null || value === undefined || value === '') return '—';
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

const WorkflowChoiceGroup = ({ label, children }) => (
  <div>
    <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-[#8d8579]">
      {label}
    </div>
    <div
      className="flex items-center gap-1 rounded-lg border border-[#d9d2c5] bg-[#f1ede4] p-0.5"
      role="group"
      aria-label={label}
    >
      {children}
    </div>
  </div>
);

const activeChoiceClass = 'h-7 rounded-md bg-[#69866f] px-3 text-[10px] font-semibold text-white shadow-sm';
const inactiveChoiceClass = 'h-7 rounded-md px-3 text-[10px] font-semibold text-[#746d62] hover:bg-[#e7e2d9] disabled:cursor-not-allowed disabled:opacity-60';
const disabledChoiceClass = 'h-7 cursor-not-allowed rounded-md px-3 text-[10px] font-semibold text-[#aaa295] opacity-70';

function CriticWarmupPanel({
  enabled,
  onEnabledChange,
  toggleDisabled,
  controlsDisabled,
  steps,
  setSteps,
  batchSize,
  setBatchSize,
  valueLearningRate,
  setValueLearningRate,
  discount,
  setDiscount,
  statusReady,
  statusLabel,
  progress,
  status,
  bundlePath,
  integrationReady,
  integrationMessage,
  sourceKind,
  sourceLabel,
  sourceReadyLabel,
  onStart,
  onStop,
  startDisabled,
  stopDisabled,
  isStarting,
  isStopping,
}) {
  return (
    <div
      className="mt-3 shrink-0 rounded-xl border border-[#d9d2c5] bg-white p-2.5"
      data-testid="flow-sde-ppo-critic-warmup"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="text-[10px] font-semibold text-[#514b42]">
            Offline value critic warm-up
          </div>
          <div className="text-[8px] text-[#91897d]">
            Checked Step 3 replay · Success + Fail required
          </div>
        </div>
        <div
          className="flex items-center gap-0.5 rounded-lg border border-[#d9d2c5] bg-[#f1ede4] p-0.5"
          role="group"
          aria-label="Critic warm-up"
        >
          <button
            type="button"
            aria-pressed={!enabled}
            disabled={toggleDisabled}
            onClick={() => onEnabledChange(false)}
            className={!enabled ? activeChoiceClass : inactiveChoiceClass}
          >
            No
          </button>
          <button
            type="button"
            aria-pressed={enabled}
            disabled={toggleDisabled}
            onClick={() => onEnabledChange(true)}
            className={enabled ? activeChoiceClass : inactiveChoiceClass}
          >
            Yes
          </button>
        </div>
      </div>

      {enabled && (
        <div className="mt-2" data-testid="critic-warmup-settings">
          <div className="grid grid-cols-2 gap-1.5 xl:grid-cols-4">
            <label className="text-[8px] font-semibold text-[#777064]">
              Steps
              <input
                aria-label="Critic warm-up steps"
                type="number"
                min={1}
                max={1000000}
                step={100}
                value={steps}
                onChange={(event) => setSteps(event.target.value)}
                disabled={controlsDisabled}
                className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
              />
            </label>
            <label className="text-[8px] font-semibold text-[#777064]">
              Batch size
              <input
                aria-label="Critic warm-up batch size"
                type="number"
                min={1}
                max={256}
                step={1}
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
                disabled={controlsDisabled}
                className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
              />
            </label>
            <label className="text-[8px] font-semibold text-[#777064]">
              Value LR
              <input
                aria-label="Critic warm-up value learning rate"
                type="number"
                min="0.00000001"
                max="1"
                step="0.00001"
                value={valueLearningRate}
                onChange={(event) => setValueLearningRate(event.target.value)}
                disabled={controlsDisabled}
                className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
              />
            </label>
            <label className="text-[8px] font-semibold text-[#777064]">
              Discount
              <input
                aria-label="Critic warm-up discount"
                type="number"
                min="0.0001"
                max="1"
                step="0.01"
                value={discount}
                onChange={(event) => setDiscount(event.target.value)}
                disabled={controlsDisabled}
                className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
              />
            </label>
          </div>

          <div className="mt-2 flex items-center justify-between gap-2 text-[8px] text-[#91897d]">
            <span>
              {statusReady ? statusLabel : 'Checking warm-up status'} · {progress}% · ETA{' '}
              {formatEta(statusValue(status, 'eta_seconds'))}
            </span>
            <span>
              Step {formatCount(statusValue(status, 'step', 'completed_steps'))}/
              {formatCount(statusValue(status, 'total_steps', 'steps'))} · Value loss{' '}
              {formatLoss(statusValue(status, 'value_loss', 'loss'))}
            </span>
          </div>
          <div
            className="mt-1 h-1.5 overflow-hidden rounded-full bg-[#ebe6dd]"
            role="progressbar"
            aria-label="Critic warm-up progress"
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={progress}
          >
            <div
              className="h-full rounded-full bg-[#69866f] transition-[width]"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="mt-2 grid grid-cols-[minmax(0,1fr)_92px_72px] gap-1.5">
            <output
              aria-label="Critic warm-up bundle path"
              title={bundlePath || 'Created after critic warm-up completes'}
              className="flex h-7 min-w-0 items-center truncate rounded-md border border-[#d9d2c5] bg-[#f5f2eb] px-2 text-[9px] text-[#6f685d]"
            >
              {bundlePath || 'Bundle path · pending'}
            </output>
            <button
              type="button"
              onClick={onStart}
              disabled={startDisabled}
              className={clsx(
                'h-7 rounded-md border text-[9px] font-semibold',
                startDisabled
                  ? 'cursor-not-allowed border-[#d9d2c5] bg-[#e9e5dc] text-[#9b9387]'
                  : 'border-[#5f7965] bg-[#69866f] text-white hover:bg-[#5f7965]'
              )}
            >
              {isStarting ? 'Starting…' : 'Train Critic'}
            </button>
            <button
              type="button"
              onClick={onStop}
              disabled={stopDisabled}
              className={clsx(
                'h-7 rounded-md border text-[9px] font-semibold',
                stopDisabled
                  ? 'cursor-not-allowed border-[#d9d2c5] bg-[#eeeae2] text-[#aaa296]'
                  : 'border-[#b77a70] bg-[#fff7f5] text-[#a45f55] hover:bg-[#f7e4df]'
              )}
            >
              {isStopping ? 'Stopping…' : 'Stop'}
            </button>
          </div>
          {integrationReady ? (
            <p
              className="mt-1.5 text-[8px] font-semibold text-[#55715d]"
              data-testid="critic-warmup-source"
            >
              Critic source: {sourceKind} · {sourceLabel} · {sourceReadyLabel}
            </p>
          ) : (
            <p className="mt-1.5 text-[8px] font-medium text-[#a8795b]">
              {integrationMessage}
            </p>
          )}
        </div>
      )}
    </div>
  );
}

function WorkflowTrainingView({
  trainingMethod,
  onTrainingMethodChange,
  selectedPolicyModel,
  onPolicyModelChange,
  algorithm,
  onAlgorithmChange,
  flowSdePpoReady,
  flowInferenceBlockedReason,
  criticWarmupEnabled,
  onCriticWarmupEnabledChange,
  warmupSteps,
  setWarmupSteps,
  warmupBatchSize,
  setWarmupBatchSize,
  warmupValueLearningRate,
  setWarmupValueLearningRate,
  warmupDiscount,
  setWarmupDiscount,
  warmupStatus,
  warmupStatusReady,
  warmupStatusLabel,
  warmupProgress,
  warmupBundlePath,
  warmupIntegrationReady,
  warmupIntegrationMessage,
  warmupSourceKind,
  warmupSourceLabel,
  warmupSourceReadyLabel,
  handleWarmupStart,
  handleWarmupStop,
  warmupStartDisabled,
  warmupStopDisabled,
  isWarmupStarting,
  isWarmupStopping,
  warmupIsRunning,
  actorTrainableGroups,
  setActorTrainableGroups,
  browserDisabled,
  criticEpochs,
  setCriticEpochs,
  actorEquivalentEpochs,
  setActorEquivalentEpochs,
  batchSize,
  setBatchSize,
  imitationSteps,
  setImitationSteps,
  imitationBatchSize,
  setImitationBatchSize,
  imitationSaveFreq,
  setImitationSaveFreq,
  statusLabel,
  displayProgress,
  jobStatus,
  modelPath,
  currentPolicyEpoch,
  datasetSelections,
  actCheckpoint,
  robotType,
  handleStart,
  handleStop,
  handleFlowSDEPPOOutcome,
  startDisabled,
  stopDisabled,
  flowOutcomeDisabled,
  isRunning,
  isStopping,
  isSubmittingOutcome,
  statusReady,
  isConversionRunning,
  trainabilityError,
}) {
  const isImitationLearning = trainingMethod === 'imitation';
  const isActSelected = selectedPolicyModel === 'act';
  const isMultiTaskDiTSelected = selectedPolicyModel === 'multi_task_dit';
  const isFlowSdePpo = (
    !isImitationLearning &&
    isMultiTaskDiTSelected &&
    algorithm === 'flow_sde_ppo'
  );
  const imitationActionChunkSize = (
    IMITATION_ACTION_CHUNK_SIZES[selectedPolicyModel] ||
    IMITATION_ACTION_CHUNK_SIZES.act
  );
  const imitationPolicyName = isMultiTaskDiTSelected
    ? 'Diffusion Transformer'
    : 'ACT';
  const isSupportedPolicy = isActSelected || isMultiTaskDiTSelected;
  const parsedRoundIndex = Number(jobStatus?.round_index);
  const targetPolicyEpoch = (
    !isImitationLearning &&
    isActSelected &&
    algorithm === 'td3' &&
    Number.isInteger(parsedRoundIndex) &&
    parsedRoundIndex >= 1
  ) ? parsedRoundIndex : Number(currentPolicyEpoch) + 1;
  const selectedPolicyLabel = {
    act: 'ACT',
    multi_task_dit: 'Diffusion Transformer',
    groot: 'GR00T',
    pi05: 'Pi0.5',
  }[selectedPolicyModel] || selectedPolicyModel;
  const workflowStartDisabled = (
    startDisabled ||
    !isSupportedPolicy ||
    (isFlowSdePpo && (
      !flowSdePpoReady ||
      (criticWarmupEnabled && !warmupIntegrationReady)
    ))
  );
  const invalidDatasetVersion = datasetSelections.find(
    (selection) => selection.version && selection.version !== 'v3.0'
  )?.version;

  const handleWorkflowStart = () => {
    if (
      !isSupportedPolicy ||
      (isFlowSdePpo && (
        !flowSdePpoReady ||
        (criticWarmupEnabled && !warmupIntegrationReady)
      ))
    ) return;
    handleStart();
  };

  const renderPolicyDiagram = () => {
    if (selectedPolicyModel === 'groot') return <GrootArchitectureDiagram />;
    if (selectedPolicyModel === 'multi_task_dit') {
      return <MultiTaskDiTArchitectureDiagram />;
    }
    if (selectedPolicyModel === 'pi05') {
      return <PI05ArchitectureDiagram disabled={browserDisabled} />;
    }
    return (
      <ACTArchitectureDiagram
        trainableGroups={isImitationLearning
          ? DEFAULT_ACT_TRAINABLE_GROUPS
          : actorTrainableGroups}
        onChange={setActorTrainableGroups}
        disabled={browserDisabled || isImitationLearning}
      />
    );
  };

  return (
    <div
      className="mt-3 flex min-h-0 min-w-0 flex-1 flex-col"
      data-testid="offline-rl-workflow-training"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <WorkflowChoiceGroup label="Policy model">
          <button
            type="button"
            aria-pressed={selectedPolicyModel === 'act'}
            disabled={browserDisabled}
            onClick={() => onPolicyModelChange('act')}
            className={selectedPolicyModel === 'act' ? activeChoiceClass : inactiveChoiceClass}
          >
            ACT
          </button>
          <button
            type="button"
            disabled={browserDisabled}
            aria-pressed={selectedPolicyModel === 'multi_task_dit'}
            onClick={() => onPolicyModelChange('multi_task_dit')}
            className={selectedPolicyModel === 'multi_task_dit'
              ? activeChoiceClass
              : inactiveChoiceClass}
          >
            Diffusion Transformer
          </button>
          <button
            type="button"
            disabled={browserDisabled || isImitationLearning}
            aria-pressed={selectedPolicyModel === 'groot'}
            onClick={() => onPolicyModelChange('groot')}
            className={selectedPolicyModel === 'groot' ? activeChoiceClass : inactiveChoiceClass}
          >
            GR00T
          </button>
          <button
            type="button"
            disabled={browserDisabled || isImitationLearning}
            aria-pressed={selectedPolicyModel === 'pi05'}
            onClick={() => onPolicyModelChange('pi05')}
            className={selectedPolicyModel === 'pi05' ? activeChoiceClass : inactiveChoiceClass}
          >
            Pi0.5
          </button>
        </WorkflowChoiceGroup>

        <div className="flex flex-wrap items-start justify-end gap-2">
          <WorkflowChoiceGroup label="Training method">
            <button
              type="button"
              aria-pressed={!isImitationLearning}
              disabled={browserDisabled}
              onClick={() => onTrainingMethodChange('reinforcement')}
              className={!isImitationLearning ? activeChoiceClass : inactiveChoiceClass}
            >
              Reinforcement Learning
            </button>
            <button
              type="button"
              aria-pressed={isImitationLearning}
              disabled={browserDisabled}
              onClick={() => onTrainingMethodChange('imitation')}
              className={isImitationLearning ? activeChoiceClass : inactiveChoiceClass}
            >
              Imitation Learning
            </button>
          </WorkflowChoiceGroup>

          <WorkflowChoiceGroup label="RL algorithm">
            <button
              type="button"
              disabled={isImitationLearning || browserDisabled}
              aria-pressed={!isImitationLearning && algorithm === 'td3'}
              onClick={() => onAlgorithmChange('td3')}
              className={!isImitationLearning && algorithm === 'td3'
                ? activeChoiceClass
                : (isImitationLearning ? disabledChoiceClass : inactiveChoiceClass)}
            >
              TD3
            </button>
            <button
              type="button"
              disabled={isImitationLearning || browserDisabled}
              aria-pressed={!isImitationLearning && algorithm === 'flow_sde_ppo'}
              onClick={() => onAlgorithmChange('flow_sde_ppo')}
              className={!isImitationLearning && algorithm === 'flow_sde_ppo'
                ? activeChoiceClass
                : (isImitationLearning ? disabledChoiceClass : inactiveChoiceClass)}
              title="PPO over Flow-SDE action-chunk trajectories"
            >
              PPO
              <span className="ml-1 rounded-full bg-white/25 px-1.5 py-0.5 text-[8px]">
                Flow-SDE
              </span>
            </button>
            <button
              type="button"
              disabled
              aria-pressed="false"
              title="SAC training backend is coming soon"
              className={disabledChoiceClass}
            >
              SAC
            </button>
          </WorkflowChoiceGroup>
        </div>
      </div>

      <div
        className="mt-2 grid min-h-0 flex-1 items-stretch gap-2 lg:grid-cols-[minmax(0,1.25fr)_minmax(210px,0.75fr)]"
        data-testid="offline-rl-training-architecture"
      >
        {renderPolicyDiagram()}

        <div className="flex h-full min-h-0 flex-col rounded-xl border border-[#e0d9ce] bg-[#f5f1e9] p-3">
          {isImitationLearning ? (
            <>
              <div
                className="flex min-h-0 flex-1 flex-col"
                data-testid="act-imitation-learning-diagram"
              >
                <div className="flex items-center justify-between gap-2">
                  <div>
                    <div className="text-[13px] font-semibold text-[#39352e]">
                      {imitationPolicyName} imitation learning
                    </div>
                    <div className="text-[9px] text-[#8d8579]">
                      {isMultiTaskDiTSelected
                        ? 'Supervised flow-matching · no reward or outcome labels required'
                        : 'Full CVAE behavior cloning · all ACT blocks trainable'}
                    </div>
                  </div>
                  <span className="rounded-full bg-[#e8ebef] px-2.5 py-1 text-[9px] font-semibold text-[#65707e]">
                    IL
                  </span>
                </div>
                <div className="mt-3 grid min-h-0 flex-1 grid-rows-[1fr_auto_1fr_auto_1fr] gap-1 text-center text-[10px] font-semibold text-[#514b42]">
                  <div className="flex items-center justify-center rounded-lg border border-[#d9d2c5] bg-white px-2">
                    3 images + robot state
                  </div>
                  <div className="text-[#aaa295]">↓</div>
                  <div className="flex items-center justify-center rounded-lg border border-[#9faf9f] bg-[#edf3ec] px-2 text-[#344a38]">
                    {isMultiTaskDiTSelected
                      ? 'Diffusion Transformer · flow-matching training'
                      : 'ACT CVAE policy · full training'}
                  </div>
                  <div className="text-[#aaa295]">↓</div>
                  <div className="flex items-center justify-center rounded-lg border border-[#d9d2c5] bg-white px-2">
                    {imitationActionChunkSize}-step action chunk
                  </div>
                </div>
              </div>

              <div className="mt-3 grid shrink-0 grid-cols-4 gap-1.5">
                <label className="text-[8px] font-semibold text-[#777064]">
                  Steps
                  <input
                    aria-label="Imitation steps"
                    type="number"
                    min={1}
                    max={1000000}
                    step={1000}
                    value={imitationSteps}
                    onChange={(event) => setImitationSteps(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <label className="text-[8px] font-semibold text-[#777064]">
                  Batch size
                  <input
                    aria-label="Imitation batch size"
                    type="number"
                    min={1}
                    max={64}
                    step={1}
                    value={imitationBatchSize}
                    onChange={(event) => setImitationBatchSize(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <label className="text-[8px] font-semibold text-[#777064]">
                  Save frequency
                  <input
                    aria-label="Imitation save frequency"
                    type="number"
                    min={1}
                    step={1000}
                    value={imitationSaveFreq}
                    onChange={(event) => setImitationSaveFreq(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <div className="text-[8px] font-semibold text-[#777064]">
                  Action chunk
                  <output
                    aria-label="Imitation action chunk"
                    className="mt-1 flex h-7 w-full items-center rounded-md border border-[#d9d2c5] bg-[#ece8df] px-2 text-[10px] text-[#403b34]"
                  >
                    {imitationActionChunkSize}
                  </output>
                </div>
              </div>
            </>
          ) : isMultiTaskDiTSelected ? (
            <>
              <CriticWarmupPanel
                enabled={criticWarmupEnabled}
                onEnabledChange={onCriticWarmupEnabledChange}
                toggleDisabled={browserDisabled || warmupIsRunning}
                controlsDisabled={browserDisabled || warmupIsRunning}
                steps={warmupSteps}
                setSteps={setWarmupSteps}
                batchSize={warmupBatchSize}
                setBatchSize={setWarmupBatchSize}
                valueLearningRate={warmupValueLearningRate}
                setValueLearningRate={setWarmupValueLearningRate}
                discount={warmupDiscount}
                setDiscount={setWarmupDiscount}
                statusReady={warmupStatusReady}
                statusLabel={warmupStatusLabel}
                progress={warmupProgress}
                status={warmupStatus}
                bundlePath={warmupBundlePath}
                integrationReady={warmupIntegrationReady}
                integrationMessage={warmupIntegrationMessage}
                sourceKind={warmupSourceKind}
                sourceLabel={warmupSourceLabel}
                sourceReadyLabel={warmupSourceReadyLabel}
                onStart={handleWarmupStart}
                onStop={handleWarmupStop}
                startDisabled={warmupStartDisabled}
                stopDisabled={warmupStopDisabled}
                isStarting={isWarmupStarting}
                isStopping={isWarmupStopping}
              />

              <FlowSDEPPOArchitectureDiagram backendReady={flowSdePpoReady} />

              <div className="mt-3 grid shrink-0 grid-cols-3 gap-1.5 text-center">
                {[
                  ['Rollout', 'Online'],
                  ['Action chunk', '16 × 22D'],
                  ['Obs encoder', 'Frozen'],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-lg border border-[#d9d2c5] bg-white px-2 py-1.5"
                  >
                    <div className="text-[8px] font-semibold text-[#8d8579]">{label}</div>
                    <div className="mt-0.5 text-[10px] font-semibold text-[#514b42]">{value}</div>
                  </div>
                ))}
              </div>

            </>
          ) : isActSelected ? (
            <>
              <TD3ArchitectureDiagram />

              <div className="mt-3 grid shrink-0 grid-cols-3 gap-1.5">
                <label className="text-[8px] font-semibold text-[#777064]">
                  Critic epochs
                  <input
                    aria-label="Critic epochs"
                    type="number"
                    min={2}
                    step={2}
                    value={criticEpochs}
                    onChange={(event) => setCriticEpochs(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <label className="text-[8px] font-semibold text-[#777064]">
                  Actor epochs
                  <input
                    aria-label="Actor equivalent epochs"
                    type="number"
                    min={1}
                    step={1}
                    value={actorEquivalentEpochs}
                    onChange={(event) => setActorEquivalentEpochs(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
                <label className="text-[8px] font-semibold text-[#777064]">
                  Batch size
                  <input
                    aria-label="Batch size"
                    type="number"
                    min={1}
                    max={64}
                    step={1}
                    title="Batch size must remain unchanged across cumulative resume rounds"
                    value={batchSize}
                    onChange={(event) => setBatchSize(event.target.value)}
                    disabled={browserDisabled}
                    className="mt-1 h-7 w-full rounded-md border border-[#d9d2c5] bg-white px-2 text-[10px] text-[#403b34] outline-none disabled:cursor-not-allowed disabled:bg-[#ece8df]"
                  />
                </label>
              </div>
            </>
          ) : (
            <div className="flex min-h-0 flex-1 items-center justify-center text-center">
              <div>
                <div className="text-[13px] font-semibold text-[#655e54]">
                  No compatible RL algorithm
                </div>
                <div className="mt-1 text-[10px] text-[#8d8579]">
                  Select ACT + TD3 or Diffusion Transformer + Flow-SDE PPO.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div
        className="mt-auto grid shrink-0 gap-2 border-t border-[#e2dcd1] pt-3 lg:grid-cols-[minmax(0,1fr)_250px]"
        data-testid="offline-rl-training-footer"
      >
        <div className="rounded-xl border border-[#e2dcd1] bg-white p-2.5">
          <div className="flex items-center justify-between gap-2 text-[10px]">
            <span className="flex min-w-0 items-center gap-2 font-semibold text-[#514b42]">
              <span>Training progress</span>
              {!isImitationLearning && isActSelected && algorithm === 'td3' && (
                <span
                  className="shrink-0 rounded-md border border-[#cfd8cd] bg-[#e8eee6] px-1.5 py-0.5 font-mono text-[9px] font-bold text-[#58705d]"
                  aria-label={`ACT-TD3 policy RL Epoch ${currentPolicyEpoch} to ${targetPolicyEpoch}`}
                >
                  RL Epoch {formatPolicyEpoch(currentPolicyEpoch)} → {formatPolicyEpoch(targetPolicyEpoch)}
                </span>
              )}
            </span>
            <span className="text-[#91897d]">
              {statusLabel} · {displayProgress}%
              {isImitationLearning && (
                <> · Step {formatCount(statusValue(jobStatus, 'step', 'completed_steps'))}
                  /{formatCount(statusValue(jobStatus, 'total_steps'))}</>
              )}
              {' '}· ETA {formatEta(statusValue(jobStatus, 'eta_seconds'))}
              {!isImitationLearning && isActSelected && algorithm === 'td3' && (
                <> · Critic replay {formatCount(statusValue(jobStatus, 'completed_epochs'))}
                  /{formatCount(statusValue(jobStatus, 'total_epochs'))}</>
              )}
            </span>
          </div>
          <div
            className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#ebe6dd]"
            role="progressbar"
            aria-label={isImitationLearning
              ? 'Imitation Learning training progress'
              : (isMultiTaskDiTSelected
                ? 'Flow-SDE PPO training progress'
                : 'Offline RL training progress')}
            aria-valuemin="0"
            aria-valuemax="100"
            aria-valuenow={displayProgress}
          >
            <div
              className="h-full rounded-full bg-[#69866f] transition-[width]"
              style={{ width: `${displayProgress}%` }}
            />
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2 text-center">
            {(isImitationLearning ? (isMultiTaskDiTSelected ? [
              ['Flow loss', formatLoss(statusValue(jobStatus, 'loss', 'flow_loss'))],
              ['Step', formatCount(statusValue(jobStatus, 'step', 'completed_steps'))],
              ['Policy', modelPath ? 'Ready' : '—'],
            ] : [
              ['Total loss', formatLoss(statusValue(jobStatus, 'loss', 'total_loss'))],
              ['L1 loss', formatLoss(statusValue(jobStatus, 'l1_loss'))],
              ['KLD loss', formatLoss(statusValue(jobStatus, 'kld_loss'))],
            ]) : isMultiTaskDiTSelected ? [
              ['Actor loss', formatLoss(statusValue(jobStatus, 'actor_loss', 'policy_loss'))],
              ['Value loss', formatLoss(statusValue(jobStatus, 'value_loss'))],
              ['Approx. KL', formatLoss(statusValue(jobStatus, 'approx_kl', 'kl'))],
            ] : [
              ['Critic loss', formatLoss(statusValue(jobStatus, 'critic_loss'))],
              ['Actor loss', formatLoss(statusValue(jobStatus, 'actor_loss'))],
              ['Policy', modelPath ? 'Ready' : '—'],
            ]).map(([label, value]) => (
              <div key={label} className="rounded-lg bg-[#f5f2eb] px-2 py-1.5">
                <div className="text-[9px] text-[#999185]">{label}</div>
                <div className="mt-0.5 truncate text-[11px] font-semibold text-[#4c473f]">{value}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col justify-between rounded-xl border border-[#e2dcd1] bg-[#f8f5ef] p-2.5">
          <div>
            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-[#575147]">
              <MdDataObject size={13} /> Training action
            </div>
            {!isSupportedPolicy ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {selectedPolicyLabel} diagram preview only. Offline RL training backend is not connected.
                {' '}Training is available for ACT and Diffusion Transformer.
              </p>
            ) : isFlowSdePpo && criticWarmupEnabled && !warmupIntegrationReady ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {warmupIntegrationMessage}
              </p>
            ) : isFlowSdePpo && !flowSdePpoReady ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                Flow-SDE PPO backend is not ready. The policy and algorithm contract can be
                {' '}reviewed, but Start Training remains disabled until backend readiness is reported.
              </p>
            ) : isFlowSdePpo && flowInferenceBlockedReason ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {flowInferenceBlockedReason}
              </p>
            ) : !isFlowSdePpo && invalidDatasetVersion ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {isImitationLearning
                  ? `${imitationPolicyName} imitation learning`
                  : 'TD3'} requires LeRobot v3.0.
                {' '}The selected {invalidDatasetVersion} dataset is view only.
              </p>
            ) : !isImitationLearning && isActSelected && trainabilityError ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {trainabilityError}
              </p>
            ) : (
              <p className="mt-1 text-[9px] leading-relaxed text-[#948c80]">
                {isImitationLearning
                  ? (datasetSelections.length
                    ? `${imitationPolicyName} imitation learning ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · ${imitationSteps} steps · batch ${imitationBatchSize} · ${imitationActionChunkSize}-step chunk · no reward or Success/Fail labels required`
                    : `Include at least one LeRobot v3.0 Data Epoch in Step 3. No base ${imitationPolicyName} checkpoint, reward, or Success/Fail label is required.`)
                  : isMultiTaskDiTSelected
                    ? (actCheckpoint && robotType
                      ? 'Diffusion Transformer + Flow-SDE PPO ready · live on-policy rollout · frozen observation encoder'
                      : 'Select a MultiTaskDiT model in Workspace Paths and a robot type on Home. No LeRobot dataset is required.')
                    : (datasetSelections.length && actCheckpoint
                      ? `ACT-TD3 ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · batch ${batchSize} · ${actorTrainableGroups.length} trainable blocks`
                      : 'Include at least one LeRobot v3.0 Data Epoch in Step 3 and select an ACT model in Workspace Paths.')}
                {!isImitationLearning && <> Robot: {robotType || 'Not selected'}</>}
              </p>
            )}
            {isConversionRunning && !isFlowSdePpo && (
              <p className="mt-1 text-[9px] font-medium text-[#a8795b]">
                Dataset conversion is running.
              </p>
            )}
          </div>
          <div className="mt-2 grid grid-cols-2 gap-1.5">
            <button
              type="button"
              onClick={handleWorkflowStart}
              disabled={workflowStartDisabled}
              className={clsx(
                'flex h-8 items-center justify-center gap-1 rounded-lg border text-[9px] font-semibold',
                workflowStartDisabled
                  ? 'cursor-not-allowed border-[#d9d2c5] bg-[#e9e5dc] text-[#9b9387]'
                  : 'border-[#5f7965] bg-[#69866f] text-white hover:bg-[#5f7965]'
              )}
            >
              <MdPlayArrow size={14} />
              {!statusReady ? 'Checking…' : isRunning ? 'Training…' : 'Start Training'}
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={stopDisabled}
              className={clsx(
                'flex h-8 items-center justify-center gap-1 rounded-lg border text-[9px] font-semibold',
                stopDisabled
                  ? 'cursor-not-allowed border-[#d9d2c5] bg-[#eeeae2] text-[#aaa296]'
                  : 'border-[#b77a70] bg-[#fff7f5] text-[#a45f55] hover:bg-[#f7e4df]'
              )}
            >
              <MdStop size={14} />
              {isStopping ? 'Stopping…' : 'Stop Training'}
            </button>
          </div>
          {isFlowSdePpo && isRunning && (
            <div className="mt-2 border-t border-[#e2dcd1] pt-2">
              <div className="mb-1.5 flex items-center justify-between gap-2 text-[8px] font-semibold text-[#777064]">
                <span>Current episode outcome</span>
                {isSubmittingOutcome ? (
                  <span className="text-[#777064]">Saving…</span>
                ) : jobStatus?.awaiting_outcome ? (
                  <span className="text-[#a06458]">Outcome required</span>
                ) : null}
              </div>
              <div
                className="grid grid-cols-3 gap-1.5"
                role="group"
                aria-label="Flow-SDE episode outcome"
              >
                <button
                  type="button"
                  onClick={() => handleFlowSDEPPOOutcome('success')}
                  disabled={flowOutcomeDisabled}
                  className="h-7 rounded-md border border-[#78937d] bg-[#edf4ed] text-[9px] font-semibold text-[#48654e] hover:bg-[#dfeade] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Success
                </button>
                <button
                  type="button"
                  onClick={() => handleFlowSDEPPOOutcome('fail')}
                  disabled={flowOutcomeDisabled}
                  className="h-7 rounded-md border border-[#bd8177] bg-[#fff4f1] text-[9px] font-semibold text-[#9d584f] hover:bg-[#f7e4df] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Fail
                </button>
                <button
                  type="button"
                  onClick={() => handleFlowSDEPPOOutcome('cancel')}
                  disabled={flowOutcomeDisabled}
                  className="h-7 rounded-md border border-[#c9c0b2] bg-[#f1ede5] text-[9px] font-semibold text-[#6f685d] hover:bg-[#e6e0d6] disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function OfflineRLTrainingSection({
  isActive = true,
  inferencePhase = InferencePhase.READY,
  onRunningChange,
  onDeploymentStateChange,
  currentPolicyEpoch = 0,
  forceFreshLineage = false,
  onFreshLineageConsumed,
  flowSdePpoReady = false,
  getFlowSDEPPOStatus,
  onStartFlowSDEPPO,
  onStopFlowSDEPPO,
  onSubmitFlowSDEPPOOutcome,
  variant = 'default',
}) {
  const dispatch = useDispatch();
  const robotType = useSelector((state) => state.tasks.robotType);
  const inferenceTaskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const datasetPath = useSelector(selectOfflineRLDatasetPath);
  const datasetSelections = useSelector(selectOfflineRLDatasetSelections, shallowEqual);
  const parentCheckpoint = useSelector(selectOfflineRLCheckpointPath);
  const actCheckpoint = inferenceTaskInfo.policyPath || '';
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus?.status || 'idle'
  );
  const [trainingMethod, setTrainingMethod] = useState('reinforcement');
  const [algorithm, setAlgorithm] = useState('td3');
  const [selectedPolicyModel, setSelectedPolicyModel] = useState('act');
  const [actorTrainableGroups, setActorTrainableGroups] = useState(
    DEFAULT_ACT_TRAINABLE_GROUPS
  );
  const [criticEpochs, setCriticEpochs] = useState('10');
  const [actorEquivalentEpochs, setActorEquivalentEpochs] = useState('5');
  const [batchSize, setBatchSize] = useState('4');
  const [imitationSteps, setImitationSteps] = useState('80000');
  const [imitationBatchSize, setImitationBatchSize] = useState('8');
  const [imitationSaveFreq, setImitationSaveFreq] = useState('10000');
  const [criticWarmupEnabled, setCriticWarmupEnabled] = useState(false);
  const [warmupSteps, setWarmupSteps] = useState('2000');
  const [warmupBatchSize, setWarmupBatchSize] = useState('8');
  const [warmupValueLearningRate, setWarmupValueLearningRate] = useState('0.0001');
  const [warmupDiscount, setWarmupDiscount] = useState('0.99');
  const [warmupStatus, setWarmupStatus] = useState({ status: 'idle' });
  const [warmupStatusReady, setWarmupStatusReady] = useState(false);
  const [isWarmupStarting, setIsWarmupStarting] = useState(false);
  const [isWarmupStopping, setIsWarmupStopping] = useState(false);
  const [jobStatus, setJobStatus] = useState({ status: 'idle' });
  const [statusReady, setStatusReady] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isSubmittingOutcome, setIsSubmittingOutcome] = useState(false);
  const [showDatasetBrowser, setShowDatasetBrowser] = useState(false);
  const [showActBrowser, setShowActBrowser] = useState(false);
  const [showParentBrowser, setShowParentBrowser] = useState(false);
  const lastAnnouncedStatus = useRef('idle');
  const statusRequestSequence = useRef(0);
  const activeStatusRequest = useRef(null);
  const isStartingRef = useRef(false);
  const isStoppingRef = useRef(false);
  const actorTrainabilityHydratedRef = useRef(false);
  const batchSizeHydratedRef = useRef(false);
  const imitationConfigHydratedRef = useRef(false);
  const warmupStatusRequestSequence = useRef(0);
  const activeWarmupStatusRequest = useRef(null);
  const isWarmupStartingRef = useRef(false);
  const isWarmupStoppingRef = useRef(false);

  const requestStatus = useCallback(async ({ isCancelled = () => false } = {}) => {
    if (
      isStartingRef.current ||
      isStoppingRef.current ||
      activeStatusRequest.current !== null
    ) {
      return null;
    }

    const requestSequence = ++statusRequestSequence.current;
    activeStatusRequest.current = requestSequence;
    try {
      const status = trainingMethod === 'imitation'
        ? await getImitationLearningStatus()
        : (selectedPolicyModel === 'multi_task_dit' && getFlowSDEPPOStatus
          ? await getFlowSDEPPOStatus()
          : await getOfflineRLStatus());
      if (
        isCancelled() ||
        isStartingRef.current ||
        isStoppingRef.current ||
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
        !isStoppingRef.current &&
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
  }, [getFlowSDEPPOStatus, selectedPolicyModel, trainingMethod]);

  const requestWarmupStatus = useCallback(async ({ isCancelled = () => false } = {}) => {
    if (
      isWarmupStartingRef.current ||
      isWarmupStoppingRef.current ||
      activeWarmupStatusRequest.current !== null
    ) {
      return null;
    }

    const requestSequence = ++warmupStatusRequestSequence.current;
    activeWarmupStatusRequest.current = requestSequence;
    try {
      const status = await getFlowSDEPPOValueWarmupStatus();
      if (
        isCancelled() ||
        isWarmupStartingRef.current ||
        isWarmupStoppingRef.current ||
        requestSequence !== warmupStatusRequestSequence.current
      ) {
        return null;
      }
      setWarmupStatus(status || { status: 'idle' });
      setWarmupStatusReady(true);
      return status;
    } catch {
      if (
        !isCancelled() &&
        !isWarmupStartingRef.current &&
        !isWarmupStoppingRef.current &&
        requestSequence === warmupStatusRequestSequence.current
      ) {
        setWarmupStatusReady(false);
      }
      return null;
    } finally {
      if (activeWarmupStatusRequest.current === requestSequence) {
        activeWarmupStatusRequest.current = null;
      }
    }
  }, []);

  const setDatasetPath = useCallback((value) => {
    dispatch(setOfflineRLDatasetPath(value));
  }, [dispatch]);

  const setActCheckpoint = useCallback((value) => {
    dispatch(setInferenceTaskInfo({ policyPath: value }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  }, [dispatch]);

  const setParentCheckpoint = useCallback((value) => {
    dispatch(setOfflineRLCheckpointPath(value));
  }, [dispatch]);

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
      if (isStartingRef.current || isStoppingRef.current) {
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
  const isImitationLearning = trainingMethod === 'imitation';
  const isFlowSdePpo = (
    !isImitationLearning &&
    selectedPolicyModel === 'multi_task_dit' &&
    algorithm === 'flow_sde_ppo'
  );
  const imitationPolicyType = selectedPolicyModel === 'multi_task_dit'
    ? 'multi_task_dit'
    : 'act';
  const imitationActionChunkSize = IMITATION_ACTION_CHUNK_SIZES[imitationPolicyType];
  const flowInferenceReady = inferencePhase === InferencePhase.READY;
  const flowInferenceBlockedReason = flowInferenceReady
    ? ''
    : `Flow-SDE PPO requires Inference READY (current: ${
      INFERENCE_PHASE_NAMES[inferencePhase] || 'UNKNOWN'
    }).`;
  useEffect(() => {
    if (!isActive || !isFlowSdePpo || !criticWarmupEnabled) {
      warmupStatusRequestSequence.current += 1;
      activeWarmupStatusRequest.current = null;
      setWarmupStatusReady(false);
      return undefined;
    }

    let cancelled = false;
    let nextPollTimer = null;
    const scheduleNextPoll = () => {
      if (!cancelled) nextPollTimer = setTimeout(poll, POLL_INTERVAL_MS);
    };
    async function poll() {
      if (cancelled) return;
      if (isWarmupStartingRef.current || isWarmupStoppingRef.current) {
        scheduleNextPoll();
        return;
      }
      try {
        await requestWarmupStatus({ isCancelled: () => cancelled });
      } finally {
        scheduleNextPoll();
      }
    }
    poll();
    return () => {
      cancelled = true;
      warmupStatusRequestSequence.current += 1;
      activeWarmupStatusRequest.current = null;
      if (nextPollTimer !== null) clearTimeout(nextPollTimer);
    };
  }, [criticWarmupEnabled, isActive, isFlowSdePpo, requestWarmupStatus]);

  const normalizedWarmupStatus = String(warmupStatus?.status || 'idle').toLowerCase();
  const warmupIsRunning = (
    isWarmupStarting || RUNNING_STATUSES.has(normalizedWarmupStatus)
  );
  const warmupIsComplete = COMPLETE_STATUSES.has(normalizedWarmupStatus);
  const warmupProgress = Number(boundedPercentage(
    statusValue(warmupStatus, 'percentage', 'progress_percentage', 'progress')
  ).toFixed(1));
  const warmupBundlePath = String(
    statusValue(warmupStatus, 'bundle_path') || ''
  ).trim();
  const warmupStatusLabel = !warmupStatusReady
    ? 'Checking'
    : (isWarmupStarting
      ? 'Starting'
      : (normalizedWarmupStatus === 'running'
        ? 'Training critic'
        : (warmupIsComplete
          ? 'Complete'
          : (['failed', 'error'].includes(normalizedWarmupStatus)
            ? 'Failed'
            : (normalizedWarmupStatus === 'stopped' ? 'Stopped' : 'Ready')))));
  const parsedCriticEpochs = Number(criticEpochs);
  const parsedActorEquivalentEpochs = Number(actorEquivalentEpochs);
  const parsedBatchSize = Number(batchSize);
  const parsedImitationSteps = Number(imitationSteps);
  const parsedImitationBatchSize = Number(imitationBatchSize);
  const parsedImitationSaveFreq = Number(imitationSaveFreq);
  const parsedWarmupSteps = Number(warmupSteps);
  const parsedWarmupBatchSize = Number(warmupBatchSize);
  const parsedWarmupValueLearningRate = Number(warmupValueLearningRate);
  const parsedWarmupDiscount = Number(warmupDiscount);
  const scheduleValid =
    Number.isInteger(parsedCriticEpochs) &&
    Number.isInteger(parsedActorEquivalentEpochs) &&
    parsedCriticEpochs > 0 &&
    parsedActorEquivalentEpochs > 0 &&
    parsedCriticEpochs === 2 * parsedActorEquivalentEpochs;
  const batchSizeValid = (
    Number.isInteger(parsedBatchSize) &&
    parsedBatchSize >= 1 &&
    parsedBatchSize <= 64
  );
  const imitationStepsValid = (
    Number.isInteger(parsedImitationSteps) &&
    parsedImitationSteps >= 1 &&
    parsedImitationSteps <= 1000000
  );
  const imitationBatchSizeValid = (
    Number.isInteger(parsedImitationBatchSize) &&
    parsedImitationBatchSize >= 1 &&
    parsedImitationBatchSize <= 64
  );
  const imitationSaveFreqValid = (
    Number.isInteger(parsedImitationSaveFreq) &&
    parsedImitationSaveFreq >= 1 &&
    parsedImitationSaveFreq <= parsedImitationSteps
  );
  const warmupConfigValid = (
    Number.isInteger(parsedWarmupSteps) &&
    parsedWarmupSteps >= 1 &&
    parsedWarmupSteps <= 1000000 &&
    Number.isInteger(parsedWarmupBatchSize) &&
    parsedWarmupBatchSize >= 1 &&
    parsedWarmupBatchSize <= 256 &&
    Number.isFinite(parsedWarmupValueLearningRate) &&
    parsedWarmupValueLearningRate > 0 &&
    parsedWarmupValueLearningRate <= 1 &&
    Number.isFinite(parsedWarmupDiscount) &&
    parsedWarmupDiscount > 0 &&
    parsedWarmupDiscount <= 1
  );
  const trainabilityError = validateActorTrainableGroups(actorTrainableGroups);
  const trainabilityValid = trainabilityError === '';
  const datasetPaths = useMemo(
    () => datasetSelections.map((selection) => String(selection.path || '').trim()).filter(Boolean),
    [datasetSelections]
  );
  const selectedDatasetVersionInvalid = datasetSelections.some(
    (selection) => selection.version && selection.version !== 'v3.0'
  );
  const flowTaskInstruction = useMemo(() => {
    const instructions = inferenceTaskInfo?.taskInstruction;
    const firstInstruction = Array.isArray(instructions) ? instructions[0] : instructions;
    return String(firstInstruction || '').trim();
  }, [inferenceTaskInfo?.taskInstruction]);
  const warmupPolicyCheckpoint = normalizeContractPath(warmupStatus?.policy_checkpoint);
  const selectedFlowPolicyCheckpoint = normalizeContractPath(actCheckpoint);
  const warmupTaskInstruction = String(warmupStatus?.task_instruction || '').trim();
  const warmupPolicyMatches = Boolean(
    warmupPolicyCheckpoint &&
    selectedFlowPolicyCheckpoint &&
    warmupPolicyCheckpoint === selectedFlowPolicyCheckpoint
  );
  const warmupTaskMatches = Boolean(
    warmupTaskInstruction &&
    flowTaskInstruction &&
    warmupTaskInstruction === flowTaskInstruction
  );
  const compatibleWarmupReady = Boolean(
    warmupStatusReady &&
    warmupIsComplete &&
    warmupBundlePath &&
    warmupPolicyMatches &&
    warmupTaskMatches
  );
  const resumeJobId = String(jobStatus?.job_id || '').trim();
  const resumeImmediatePolicy = normalizeContractPath(jobStatus?.policy_checkpoint);
  const resumeLineagePolicy = normalizeContractPath(jobStatus?.lineage_policy_checkpoint);
  const resumeModelPath = normalizeContractPath(jobStatus?.model_path);
  const resumeCheckpointPath = normalizeContractPath(jobStatus?.checkpoint_path);
  const resumeTaskInstruction = String(jobStatus?.task_instruction || '').trim();
  const resumeContractPresent = Boolean(
    statusReady &&
    isComplete &&
    resumeJobId &&
    resumeImmediatePolicy &&
    resumeModelPath &&
    resumeCheckpointPath &&
    resumeTaskInstruction
  );
  const resumePolicyMatches = Boolean(
    resumeContractPresent &&
    selectedFlowPolicyCheckpoint &&
    [resumeLineagePolicy, resumeImmediatePolicy, resumeModelPath]
      .filter(Boolean)
      .includes(selectedFlowPolicyCheckpoint)
  );
  const resumeTaskMatches = Boolean(
    resumeContractPresent &&
    flowTaskInstruction &&
    resumeTaskInstruction === flowTaskInstruction
  );
  const ppoResumeReady = Boolean(
    resumeContractPresent && resumePolicyMatches && resumeTaskMatches
  );
  const ppoResumeMismatch = Boolean(
    resumeContractPresent && (!resumePolicyMatches || !resumeTaskMatches)
  );
  // A compatible completed PPO trainer state is newer than an offline warm-up
  // and therefore takes precedence. An unrelated recovered PPO job must not
  // shadow a compatible warm-up for the currently selected policy and task.
  const warmupIntegrationReady = Boolean(
    ppoResumeReady || compatibleWarmupReady
  );
  const warmupSourceKind = ppoResumeReady ? 'PPO' : 'Warm-up';
  const warmupSourceLabel = ppoResumeReady
    ? shortWarmupSource(jobStatus, resumeCheckpointPath)
    : shortWarmupSource(warmupStatus, warmupBundlePath);
  const warmupSourceReadyLabel = ppoResumeReady
    ? 'Ready to continue'
    : 'Ready for online PPO';
  const warmupIntegrationMessage = ppoResumeMismatch && !compatibleWarmupReady
    ? (!resumePolicyMatches
      ? 'Completed PPO checkpoint belongs to a different policy lineage.'
      : 'Completed PPO checkpoint belongs to a different task instruction.')
    : (!warmupStatusReady
      ? 'Checking for a compatible completed critic warm-up.'
      : (warmupIsRunning
        ? 'Critic warm-up is running. Online PPO starts after it completes.'
        : (!warmupIsComplete
          ? 'Train the critic to completion, or select No to start PPO with a new critic.'
          : (!warmupBundlePath
            ? 'Completed critic warm-up has no verified bundle path.'
            : (!warmupPolicyMatches
              ? 'Completed critic warm-up belongs to a different policy checkpoint.'
              : (!warmupTaskMatches
                ? 'Completed critic warm-up belongs to a different task instruction.'
                : 'Completed critic warm-up is not ready for online PPO.'))))));

  const resetStatusChannel = () => {
    statusRequestSequence.current += 1;
    activeStatusRequest.current = null;
    actorTrainabilityHydratedRef.current = false;
    batchSizeHydratedRef.current = false;
    imitationConfigHydratedRef.current = false;
    lastAnnouncedStatus.current = 'idle';
    setJobStatus({ status: 'idle' });
    setStatusReady(false);
  };

  const handleTrainingMethodChange = (nextMethod) => {
    if (interactionLocked || nextMethod === trainingMethod) return;
    resetStatusChannel();
    if (nextMethod === 'reinforcement') {
      setAlgorithm(selectedPolicyModel === 'multi_task_dit' ? 'flow_sde_ppo' : 'td3');
    }
    setTrainingMethod(nextMethod);
  };

  const handlePolicyModelChange = (nextPolicyModel) => {
    if (interactionLocked || nextPolicyModel === selectedPolicyModel) return;
    if (
      nextPolicyModel === 'multi_task_dit' ||
      selectedPolicyModel === 'multi_task_dit'
    ) {
      resetStatusChannel();
    }
    setSelectedPolicyModel(nextPolicyModel);
    if (nextPolicyModel === 'act') {
      setAlgorithm('td3');
      return;
    }
    if (nextPolicyModel === 'multi_task_dit') {
      setAlgorithm('flow_sde_ppo');
      return;
    }
    setAlgorithm('');
    if (trainingMethod === 'imitation') setTrainingMethod('reinforcement');
  };

  const handleAlgorithmChange = (nextAlgorithm) => {
    if (interactionLocked || nextAlgorithm === algorithm) return;
    const nextPolicyModel = nextAlgorithm === 'flow_sde_ppo' ? 'multi_task_dit' : 'act';
    resetStatusChannel();
    setTrainingMethod('reinforcement');
    setSelectedPolicyModel(nextPolicyModel);
    setAlgorithm(nextAlgorithm);
  };

  useEffect(() => {
    if (isImitationLearning || actorTrainabilityHydratedRef.current || normalizedStatus === 'idle') {
      return;
    }
    const reportedGroups = jobStatus?.actor_trainable_groups;
    if (!Array.isArray(reportedGroups)) return;
    const reportedSet = new Set(reportedGroups);
    const canonicalGroups = DEFAULT_ACT_TRAINABLE_GROUPS.filter((group) => (
      reportedSet.has(group)
    ));
    if (validateActorTrainableGroups(canonicalGroups)) return;
    setActorTrainableGroups(canonicalGroups);
    actorTrainabilityHydratedRef.current = true;
  }, [isImitationLearning, jobStatus?.actor_trainable_groups, normalizedStatus]);

  useEffect(() => {
    if (isImitationLearning || batchSizeHydratedRef.current || normalizedStatus === 'idle') return;
    const reportedBatchSize = Number(jobStatus?.batch_size);
    if (!Number.isInteger(reportedBatchSize) || reportedBatchSize < 1 || reportedBatchSize > 64) {
      return;
    }
    setBatchSize(String(reportedBatchSize));
    batchSizeHydratedRef.current = true;
  }, [isImitationLearning, jobStatus?.batch_size, normalizedStatus]);

  useEffect(() => {
    if (!isImitationLearning || imitationConfigHydratedRef.current || normalizedStatus === 'idle') {
      return;
    }
    const reportedSteps = Number(statusValue(jobStatus, 'total_steps', 'steps'));
    const reportedBatchSize = Number(jobStatus?.batch_size);
    const reportedSaveFreq = Number(jobStatus?.save_freq);
    if (Number.isInteger(reportedSteps) && reportedSteps >= 1 && reportedSteps <= 1000000) {
      setImitationSteps(String(reportedSteps));
    }
    if (Number.isInteger(reportedBatchSize) && reportedBatchSize >= 1 && reportedBatchSize <= 64) {
      setImitationBatchSize(String(reportedBatchSize));
    }
    if (Number.isInteger(reportedSaveFreq) && reportedSaveFreq >= 1) {
      setImitationSaveFreq(String(reportedSaveFreq));
    }
    imitationConfigHydratedRef.current = true;
  }, [
    isImitationLearning,
    jobStatus,
    normalizedStatus,
  ]);

  useEffect(() => {
    if (!RUNNING_STATUSES.has(normalizedStatus)) setIsStopping(false);
  }, [normalizedStatus]);

  useEffect(() => {
    if (onRunningChange) onRunningChange(interactionLocked || warmupIsRunning);
  }, [interactionLocked, onRunningChange, warmupIsRunning]);

  useEffect(() => {
    if (normalizedStatus === lastAnnouncedStatus.current) return;
    const methodLabel = isImitationLearning ? 'Imitation Learning' : 'Offline RL';
    if (isComplete) toast.success(`${methodLabel} training completed`);
    if (isFailed) {
      toast.error(jobStatus?.message || `${methodLabel} training failed`);
    }
    if (normalizedStatus === 'stopped') {
      toast.success(`${methodLabel} training stopped`);
    }
    lastAnnouncedStatus.current = normalizedStatus;
  }, [isComplete, isFailed, isImitationLearning, jobStatus?.message, normalizedStatus]);

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

  useEffect(() => {
    if (!onDeploymentStateChange) return;
    const reportedPolicyType = ['act', 'multi_task_dit'].includes(jobStatus?.policy_type)
      ? jobStatus.policy_type
      : selectedPolicyModel;
    const reportedRoundIndex = Number(jobStatus?.round_index);
    const deployedRLEpoch = (
      !isImitationLearning &&
      selectedPolicyModel === 'act' &&
      algorithm === 'td3' &&
      Number.isInteger(reportedRoundIndex) &&
      reportedRoundIndex >= 1
    ) ? reportedRoundIndex : Number(currentPolicyEpoch);
    onDeploymentStateChange({
      ready: statusReady && isComplete && Boolean(modelPath.trim()),
      modelPath: isComplete ? modelPath.trim() : '',
      serviceType: 'lerobot',
      policyType: reportedPolicyType,
      rlEpoch: deployedRLEpoch,
    });
  }, [
    algorithm,
    currentPolicyEpoch,
    isComplete,
    isImitationLearning,
    jobStatus?.round_index,
    jobStatus?.policy_type,
    modelPath,
    onDeploymentStateChange,
    selectedPolicyModel,
    statusReady,
  ]);

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
    if (!statusReady) return 'Wait for training status to load';
    if (isFlowSdePpo) {
      if (!actCheckpoint.trim()) return 'Select the MultiTaskDiT checkpoint';
      if (!robotType?.trim()) return 'Select a robot type on the Home page first';
      if (!flowInferenceReady) return flowInferenceBlockedReason;
      if (!flowSdePpoReady || typeof onStartFlowSDEPPO !== 'function') {
        return 'Flow-SDE PPO backend is not ready';
      }
      if (criticWarmupEnabled && !warmupIntegrationReady) {
        return warmupIntegrationMessage;
      }
      return '';
    }
    if (isConversionRunning) return 'Wait for dataset conversion to finish';
    if (!datasetPaths.length) return 'Include at least one LeRobot v3 Data Epoch';
    if (selectedDatasetVersionInvalid) {
      const trainingLabel = isImitationLearning
        ? `${imitationPolicyType === 'multi_task_dit' ? 'Diffusion Transformer' : 'ACT'} imitation learning`
        : 'TD3';
      return `${trainingLabel} requires LeRobot v3.0`;
    }
    if (isImitationLearning) {
      if (!imitationStepsValid) return 'Imitation steps must be an integer from 1 to 1,000,000';
      if (!imitationBatchSizeValid) return 'Imitation batch size must be an integer from 1 to 64';
      if (!imitationSaveFreqValid) {
        return 'Imitation save frequency must be an integer from 1 through the total steps';
      }
      return '';
    }
    if (!actCheckpoint.trim()) {
      return isFlowSdePpo
        ? 'Select the MultiTaskDiT checkpoint'
        : 'Select the original ACT checkpoint';
    }
    if (!robotType?.trim()) return 'Select a robot type on the Home page first';
    if (algorithm !== 'td3') return 'Only TD3 is available';
    if (trainabilityError) return trainabilityError;
    if (!batchSizeValid) return 'Batch size must be an integer from 1 to 64';
    if (!scheduleValid) {
      return 'TD3 requires Critic epochs = 2 × Actor equivalent epochs';
    }
    return '';
  };

  const validateWarmupRequest = () => {
    if (!isFlowSdePpo || !criticWarmupEnabled) {
      return 'Select Diffusion Transformer + Flow-SDE PPO and enable Critic warm-up';
    }
    if (!warmupStatusReady) return 'Wait for critic warm-up status to load';
    if (isConversionRunning) return 'Wait for dataset conversion to finish';
    if (!datasetPaths.length) return 'Include at least one checked LeRobot v3.0 Data Epoch';
    if (selectedDatasetVersionInvalid) return 'Critic warm-up requires LeRobot v3.0';
    if (!actCheckpoint.trim()) return 'Select the MultiTaskDiT checkpoint';
    if (!flowTaskInstruction) return 'Enter a task instruction for critic warm-up';
    if (!warmupConfigValid) {
      return 'Warm-up settings require valid steps, batch size, value LR, and discount';
    }
    if (!flowSdePpoReady) return 'Flow-SDE PPO backend is not ready';
    return '';
  };

  const handleCriticWarmupEnabledChange = (enabled) => {
    if (warmupIsRunning || Boolean(enabled) === criticWarmupEnabled) return;
    warmupStatusRequestSequence.current += 1;
    activeWarmupStatusRequest.current = null;
    setWarmupStatusReady(false);
    setCriticWarmupEnabled(Boolean(enabled));
  };

  const handleWarmupStart = async () => {
    const validationError = validateWarmupRequest();
    if (validationError) {
      toast.error(validationError);
      return;
    }

    isWarmupStartingRef.current = true;
    warmupStatusRequestSequence.current += 1;
    activeWarmupStatusRequest.current = null;
    setIsWarmupStarting(true);
    try {
      const result = await startFlowSDEPPOValueWarmup({
        policy_checkpoint: actCheckpoint.trim(),
        dataset_paths: datasetPaths,
        policy_type: 'multi_task_dit',
        task_instruction: flowTaskInstruction,
        steps: parsedWarmupSteps,
        batch_size: parsedWarmupBatchSize,
        value_learning_rate: parsedWarmupValueLearningRate,
        discount: parsedWarmupDiscount,
      });
      setWarmupStatus(result || { status: 'running' });
      setWarmupStatusReady(true);
      toast.success('Value critic warm-up started');
    } catch (error) {
      toast.error(`Value critic warm-up start failed: ${error.message}`);
      setWarmupStatusReady(false);
      isWarmupStartingRef.current = false;
      setIsWarmupStarting(false);
      await requestWarmupStatus();
      return;
    }
    isWarmupStartingRef.current = false;
    setIsWarmupStarting(false);
  };

  const handleWarmupStop = async () => {
    const jobId = String(warmupStatus?.job_id || '').trim();
    if (!jobId || !RUNNING_STATUSES.has(normalizedWarmupStatus) || isWarmupStopping) return;
    if (!window.confirm(
      'Stop the current value critic warm-up job?\n\n' +
      'The backend will stop at a safe boundary; any completed bundle remains on disk.'
    )) return;

    isWarmupStoppingRef.current = true;
    warmupStatusRequestSequence.current += 1;
    activeWarmupStatusRequest.current = null;
    setIsWarmupStopping(true);
    try {
      const result = await stopFlowSDEPPOValueWarmup(jobId);
      setWarmupStatus(result || { ...warmupStatus, status: 'running' });
      toast.success('Value critic warm-up stop requested');
    } catch (error) {
      toast.error(`Value critic warm-up stop failed: ${error.message}`);
      setWarmupStatusReady(false);
      isWarmupStoppingRef.current = false;
      setIsWarmupStopping(false);
      await requestWarmupStatus();
      return;
    }
    isWarmupStoppingRef.current = false;
    setIsWarmupStopping(false);
  };

  const completedDatasetPaths = Array.isArray(jobStatus?.dataset_paths) &&
    jobStatus.dataset_paths.length
    ? jobStatus.dataset_paths.map((path) => String(path || '').trim()).filter(Boolean)
    : [String(jobStatus?.dataset_path || '').trim()].filter(Boolean);
  const completedBaseActCheckpoint = String(jobStatus?.act_checkpoint || '').trim();
  const selectedActCheckpoint = actCheckpoint.trim();
  const selectedModelMatchesCompletedLineage = (
    selectedActCheckpoint === completedBaseActCheckpoint ||
    selectedActCheckpoint === modelPath.trim()
  );
  const selectedReplayExtendsCompletedReplay = (
    completedDatasetPaths.length > 0 &&
    completedDatasetPaths.length < datasetPaths.length &&
    completedDatasetPaths.every((path, index) => path === datasetPaths[index])
  );
  const canAutoResumeWorkflow = (
    variant === 'workflow' &&
    !forceFreshLineage &&
    isComplete &&
    Boolean(checkpointPath.trim()) &&
    Boolean(completedBaseActCheckpoint) &&
    selectedReplayExtendsCompletedReplay &&
    selectedModelMatchesCompletedLineage
  );

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
      const result = isImitationLearning
        ? await startImitationLearningTraining({
          // Keep the legacy scalar together with the authoritative ordered
          // roots so the ACT-IL adapter can train every checked Data Epoch.
          dataset_path: datasetPaths[0],
          dataset_paths: datasetPaths,
          policy_type: imitationPolicyType,
          steps: parsedImitationSteps,
          batch_size: parsedImitationBatchSize,
          save_freq: parsedImitationSaveFreq,
          chunk_size: imitationActionChunkSize,
          ...(imitationPolicyType === 'multi_task_dit' && flowTaskInstruction
            ? { task_instruction: flowTaskInstruction }
            : {}),
        })
        : isFlowSdePpo
          ? await onStartFlowSDEPPO({
            policy_type: 'multi_task_dit',
            policy_checkpoint: criticWarmupEnabled && ppoResumeReady
              ? resumeModelPath
              : selectedActCheckpoint,
            algorithm: 'flow_sde_ppo',
            robot_type: robotType.trim(),
            ...(criticWarmupEnabled
              ? (ppoResumeReady
                ? { resume_checkpoint: resumeCheckpointPath }
                : { value_warmup_bundle: warmupBundlePath })
              : {}),
            ...(flowTaskInstruction
              ? { task_instruction: flowTaskInstruction }
              : {}),
          })
          : await startOfflineRLTraining({
          // Keep the first root in the legacy scalar field while the ordered
          // list is authoritative for immutable multi-epoch replay.
          dataset_path: datasetPaths[0],
          dataset_paths: datasetPaths,
          // A completed server job is the sole resume source for the compact
          // workflow. This keeps the immutable base ACT path even after Deploy
          // Policy changes the inference model to the trained output. A hidden
          // Redux checkpoint is never submitted by this variant.
          act_checkpoint: canAutoResumeWorkflow
            ? completedBaseActCheckpoint
            : selectedActCheckpoint,
          parent_checkpoint: variant === 'workflow'
            ? (canAutoResumeWorkflow ? checkpointPath.trim() : '')
            : parentCheckpoint.trim(),
          algorithm,
          robot_type: robotType.trim(),
          critic_epochs: parsedCriticEpochs,
          actor_equivalent_epochs: parsedActorEquivalentEpochs,
          batch_size: parsedBatchSize,
          actor_trainable_groups: actorTrainableGroups,
        });
      setJobStatus(result || { status: 'starting' });
      if (
        forceFreshLineage &&
        !isImitationLearning &&
        !isFlowSdePpo
      ) {
        onFreshLineageConsumed?.();
      }
      const methodLabel = isImitationLearning
        ? 'Imitation Learning'
        : (isFlowSdePpo ? 'Flow-SDE PPO' : 'Offline RL');
      toast.success(`${methodLabel} training started`);
    } catch (error) {
      const methodLabel = isImitationLearning
        ? 'Imitation Learning'
        : (isFlowSdePpo ? 'Flow-SDE PPO' : 'Offline RL');
      toast.error(`${methodLabel} start failed: ${error.message}`);
      setStatusReady(false);
      isStartingRef.current = false;
      setIsStarting(false);
      await requestStatus();
      return;
    }
    isStartingRef.current = false;
    setIsStarting(false);
  };

  const handleStop = async () => {
    const jobId = String(jobStatus?.job_id || '').trim();
    if (!jobId || !RUNNING_STATUSES.has(normalizedStatus) || isStopping) return;
    if (isFlowSdePpo && typeof onStopFlowSDEPPO !== 'function') return;
    if (!window.confirm(
      `Stop the current ${isImitationLearning ? 'Imitation Learning' : 'Offline RL'} training job?\n\n` +
      'The current job will stop at a safe boundary and will not export a deployable policy.'
    )) return;

    isStoppingRef.current = true;
    statusRequestSequence.current += 1;
    activeStatusRequest.current = null;
    setIsStopping(true);
    try {
      const result = isImitationLearning
        ? await stopImitationLearningTraining(jobId)
        : (isFlowSdePpo
          ? await onStopFlowSDEPPO(jobId)
          : await stopOfflineRLTraining(jobId));
      setJobStatus(result || { ...jobStatus, status: 'running' });
      toast.success(`${isImitationLearning ? 'Imitation Learning' : 'Offline RL'} stop requested`);
    } catch (error) {
      toast.error(`${isImitationLearning ? 'Imitation Learning' : 'Offline RL'} stop failed: ${error.message}`);
      setStatusReady(false);
      isStoppingRef.current = false;
      setIsStopping(false);
      await requestStatus();
      return;
    }
    isStoppingRef.current = false;
  };

  const handleFlowSDEPPOOutcome = async (outcome) => {
    const jobId = String(jobStatus?.job_id || '').trim();
    if (
      !isFlowSdePpo ||
      !RUNNING_STATUSES.has(normalizedStatus) ||
      !jobId ||
      typeof onSubmitFlowSDEPPOOutcome !== 'function' ||
      isSubmittingOutcome
    ) return;

    setIsSubmittingOutcome(true);
    try {
      const result = await onSubmitFlowSDEPPOOutcome(jobId, outcome);
      setJobStatus(result || jobStatus);
      toast.success(`Episode marked ${outcome}`);
    } catch (error) {
      toast.error(`Episode outcome failed: ${error.message}`);
    } finally {
      setIsSubmittingOutcome(false);
    }
  };

  const browserDisabled = interactionLocked || warmupIsRunning || !isActive;
  const selectedTrainingConfigurationValid = isImitationLearning
    ? imitationStepsValid && imitationBatchSizeValid && imitationSaveFreqValid
    : (isFlowSdePpo
      ? (
        flowSdePpoReady &&
        typeof onStartFlowSDEPPO === 'function' &&
        Boolean(actCheckpoint.trim()) &&
        Boolean(robotType?.trim()) &&
        flowInferenceReady
      )
      : scheduleValid && batchSizeValid && trainabilityValid);
  const startDisabled = (
    interactionLocked ||
    (!isFlowSdePpo && isConversionRunning) ||
    (isFlowSdePpo && criticWarmupEnabled && !warmupIntegrationReady) ||
    !selectedTrainingConfigurationValid ||
    (!isFlowSdePpo && selectedDatasetVersionInvalid) ||
    !isActive
  );
  const stopDisabled = (
    !isActive ||
    !statusReady ||
    !RUNNING_STATUSES.has(normalizedStatus) ||
    !String(jobStatus?.job_id || '').trim() ||
    (isFlowSdePpo && typeof onStopFlowSDEPPO !== 'function') ||
    isStopping
  );
  const flowOutcomeDisabled = (
    !isActive ||
    !isFlowSdePpo ||
    !RUNNING_STATUSES.has(normalizedStatus) ||
    jobStatus?.awaiting_outcome !== true ||
    !String(jobStatus?.job_id || '').trim() ||
    typeof onSubmitFlowSDEPPOOutcome !== 'function' ||
    isSubmittingOutcome
  );
  const warmupStartDisabled = (
    !isActive ||
    !isFlowSdePpo ||
    !criticWarmupEnabled ||
    !warmupStatusReady ||
    warmupIsRunning ||
    isRunning ||
    isConversionRunning ||
    !flowSdePpoReady ||
    !datasetPaths.length ||
    selectedDatasetVersionInvalid ||
    !actCheckpoint.trim() ||
    !flowTaskInstruction ||
    !warmupConfigValid
  );
  const warmupStopDisabled = (
    !isActive ||
    !criticWarmupEnabled ||
    !warmupStatusReady ||
    !RUNNING_STATUSES.has(normalizedWarmupStatus) ||
    !String(warmupStatus?.job_id || '').trim() ||
    isWarmupStopping
  );

  if (variant === 'workflow') {
    return (
      <WorkflowTrainingView
        trainingMethod={trainingMethod}
        onTrainingMethodChange={handleTrainingMethodChange}
        selectedPolicyModel={selectedPolicyModel}
        onPolicyModelChange={handlePolicyModelChange}
        algorithm={algorithm}
        onAlgorithmChange={handleAlgorithmChange}
        flowSdePpoReady={flowSdePpoReady}
        flowInferenceBlockedReason={flowInferenceBlockedReason}
        criticWarmupEnabled={criticWarmupEnabled}
        onCriticWarmupEnabledChange={handleCriticWarmupEnabledChange}
        warmupSteps={warmupSteps}
        setWarmupSteps={setWarmupSteps}
        warmupBatchSize={warmupBatchSize}
        setWarmupBatchSize={setWarmupBatchSize}
        warmupValueLearningRate={warmupValueLearningRate}
        setWarmupValueLearningRate={setWarmupValueLearningRate}
        warmupDiscount={warmupDiscount}
        setWarmupDiscount={setWarmupDiscount}
        warmupStatus={warmupStatus}
        warmupStatusReady={warmupStatusReady}
        warmupStatusLabel={warmupStatusLabel}
        warmupProgress={warmupProgress}
        warmupBundlePath={warmupBundlePath}
        warmupIntegrationReady={warmupIntegrationReady}
        warmupIntegrationMessage={warmupIntegrationMessage}
        warmupSourceKind={warmupSourceKind}
        warmupSourceLabel={warmupSourceLabel}
        warmupSourceReadyLabel={warmupSourceReadyLabel}
        handleWarmupStart={handleWarmupStart}
        handleWarmupStop={handleWarmupStop}
        warmupStartDisabled={warmupStartDisabled}
        warmupStopDisabled={warmupStopDisabled}
        isWarmupStarting={isWarmupStarting}
        isWarmupStopping={isWarmupStopping}
        warmupIsRunning={warmupIsRunning}
        actorTrainableGroups={actorTrainableGroups}
        setActorTrainableGroups={setActorTrainableGroups}
        browserDisabled={browserDisabled}
        criticEpochs={criticEpochs}
        setCriticEpochs={setCriticEpochs}
        actorEquivalentEpochs={actorEquivalentEpochs}
        setActorEquivalentEpochs={setActorEquivalentEpochs}
        batchSize={batchSize}
        setBatchSize={setBatchSize}
        imitationSteps={imitationSteps}
        setImitationSteps={setImitationSteps}
        imitationBatchSize={imitationBatchSize}
        setImitationBatchSize={setImitationBatchSize}
        imitationSaveFreq={imitationSaveFreq}
        setImitationSaveFreq={setImitationSaveFreq}
        statusLabel={statusLabel}
        displayProgress={displayProgress}
        jobStatus={jobStatus}
        modelPath={modelPath}
        currentPolicyEpoch={currentPolicyEpoch}
        datasetSelections={datasetSelections}
        actCheckpoint={actCheckpoint}
        robotType={robotType}
        handleStart={handleStart}
        handleStop={handleStop}
        handleFlowSDEPPOOutcome={handleFlowSDEPPOOutcome}
        startDisabled={startDisabled}
        stopDisabled={stopDisabled}
        flowOutcomeDisabled={flowOutcomeDisabled}
        isRunning={isRunning}
        isStopping={isStopping}
        isSubmittingOutcome={isSubmittingOutcome}
        statusReady={statusReady}
        isConversionRunning={isConversionRunning}
        trainabilityError={trainabilityError}
      />
    );
  }

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

          <div className="grid grid-cols-2 gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs text-gray-600 sm:grid-cols-5">
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
            <label className="flex flex-col gap-1">
              <span className="text-gray-400">Batch size</span>
              <input
                aria-label="Batch size"
                type="number"
                min={1}
                max={64}
                step={1}
                title="Batch size must remain unchanged across cumulative resume rounds"
                value={batchSize}
                onChange={(event) => setBatchSize(event.target.value)}
                disabled={browserDisabled}
                className="h-8 rounded-md border border-gray-300 bg-white px-2 text-sm font-semibold disabled:bg-gray-100"
              />
            </label>
          </div>
          <p className="-mt-2 text-xs text-gray-500">
            Round size is inferred from dataset growth. TD3 policy delay stays at 2,
            so Critic epochs must be exactly twice Actor equivalent epochs. A resumed
            round keeps its batch size; a fresh training lineage may choose a new value.
          </p>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleStart}
              disabled={startDisabled}
              className="flex h-11 items-center justify-center gap-2 rounded-lg bg-indigo-600 px-6 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-gray-300 disabled:text-gray-500"
            >
              <MdPlayArrow size={21} />
              {!statusReady ? 'Checking status…' : isRunning ? 'Training…' : 'Start Training'}
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={stopDisabled}
              className="flex h-11 items-center justify-center gap-2 rounded-lg border border-red-300 bg-red-50 px-5 text-sm font-semibold text-red-700 hover:bg-red-100 disabled:cursor-not-allowed disabled:border-gray-200 disabled:bg-gray-100 disabled:text-gray-400"
            >
              <MdStop size={20} />
              {isStopping ? 'Stopping…' : 'Stop Training'}
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
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">
                {displayProgress}% · ETA {formatEta(statusValue(jobStatus, 'eta_seconds'))}
              </span>
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
