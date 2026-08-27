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
  getACTTD3CriticWarmupStatus,
  getFlowSDEPPOValueWarmupStatus,
  getImitationLearningStatus,
  getOfflineRLDatasetInfo,
  getOfflineRLStatus,
  startACTTD3CriticWarmup,
  startFlowSDEPPOValueWarmup,
  startImitationLearningTraining,
  startOfflineRLTraining,
  stopACTTD3CriticWarmup,
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
import ACTTD3TrainingLoop, {
  ImitationLearningCard,
  PolicyTrainingLoopLayout,
} from './ACTTD3TrainingLoop';
import FlowSDEPPOArchitectureDiagram from './FlowSDEPPOArchitectureDiagram';
import GrootArchitectureDiagram from './GrootArchitectureDiagram';
import MultiTaskDiTArchitectureDiagram from './MultiTaskDiTArchitectureDiagram';
import PI05ArchitectureDiagram from './PI05ArchitectureDiagram';
import RLTArchitectureDiagram, {
  DEFAULT_RLT_TRAINABLE_GROUPS,
} from './RLTArchitectureDiagram';
import TD3ArchitectureDiagram from './TD3ArchitectureDiagram';
import TrainingLossChart from './TrainingLossChart';

const POLL_INTERVAL_MS = 2000;
const DEFAULT_ACT_CRITIC_WARMUP_UPDATES = 5000;
const IMITATION_ACTION_CHUNK_SIZES = Object.freeze({
  act: 30,
  multi_task_dit: 16,
});
const DEFAULT_ALGORITHM_BY_POLICY = Object.freeze({
  act: 'td3',
  multi_task_dit: 'flow_sde_ppo',
  groot: 'rlt',
  pi05: 'rlt',
});

export const resolveTrainingPolicyModel = (taskInfo = {}) => {
  const serviceType = String(taskInfo.serviceType || '').trim().toLowerCase();
  const policyType = String(taskInfo.policyType || '').trim().toLowerCase();
  if (serviceType === 'groot') return 'groot';
  if (serviceType !== 'lerobot') return null;
  if (['act', 'multi_task_dit', 'pi05'].includes(policyType)) return policyType;
  return null;
};
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
const ACT_TD3_ALGORITHMS = new Set(['td3']);
const TD3_ACTOR_OBJECTIVES = new Set(['td3', 'td3_bc']);

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

const optionalFiniteNumber = (value) => {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') {
    return null;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const normalizeLossHistory = (history) => {
  const actor = [];
  const critic = [];
  (Array.isArray(history) ? history : []).forEach((point) => {
    const step = optionalFiniteNumber(point?.step);
    if (step === null) return;
    const actorLoss = optionalFiniteNumber(point?.actor_loss);
    const criticLoss = optionalFiniteNumber(point?.critic_loss);
    if (actorLoss !== null) actor.push({ step, loss: actorLoss });
    if (criticLoss !== null) critic.push({ step, loss: criticLoss });
  });
  return { actor, critic };
};

const appendLossSample = (series, point) => {
  if (!point || !Number.isFinite(point.step) || !Number.isFinite(point.loss)) return series;
  const next = [...series];
  const last = next[next.length - 1];
  if (last?.step === point.step) next[next.length - 1] = point;
  else next.push(point);
  return next.slice(-500);
};

const normalizeContractPath = (value) => {
  const normalized = String(value || '').trim();
  if (normalized === '/') return normalized;
  return normalized.replace(/\/+$/, '');
};

const actPolicyPathsEquivalent = (left, right) => {
  const selected = normalizeContractPath(left);
  const reported = normalizeContractPath(right);
  if (!selected || !reported) return false;
  return (
    selected === reported ||
    `${selected}/pretrained_model` === reported ||
    selected === `${reported}/pretrained_model`
  );
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
}) {
  return (
    <div
      className="flex h-full min-h-0 flex-col rounded-2xl border border-[#decfc3] bg-white p-4 shadow-[0_8px_24px_rgba(75,66,51,0.07)]"
      data-testid="diffusion-critic-warmup-card"
      role="region"
      aria-label="Diffusion Policy critic warm-up"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[#aa795f]">
            Value initialization
          </div>
          <div className="mt-0.5 text-[14px] font-semibold text-[#38342e]">
            Critic Warm-up
          </div>
          <div className="mt-1 text-[10px] text-[#8b8378]">
            Pretrain Diffusion state values before policy optimization
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-[#f5e9df] px-2.5 py-1 text-[9px] font-bold text-[#9b6245]">
          Critic
        </span>
      </div>

      <div className="mt-3 grid grid-cols-[0.8fr_auto_1.2fr] items-stretch gap-2">
        <div
          className="rounded-xl border border-[#ded9d1] bg-[#f4f2ee] p-3 text-[#777068]"
          aria-label="Diffusion policy: Frozen; no gradients"
        >
          <div className="text-[8px] font-bold uppercase tracking-[0.12em]">Actor</div>
          <div className="mt-1 text-[12px] font-semibold">Diffusion Policy</div>
          <output
            aria-label="Critic warm-up Diffusion policy mode"
            className="mt-1 inline-flex rounded-full bg-[#dedbd5] px-2 py-0.5 text-[8px] font-bold text-[#756e66]"
          >
            Frozen
          </output>
        </div>
        <div className="flex items-center text-[14px] font-semibold text-[#aaa196]" aria-hidden="true">
          →
        </div>
        <div className="rounded-xl border border-[#e1bca4] bg-[#fbede3] p-3 text-[#754832]">
          <div className="text-[8px] font-bold uppercase tracking-[0.12em] text-[#ad7251]">
            Value function
          </div>
          <div className="mt-1 text-[12px] font-semibold">Value Critic Network</div>
          <div className="mt-0.5 text-[8px] text-[#9a674d]">State value V(s) · offline targets</div>
          <span className="mt-1 inline-flex rounded-full bg-[#d9895f] px-2 py-0.5 text-[8px] font-bold text-white">
            Trainable
          </span>
        </div>
      </div>

      <div className="mt-3 flex min-h-0 flex-1 flex-col" data-testid="critic-warmup-settings">
          <div className="grid grid-cols-2 gap-2">
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
                className="mt-1 h-8 w-full rounded-lg border border-[#ddc9b9] bg-white px-2.5 text-[10px] font-semibold text-[#4a4038] outline-none transition focus:border-[#bd8564] focus:ring-2 focus:ring-[#ead7ca] disabled:cursor-not-allowed disabled:bg-[#eeeae4] disabled:text-[#999187]"
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
                className="mt-1 h-8 w-full rounded-lg border border-[#ddc9b9] bg-white px-2.5 text-[10px] font-semibold text-[#4a4038] outline-none transition focus:border-[#bd8564] focus:ring-2 focus:ring-[#ead7ca] disabled:cursor-not-allowed disabled:bg-[#eeeae4] disabled:text-[#999187]"
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
                className="mt-1 h-8 w-full rounded-lg border border-[#ddc9b9] bg-white px-2.5 text-[10px] font-semibold text-[#4a4038] outline-none transition focus:border-[#bd8564] focus:ring-2 focus:ring-[#ead7ca] disabled:cursor-not-allowed disabled:bg-[#eeeae4] disabled:text-[#999187]"
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
                className="mt-1 h-8 w-full rounded-lg border border-[#ddc9b9] bg-white px-2.5 text-[10px] font-semibold text-[#4a4038] outline-none transition focus:border-[#bd8564] focus:ring-2 focus:ring-[#ead7ca] disabled:cursor-not-allowed disabled:bg-[#eeeae4] disabled:text-[#999187]"
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
              {formatCount(statusValue(status, 'total_steps', 'steps'))} · Critic loss{' '}
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

          <div
            className="mt-3 rounded-xl border border-[#ebe3da] bg-[#faf8f4] px-3 py-2 text-[9px]"
            aria-label="Diffusion critic warm-up checkpoint"
          >
            <div className="font-semibold text-[#645b52]">Critic bundle</div>
            <output
              aria-label="Critic warm-up bundle path"
              title={bundlePath || 'Created after critic warm-up completes'}
              className="mt-1 block truncate font-mono text-[8px] text-[#998f85]"
            >
              {bundlePath || 'Bundle path · pending'}
            </output>
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
  td3ActorObjective,
  onTD3ActorObjectiveChange,
  flowSdePpoReady,
  flowInferenceBlockedReason,
  flowTaskInstruction,
  ppoResumeReady,
  compatibleWarmupReady,
  warmupSteps,
  setWarmupSteps,
  warmupBatchSize,
  setWarmupBatchSize,
  warmupValueLearningRate,
  setWarmupValueLearningRate,
  warmupDiscount,
  setWarmupDiscount,
  actorTrainableGroups,
  setActorTrainableGroups,
  rltTrainableGroups,
  setRltTrainableGroups,
  browserDisabled,
  selectionDisabled,
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
  imitationActionChunkSize,
  setImitationActionChunkSize,
  criticWarmupUpdates,
  setCriticWarmupUpdates,
  statusLabel,
  displayProgress,
  jobStatus,
  modelPath,
  currentPolicyEpoch,
  datasetSelections,
  trainingReplayDatasets,
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
  onCompactLayoutChange,
}) {
  const isReinforcementLearning = trainingMethod === 'reinforcement';
  const isImitationLearning = trainingMethod === 'imitation';
  const isCriticWarmup = trainingMethod === 'critic';
  const isActSelected = selectedPolicyModel === 'act';
  const isMultiTaskDiTSelected = selectedPolicyModel === 'multi_task_dit';
  const isRltPolicySelected = ['groot', 'pi05'].includes(selectedPolicyModel);
  const isDiffusionCriticWarmup = isCriticWarmup && isMultiTaskDiTSelected;
  const criticModelUnsupported = (
    isCriticWarmup && !isActSelected && !isMultiTaskDiTSelected
  );
  const isFlowSdePpo = (
    isReinforcementLearning &&
    isMultiTaskDiTSelected &&
    algorithm === 'flow_sde_ppo'
  );
  const td3Available = isReinforcementLearning && isActSelected && !browserDisabled;
  const isActTD3 = (
    isReinforcementLearning &&
    isActSelected &&
    ACT_TD3_ALGORITHMS.has(algorithm)
  );
  const isActTrainingLoop = (
    isActSelected &&
    (isImitationLearning || isCriticWarmup || isActTD3)
  );
  const [sampledLossHistory, setSampledLossHistory] = useState({
    jobId: '',
    actor: [],
    critic: [],
  });
  const persistedLossHistory = useMemo(
    () => normalizeLossHistory(jobStatus?.loss_history),
    [jobStatus?.loss_history]
  );
  useEffect(() => {
    if (!isActTD3 || persistedLossHistory.actor.length || persistedLossHistory.critic.length) {
      return;
    }
    const jobId = String(jobStatus?.job_id || 'pending');
    const stepCandidates = [
      jobStatus?.completed_critic_updates,
      jobStatus?.completed_epochs,
      jobStatus?.percentage,
    ].map(optionalFiniteNumber);
    const step = stepCandidates.find((value) => value !== null);
    if (step === undefined) return;
    const actorLoss = optionalFiniteNumber(jobStatus?.actor_loss);
    const criticLoss = optionalFiniteNumber(jobStatus?.critic_loss);
    if (actorLoss === null && criticLoss === null) return;

    setSampledLossHistory((current) => {
      const base = current.jobId === jobId
        ? current
        : { jobId, actor: [], critic: [] };
      return {
        jobId,
        actor: appendLossSample(
          base.actor,
          actorLoss !== null ? { step, loss: actorLoss } : null
        ),
        critic: appendLossSample(
          base.critic,
          criticLoss !== null ? { step, loss: criticLoss } : null
        ),
      };
    });
  }, [
    isActTD3,
    jobStatus?.actor_loss,
    jobStatus?.completed_critic_updates,
    jobStatus?.completed_epochs,
    jobStatus?.critic_loss,
    jobStatus?.job_id,
    jobStatus?.percentage,
    persistedLossHistory.actor.length,
    persistedLossHistory.critic.length,
  ]);
  const actorLossHistory = persistedLossHistory.actor.length
    ? persistedLossHistory.actor
    : sampledLossHistory.actor;
  const criticLossHistory = persistedLossHistory.critic.length
    ? persistedLossHistory.critic
    : sampledLossHistory.critic;
  const isPureTD3 = isActTD3 && td3ActorObjective === 'td3';
  const flowSdePpoAvailable = (
    isReinforcementLearning &&
    isMultiTaskDiTSelected &&
    !browserDisabled
  );
  const rltAvailable = (
    isReinforcementLearning &&
    isRltPolicySelected &&
    !browserDisabled
  );
  const isRltLayout = (
    isReinforcementLearning &&
    isRltPolicySelected &&
    algorithm === 'rlt'
  );
  const isRlt = rltAvailable && algorithm === 'rlt';
  const isCompactWorkflowLayout = (
    isRltLayout ||
    isActTrainingLoop ||
    isImitationLearning ||
    (
      isMultiTaskDiTSelected &&
      (isDiffusionCriticWarmup || isFlowSdePpo)
    )
  );

  useEffect(() => {
    onCompactLayoutChange?.(isCompactWorkflowLayout);
    return () => onCompactLayoutChange?.(false);
  }, [isCompactWorkflowLayout, onCompactLayoutChange]);
  const displayedImitationActionChunkSize = isMultiTaskDiTSelected
    ? IMITATION_ACTION_CHUNK_SIZES.multi_task_dit
    : imitationActionChunkSize;
  const imitationPolicyName = {
    act: 'ACT',
    multi_task_dit: 'Diffusion Transformer',
    groot: 'GR00T',
    pi05: 'Pi0.5',
  }[selectedPolicyModel] || selectedPolicyModel;
  const imitationCardPresentation = {
    multi_task_dit: {
      title: 'Diffusion Transformer imitation learning',
      description: 'Supervised flow-matching · no reward or outcome labels required',
      objectiveEyebrow: 'Diffusion objective',
      objectiveTitle: 'Flow-Matching Reconstruction',
      objectiveDetail: 'Noise-conditioned velocity regression over demonstrated action chunks',
      actionChunkDisabled: true,
      actionChunkTitle: 'Diffusion Transformer horizon is fixed by its model contract',
    },
    groot: {
      title: 'GR00T imitation learning',
      description: 'Supervised GR00T action-chunk fine-tuning preview',
      objectiveEyebrow: 'GR00T objective',
      objectiveTitle: 'Flow-Matching Action Reconstruction',
      objectiveDetail: 'Supervised action-flow matching over demonstrated action chunks',
      actionChunkDisabled: false,
      actionChunkTitle: 'GR00T action horizon preview',
    },
    pi05: {
      title: 'Pi0.5 imitation learning',
      description: 'Supervised Pi0.5 action-chunk fine-tuning preview',
      objectiveEyebrow: 'Pi0.5 objective',
      objectiveTitle: 'Flow-Matching Action Reconstruction',
      objectiveDetail: 'Supervised action-flow matching over demonstrated action chunks',
      actionChunkDisabled: false,
      actionChunkTitle: 'Pi0.5 action horizon preview',
    },
  }[selectedPolicyModel] || {
    title: `${imitationPolicyName} imitation learning`,
    description: `Fit ${imitationPolicyName} action chunks to recorded demonstrations`,
    objectiveEyebrow: `${imitationPolicyName} objective`,
    objectiveTitle: 'Action Chunk Reconstruction',
    objectiveDetail: 'Supervised reconstruction of demonstrated action chunks',
    actionChunkDisabled: false,
    actionChunkTitle: '',
  };
  const isSupportedPolicy = isActSelected || isMultiTaskDiTSelected;
  const parsedRoundIndex = Number(jobStatus?.round_index);
  const targetPolicyEpoch = (
    isReinforcementLearning &&
    isActSelected &&
    ACT_TD3_ALGORITHMS.has(algorithm) &&
    Number.isInteger(parsedRoundIndex) &&
    parsedRoundIndex >= 1
  ) ? parsedRoundIndex : Number(currentPolicyEpoch) + 1;
  const selectedPolicyLabel = {
    act: 'ACT',
    multi_task_dit: 'Diffusion Transformer',
    groot: 'GR00T',
    pi05: 'Pi0.5',
  }[selectedPolicyModel] || selectedPolicyModel;
  const sharedLoopReplayStep = isImitationLearning
    ? 'Replay Buffer → IL'
    : (isCriticWarmup
      ? 'Replay Buffer → Critic'
      : (isRlt
        ? 'Replay Buffer → RLT'
        : (isFlowSdePpo ? 'Rollout Buffer → PPO' : 'Replay Buffer → Training')));
  const sharedLoopRegionLabel = `${selectedPolicyLabel} ${
    isImitationLearning
      ? 'imitation'
      : (isCriticWarmup ? 'critic warm-up' : (isRlt ? 'RLT' : 'reinforcement'))
  } training loop`;
  const workflowStartDisabled = (
    startDisabled ||
    !isSupportedPolicy ||
    criticModelUnsupported ||
    (isFlowSdePpo && !flowSdePpoReady)
  );
  const invalidDatasetVersion = datasetSelections.find(
    (selection) => selection.version && selection.version !== 'v3.0'
  )?.version;
  const reportedCriticCheckpointPath = String(
    statusValue(jobStatus, 'checkpoint_path') || ''
  ).trim();
  const reportedCriticActCheckpoint = String(jobStatus?.act_checkpoint || '').trim();
  const criticCheckpointMatchesPolicy = Boolean(
    reportedCriticCheckpointPath &&
    actPolicyPathsEquivalent(actCheckpoint, reportedCriticActCheckpoint)
  );
  const criticCheckpointPath = criticCheckpointMatchesPolicy
    ? reportedCriticCheckpointPath
    : '';
  const criticCheckpointGuidance = reportedCriticCheckpointPath && !criticCheckpointMatchesPolicy
    ? 'Saved critic belongs to a different or unverified ACT policy'
    : (actCheckpoint
      ? 'Resolved by backend under selected ACT policy/critic/latest.pt'
      : 'Select an ACT policy');
  const configuredCriticWarmupUpdates = Number(criticWarmupUpdates);
  const criticWarmupTotalUpdates = statusValue(
    jobStatus,
    'total_critic_updates'
  ) ?? (
    Number.isInteger(configuredCriticWarmupUpdates) && configuredCriticWarmupUpdates > 0
      ? configuredCriticWarmupUpdates
      : DEFAULT_ACT_CRITIC_WARMUP_UPDATES
  );
  const handleWorkflowStart = () => {
    if (
      !isSupportedPolicy ||
      criticModelUnsupported ||
      (isFlowSdePpo && !flowSdePpoReady)
    ) return;
    handleStart();
  };

  const progressMetrics = isImitationLearning ? (isMultiTaskDiTSelected ? [
    {
      label: 'Flow loss',
      value: formatLoss(statusValue(jobStatus, 'loss', 'flow_loss')),
      tone: 'actor',
    },
    {
      label: 'Step',
      value: formatCount(statusValue(jobStatus, 'step', 'completed_steps')),
      tone: 'neutral',
    },
    {
      label: 'Policy',
      value: modelPath ? 'Ready' : '—',
      tone: 'neutral',
    },
  ] : [
    {
      label: 'Total loss',
      value: formatLoss(statusValue(jobStatus, 'loss', 'total_loss')),
      tone: 'critic',
    },
    {
      label: 'L1 loss',
      value: formatLoss(statusValue(jobStatus, 'l1_loss')),
      tone: 'actor',
    },
    {
      label: 'KLD loss',
      value: formatLoss(statusValue(jobStatus, 'kld_loss')),
      tone: 'neutral',
    },
  ]) : isCriticWarmup ? (isDiffusionCriticWarmup ? [
    {
      label: 'Critic loss',
      value: formatLoss(statusValue(jobStatus, 'value_loss', 'loss')),
      tone: 'critic',
    },
    {
      label: 'Step',
      value: formatCount(statusValue(jobStatus, 'step', 'completed_steps')),
      tone: 'neutral',
    },
    { label: 'Policy', value: 'Frozen', tone: 'neutral' },
  ] : [
    {
      label: 'Critic loss',
      value: formatLoss(statusValue(jobStatus, 'critic_loss')),
      tone: 'critic',
    },
    {
      label: 'Target mean',
      value: formatLoss(statusValue(jobStatus, 'target_mean')),
      tone: 'neutral',
    },
    {
      label: 'Actor',
      value: statusValue(jobStatus, 'actor_exactly_unchanged') === false
        ? 'Changed'
        : (statusValue(jobStatus, 'actor_exactly_unchanged') === true
          ? 'Unchanged'
          : 'Frozen'),
      tone: 'actor',
    },
  ]) : isRlt ? [
    {
      label: 'Critic loss',
      value: formatLoss(statusValue(jobStatus, 'critic_loss')),
      tone: 'critic',
    },
    {
      label: 'Action MLP loss',
      value: formatLoss(statusValue(jobStatus, 'actor_loss')),
      tone: 'actor',
    },
  ] : isMultiTaskDiTSelected ? [
    {
      label: 'Actor loss',
      value: formatLoss(statusValue(jobStatus, 'actor_loss', 'policy_loss')),
      tone: 'actor',
    },
    {
      label: 'Critic loss',
      value: formatLoss(statusValue(jobStatus, 'value_loss')),
      tone: 'critic',
    },
    {
      label: 'Approx. KL',
      value: formatLoss(statusValue(jobStatus, 'approx_kl', 'kl')),
      tone: 'neutral',
    },
  ] : [
    {
      label: 'Critic loss',
      value: formatLoss(statusValue(jobStatus, 'critic_loss')),
      tone: 'critic',
    },
    {
      label: 'Actor loss',
      value: formatLoss(statusValue(jobStatus, 'actor_loss')),
      tone: 'actor',
    },
    {
      label: 'Policy',
      value: modelPath ? 'Ready' : '—',
      tone: 'neutral',
    },
  ];
  const progressDetailLabel = isActTD3
    ? `Critic replay ${formatCount(statusValue(jobStatus, 'completed_epochs'))}/${formatCount(statusValue(jobStatus, 'total_epochs'))}`
    : (isImitationLearning
      ? `Step ${formatCount(statusValue(jobStatus, 'step', 'completed_steps'))}/${formatCount(statusValue(jobStatus, 'total_steps'))}`
      : (isCriticWarmup
        ? (isDiffusionCriticWarmup
          ? `Step ${formatCount(statusValue(jobStatus, 'step', 'completed_steps'))}/${formatCount(statusValue(jobStatus, 'total_steps', 'steps'))}`
          : `Update ${formatCount(statusValue(jobStatus, 'completed_critic_updates'))}/${formatCount(criticWarmupTotalUpdates)}`)
        : ''));
  const progressAriaLabel = isActTD3
    ? 'Training loss progress'
    : (isImitationLearning
      ? 'Imitation Learning training progress'
      : (isCriticWarmup
        ? `${isDiffusionCriticWarmup ? 'Diffusion' : 'ACT'} critic warm-up progress`
        : (isRlt
          ? 'RLT training progress'
          : (isMultiTaskDiTSelected
            ? 'Flow-SDE PPO training progress'
            : 'Offline RL training progress'))));

  const renderPolicyDiagram = () => {
    if (selectedPolicyModel === 'groot') {
      return <GrootArchitectureDiagram mode={isRlt ? 'rlt' : 'finetune'} />;
    }
    if (selectedPolicyModel === 'multi_task_dit') {
      return <MultiTaskDiTArchitectureDiagram criticOnly={isDiffusionCriticWarmup} />;
    }
    if (selectedPolicyModel === 'pi05') {
      return <PI05ArchitectureDiagram disabled={browserDisabled} />;
    }
    return (
      <ACTArchitectureDiagram
        trainableGroups={isCriticWarmup
          ? []
          : (isPureTD3
            ? actorTrainableGroups.filter((group) => group !== 'cvae_encoder')
            : actorTrainableGroups)}
        onChange={(groups) => setActorTrainableGroups(
          isPureTD3
            ? groups.filter((group) => group !== 'cvae_encoder')
            : groups
        )}
        disabled={browserDisabled || isCriticWarmup}
      />
    );
  };

  return (
    <div
      className={clsx(
        'mt-3 grid min-h-0 min-w-0 overflow-hidden',
        isCompactWorkflowLayout
          ? 'flex-none grid-rows-[auto_auto_auto]'
          : 'flex-1 grid-rows-[auto_minmax(0,1fr)_auto]'
      )}
      data-testid="offline-rl-workflow-training"
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <WorkflowChoiceGroup label="Policy model">
          <button
            type="button"
            aria-pressed={selectedPolicyModel === 'act'}
            disabled={selectionDisabled}
            onClick={() => onPolicyModelChange('act')}
            className={selectedPolicyModel === 'act' ? activeChoiceClass : inactiveChoiceClass}
          >
            ACT
          </button>
          <button
            type="button"
            disabled={selectionDisabled}
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
            disabled={selectionDisabled}
            aria-pressed={selectedPolicyModel === 'groot'}
            onClick={() => onPolicyModelChange('groot')}
            className={selectedPolicyModel === 'groot'
              ? activeChoiceClass
              : inactiveChoiceClass}
          >
            GR00T
          </button>
          <button
            type="button"
            disabled={selectionDisabled}
            aria-pressed={selectedPolicyModel === 'pi05'}
            onClick={() => onPolicyModelChange('pi05')}
            className={selectedPolicyModel === 'pi05'
              ? activeChoiceClass
              : inactiveChoiceClass}
          >
            Pi0.5
          </button>
        </WorkflowChoiceGroup>

        <div className="flex flex-wrap items-start justify-end gap-2">
          <WorkflowChoiceGroup label="Training method">
            <button
              type="button"
              aria-label="Imitation Learning"
              aria-pressed={isImitationLearning}
              disabled={selectionDisabled}
              onClick={() => onTrainingMethodChange('imitation')}
              className={isImitationLearning ? activeChoiceClass : inactiveChoiceClass}
            >
              IL
            </button>
            <button
              type="button"
              aria-label="Critic Warm-up"
              aria-pressed={isCriticWarmup}
              disabled={selectionDisabled || (
                !isActSelected && !isMultiTaskDiTSelected && !isCriticWarmup
              )}
              onClick={() => onTrainingMethodChange('critic')}
              className={isCriticWarmup ? activeChoiceClass : inactiveChoiceClass}
              title={isActSelected
                ? 'Warm up ACT-TD3 critics with the ACT actor frozen'
                : (isMultiTaskDiTSelected
                  ? 'Warm up the Flow-SDE value critic with the Diffusion policy frozen'
                  : 'Select ACT or Diffusion Transformer to run critic warm-up')}
            >
              Critic
            </button>
            <button
              type="button"
              aria-label="Reinforcement Learning"
              aria-pressed={isReinforcementLearning}
              disabled={selectionDisabled}
              onClick={() => onTrainingMethodChange('reinforcement')}
              className={isReinforcementLearning ? activeChoiceClass : inactiveChoiceClass}
            >
              RL
            </button>
          </WorkflowChoiceGroup>

          <div className="flex flex-col gap-2">
            <WorkflowChoiceGroup label="RL algorithm">
              <button
                type="button"
                disabled={!td3Available}
                aria-pressed={isReinforcementLearning && algorithm === 'td3'}
                onClick={() => onAlgorithmChange('td3')}
                className={isReinforcementLearning && algorithm === 'td3'
                  ? activeChoiceClass
                  : (!td3Available ? disabledChoiceClass : inactiveChoiceClass)}
              >
                TD3
              </button>
              <button
                type="button"
                disabled={!flowSdePpoAvailable}
                aria-pressed={isReinforcementLearning && algorithm === 'flow_sde_ppo'}
                onClick={() => onAlgorithmChange('flow_sde_ppo')}
                className={isReinforcementLearning && algorithm === 'flow_sde_ppo'
                  ? activeChoiceClass
                  : (!flowSdePpoAvailable ? disabledChoiceClass : inactiveChoiceClass)}
                title="PPO over Flow-SDE action-chunk trajectories"
              >
                PPO
                <span className="ml-1 rounded-full bg-white/25 px-1.5 py-0.5 text-[8px]">
                  Flow-SDE
                </span>
              </button>
              <button
                type="button"
                disabled={!rltAvailable}
                aria-pressed={isReinforcementLearning && algorithm === 'rlt'}
                onClick={() => onAlgorithmChange('rlt')}
                className={isReinforcementLearning && algorithm === 'rlt'
                  ? activeChoiceClass
                  : (!rltAvailable ? disabledChoiceClass : inactiveChoiceClass)}
                title="RL Token Transformer with a lightweight Action MLP"
              >
                RLT
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
      </div>

      <div
        className="mt-2 grid min-h-0 flex-1 items-stretch gap-2 overflow-y-auto overscroll-contain pr-1"
        data-testid="offline-rl-training-architecture"
      >
        {isActTrainingLoop ? (
          <ACTTD3TrainingLoop
            mode={trainingMethod}
            trainableGroups={isCriticWarmup
              ? []
              : (isPureTD3
                ? actorTrainableGroups.filter((group) => group !== 'cvae_encoder')
                : actorTrainableGroups)}
            onTrainableGroupsChange={(groups) => {
              if (isCriticWarmup) return;
              setActorTrainableGroups(
                isPureTD3
                  ? groups.filter((group) => group !== 'cvae_encoder')
                  : groups
              );
            }}
            lockedGroups={isPureTD3 ? ['cvae_encoder'] : []}
            datasets={trainingReplayDatasets}
            actorObjective={td3ActorObjective}
            onActorObjectiveChange={onTD3ActorObjectiveChange}
            criticEpochs={criticEpochs}
            onCriticEpochsChange={setCriticEpochs}
            actorEpochs={actorEquivalentEpochs}
            onActorEpochsChange={setActorEquivalentEpochs}
            batchSize={batchSize}
            onBatchSizeChange={setBatchSize}
            imitationSteps={imitationSteps}
            onImitationStepsChange={setImitationSteps}
            imitationBatchSize={imitationBatchSize}
            onImitationBatchSizeChange={setImitationBatchSize}
            imitationSaveFreq={imitationSaveFreq}
            onImitationSaveFreqChange={setImitationSaveFreq}
            imitationActionChunkSize={displayedImitationActionChunkSize}
            onImitationActionChunkSizeChange={setImitationActionChunkSize}
            criticWarmupBatchSize={batchSize}
            onCriticWarmupBatchSizeChange={setBatchSize}
            criticWarmupUpdates={criticWarmupUpdates}
            onCriticWarmupUpdatesChange={setCriticWarmupUpdates}
            criticCheckpointPath={criticCheckpointPath}
            criticGuidance={criticCheckpointGuidance}
            policyDisabled={browserDisabled || isCriticWarmup}
            disabled={browserDisabled}
            fitContent={isCompactWorkflowLayout}
            updated={Boolean(
              !isCriticWarmup &&
              modelPath &&
              COMPLETE_STATUSES.has(String(jobStatus?.status || '').toLowerCase())
            )}
          />
        ) : (
          <PolicyTrainingLoopLayout
            regionLabel={sharedLoopRegionLabel}
            testId="policy-training-loop"
            trainingMode={trainingMethod}
            policyModel={selectedPolicyModel}
            policyLabel={selectedPolicyLabel}
            policyTestId="training-policy-stage"
            policyNode={renderPolicyDiagram()}
            datasets={trainingReplayDatasets}
            trainingNode={isImitationLearning ? (
              <ImitationLearningCard
                policyLabel={imitationPolicyName}
                title={imitationCardPresentation.title}
                description={imitationCardPresentation.description}
                objectiveEyebrow={imitationCardPresentation.objectiveEyebrow}
                objectiveTitle={imitationCardPresentation.objectiveTitle}
                objectiveDetail={imitationCardPresentation.objectiveDetail}
                titleId={`${selectedPolicyModel}-imitation-algorithm-title`}
                testId={`${selectedPolicyModel}-imitation-algorithm-card`}
                steps={imitationSteps}
                onStepsChange={setImitationSteps}
                batchSize={imitationBatchSize}
                onBatchSizeChange={setImitationBatchSize}
                saveFreq={imitationSaveFreq}
                onSaveFreqChange={setImitationSaveFreq}
                actionChunkSize={displayedImitationActionChunkSize}
                onActionChunkSizeChange={setImitationActionChunkSize}
                actionChunkDisabled={imitationCardPresentation.actionChunkDisabled}
                actionChunkTitle={imitationCardPresentation.actionChunkTitle}
                disabled={browserDisabled}
              />
            ) : (
              <div
                className={clsx(
                  'flex min-h-0 flex-col rounded-2xl border border-[#decfc3] bg-white p-4 shadow-[0_8px_24px_rgba(75,66,51,0.07)]',
                  isCompactWorkflowLayout ? 'h-fit' : 'h-full'
                )}
                data-testid="training-algorithm-card"
              >
          {isDiffusionCriticWarmup ? (
            <CriticWarmupPanel
              controlsDisabled={browserDisabled}
              steps={warmupSteps}
              setSteps={setWarmupSteps}
              batchSize={warmupBatchSize}
              setBatchSize={setWarmupBatchSize}
              valueLearningRate={warmupValueLearningRate}
              setValueLearningRate={setWarmupValueLearningRate}
              discount={warmupDiscount}
              setDiscount={setWarmupDiscount}
              statusReady={statusReady}
              statusLabel={statusLabel}
              progress={displayProgress}
              status={jobStatus}
              bundlePath={String(statusValue(jobStatus, 'bundle_path') || '').trim()}
              integrationReady={Boolean(
                COMPLETE_STATUSES.has(String(jobStatus?.status || '').toLowerCase()) &&
                String(statusValue(jobStatus, 'bundle_path') || '').trim()
              )}
              integrationMessage="The completed bundle is saved with the frozen Diffusion policy and value critic."
              sourceKind="Warm-up"
              sourceLabel={shortWarmupSource(
                jobStatus,
                String(statusValue(jobStatus, 'bundle_path') || '').trim()
              )}
              sourceReadyLabel="Ready for online PPO"
            />
          ) : isCriticWarmup ? (
            <div className="flex h-full items-center justify-center rounded-2xl border border-dashed border-[#d9d2c5] bg-[#f8f5ef] px-5 text-center">
              <div>
                <div className="text-[13px] font-semibold text-[#655e54]">
                  No compatible critic workflow
                </div>
                <div className="mt-1 text-[10px] text-[#8d8579]">
                  Select ACT or Diffusion Transformer.
                </div>
              </div>
            </div>
          ) : isRlt ? (
            <RLTArchitectureDiagram
              policyLabel={selectedPolicyLabel}
              trainableGroups={rltTrainableGroups}
              onChange={setRltTrainableGroups}
              disabled={browserDisabled}
            />
          ) : isFlowSdePpo ? (
            <FlowSDEPPOArchitectureDiagram backendReady={flowSdePpoReady} />
          ) : isActSelected ? (
            <>
              <TD3ArchitectureDiagram actorObjective={td3ActorObjective} />

              <div
                className="mt-2 rounded-lg border border-[#d9d2c5] bg-white px-2.5 py-2 text-[9px] text-[#6f685d]"
                aria-label="ACT TD3 actor objective"
              >
                <span className="font-semibold text-[#514b42]">
                  {isPureTD3 ? 'Pure TD3 actor' : 'TD3+BC actor'}
                </span>
                <span className="float-right font-mono font-semibold text-[#5f7664]">
                  {isPureTD3 ? '-Q1' : '-Q1 + CVAE/BC'}
                </span>
                <div className="mt-1 clear-both text-[8px] text-[#8d8579]">
                  {isPureTD3
                    ? 'All replay rows train the critics; CVAE encoder is frozen.'
                    : 'Critics use all rows; CVAE and deterministic BC use success rows only.'}
                </div>
              </div>

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
            )}
            replayStep={sharedLoopReplayStep}
            returnStep="Policy update → next cycle"
            connectorTestId="policy-training-loop-connectors"
            fitContent={isCompactWorkflowLayout}
            updated={Boolean(
              modelPath &&
              COMPLETE_STATUSES.has(String(jobStatus?.status || '').toLowerCase())
            )}
          />
        )}
      </div>

      <div
        className={clsx(
          'grid shrink-0 items-stretch gap-2 border-t border-[#e2dcd1] pt-3 xl:grid-cols-[minmax(0,1fr)_220px]',
          isCompactWorkflowLayout ? 'mt-3' : 'mt-auto'
        )}
        data-testid="offline-rl-training-footer"
      >
        <div
          className="rounded-xl border border-[#e2dcd1] bg-[#f8f5ef] p-2.5"
          data-testid="offline-rl-training-progress-card"
        >
          <div className="flex items-center justify-between gap-2 text-[10px]">
            <span className="flex min-w-0 items-center gap-2 font-semibold text-[#514b42]">
              <span>{isCriticWarmup ? 'Critic warm-up progress' : 'Training progress'}</span>
              {isActTD3 && (
                <span
                  className="shrink-0 rounded-md border border-[#cfd8cd] bg-[#e8eee6] px-1.5 py-0.5 font-mono text-[9px] font-bold text-[#58705d]"
                  aria-label={`ACT-TD3 policy RL Epoch ${currentPolicyEpoch} to ${targetPolicyEpoch}`}
                >
                  RL Epoch {formatPolicyEpoch(currentPolicyEpoch)} → {formatPolicyEpoch(targetPolicyEpoch)}
                </span>
              )}
            </span>
          </div>
          <div className="mt-2">
            <TrainingLossChart
              actorLossHistory={actorLossHistory}
              criticLossHistory={criticLossHistory}
              metrics={isActTD3 ? null : progressMetrics}
              percentage={displayProgress}
              status={jobStatus?.status || 'idle'}
              displayStatus={statusLabel}
              etaSeconds={Number(statusValue(jobStatus, 'eta_seconds'))}
              showEta
              detailLabel={progressDetailLabel}
              progressLabel={progressAriaLabel}
              expandable={isReinforcementLearning}
              rlMetricHistory={jobStatus?.rl_metric_history}
            />
          </div>
        </div>

        <div className="flex flex-col justify-between rounded-xl border border-[#e2dcd1] bg-[#f8f5ef] p-2.5">
          <div>
            <div className="flex items-center gap-1.5 text-[10px] font-semibold text-[#575147]">
              <MdDataObject size={13} /> Training action
            </div>
            {isCriticWarmup && !isActSelected && !isMultiTaskDiTSelected ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {selectedPolicyLabel} critic warm-up is not connected. Select ACT or Diffusion
                {' '}Transformer; the selected model remains available for inspection.
              </p>
            ) : isImitationLearning && ['groot', 'pi05'].includes(selectedPolicyModel) ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {selectedPolicyLabel} imitation-learning preview is available, but its training backend is not
                {' '}connected yet. Start Training remains disabled.
              </p>
            ) : isRlt ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {selectedPolicyModel === 'pi05'
                  ? 'Pi0.5 + RLT architecture is available, but a Pi0.5-compatible RLT checkpoint is required. The current showroom RLT artifact is GR00T-only.'
                  : 'GR00T + RLT architecture is configured for shadow verification.'}
                {' '}The training backend is not connected yet, so Start Training remains disabled.
              </p>
            ) : !isSupportedPolicy ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {selectedPolicyLabel} diagram preview only. Offline RL training backend is not connected.
                {' '}Training is available for ACT and Diffusion Transformer.
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
                  : (isCriticWarmup
                    ? `${isDiffusionCriticWarmup ? 'Diffusion' : 'ACT'} critic warm-up`
                    : 'TD3')} requires LeRobot v3.0.
                {' '}The selected {invalidDatasetVersion} dataset is view only.
              </p>
            ) : isActSelected && (isReinforcementLearning || isImitationLearning) && trainabilityError ? (
              <p className="mt-1 text-[9px] leading-relaxed text-[#a06458]" role="alert">
                {trainabilityError}
              </p>
            ) : (
              <p className="mt-1 text-[9px] leading-relaxed text-[#948c80]">
                {isImitationLearning
                  ? (datasetSelections.length
                    ? `${imitationPolicyName} imitation learning ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · ${imitationSteps} steps · batch ${imitationBatchSize} · ${displayedImitationActionChunkSize}-step chunk${isActSelected ? ` · ${actorTrainableGroups.length} trainable blocks` : ''} · no reward or Success/Fail labels required`
                    : `Include at least one LeRobot v3.0 Data Epoch in Step 3. No base ${imitationPolicyName} checkpoint, reward, or Success/Fail label is required.`)
                  : isCriticWarmup
                    ? (isDiffusionCriticWarmup
                      ? (datasetSelections.length && actCheckpoint && flowTaskInstruction
                        ? `Diffusion value critic ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · ${warmupSteps} steps · batch ${warmupBatchSize} · policy frozen · Success + Fail required`
                        : 'Include LeRobot v3.0 Success + Fail replay, select a MultiTaskDiT policy, and enter a task instruction.')
                      : (datasetSelections.length && actCheckpoint
                        ? `ACT critic warm-up ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · batch ${batchSize} · ACT actor frozen · Success + Fail required`
                        : 'Include at least one LeRobot v3.0 Data Epoch in Step 3 and select an ACT policy. The critic is saved under that policy.'))
                  : isMultiTaskDiTSelected
                    ? (actCheckpoint && robotType
                      ? `Diffusion Transformer + Flow-SDE PPO ready · ${ppoResumeReady
                        ? 'continuing the compatible PPO critic'
                        : (compatibleWarmupReady
                          ? 'compatible offline critic bundle attached automatically'
                          : 'fresh value critic initialization')} · frozen observation encoder`
                      : 'Select a MultiTaskDiT model in Workspace Paths and a robot type on Home. No LeRobot dataset is required.')
                    : (datasetSelections.length && actCheckpoint
                      ? `ACT-${isPureTD3 ? 'TD3' : 'TD3+BC'} ready · ${datasetSelections.length} Data Epoch${datasetSelections.length === 1 ? '' : 's'} · batch ${batchSize} · ${isPureTD3 ? actorTrainableGroups.filter((group) => group !== 'cvae_encoder').length : actorTrainableGroups.length} trainable blocks`
                      : 'Include at least one LeRobot v3.0 Data Epoch in Step 3 and select an ACT model in Workspace Paths.')}
                {!isImitationLearning && !isDiffusionCriticWarmup && (
                  <> Robot: {robotType || 'Not selected'}</>
                )}
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
              {!statusReady
                ? 'Checking…'
                : (isRunning
                  ? (isCriticWarmup ? 'Warming Critic…' : 'Training…')
                  : (isCriticWarmup ? 'Start Critic Warm-up' : 'Start Training'))}
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
              {isStopping
                ? 'Stopping…'
                : (isCriticWarmup ? 'Stop Critic Warm-up' : 'Stop Training')}
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
  onTrainingMethodStateChange,
  currentPolicyEpoch = 0,
  forceFreshLineage = false,
  onFreshLineageConsumed,
  flowSdePpoReady = false,
  getFlowSDEPPOStatus,
  onStartFlowSDEPPO,
  onStopFlowSDEPPO,
  onSubmitFlowSDEPPOOutcome,
  onCompactLayoutChange,
  variant = 'default',
}) {
  const dispatch = useDispatch();
  const robotType = useSelector((state) => state.tasks.robotType);
  const inferenceTaskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const datasetPath = useSelector(selectOfflineRLDatasetPath);
  const datasetSelections = useSelector(selectOfflineRLDatasetSelections, shallowEqual);
  const parentCheckpoint = useSelector(selectOfflineRLCheckpointPath);
  const actCheckpoint = inferenceTaskInfo.policyPath || '';
  const inferencePolicyModel = resolveTrainingPolicyModel(inferenceTaskInfo);
  const inferenceModelKey = [
    String(inferenceTaskInfo.serviceType || '').trim(),
    String(inferenceTaskInfo.policyType || '').trim(),
  ].join(':');
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus?.status || 'idle'
  );
  const [trainingMethod, setTrainingMethod] = useState('reinforcement');
  const [selectedPolicyModel, setSelectedPolicyModel] = useState(
    () => inferencePolicyModel || 'act'
  );
  const [algorithm, setAlgorithm] = useState(
    () => DEFAULT_ALGORITHM_BY_POLICY[inferencePolicyModel || 'act'] || ''
  );
  const [td3ActorObjective, setTD3ActorObjective] = useState('td3_bc');
  const [actorTrainableGroups, setActorTrainableGroups] = useState(
    DEFAULT_ACT_TRAINABLE_GROUPS
  );
  const [rltTrainableGroups, setRltTrainableGroups] = useState(
    DEFAULT_RLT_TRAINABLE_GROUPS
  );
  const [criticEpochs, setCriticEpochs] = useState('10');
  const [actorEquivalentEpochs, setActorEquivalentEpochs] = useState('5');
  const [batchSize, setBatchSize] = useState('4');
  const [imitationSteps, setImitationSteps] = useState('80000');
  const [imitationBatchSize, setImitationBatchSize] = useState('8');
  const [imitationSaveFreq, setImitationSaveFreq] = useState('10000');
  const [imitationActionChunkSize, setImitationActionChunkSize] = useState(
    String(IMITATION_ACTION_CHUNK_SIZES.act)
  );
  const [criticWarmupUpdates, setCriticWarmupUpdates] = useState(
    String(DEFAULT_ACT_CRITIC_WARMUP_UPDATES)
  );
  const [warmupSteps, setWarmupSteps] = useState('2000');
  const [warmupBatchSize, setWarmupBatchSize] = useState('8');
  const [warmupValueLearningRate, setWarmupValueLearningRate] = useState('0.0001');
  const [warmupDiscount, setWarmupDiscount] = useState('0.99');
  const [warmupStatus, setWarmupStatus] = useState({ status: 'idle' });
  const [warmupStatusReady, setWarmupStatusReady] = useState(false);
  const [jobStatus, setJobStatus] = useState({ status: 'idle' });
  const [trainingReplayDatasets, setTrainingReplayDatasets] = useState([]);
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
  const td3ObjectiveHydratedJobRef = useRef('');
  const td3ObjectiveUserSelectedRef = useRef(false);
  const batchSizeHydratedRef = useRef(false);
  const imitationConfigHydratedRef = useRef(false);
  const criticWarmupConfigHydratedRef = useRef(false);
  const warmupStatusRequestSequence = useRef(0);
  const activeWarmupStatusRequest = useRef(null);
  const lastObservedInferenceModelKeyRef = useRef(inferenceModelKey);

  useEffect(() => {
    onTrainingMethodStateChange?.(trainingMethod);
  }, [onTrainingMethodStateChange, trainingMethod]);

  useEffect(() => {
    let cancelled = false;
    const selected = datasetSelections
      .map((selection) => ({
        ...selection,
        path: String(selection?.path || '').trim(),
      }))
      .filter((selection) => selection.path);

    // Render the selected roots immediately, then replace each row with the
    // existing read-only dataset summary. Counts are never guessed while the
    // metadata request is pending or unavailable.
    setTrainingReplayDatasets(selected);
    if (!selected.length) return () => {
      cancelled = true;
    };

    Promise.all(selected.map(async (selection) => {
      try {
        const summary = await getOfflineRLDatasetInfo(selection.path);
        return {
          ...selection,
          ...(summary || {}),
          path: selection.path,
        };
      } catch {
        return selection;
      }
    })).then((summaries) => {
      if (!cancelled) setTrainingReplayDatasets(summaries);
    });

    return () => {
      cancelled = true;
    };
  }, [datasetSelections]);

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
      const backendSupported = trainingMethod === 'critic'
        ? ['act', 'multi_task_dit'].includes(selectedPolicyModel)
        : ['act', 'multi_task_dit'].includes(selectedPolicyModel);
      if (!backendSupported) {
        if (!isCancelled() && requestSequence === statusRequestSequence.current) {
          setJobStatus({ status: 'idle' });
          setStatusReady(true);
        }
        return { status: 'idle' };
      }
      const status = trainingMethod === 'imitation'
        ? await getImitationLearningStatus()
        : (trainingMethod === 'critic'
          ? (selectedPolicyModel === 'multi_task_dit'
            ? await getFlowSDEPPOValueWarmupStatus()
            : await getACTTD3CriticWarmupStatus())
          : (selectedPolicyModel === 'multi_task_dit' && getFlowSDEPPOStatus
            ? await getFlowSDEPPOStatus()
            : await getOfflineRLStatus()));
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
      if (trainingMethod === 'critic' && selectedPolicyModel === 'multi_task_dit') {
        setWarmupStatus(status || { status: 'idle' });
        setWarmupStatusReady(true);
      }
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
    if (activeWarmupStatusRequest.current !== null) {
      return null;
    }

    const requestSequence = ++warmupStatusRequestSequence.current;
    activeWarmupStatusRequest.current = requestSequence;
    try {
      const status = await getFlowSDEPPOValueWarmupStatus();
      if (
        isCancelled() ||
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
  const isReinforcementLearning = trainingMethod === 'reinforcement';
  const isImitationLearning = trainingMethod === 'imitation';
  const isCriticWarmup = trainingMethod === 'critic';
  const isDiffusionCriticWarmup = (
    isCriticWarmup && selectedPolicyModel === 'multi_task_dit'
  );
  const isActCriticWarmup = isCriticWarmup && selectedPolicyModel === 'act';
  const selectedPolicyBackendSupported = ['act', 'multi_task_dit'].includes(
    selectedPolicyModel
  );
  const isFlowSdePpo = (
    isReinforcementLearning &&
    selectedPolicyModel === 'multi_task_dit' &&
    algorithm === 'flow_sde_ppo'
  );
  const imitationPolicyType = selectedPolicyModel === 'multi_task_dit'
    ? 'multi_task_dit'
    : 'act';
  const effectiveImitationActionChunkSize = imitationPolicyType === 'multi_task_dit'
    ? IMITATION_ACTION_CHUNK_SIZES.multi_task_dit
    : Number(imitationActionChunkSize);
  const flowInferenceReady = inferencePhase === InferencePhase.READY;
  const flowInferenceBlockedReason = flowInferenceReady
    ? ''
    : `Flow-SDE PPO requires Inference READY (current: ${
      INFERENCE_PHASE_NAMES[inferencePhase] || 'UNKNOWN'
    }).`;
  useEffect(() => {
    if (!isActive || !isFlowSdePpo) {
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
  }, [isActive, isFlowSdePpo, requestWarmupStatus]);

  const normalizedWarmupStatus = String(warmupStatus?.status || 'idle').toLowerCase();
  const warmupIsRunning = Boolean(
    isFlowSdePpo && RUNNING_STATUSES.has(normalizedWarmupStatus)
  );
  // A status channel can be temporarily unavailable after changing methods.
  // Keep configuration and Start locked until it recovers, but never trap the
  // user in that method. Only a confirmed launch may lock method/model tabs.
  const selectionLocked = isRunning || warmupIsRunning || !isActive;
  const warmupIsComplete = COMPLETE_STATUSES.has(normalizedWarmupStatus);
  const warmupBundlePath = String(
    statusValue(warmupStatus, 'bundle_path') || ''
  ).trim();
  const parsedCriticEpochs = Number(criticEpochs);
  const parsedActorEquivalentEpochs = Number(actorEquivalentEpochs);
  const parsedBatchSize = Number(batchSize);
  const parsedImitationSteps = Number(imitationSteps);
  const parsedImitationBatchSize = Number(imitationBatchSize);
  const parsedImitationSaveFreq = Number(imitationSaveFreq);
  const parsedImitationActionChunkSize = Number(imitationActionChunkSize);
  const parsedCriticWarmupUpdates = Number(criticWarmupUpdates);
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
  const imitationActionChunkSizeValid = (
    imitationPolicyType === 'multi_task_dit' || (
      Number.isInteger(parsedImitationActionChunkSize) &&
      parsedImitationActionChunkSize >= 1 &&
      parsedImitationActionChunkSize <= 100
    )
  );
  const criticWarmupUpdatesValid = (
    Number.isInteger(parsedCriticWarmupUpdates) &&
    parsedCriticWarmupUpdates >= 1 &&
    parsedCriticWarmupUpdates <= 1000000
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
  const effectiveActorTrainableGroups = (
    isReinforcementLearning &&
    selectedPolicyModel === 'act' &&
    algorithm === 'td3' &&
    td3ActorObjective === 'td3'
  )
    ? actorTrainableGroups.filter((group) => group !== 'cvae_encoder')
    : actorTrainableGroups;
  const trainabilityError = validateActorTrainableGroups(effectiveActorTrainableGroups);
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
  // A compatible completed PPO trainer state is newer than an offline warm-up
  // and therefore takes precedence. An unrelated recovered PPO job must not
  // shadow a compatible warm-up for the currently selected policy and task.

  const resetStatusChannel = useCallback(() => {
    statusRequestSequence.current += 1;
    activeStatusRequest.current = null;
    actorTrainabilityHydratedRef.current = false;
    batchSizeHydratedRef.current = false;
    imitationConfigHydratedRef.current = false;
    criticWarmupConfigHydratedRef.current = false;
    lastAnnouncedStatus.current = 'idle';
    setJobStatus({ status: 'idle' });
    setStatusReady(false);
  }, []);

  useEffect(() => {
    if (lastObservedInferenceModelKeyRef.current === inferenceModelKey) return;
    if (selectionLocked) return;

    lastObservedInferenceModelKeyRef.current = inferenceModelKey;
    if (!inferencePolicyModel) return;

    resetStatusChannel();
    td3ObjectiveHydratedJobRef.current = '';
    td3ObjectiveUserSelectedRef.current = false;
    setSelectedPolicyModel(inferencePolicyModel);
    setAlgorithm(DEFAULT_ALGORITHM_BY_POLICY[inferencePolicyModel] || '');
    if (
      trainingMethod === 'imitation' &&
      !['act', 'multi_task_dit', 'groot', 'pi05'].includes(inferencePolicyModel)
    ) {
      setTrainingMethod('reinforcement');
    }
  }, [
    inferenceModelKey,
    inferencePolicyModel,
    resetStatusChannel,
    selectionLocked,
    trainingMethod,
  ]);

  const handleTrainingMethodChange = (nextMethod) => {
    if (selectionLocked || nextMethod === trainingMethod) return;
    resetStatusChannel();
    if (nextMethod === 'reinforcement') {
      setAlgorithm(DEFAULT_ALGORITHM_BY_POLICY[selectedPolicyModel] || '');
    }
    setTrainingMethod(nextMethod);
  };

  const handlePolicyModelChange = (nextPolicyModel) => {
    if (
      selectionLocked ||
      nextPolicyModel === selectedPolicyModel
    ) return;
    resetStatusChannel();
    td3ObjectiveHydratedJobRef.current = '';
    td3ObjectiveUserSelectedRef.current = false;
    setSelectedPolicyModel(nextPolicyModel);
    setAlgorithm(DEFAULT_ALGORITHM_BY_POLICY[nextPolicyModel] || '');
  };

  const handleAlgorithmChange = (nextAlgorithm) => {
    if (!isReinforcementLearning || interactionLocked || nextAlgorithm === algorithm) return;
    const compatible = (
      (ACT_TD3_ALGORITHMS.has(nextAlgorithm) && selectedPolicyModel === 'act') ||
      (nextAlgorithm === 'flow_sde_ppo' && selectedPolicyModel === 'multi_task_dit') ||
      (nextAlgorithm === 'rlt' && ['groot', 'pi05'].includes(selectedPolicyModel))
    );
    if (!compatible) return;
    resetStatusChannel();
    setAlgorithm(nextAlgorithm);
  };

  const handleTD3ActorObjectiveChange = (nextObjective) => {
    if (
      !isReinforcementLearning ||
      selectedPolicyModel !== 'act' ||
      algorithm !== 'td3' ||
      interactionLocked ||
      !TD3_ACTOR_OBJECTIVES.has(nextObjective) ||
      nextObjective === td3ActorObjective
    ) return;
    td3ObjectiveUserSelectedRef.current = true;
    setTD3ActorObjective(nextObjective);
  };

  useEffect(() => {
    const reportedObjective = String(jobStatus?.actor_objective || '').trim().toLowerCase();
    if (
      !isReinforcementLearning ||
      selectedPolicyModel !== 'act' ||
      algorithm !== 'td3' ||
      normalizedStatus === 'idle' ||
      !TD3_ACTOR_OBJECTIVES.has(reportedObjective) ||
      td3ObjectiveUserSelectedRef.current
    ) return;

    const statusIdentity = [
      String(jobStatus?.job_id || 'status'),
      reportedObjective,
      String(jobStatus?.act_checkpoint || ''),
    ].join(':');
    if (td3ObjectiveHydratedJobRef.current === statusIdentity) return;

    setTD3ActorObjective(reportedObjective);
    td3ObjectiveHydratedJobRef.current = statusIdentity;
  }, [
    algorithm,
    isReinforcementLearning,
    jobStatus?.act_checkpoint,
    jobStatus?.actor_objective,
    jobStatus?.job_id,
    normalizedStatus,
    selectedPolicyModel,
  ]);

  useEffect(() => {
    const supportsActorTrainability = (
      isReinforcementLearning || (isImitationLearning && selectedPolicyModel === 'act')
    );
    if (!supportsActorTrainability || actorTrainabilityHydratedRef.current || normalizedStatus === 'idle') {
      return;
    }
    const reportedGroups = isImitationLearning
      ? jobStatus?.trainable_groups
      : jobStatus?.actor_trainable_groups;
    if (!Array.isArray(reportedGroups)) return;
    const reportedSet = new Set(reportedGroups);
    const canonicalGroups = DEFAULT_ACT_TRAINABLE_GROUPS.filter((group) => (
      reportedSet.has(group)
    ));
    if (validateActorTrainableGroups(canonicalGroups)) return;
    setActorTrainableGroups(canonicalGroups);
    actorTrainabilityHydratedRef.current = true;
  }, [
    isImitationLearning,
    isReinforcementLearning,
    jobStatus?.actor_trainable_groups,
    jobStatus?.trainable_groups,
    normalizedStatus,
    selectedPolicyModel,
  ]);

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
    const reportedChunkSize = Number(jobStatus?.chunk_size);
    if (Number.isInteger(reportedSteps) && reportedSteps >= 1 && reportedSteps <= 1000000) {
      setImitationSteps(String(reportedSteps));
    }
    if (Number.isInteger(reportedBatchSize) && reportedBatchSize >= 1 && reportedBatchSize <= 64) {
      setImitationBatchSize(String(reportedBatchSize));
    }
    if (Number.isInteger(reportedSaveFreq) && reportedSaveFreq >= 1) {
      setImitationSaveFreq(String(reportedSaveFreq));
    }
    if (
      imitationPolicyType === 'act' &&
      Number.isInteger(reportedChunkSize) &&
      reportedChunkSize >= 1 &&
      reportedChunkSize <= 100
    ) {
      setImitationActionChunkSize(String(reportedChunkSize));
    }
    imitationConfigHydratedRef.current = true;
  }, [
    isImitationLearning,
    imitationPolicyType,
    jobStatus,
    normalizedStatus,
  ]);

  useEffect(() => {
    if (!isCriticWarmup || criticWarmupConfigHydratedRef.current || normalizedStatus === 'idle') {
      return;
    }
    if (isDiffusionCriticWarmup) {
      const reportedSteps = Number(statusValue(jobStatus, 'total_steps', 'steps'));
      const reportedBatchSize = Number(jobStatus?.batch_size);
      const reportedValueLearningRate = Number(jobStatus?.value_learning_rate);
      const reportedDiscount = Number(jobStatus?.discount);
      if (Number.isInteger(reportedSteps) && reportedSteps >= 1 && reportedSteps <= 1000000) {
        setWarmupSteps(String(reportedSteps));
      }
      if (
        Number.isInteger(reportedBatchSize) &&
        reportedBatchSize >= 1 &&
        reportedBatchSize <= 256
      ) {
        setWarmupBatchSize(String(reportedBatchSize));
      }
      if (Number.isFinite(reportedValueLearningRate) && reportedValueLearningRate > 0) {
        setWarmupValueLearningRate(String(reportedValueLearningRate));
      }
      if (Number.isFinite(reportedDiscount) && reportedDiscount > 0 && reportedDiscount <= 1) {
        setWarmupDiscount(String(reportedDiscount));
      }
      criticWarmupConfigHydratedRef.current = true;
      return;
    }
    const reportedUpdates = Number(jobStatus?.total_critic_updates);
    if (
      Number.isInteger(reportedUpdates) &&
      reportedUpdates >= 1 &&
      reportedUpdates <= 1000000
    ) {
      setCriticWarmupUpdates(String(reportedUpdates));
      criticWarmupConfigHydratedRef.current = true;
    }
  }, [isCriticWarmup, isDiffusionCriticWarmup, jobStatus, normalizedStatus]);

  useEffect(() => {
    if (!RUNNING_STATUSES.has(normalizedStatus)) setIsStopping(false);
  }, [normalizedStatus]);

  useEffect(() => {
    if (onRunningChange) onRunningChange(interactionLocked || warmupIsRunning);
  }, [interactionLocked, onRunningChange, warmupIsRunning]);

  useEffect(() => {
    if (normalizedStatus === lastAnnouncedStatus.current) return;
    const methodLabel = isImitationLearning
      ? 'Imitation Learning'
      : (isCriticWarmup ? 'Critic Warm-up' : 'Offline RL');
    if (isComplete) toast.success(`${methodLabel} training completed`);
    if (isFailed) {
      toast.error(jobStatus?.message || `${methodLabel} training failed`);
    }
    if (normalizedStatus === 'stopped') {
      toast.success(`${methodLabel} training stopped`);
    }
    lastAnnouncedStatus.current = normalizedStatus;
  }, [isComplete, isCriticWarmup, isFailed, isImitationLearning, jobStatus?.message, normalizedStatus]);

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
    if (isCriticWarmup) {
      onDeploymentStateChange({
        ready: false,
        modelPath: '',
        serviceType: 'lerobot',
        policyType: selectedPolicyModel,
        rlEpoch: Number(currentPolicyEpoch),
        lineageMode: 'unchanged',
      });
      return;
    }
    const reportedPolicyType = ['act', 'multi_task_dit'].includes(jobStatus?.policy_type)
      ? jobStatus.policy_type
      : selectedPolicyModel;
    const reportedRoundIndex = Number(jobStatus?.round_index);
    const deployedRLEpoch = isImitationLearning
      ? 0
      : (
        selectedPolicyModel === 'act' &&
        ACT_TD3_ALGORITHMS.has(algorithm) &&
        Number.isInteger(reportedRoundIndex) &&
        reportedRoundIndex >= 1
      ) ? reportedRoundIndex : Number(currentPolicyEpoch) + 1;
    const lineageMode = isImitationLearning
      ? 'new'
      : 'advance';
    onDeploymentStateChange({
      ready: selectedPolicyBackendSupported && statusReady && isComplete && Boolean(modelPath.trim()),
      modelPath: isComplete ? modelPath.trim() : '',
      serviceType: 'lerobot',
      policyType: reportedPolicyType,
      rlEpoch: deployedRLEpoch,
      lineageMode,
    });
  }, [
    algorithm,
    currentPolicyEpoch,
    isComplete,
    isCriticWarmup,
    isImitationLearning,
    jobStatus?.round_index,
    jobStatus?.policy_type,
    modelPath,
    onDeploymentStateChange,
    selectedPolicyBackendSupported,
    selectedPolicyModel,
    statusReady,
  ]);

  const statusLabel = useMemo(() => {
    if (!statusReady) return 'Checking';
    if (isStarting || normalizedStatus === 'starting') return 'Starting';
    if (normalizedStatus === 'running') {
      return isCriticWarmup ? 'Training critic' : 'Training';
    }
    if (isComplete) return 'Complete';
    if (isFailed) return 'Failed';
    if (normalizedStatus === 'stopped') return 'Stopped';
    return 'Ready';
  }, [isComplete, isCriticWarmup, isFailed, isStarting, normalizedStatus, statusReady]);

  const validateRequest = () => {
    if (!statusReady) return 'Wait for training status to load';
    if (
      isCriticWarmup &&
      !['act', 'multi_task_dit'].includes(selectedPolicyModel)
    ) {
      return 'Critic warm-up is available for ACT and Diffusion Transformer';
    }
    if (
      !isCriticWarmup &&
      !['act', 'multi_task_dit'].includes(selectedPolicyModel)
    ) {
      const policyLabel = selectedPolicyModel === 'pi05' ? 'Pi0.5' : 'GR00T';
      return `${policyLabel} training backend is not connected`;
    }
    if (isFlowSdePpo) {
      if (!actCheckpoint.trim()) return 'Select the MultiTaskDiT checkpoint';
      if (!robotType?.trim()) return 'Select a robot type on the Home page first';
      if (!flowInferenceReady) return flowInferenceBlockedReason;
      if (!flowSdePpoReady || typeof onStartFlowSDEPPO !== 'function') {
        return 'Flow-SDE PPO backend is not ready';
      }
      return '';
    }
    if (isConversionRunning) return 'Wait for dataset conversion to finish';
    if (!datasetPaths.length) return 'Include at least one LeRobot v3 Data Epoch';
    if (selectedDatasetVersionInvalid) {
      const trainingLabel = isImitationLearning
        ? `${imitationPolicyType === 'multi_task_dit' ? 'Diffusion Transformer' : 'ACT'} imitation learning`
        : (isCriticWarmup
          ? `${isDiffusionCriticWarmup ? 'Diffusion' : 'ACT'} critic warm-up`
          : 'TD3');
      return `${trainingLabel} requires LeRobot v3.0`;
    }
    if (isCriticWarmup) {
      if (isDiffusionCriticWarmup) {
        if (!actCheckpoint.trim()) return 'Select the MultiTaskDiT checkpoint';
        if (!flowTaskInstruction) return 'Enter a task instruction for critic warm-up';
        if (!warmupConfigValid) {
          return 'Warm-up settings require valid steps, batch size, value LR, and discount';
        }
        return '';
      }
      if (!isActCriticWarmup) {
        return 'Critic warm-up is available for ACT and Diffusion Transformer';
      }
      if (!actCheckpoint.trim()) return 'Select the ACT policy checkpoint';
      if (!robotType?.trim()) return 'Select a robot type on the Home page first';
      if (!batchSizeValid) {
        return 'Critic warm-up batch size must be an integer from 1 to 64';
      }
      if (!criticWarmupUpdatesValid) {
        return 'Critic warm-up updates must be an integer from 1 to 1,000,000';
      }
      return '';
    }
    if (isImitationLearning) {
      if (!imitationStepsValid) return 'Imitation steps must be an integer from 1 to 1,000,000';
      if (!imitationBatchSizeValid) return 'Imitation batch size must be an integer from 1 to 64';
      if (!imitationSaveFreqValid) {
        return 'Imitation save frequency must be an integer from 1 through the total steps';
      }
      if (!imitationActionChunkSizeValid) {
        return 'ACT imitation action chunk must be an integer from 1 to 100';
      }
      if (imitationPolicyType === 'act' && trainabilityError) return trainabilityError;
      return '';
    }
    if (!actCheckpoint.trim()) {
      return isFlowSdePpo
        ? 'Select the MultiTaskDiT checkpoint'
        : 'Select the original ACT checkpoint';
    }
    if (!robotType?.trim()) return 'Select a robot type on the Home page first';
    if (!ACT_TD3_ALGORITHMS.has(algorithm)) {
      return 'Select TD3 for ACT';
    }
    if (!TD3_ACTOR_OBJECTIVES.has(td3ActorObjective)) {
      return 'Select a valid TD3 loss option';
    }
    if (trainabilityError) return trainabilityError;
    if (!batchSizeValid) return 'Batch size must be an integer from 1 to 64';
    if (!scheduleValid) {
      return 'TD3 requires Critic epochs = 2 × Actor equivalent epochs';
    }
    return '';
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
    isReinforcementLearning &&
    !forceFreshLineage &&
    isComplete &&
    Boolean(checkpointPath.trim()) &&
    Boolean(completedBaseActCheckpoint) &&
    selectedReplayExtendsCompletedReplay &&
    selectedModelMatchesCompletedLineage &&
    jobStatus?.algorithm === algorithm &&
    jobStatus?.actor_objective === td3ActorObjective
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
          chunk_size: effectiveImitationActionChunkSize,
          ...(imitationPolicyType === 'act'
            ? { trainable_groups: actorTrainableGroups }
            : {}),
          ...(imitationPolicyType === 'multi_task_dit' && flowTaskInstruction
            ? { task_instruction: flowTaskInstruction }
            : {}),
        })
        : isCriticWarmup
          ? (isDiffusionCriticWarmup
            ? await startFlowSDEPPOValueWarmup({
              policy_checkpoint: selectedActCheckpoint,
              dataset_paths: datasetPaths,
              policy_type: 'multi_task_dit',
              task_instruction: flowTaskInstruction,
              steps: parsedWarmupSteps,
              batch_size: parsedWarmupBatchSize,
              value_learning_rate: parsedWarmupValueLearningRate,
              discount: parsedWarmupDiscount,
            })
            : await startACTTD3CriticWarmup({
              dataset_path: datasetPaths[0],
              dataset_paths: datasetPaths,
              act_checkpoint: selectedActCheckpoint,
              robot_type: robotType.trim(),
              batch_size: parsedBatchSize,
              critic_updates: parsedCriticWarmupUpdates,
            }))
        : isFlowSdePpo
          ? await onStartFlowSDEPPO({
            policy_type: 'multi_task_dit',
            policy_checkpoint: ppoResumeReady
              ? resumeModelPath
              : selectedActCheckpoint,
            algorithm: 'flow_sde_ppo',
            robot_type: robotType.trim(),
            ...(ppoResumeReady
              ? { resume_checkpoint: resumeCheckpointPath }
              : (compatibleWarmupReady
                ? { value_warmup_bundle: warmupBundlePath }
                : {})),
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
          algorithm: 'td3',
          actor_objective: td3ActorObjective,
          robot_type: robotType.trim(),
          critic_epochs: parsedCriticEpochs,
          actor_equivalent_epochs: parsedActorEquivalentEpochs,
          batch_size: parsedBatchSize,
          actor_trainable_groups: (
            td3ActorObjective === 'td3'
              ? actorTrainableGroups.filter((group) => group !== 'cvae_encoder')
              : actorTrainableGroups
          ),
        });
      setJobStatus(result || { status: 'starting' });
      if (isDiffusionCriticWarmup) {
        setWarmupStatus(result || { status: 'starting' });
        setWarmupStatusReady(true);
      }
      if (
        forceFreshLineage &&
        isReinforcementLearning &&
        !isFlowSdePpo
      ) {
        onFreshLineageConsumed?.();
      }
      const methodLabel = isImitationLearning
        ? 'Imitation Learning'
        : (isCriticWarmup
          ? 'Critic Warm-up'
          : (isFlowSdePpo ? 'Flow-SDE PPO' : 'Offline RL'));
      toast.success(`${methodLabel} training started`);
    } catch (error) {
      const methodLabel = isImitationLearning
        ? 'Imitation Learning'
        : (isCriticWarmup
          ? 'Critic Warm-up'
          : (isFlowSdePpo ? 'Flow-SDE PPO' : 'Offline RL'));
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
    const methodLabel = isImitationLearning
      ? 'Imitation Learning'
      : (isCriticWarmup ? 'Critic Warm-up' : 'Offline RL');
    if (!window.confirm(
      `Stop the current ${methodLabel} training job?\n\n` +
      'The current job will stop at a safe boundary and will not export a deployable policy.'
    )) return;

    isStoppingRef.current = true;
    statusRequestSequence.current += 1;
    activeStatusRequest.current = null;
    setIsStopping(true);
    try {
      const result = isImitationLearning
        ? await stopImitationLearningTraining(jobId)
        : (isCriticWarmup
          ? (isDiffusionCriticWarmup
            ? await stopFlowSDEPPOValueWarmup(jobId)
            : await stopACTTD3CriticWarmup(jobId))
          : (isFlowSdePpo
            ? await onStopFlowSDEPPO(jobId)
            : await stopOfflineRLTraining(jobId)));
      setJobStatus(result || { ...jobStatus, status: 'running' });
      if (isDiffusionCriticWarmup) {
        setWarmupStatus(result || { ...jobStatus, status: 'running' });
        setWarmupStatusReady(true);
      }
      toast.success(`${methodLabel} stop requested`);
    } catch (error) {
      toast.error(`${methodLabel} stop failed: ${error.message}`);
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
    ? imitationStepsValid && imitationBatchSizeValid && imitationSaveFreqValid &&
      imitationActionChunkSizeValid &&
      (imitationPolicyType !== 'act' || trainabilityValid)
    : (isCriticWarmup
      ? (isDiffusionCriticWarmup
        ? (
          warmupConfigValid &&
          Boolean(datasetPaths.length) &&
          Boolean(actCheckpoint.trim()) &&
          Boolean(flowTaskInstruction)
        )
        : (
          isActCriticWarmup &&
          batchSizeValid &&
          criticWarmupUpdatesValid &&
          Boolean(datasetPaths.length) &&
          Boolean(actCheckpoint.trim()) &&
          Boolean(robotType?.trim())
        ))
      : (isFlowSdePpo
      ? (
        flowSdePpoReady &&
        typeof onStartFlowSDEPPO === 'function' &&
        Boolean(actCheckpoint.trim()) &&
        Boolean(robotType?.trim()) &&
        flowInferenceReady
      )
      : scheduleValid && batchSizeValid && trainabilityValid));
  const startDisabled = (
    interactionLocked ||
    (!isFlowSdePpo && isConversionRunning) ||
    (isFlowSdePpo && (!warmupStatusReady || warmupIsRunning)) ||
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
  if (variant === 'workflow') {
    return (
      <WorkflowTrainingView
        trainingMethod={trainingMethod}
        onTrainingMethodChange={handleTrainingMethodChange}
        selectedPolicyModel={selectedPolicyModel}
        onPolicyModelChange={handlePolicyModelChange}
        algorithm={algorithm}
        onAlgorithmChange={handleAlgorithmChange}
        td3ActorObjective={td3ActorObjective}
        onTD3ActorObjectiveChange={handleTD3ActorObjectiveChange}
        flowSdePpoReady={flowSdePpoReady}
        flowInferenceBlockedReason={flowInferenceBlockedReason}
        flowTaskInstruction={flowTaskInstruction}
        ppoResumeReady={ppoResumeReady}
        compatibleWarmupReady={compatibleWarmupReady}
        warmupSteps={warmupSteps}
        setWarmupSteps={setWarmupSteps}
        warmupBatchSize={warmupBatchSize}
        setWarmupBatchSize={setWarmupBatchSize}
        warmupValueLearningRate={warmupValueLearningRate}
        setWarmupValueLearningRate={setWarmupValueLearningRate}
        warmupDiscount={warmupDiscount}
        setWarmupDiscount={setWarmupDiscount}
        actorTrainableGroups={actorTrainableGroups}
        setActorTrainableGroups={setActorTrainableGroups}
        rltTrainableGroups={rltTrainableGroups}
        setRltTrainableGroups={setRltTrainableGroups}
        browserDisabled={browserDisabled}
        selectionDisabled={selectionLocked}
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
        imitationActionChunkSize={imitationActionChunkSize}
        setImitationActionChunkSize={setImitationActionChunkSize}
        criticWarmupUpdates={criticWarmupUpdates}
        setCriticWarmupUpdates={setCriticWarmupUpdates}
        statusLabel={statusLabel}
        displayProgress={displayProgress}
        jobStatus={jobStatus}
        modelPath={modelPath}
        currentPolicyEpoch={currentPolicyEpoch}
        datasetSelections={datasetSelections}
        trainingReplayDatasets={trainingReplayDatasets}
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
        onCompactLayoutChange={onCompactLayoutChange}
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

          <div className="flex flex-col gap-1.5">
            <label htmlFor="offline-rl-td3-loss-option" className="text-sm font-medium text-gray-600">
              Loss option
            </label>
            <select
              id="offline-rl-td3-loss-option"
              value={td3ActorObjective}
              onChange={(event) => handleTD3ActorObjectiveChange(event.target.value)}
              disabled={browserDisabled || algorithm !== 'td3'}
              className="h-10 rounded-md border border-gray-300 bg-white px-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:cursor-not-allowed disabled:bg-gray-100"
            >
              <option value="td3">TD3</option>
              <option value="td3_bc">TD3-BC</option>
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
