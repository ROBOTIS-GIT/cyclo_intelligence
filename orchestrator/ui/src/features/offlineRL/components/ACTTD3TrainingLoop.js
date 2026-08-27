// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useId, useRef, useState } from 'react';
import clsx from 'clsx';
import { MdArrowDownward, MdCheck } from 'react-icons/md';

import ACTArchitectureDiagram, {
  DEFAULT_ACT_TRAINABLE_GROUPS,
} from './ACTArchitectureDiagram';
import TrainingReplayBufferCard, {
  DEFAULT_TRAINING_REPLAY_CAPACITY,
} from './TrainingReplayBufferCard';

const inputClass = (
  'mt-1 h-8 w-full rounded-lg border border-[#ddc9b9] bg-white px-2.5 '
  + 'text-[10px] font-semibold text-[#4a4038] outline-none transition '
  + 'focus:border-[#bd8564] focus:ring-2 focus:ring-[#ead7ca] '
  + 'disabled:cursor-not-allowed disabled:bg-[#eeeae4] disabled:text-[#999187]'
);

const objectiveOptions = [
  {
    value: 'td3',
    label: 'TD3',
    detail: '-Q1',
  },
  {
    value: 'td3_bc',
    label: 'TD3-BC',
    detail: '-Q1 + success BC',
  },
];

const modeContent = {
  reinforcement: {
    regionLabel: 'ACT TD3 training loop',
    replayStep: 'Replay Buffer → TD3',
    returnStep: 'Policy update → next cycle',
  },
  imitation: {
    regionLabel: 'ACT imitation training loop',
    replayStep: 'Replay Buffer → IL',
    returnStep: 'Policy update → next cycle',
  },
  critic: {
    regionLabel: 'ACT critic warm-up loop',
    replayStep: 'Replay Buffer → Critic',
    returnStep: 'Warm critic → RL',
  },
};

function normalizeMode(mode) {
  const normalized = String(mode || '').trim().toLowerCase().replaceAll('-', '_');
  if (['imitation', 'il', 'imitation_learning'].includes(normalized)) return 'imitation';
  if (['critic', 'critic_warmup', 'warmup'].includes(normalized)) return 'critic';
  return 'reinforcement';
}

function NumberSetting({
  label,
  ariaLabel = label,
  value,
  onChange,
  disabled,
  min,
  max,
  step = 1,
  title,
}) {
  return (
    <label className="min-w-0 text-[9px] font-semibold text-[#776b62]">
      {label}
      <input
        aria-label={ariaLabel}
        type="number"
        min={min}
        max={max}
        step={step}
        title={title}
        value={value}
        onChange={(event) => {
          if (!disabled) onChange?.(event.target.value);
        }}
        disabled={disabled}
        className={inputClass}
      />
    </label>
  );
}

function TrainingCardShell({
  eyebrow,
  title,
  description,
  badge,
  titleId,
  testId,
  children,
}) {
  return (
    <section
      className="h-full min-w-0 rounded-2xl border border-[#decfc3] bg-white p-4 shadow-[0_8px_24px_rgba(75,66,51,0.07)]"
      aria-labelledby={titleId}
      data-testid={testId}
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[9px] font-semibold uppercase tracking-[0.14em] text-[#aa795f]">
            {eyebrow}
          </div>
          <h3 id={titleId} className="mt-0.5 text-[14px] font-semibold text-[#38342e]">
            {title}
          </h3>
          <p className="mt-1 text-[10px] text-[#8b8378]">
            {description}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-[#f5e9df] px-2.5 py-1 text-[9px] font-bold text-[#9b6245]">
          {badge}
        </span>
      </div>
      {children}
    </section>
  );
}

function AlgorithmCard({
  actorObjective,
  onActorObjectiveChange,
  criticEpochs,
  onCriticEpochsChange,
  actorEpochs,
  onActorEpochsChange,
  batchSize,
  onBatchSizeChange,
  disabled,
}) {
  return (
    <TrainingCardShell
      eyebrow="Training"
      title="RL Algorithm"
      description="Chunk-level actor and critic update"
      badge="TD3"
      titleId="act-td3-algorithm-title"
      testId="act-td3-algorithm-card"
    >
      <div className="mt-3 rounded-xl border border-[#e1bca4] bg-[#fbede3] p-3 text-[#754832]">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[8px] font-bold uppercase tracking-[0.12em] text-[#ad7251]">
            Value function
          </span>
          <span className="rounded-full bg-[#d9895f] px-2 py-0.5 text-[8px] font-semibold text-white">
            Trainable
          </span>
        </div>
        <div className="mt-1 text-[13px] font-semibold">Critic Network</div>
        <div className="mt-0.5 text-[9px] text-[#9a674d]">
          Twin chunk Q-functions · clipped Bellman target
        </div>
      </div>

      <fieldset className="mt-3" disabled={disabled}>
        <legend className="text-[9px] font-semibold text-[#776b62]">Loss option</legend>
        <div className="mt-1 grid grid-cols-2 gap-1 rounded-xl bg-[#f1ede7] p-1">
          {objectiveOptions.map((option) => {
            const selected = actorObjective === option.value;
            return (
              <button
                key={option.value}
                type="button"
                aria-pressed={selected}
                aria-label={`${option.label} loss`}
                disabled={disabled}
                onClick={() => onActorObjectiveChange?.(option.value)}
                className={clsx(
                  'rounded-lg border px-2 py-1.5 text-left transition focus:outline-none focus:ring-2 focus:ring-[#d7aa8e]',
                  selected
                    ? 'border-[#d6a486] bg-white text-[#71472f] shadow-sm'
                    : 'border-transparent text-[#8c8278] hover:bg-white/60',
                  disabled && 'cursor-not-allowed opacity-60'
                )}
              >
                <span className="block text-[10px] font-semibold">{option.label}</span>
                <span className="mt-0.5 block truncate text-[8px] opacity-75">{option.detail}</span>
              </button>
            );
          })}
        </div>
      </fieldset>

      <div className="mt-3 grid grid-cols-3 gap-2">
        <NumberSetting
          label="Critic epochs"
          value={criticEpochs}
          onChange={onCriticEpochsChange}
          disabled={disabled}
          min={2}
          step={2}
        />
        <NumberSetting
          label="Actor epochs"
          value={actorEpochs}
          onChange={onActorEpochsChange}
          disabled={disabled}
          min={1}
        />
        <NumberSetting
          label="Batch size"
          value={batchSize}
          onChange={onBatchSizeChange}
          disabled={disabled}
          min={1}
          max={64}
        />
      </div>

    </TrainingCardShell>
  );
}

export function ImitationLearningCard({
  policyLabel = 'ACT',
  title = 'Imitation Learning',
  description = '',
  objectiveEyebrow = '',
  objectiveTitle = 'Action Chunk Reconstruction',
  objectiveDetail = 'CVAE reconstruction + latent KL regularization',
  titleId = 'act-imitation-algorithm-title',
  testId = 'act-imitation-algorithm-card',
  steps,
  onStepsChange,
  batchSize,
  onBatchSizeChange,
  saveFreq,
  onSaveFreqChange,
  actionChunkSize,
  onActionChunkSizeChange,
  actionChunkDisabled = false,
  actionChunkTitle = '',
  disabled,
}) {
  const resolvedDescription = description || (
    `Fit ${policyLabel} action chunks to recorded demonstrations`
  );
  const resolvedObjectiveEyebrow = objectiveEyebrow || `${policyLabel} objective`;

  return (
    <TrainingCardShell
      eyebrow="Supervised training"
      title={title}
      description={resolvedDescription}
      badge="IL"
      titleId={titleId}
      testId={testId}
    >
      <div className="mt-3 rounded-xl border border-[#cfd5e7] bg-[#f2f4fa] p-3 text-[#4b587b]">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[8px] font-bold uppercase tracking-[0.12em] text-[#7180a4]">
            {resolvedObjectiveEyebrow}
          </span>
          <span className="rounded-full bg-[#66759b] px-2 py-0.5 text-[8px] font-semibold text-white">
            Trainable
          </span>
        </div>
        <div className="mt-1 text-[13px] font-semibold">{objectiveTitle}</div>
        <div className="mt-0.5 text-[9px] text-[#6f7890]">
          {objectiveDetail}
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <NumberSetting
          label="Training steps"
          ariaLabel="Imitation steps"
          value={steps}
          onChange={onStepsChange}
          disabled={disabled}
          min={1}
        />
        <NumberSetting
          label="Batch size"
          ariaLabel="Imitation batch size"
          value={batchSize}
          onChange={onBatchSizeChange}
          disabled={disabled}
          min={1}
          max={1024}
        />
        <NumberSetting
          label="Save frequency"
          ariaLabel="Imitation save frequency"
          value={saveFreq}
          onChange={onSaveFreqChange}
          disabled={disabled}
          min={1}
        />
        <NumberSetting
          label="Action chunk size"
          ariaLabel="Imitation action chunk"
          value={actionChunkSize}
          onChange={onActionChunkSizeChange}
          disabled={disabled || actionChunkDisabled}
          min={1}
          max={100}
          title={actionChunkTitle || undefined}
        />
      </div>
    </TrainingCardShell>
  );
}

function CriticWarmupCard({
  batchSize,
  onBatchSizeChange,
  updates,
  onUpdatesChange,
  checkpointPath,
  guidance,
  disabled,
}) {
  const checkpoint = String(checkpointPath || '').trim();
  const checkpointDisplay = checkpoint || guidance || 'Created beside the ACT policy';

  return (
    <TrainingCardShell
      eyebrow="Value initialization"
      title="Critic Warm-up"
      description="Pretrain ACT chunk values before policy optimization"
      badge="Critic"
      titleId="act-critic-warmup-title"
      testId="act-critic-warmup-card"
    >
      <div className="mt-3 grid grid-cols-[0.8fr_auto_1.2fr] items-stretch gap-2">
        <div
          className="rounded-xl border border-[#ded9d1] bg-[#f4f2ee] p-3 text-[#777068]"
          aria-label="ACT actor: Frozen; no gradients"
        >
          <div className="text-[8px] font-bold uppercase tracking-[0.12em]">Actor</div>
          <div className="mt-1 text-[12px] font-semibold">ACT Policy</div>
          <output
            aria-label="Critic warm-up ACT actor mode"
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
          <div className="mt-1 text-[12px] font-semibold">Critic Network</div>
          <span className="mt-1 inline-flex rounded-full bg-[#d9895f] px-2 py-0.5 text-[8px] font-bold text-white">
            Trainable
          </span>
        </div>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2">
        <NumberSetting
          label="Warm-up updates"
          ariaLabel="Critic warm-up total updates"
          value={updates}
          onChange={onUpdatesChange}
          disabled={disabled}
          min={1}
        />
        <NumberSetting
          label="Batch size"
          ariaLabel="Critic warm-up batch size"
          value={batchSize}
          onChange={onBatchSizeChange}
          disabled={disabled}
          min={1}
          max={1024}
        />
      </div>

      <div
        className="mt-3 rounded-xl border border-[#ebe3da] bg-[#faf8f4] px-3 py-2 text-[9px]"
        aria-label="ACT critic warm-up checkpoint"
      >
        <div className="font-semibold text-[#645b52]">Critic checkpoint</div>
        <div
          className="mt-1 truncate font-mono text-[8px] text-[#998f85]"
          title={checkpointDisplay}
          aria-label="Critic checkpoint path"
        >
          {checkpointDisplay}
        </div>
        {checkpoint && guidance && (
          <div className="mt-1.5 text-[8px] leading-3 text-[#8b8278]">
            {guidance}
          </div>
        )}
      </div>
    </TrainingCardShell>
  );
}

const emptyConnectorGeometry = {
  width: 1,
  height: 1,
  paths: [
    { id: 'policy-to-replay', d: '' },
    { id: 'replay-to-algorithm', d: '' },
    { id: 'algorithm-to-policy', d: '' },
  ],
};

const rounded = (value) => Math.round(value * 10) / 10;

const relativeRect = (containerRect, rect) => ({
  left: rect.left - containerRect.left,
  right: rect.right - containerRect.left,
  top: rect.top - containerRect.top,
  bottom: rect.bottom - containerRect.top,
  width: rect.width,
  height: rect.height,
});

const point = (x, y) => ({ x: rounded(x), y: rounded(y) });

/**
 * Builds connector paths from the cards' real rendered bounds. Keeping this
 * calculation independent from React makes the edge-center contract directly
 * testable and prevents policy-specific card heights from changing the loop.
 */
export function buildLoopConnectorGeometry({
  containerRect,
  policyRect,
  replayRect,
  algorithmRect,
}) {
  if (!containerRect || !policyRect || !replayRect || !algorithmRect) {
    return emptyConnectorGeometry;
  }

  const policy = relativeRect(containerRect, policyRect);
  const replay = relativeRect(containerRect, replayRect);
  const algorithm = relativeRect(containerRect, algorithmRect);
  const policyToReplayStart = point(policy.right, policy.top + (policy.height / 2));
  const policyToReplayEnd = point(replay.left, replay.top + (replay.height / 2));
  const replayToAlgorithmStart = point(replay.left + (replay.width / 2), replay.bottom);
  const replayToAlgorithmEnd = point(algorithm.right, algorithm.top + (algorithm.height / 2));
  const algorithmToPolicyStart = point(algorithm.left, algorithm.top + (algorithm.height / 2));
  const algorithmToPolicyEnd = point(policy.left + (policy.width / 2), policy.bottom);

  const horizontalGap = Math.max(18, Math.abs(policyToReplayEnd.x - policyToReplayStart.x) * 0.38);
  const rightTurn = Math.max(24, Math.min(48, Math.abs(
    replayToAlgorithmEnd.y - replayToAlgorithmStart.y
  ) * 0.35));
  const leftTurn = Math.max(24, Math.min(48, Math.abs(
    algorithmToPolicyEnd.y - algorithmToPolicyStart.y
  ) * 0.35));

  return {
    width: Math.max(1, rounded(containerRect.width)),
    height: Math.max(1, rounded(containerRect.height)),
    paths: [
      {
        id: 'policy-to-replay',
        start: policyToReplayStart,
        end: policyToReplayEnd,
        d: [
          `M ${policyToReplayStart.x} ${policyToReplayStart.y}`,
          `C ${rounded(policyToReplayStart.x + horizontalGap)} ${policyToReplayStart.y}`,
          `${rounded(policyToReplayEnd.x - horizontalGap)} ${policyToReplayEnd.y}`,
          `${policyToReplayEnd.x} ${policyToReplayEnd.y}`,
        ].join(' '),
      },
      {
        id: 'replay-to-algorithm',
        start: replayToAlgorithmStart,
        end: replayToAlgorithmEnd,
        d: [
          `M ${replayToAlgorithmStart.x} ${replayToAlgorithmStart.y}`,
          `C ${replayToAlgorithmStart.x} ${rounded(replayToAlgorithmStart.y + rightTurn)}`,
          `${rounded(replayToAlgorithmEnd.x + rightTurn)} ${replayToAlgorithmEnd.y}`,
          `${replayToAlgorithmEnd.x} ${replayToAlgorithmEnd.y}`,
        ].join(' '),
      },
      {
        id: 'algorithm-to-policy',
        start: algorithmToPolicyStart,
        end: algorithmToPolicyEnd,
        d: [
          `M ${algorithmToPolicyStart.x} ${algorithmToPolicyStart.y}`,
          `C ${rounded(algorithmToPolicyStart.x - leftTurn)} ${algorithmToPolicyStart.y}`,
          `${algorithmToPolicyEnd.x} ${rounded(algorithmToPolicyEnd.y + leftTurn)}`,
          `${algorithmToPolicyEnd.x} ${algorithmToPolicyEnd.y}`,
        ].join(' '),
      },
    ],
  };
}

function DesktopLoopConnector({
  containerRef,
  policyRef,
  replayRef,
  algorithmRef,
  testId = 'policy-training-loop-connectors',
}) {
  const reactId = useId();
  const markerId = `policy-training-loop-arrow-${reactId.replaceAll(':', '')}`;
  const [geometry, setGeometry] = useState(emptyConnectorGeometry);

  useEffect(() => {
    const elements = [
      containerRef.current,
      policyRef.current,
      replayRef.current,
      algorithmRef.current,
    ];
    if (elements.some((element) => !element)) return undefined;

    const requestFrame = window.requestAnimationFrame
      ? window.requestAnimationFrame.bind(window)
      : (callback) => window.setTimeout(callback, 0);
    const cancelFrame = window.cancelAnimationFrame
      ? window.cancelAnimationFrame.bind(window)
      : window.clearTimeout.bind(window);
    let frameId = null;

    const measure = () => {
      if (frameId !== null) return;
      frameId = requestFrame(() => {
        frameId = null;
        setGeometry(buildLoopConnectorGeometry({
          containerRect: containerRef.current?.getBoundingClientRect(),
          policyRect: policyRef.current?.getBoundingClientRect(),
          replayRect: replayRef.current?.getBoundingClientRect(),
          algorithmRect: algorithmRef.current?.getBoundingClientRect(),
        }));
      });
    };

    const observer = typeof ResizeObserver === 'function'
      ? new ResizeObserver(measure)
      : null;
    elements.forEach((element) => observer?.observe(element));
    window.addEventListener('resize', measure);
    measure();

    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', measure);
      if (frameId !== null) cancelFrame(frameId);
    };
  }, [algorithmRef, containerRef, policyRef, replayRef]);

  return (
    <svg
      viewBox={`0 0 ${geometry.width} ${geometry.height}`}
      preserveAspectRatio="none"
      className="pointer-events-none absolute inset-0 z-20 hidden h-full w-full overflow-visible 2xl:block"
      aria-hidden="true"
      data-testid={testId}
    >
      <defs>
        <marker
          id={markerId}
          markerWidth="10"
          markerHeight="10"
          refX="9"
          refY="5"
          orient="auto"
          markerUnits="userSpaceOnUse"
          viewBox="0 0 10 10"
        >
          <path d="M 0 1 L 9 5 L 0 9 Z" fill="#8f887d" />
        </marker>
      </defs>
      {geometry.paths.map((connector) => (
        <path
          key={connector.id}
          data-connector={connector.id}
          d={connector.d}
          fill="none"
          stroke="#8f887d"
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
          markerEnd={`url(#${markerId})`}
        />
      ))}
    </svg>
  );
}

function MobileStep({ children }) {
  return (
    <div className="flex items-center justify-center py-1.5 text-[#8f897f] 2xl:hidden">
      <MdArrowDownward size={14} className="mr-1" aria-hidden="true" />
      <span className="text-[8px] font-bold uppercase tracking-[0.12em]">{children}</span>
    </div>
  );
}

/**
 * Shared presentation shell for every policy training workflow.
 *
 * Algorithm support, validation, and request submission deliberately stay in
 * the parent. This component only keeps Policy -> Replay Buffer -> Training in
 * the same visual order for ACT, GR00T, Diffusion Policy, and Pi0.5.
 */
export function PolicyTrainingLoopLayout({
  regionLabel = 'Policy training loop',
  testId = 'policy-training-loop',
  trainingMode = 'reinforcement',
  policyModel = '',
  policyLabel = 'Policy',
  policyTestId = 'training-policy-stage',
  policyNode,
  datasets = [],
  capacityEpisodes = DEFAULT_TRAINING_REPLAY_CAPACITY,
  onInspectDataset = null,
  trainingNode,
  replayStep = 'Replay Buffer -> Training',
  returnStep = 'Policy update -> next cycle',
  connectorTestId = 'policy-training-loop-connectors',
  updated = false,
  fitContent = false,
}) {
  const containerRef = useRef(null);
  const policyRef = useRef(null);
  const replayRef = useRef(null);
  const algorithmRef = useRef(null);

  return (
    <section
      ref={containerRef}
      aria-label={regionLabel}
      className={clsx(
        'relative min-w-0 rounded-2xl border border-[#ded8ce] bg-[#f5f1e9] p-3.5',
        fitContent && 'self-start'
      )}
      data-testid={testId}
      data-training-mode={trainingMode}
      data-policy-model={policyModel || undefined}
      data-fit-content={fitContent ? 'true' : 'false'}
    >
      <DesktopLoopConnector
        containerRef={containerRef}
        policyRef={policyRef}
        replayRef={replayRef}
        algorithmRef={algorithmRef}
        testId={connectorTestId}
      />

      <div className="relative z-10 grid min-w-0 gap-3 2xl:grid-cols-2 2xl:gap-x-12 2xl:gap-y-12">
        <div
          ref={policyRef}
          className={clsx(
            'relative min-h-0 rounded-2xl transition-shadow',
            updated && 'ring-2 ring-[#66759b] ring-offset-2 ring-offset-[#f5f1e9]'
          )}
          aria-label={`${policyLabel} policy: ${updated ? 'updated' : 'current'}`}
          data-testid={policyTestId}
        >
          <span className="sr-only">Policy</span>
          {updated && (
            <span className="absolute -right-1 -top-2 z-20 flex items-center gap-1 rounded-full bg-[#53658f] px-2.5 py-1 text-[8px] font-bold uppercase tracking-[0.08em] text-white shadow-md">
              <MdCheck size={11} aria-hidden="true" /> Updated policy
            </span>
          )}
          {policyNode}
        </div>

        <MobileStep>Policy -&gt; deployed data</MobileStep>

        <div
          ref={replayRef}
          className="min-w-0"
          data-testid="training-replay-buffer-stage"
        >
          <TrainingReplayBufferCard
            datasets={datasets}
            capacityEpisodes={capacityEpisodes}
            onInspectDataset={onInspectDataset}
            className="h-full"
          />
        </div>

        <MobileStep>{replayStep}</MobileStep>

        <div
          ref={algorithmRef}
          className="min-w-0 2xl:col-span-2 2xl:w-[calc(50%-1.5rem)] 2xl:justify-self-center"
          data-testid="training-algorithm-stage"
        >
          {trainingNode}
        </div>

        <MobileStep>{returnStep}</MobileStep>
      </div>
    </section>
  );
}

/**
 * Presentation-only ACT training loop shared by RL, imitation, and critic
 * warm-up workflows.
 *
 * State, validation, persistence, and training submission remain owned by the
 * parent. Every interactive value is controlled through the props below.
 */
export default function ACTTD3TrainingLoop({
  mode = 'reinforcement',
  trainableGroups = DEFAULT_ACT_TRAINABLE_GROUPS,
  onTrainableGroupsChange = null,
  lockedGroups = [],
  datasets = [],
  capacityEpisodes = DEFAULT_TRAINING_REPLAY_CAPACITY,
  onInspectDataset = null,
  actorObjective = 'td3_bc',
  onActorObjectiveChange = null,
  criticEpochs = '10',
  onCriticEpochsChange = null,
  actorEpochs = '5',
  onActorEpochsChange = null,
  batchSize = '4',
  onBatchSizeChange = null,
  imitationSteps = '80000',
  onImitationStepsChange = null,
  imitationBatchSize = '8',
  onImitationBatchSizeChange = null,
  imitationSaveFreq = '10000',
  onImitationSaveFreqChange = null,
  imitationActionChunkSize = '30',
  onImitationActionChunkSizeChange = null,
  criticWarmupBatchSize = '256',
  onCriticWarmupBatchSizeChange = null,
  criticWarmupUpdates = '1000',
  onCriticWarmupUpdatesChange = null,
  criticCheckpointPath = '',
  criticGuidance = '',
  policyDisabled = false,
  disabled = false,
  fitContent = false,
  updated = false,
}) {
  const resolvedMode = normalizeMode(mode);
  const content = modeContent[resolvedMode];
  const isPolicyDisabled = disabled || policyDisabled || resolvedMode === 'critic';
  let trainingNode = null;
  if (resolvedMode === 'imitation') {
    trainingNode = (
      <div data-testid="act-imitation-learning-diagram">
        <ImitationLearningCard
          steps={imitationSteps}
          onStepsChange={onImitationStepsChange}
          batchSize={imitationBatchSize}
          onBatchSizeChange={onImitationBatchSizeChange}
          saveFreq={imitationSaveFreq}
          onSaveFreqChange={onImitationSaveFreqChange}
          actionChunkSize={imitationActionChunkSize}
          onActionChunkSizeChange={onImitationActionChunkSizeChange}
          disabled={disabled}
        />
      </div>
    );
  } else if (resolvedMode === 'critic') {
    trainingNode = (
      <CriticWarmupCard
        batchSize={criticWarmupBatchSize}
        onBatchSizeChange={onCriticWarmupBatchSizeChange}
        updates={criticWarmupUpdates}
        onUpdatesChange={onCriticWarmupUpdatesChange}
        checkpointPath={criticCheckpointPath}
        guidance={criticGuidance}
        disabled={disabled}
      />
    );
  } else {
    trainingNode = (
      <AlgorithmCard
        actorObjective={actorObjective}
        onActorObjectiveChange={onActorObjectiveChange}
        criticEpochs={criticEpochs}
        onCriticEpochsChange={onCriticEpochsChange}
        actorEpochs={actorEpochs}
        onActorEpochsChange={onActorEpochsChange}
        batchSize={batchSize}
        onBatchSizeChange={onBatchSizeChange}
        disabled={disabled}
      />
    );
  }

  return (
    <PolicyTrainingLoopLayout
      regionLabel={content.regionLabel}
      testId="act-td3-training-loop"
      trainingMode={resolvedMode}
      policyModel="act"
      policyLabel="ACT"
      policyTestId="act-td3-policy-stage"
      policyNode={(
        <ACTArchitectureDiagram
          trainableGroups={resolvedMode === 'critic' ? [] : trainableGroups}
          onChange={onTrainableGroupsChange || (() => {})}
          disabled={isPolicyDisabled}
          lockedGroups={lockedGroups}
        />
      )}
      datasets={datasets}
      capacityEpisodes={capacityEpisodes}
      onInspectDataset={onInspectDataset}
      trainingNode={trainingNode}
      replayStep={content.replayStep}
      returnStep={content.returnStep}
      connectorTestId="act-td3-loop-connectors"
      fitContent={fitContent}
      updated={updated}
    />
  );
}
