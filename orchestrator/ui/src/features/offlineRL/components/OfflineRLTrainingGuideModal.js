// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import {
  MdAccountTree,
  MdArrowBack,
  MdArrowForward,
  MdCheckCircle,
  MdDataObject,
  MdInfoOutline,
  MdModelTraining,
  MdOutlineDataset,
  MdPlayArrow,
} from 'react-icons/md';

const GUIDE_TABS = [
  {
    id: 'quick-start',
    label: 'Quick Start',
    detail: 'End-to-end workflow',
    icon: MdPlayArrow,
  },
  {
    id: 'act',
    label: 'ACT',
    detail: 'IL · Critic · TD3',
    icon: MdAccountTree,
  },
  {
    id: 'diffusion',
    label: 'Diffusion Policy',
    detail: 'IL · Critic · Flow-SDE PPO',
    icon: MdDataObject,
  },
  {
    id: 'groot',
    label: 'GR00T',
    detail: 'RLT integration',
    icon: MdModelTraining,
  },
  {
    id: 'pi05',
    label: 'Pi0.5',
    detail: 'Architecture preview',
    icon: MdOutlineDataset,
  },
];

const STATUS_STYLES = {
  ready: 'border-[#b8c9b6] bg-[#e5eee3] text-[#4f6b55]',
  online: 'border-[#b8c8d7] bg-[#e7eef4] text-[#526b80]',
  staged: 'border-[#ddcba6] bg-[#f5eddc] text-[#876d3e]',
  planned: 'border-[#d8d1c5] bg-[#efebe3] text-[#7d7569]',
};

const MODEL_GUIDES = {
  act: {
    testId: 'training-guide-act',
    title: 'ACT',
    subtitle: 'Action Chunking Transformer',
    status: 'Ready',
    tone: 'ready',
    summary: 'ACT learns a 30-step action chunk from three synchronized camera views and robot state. Cyclo connects supervised IL, independent critic warm-up, and offline TD3 to the same policy lineage.',
    methods: ['IL', 'Critic', 'TD3'],
    flow: ['3 RGB images + robot state', 'Visual / state encoders', 'Action Module', '30-step action chunk'],
    contract: [
      ['Observation', 'Head, left-wrist, and right-wrist RGB images with robot state.'],
      ['Action', 'A 30-step chunk by default. IL exposes the horizon, but the checkpoint and runtime contract must match.'],
      ['Data', 'LeRobot v3.0 demonstrations. Critic and TD3 require success/fail episode outcomes.'],
      ['TD3 critic source', 'Automatic priority: resume checkpoint → previous round → policy warm-up → random initialization.'],
      ['Execution', 'The current showroom control path consumes actions at the configured 15 Hz contract.'],
    ],
    recipes: [
      {
        name: 'Imitation Learning',
        status: 'Ready',
        tone: 'ready',
        when: 'Create the Epoch 0 policy or continue supervised fine-tuning.',
        data: 'Demonstrations; outcome labels are not required.',
        configure: 'Trainable ACT blocks, steps, batch size, save frequency, and action-chunk horizon.',
        output: 'An ACT pretrained_model checkpoint ready for inference or later RL.',
      },
      {
        name: 'Critic Warm-up',
        status: 'Ready',
        tone: 'ready',
        when: 'Initialize chunk Q-functions before changing the actor.',
        data: 'Labeled replay with at least one success and one failure.',
        configure: 'Critic updates and batch size. The ACT actor remains frozen.',
        output: 'A critic checkpoint attached to the selected ACT policy lineage.',
      },
      {
        name: 'TD3',
        status: 'Ready',
        tone: 'ready',
        when: 'Improve the converged ACT actor from offline success/failure replay.',
        data: 'Selected labeled LeRobot v3.0 datasets.',
        configure: 'TD3 or TD3-BC loss, critic/actor epochs, batch size, and trainable ACT blocks. Cyclo selects the critic source automatically using the documented priority.',
        output: 'Updated ACT policy, twin critics, optimizer state, and the next RL epoch.',
      },
    ],
    steps: [
      'Select ACT and the base Model path in Inference Settings.',
      'Deploy one or more compatible LeRobot v3.0 datasets into the Training Replay Buffer.',
      'Choose IL, Critic, or RL. For RL, select TD3 and its loss option.',
      'Set trainable blocks and training settings, then verify that Start Training is enabled.',
      'Monitor progress and losses; deploy the completed policy only after inference validation.',
    ],
    monitor: [
      'IL: total, L1, and KL losses should stabilize without diverging.',
      'Critic/TD3: inspect critic and actor loss together; neither alone is a success metric.',
      'Compare pre/post-policy rollout video and episode success rate before deployment.',
      'Keep training state and critic artifacts for cumulative replay rounds.',
    ],
    limitations: [
      'TD3 is offline in this workflow; it does not collect fresh on-policy transitions while training.',
      'Changing the action horizon or action ordering without a matching dataset/runtime contract invalidates the checkpoint.',
      'TD3-BC clones success behavior only; failed actions still train the critics but are not behavior-cloned.',
    ],
  },
  diffusion: {
    testId: 'training-guide-diffusion',
    title: 'Diffusion Policy',
    subtitle: 'MultiTaskDiT · Flow-Matching action policy',
    status: 'Ready',
    tone: 'ready',
    summary: 'The current Diffusion Policy freezes observation conditioning and trains a Flow-Matching DiT action module. It supports supervised flow-matching IL, an independent offline value-critic workflow, and online Flow-SDE PPO.',
    methods: ['IL', 'Critic', 'Flow-SDE PPO'],
    flow: ['3 RGB images + task + 22D state', 'Frozen conditioning', 'Flow-Matching DiT', '16 × 22D action chunk'],
    contract: [
      ['Observation', 'Three RGB views, a task instruction, and a 22D robot state.'],
      ['Action', 'A fixed 16 × 22D chunk for the current MultiTaskDiT checkpoint.'],
      ['IL data', 'LeRobot v3.0 demonstrations; success/fail outcomes are optional.'],
      ['Critic data', 'Checked LeRobot v3.0 replay with success/fail outcomes, the selected policy checkpoint, and the same task instruction used by PPO.'],
      ['PPO rollout', 'Live simulator transitions with action-step ACK and a terminal success/fail outcome.'],
    ],
    recipes: [
      {
        name: 'Flow-Matching IL',
        status: 'Ready',
        tone: 'ready',
        when: 'Create or refine the initial diffusion action policy from demonstrations.',
        data: 'Synchronized images, state, task instruction, and recorded action chunks.',
        configure: 'Steps, batch size, and save frequency. The 16-step horizon is fixed.',
        output: 'A MultiTaskDiT pretrained_model checkpoint for standard inference or PPO.',
      },
      {
        name: 'Offline Value Critic',
        status: 'Ready',
        tone: 'ready',
        when: 'Warm up the value function independently before starting online PPO.',
        data: 'Checked LeRobot v3.0 replay with both successful and failed episodes.',
        configure: 'Select Critic, then set optimizer steps, batch size, value learning rate, discount, policy checkpoint, and task instruction.',
        output: 'A value-warm-up bundle bound to the exact Diffusion Policy checkpoint and task instruction.',
      },
      {
        name: 'Flow-SDE PPO',
        status: 'Online',
        tone: 'online',
        when: 'Fine-tune the current policy using fresh simulator rollouts.',
        data: 'On-policy latent trajectory, log-probability, reward/outcome, and done state.',
        configure: 'Choose RL → PPO, verify the task instruction and rollout transport, then start the online update. Critic training is not embedded in PPO.',
        output: 'Updated DiT actor, value state, optimizer state, and PPO training state. A compatible completed critic bundle is reused automatically; otherwise PPO initializes a fresh value head.',
      },
    ],
    steps: [
      'Select Diffusion Transformer in Inference Settings and verify the task instruction.',
      'For IL or Critic, deploy compatible LeRobot v3.0 data. Critic replay must include Success and Fail outcomes.',
      'Choose Critic to train the value head independently. Its policy checkpoint and task instruction define bundle compatibility.',
      'Choose RL → PPO and start cyclo_lab with action-step transport. Cyclo automatically reuses a compatible completed critic bundle.',
      'Provide a terminal Success, Fail, or Cancel outcome for each PPO episode, then validate repeated rollouts before Policy Deploy.',
    ],
    monitor: [
      'IL: monitor flow loss and compare sampled action chunks with demonstrations.',
      'Critic: monitor value loss and confirm the completed bundle reports the selected policy checkpoint and task instruction.',
      'PPO: success rate is primary; actor loss, value loss, KL, and clip fraction are diagnostics.',
      'A missing action-step ACK means the simulator did not acknowledge the requested action.',
      'Preserve value/optimizer state when continuing the next PPO update.',
    ],
    limitations: [
      'PPO is on-policy. A previously converted LeRobot dataset cannot recreate the old-policy log-probability required for a valid PPO ratio.',
      'A critic bundle is reused only when both its policy checkpoint and task instruction match the PPO request; stale bundles are not attached.',
      'The current action horizon is fixed at 16 steps by the model contract.',
      'Real-robot PPO requires a separate safety and rollout qualification process.',
    ],
  },
  groot: {
    testId: 'training-guide-groot',
    title: 'GR00T N1.7',
    subtitle: 'Frozen VLA policy with an RLT action adapter',
    status: 'Staged',
    tone: 'staged',
    summary: 'Cyclo can inspect the GR00T + RLT architecture and route a qualified RLT bundle during inference. The RLT training request is intentionally not connected to Start Training yet.',
    methods: ['RLT inference', 'Training staged'],
    flow: ['3 RGB images + state + language', 'Frozen GR00T N1.7', 'RL Token Encoder + Action MLP', '10 × 19 @ 15 Hz'],
    contract: [
      ['Base policy', 'GR00T N1.7 remains frozen in the current RLT design.'],
      ['Reference action', 'The frozen Flow-Matching head supplies a 16 × 19 reference chunk.'],
      ['RLT action', 'RL token features and the reference action produce a 10 × 19 chunk at 15 Hz.'],
      ['Deployment', 'RLT routing requires a compatible, deployment-qualified bundle.'],
    ],
    recipes: [
      {
        name: 'Cyclo RLT',
        status: 'Staged',
        tone: 'staged',
        when: 'Use only for architecture inspection and qualified-bundle inference today.',
        data: 'Future training requires GR00T token features, reference actions, rewards, and transitions.',
        configure: 'RL Token Encoder / Action MLP trainability is visible, but Start Training is not wired.',
        output: 'A future bundle must include adapter weights and its base-policy compatibility metadata.',
      },
      {
        name: 'RLinf GR00T PPO',
        status: 'Reference',
        tone: 'online',
        when: 'External reference for simulator-based actor-critic training; it is not the Cyclo RLT backend.',
        data: 'Multi-view RGB, proprioception, language prompt, simulator reward, and online rollout.',
        configure: 'The maintained N1.7 example uses a value head, 16 action chunks, and four denoising steps.',
        output: 'GR00T actor/value checkpoints produced by the separate RLinf training stack.',
      },
    ],
    steps: [
      'Select GR00T N1.7 and a compatible base policy path in Inference Settings.',
      'Choose RLT and inspect the frozen GR00T / adapter boundary.',
      'For inference testing, select a qualified RLT Bundle Path before starting inference.',
      'Switch between VLA Action and RLT Action only through the runtime action router.',
      'Do not expect Start Training to launch GR00T RLT until the trainer/checkpoint contract is connected.',
    ],
    monitor: [
      'Compare VLA Action and RLT Action rollouts from the same initial condition.',
      'Watch action continuity, oscillation, success rate, and fallback routing.',
      'Record the base-policy identity and adapter bundle together for reproducibility.',
      'For external RLinf PPO, success rate is primary and actor/value/KL metrics are diagnostic.',
    ],
    limitations: [
      'Cyclo GR00T RLT training is not connected; the current Training page is a staged architecture view.',
      'RLinf GR00T PPO and Cyclo RLT are different algorithms and runtime contracts.',
      'A GR00T-only RLT bundle is not automatically compatible with Pi0.5 or another base checkpoint.',
    ],
    reference: {
      label: 'RLinf: RL on GR00T Models',
      href: 'https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/gr00t.html',
      detail: 'Official external example covering SFT cold start, online PPO, launch configuration, metrics, and evaluation. Use it as a design reference, not as evidence that Cyclo RLT training is connected.',
    },
  },
  pi05: {
    testId: 'training-guide-pi05',
    title: 'Pi0.5',
    subtitle: 'Vision-language-action policy preview',
    status: 'Preview',
    tone: 'planned',
    summary: 'Pi0.5 is currently a topology and workflow preview in Cyclo. Its training adapter, trainability contract, checkpoint loader, and qualified RLT bundle are not connected.',
    methods: ['IL preview', 'RLT preview'],
    flow: ['Images + task + robot state', 'Vision-language encoder', 'Action conditioning', 'Flow-matched action chunk'],
    contract: [
      ['Observation', 'Multi-view RGB, language instruction, and robot proprioception.'],
      ['Action', 'Flow-matched continuous action chunk; the Cyclo deployment contract is not finalized.'],
      ['Training', 'No Pi0.5 trainer or Frozen/Fire submission contract is connected.'],
      ['Deployment', 'Requires a Pi0.5-specific runtime and compatible adapter bundle.'],
    ],
    recipes: [
      {
        name: 'Imitation Learning',
        status: 'Preview',
        tone: 'planned',
        when: 'Planned for supervised adaptation after the model/data adapter is defined.',
        data: 'Future data must match Pi0.5 image, state, language, and action normalization contracts.',
        configure: 'Trainable modules, chunk horizon, optimizer, and checkpoint format are not exposed yet.',
        output: 'No Cyclo Pi0.5 training checkpoint is currently produced.',
      },
      {
        name: 'RLT',
        status: 'Preview',
        tone: 'planned',
        when: 'Planned only after token-feature extraction and a Pi0.5-compatible action adapter exist.',
        data: 'Future transitions require reference actions, token features, rewards, and terminal state.',
        configure: 'The UI does not submit Pi0.5 RLT parameters today.',
        output: 'A future bundle must identify the Pi0.5 base policy and adapter contract.',
      },
    ],
    steps: [
      'Use the Pi0.5 tab to review the intended policy and RLT boundaries.',
      'Define and validate the Pi0.5 observation/action normalization adapter.',
      'Connect the trainability, optimizer, checkpoint, and resume contracts.',
      'Add a smoke test that verifies the expected action chunk from three cameras, state, and language.',
      'Enable Start Training only after inference and checkpoint round-trip validation pass.',
    ],
    monitor: [
      'No Pi0.5 training metrics are emitted by Cyclo yet.',
      'First validate tensor shapes, normalization, and deterministic checkpoint reload.',
      'Then add loss, rollout success, and before/after video comparisons.',
    ],
    limitations: [
      'The current Pi0.5 page must not be interpreted as a working trainer.',
      'GR00T RLT weights cannot be reused without a Pi0.5-specific compatibility layer.',
      'Start Training remains disabled until backend qualification is complete.',
    ],
  },
};

function StatusPill({ tone = 'ready', children }) {
  return (
    <span className={`shrink-0 rounded-full border px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.07em] ${STATUS_STYLES[tone]}`}>
      {children}
    </span>
  );
}

function GuideCard({ title, subtitle, status, tone, children, className = '' }) {
  return (
    <article className={`flex min-w-0 flex-col rounded-xl border border-[#ded8cc] bg-[#fbfaf6] p-4 shadow-[0_3px_10px_rgba(69,61,47,0.035)] ${className}`}>
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[13px] font-semibold text-[#39352e]">{title}</h3>
          {subtitle && (
            <p className="mt-0.5 text-[10px] leading-relaxed text-[#8a8276]">
              {subtitle}
            </p>
          )}
        </div>
        {status && <StatusPill tone={tone}>{status}</StatusPill>}
      </div>
      <div className="mt-3 min-h-0 flex-1">{children}</div>
    </article>
  );
}

function NetworkFlow({ nodes }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" aria-label={nodes.join(' to ')}>
      {nodes.map((node, index) => (
        <React.Fragment key={node}>
          <span className="rounded-lg border border-[#d8d1c5] bg-white px-2.5 py-2 text-[10px] font-semibold text-[#514b42]">
            {node}
          </span>
          {index < nodes.length - 1 && (
            <MdArrowForward className="shrink-0 text-[#aaa295]" size={14} aria-hidden="true" />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

function BulletList({ items }) {
  return (
    <ul className="space-y-2 text-[10px] leading-relaxed text-[#625c52]">
      {items.map((item) => (
        <li key={item} className="flex items-start gap-2">
          <MdCheckCircle className="mt-0.5 shrink-0 text-[#69866f]" size={13} aria-hidden="true" />
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function ContractGrid({ items }) {
  return (
    <dl className="divide-y divide-[#ece6dc] rounded-xl border border-[#e1dbd0] bg-white">
      {items.map(([label, value]) => (
        <div key={label} className="grid gap-1 px-3 py-2.5 sm:grid-cols-[110px_1fr] sm:gap-3">
          <dt className="text-[9px] font-bold uppercase tracking-[0.08em] text-[#8c8377]">{label}</dt>
          <dd className="text-[10px] leading-relaxed text-[#5f584e]">{value}</dd>
        </div>
      ))}
    </dl>
  );
}

function TrainingSteps({ items }) {
  return (
    <ol className="grid gap-2 lg:grid-cols-5">
      {items.map((item, index) => (
        <li key={item} className="rounded-xl border border-[#ded8cc] bg-white p-3">
          <div className="font-mono text-[9px] font-bold text-[#69866f]">
            STEP {String(index + 1).padStart(2, '0')}
          </div>
          <p className="mt-1.5 text-[9px] leading-relaxed text-[#6f685d]">{item}</p>
        </li>
      ))}
    </ol>
  );
}

function QuickStartPanel() {
  const steps = [
    'Select the policy and base checkpoint in Inference Settings.',
    'Collect or convert data, then deploy the selected datasets to Training.',
    'Choose IL, Critic, or RL and an algorithm enabled for the selected model.',
    'Configure trainable blocks, steps or epochs, batch size, and the action contract.',
    'Train, validate with success rate and rollout video, then deploy the policy.',
  ];
  const compatibility = [
    ['ACT', 'Ready', 'Ready', 'TD3', '—', '—'],
    ['Diffusion Policy', 'Ready', 'Ready', '—', 'Flow-SDE', '—'],
    ['GR00T', 'Preview', '—', '—', 'External ref.', 'Staged'],
    ['Pi0.5', 'Preview', '—', '—', '—', 'Preview'],
  ];

  return (
    <div className="space-y-4" data-testid="training-guide-quick-start">
      <div className="rounded-xl border border-[#cfd8cd] bg-[#edf3ec] px-4 py-3">
        <div className="flex items-start gap-2.5">
          <MdInfoOutline className="mt-0.5 shrink-0 text-[#5d7863]" size={17} aria-hidden="true" />
          <div>
            <h3 className="text-xs font-semibold text-[#3d5743]">Start from the current backend contract</h3>
            <p className="mt-1 text-[10px] leading-relaxed text-[#607064]">
              A disabled combination is a preview, not a runnable trainer. Use the model tabs to check data, action, label, and checkpoint requirements before starting.
            </p>
          </div>
        </div>
      </div>

      <GuideCard title="Training workflow" subtitle="The same five decisions apply to every connected model.">
        <TrainingSteps items={steps} />
      </GuideCard>

      <GuideCard title="Compatibility" subtitle="Availability in the current Cyclo Training Pipeline.">
        <div className="overflow-x-auto rounded-xl border border-[#e1dbd0] bg-white">
          <table className="w-full min-w-[650px] text-left text-[10px] text-[#625c52]">
            <thead className="bg-[#f1ede5] text-[9px] uppercase tracking-[0.08em] text-[#81796e]">
              <tr>
                {['Policy', 'IL', 'Critic', 'TD3', 'PPO', 'RLT'].map((label) => (
                  <th key={label} scope="col" className="px-3 py-2.5 font-bold">{label}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-[#ece6dc]">
              {compatibility.map(([model, ...cells]) => (
                <tr key={model}>
                  <th scope="row" className="px-3 py-2.5 font-semibold text-[#3f3a33]">{model}</th>
                  {cells.map((cell, index) => (
                    <td key={`${model}-${index}`} className="px-3 py-2.5">{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GuideCard>

      <div className="grid gap-3 lg:grid-cols-3">
        <GuideCard title="IL data" status="Outcome optional" tone="ready">
          <p className="text-[10px] leading-relaxed text-[#625c52]">
            Demonstrations train the policy directly. ACT and Diffusion Policy IL do not require episode outcome labels.
          </p>
        </GuideCard>
        <GuideCard title="Offline value / TD3" status="Outcome required" tone="staged">
          <p className="text-[10px] leading-relaxed text-[#625c52]">
            ACT critic warm-up, Diffusion Policy value-critic training, and TD3 require labeled replay with both successful and failed episodes.
          </p>
        </GuideCard>
        <GuideCard title="Online PPO" status="Live rollout" tone="online">
          <p className="text-[10px] leading-relaxed text-[#625c52]">
            PPO needs current-policy simulator rollouts, action-step acknowledgement, and a terminal outcome for each episode.
          </p>
        </GuideCard>
      </div>
    </div>
  );
}

function ModelGuidePanel({ guide }) {
  return (
    <div className="space-y-4" data-testid={guide.testId}>
      <section className="rounded-2xl border border-[#d9d2c5] bg-[#fbfaf6] p-5 shadow-[0_6px_20px_rgba(69,61,47,0.05)]">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[9px] font-bold uppercase tracking-[0.14em] text-[#91897d]">Model guide</p>
            <h2 className="mt-1 text-xl font-semibold tracking-[-0.02em] text-[#302d28]">{guide.title}</h2>
            <p className="mt-1 text-[11px] text-[#7e766b]">{guide.subtitle}</p>
          </div>
          <StatusPill tone={guide.tone}>{guide.status}</StatusPill>
        </div>
        <p className="mt-4 max-w-4xl text-[11px] leading-5 text-[#5f584e]">{guide.summary}</p>
        <div className="mt-4 flex flex-wrap gap-2">
          {guide.methods.map((method) => (
            <span key={method} className="rounded-full border border-[#d4cec2] bg-white px-3 py-1 text-[9px] font-semibold text-[#5f584e]">
              {method}
            </span>
          ))}
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <GuideCard title="Policy architecture" subtitle="Observation-to-action contract used by this guide.">
          <NetworkFlow nodes={guide.flow} />
        </GuideCard>
        <GuideCard title="Input / output contract" subtitle="Verify these fields before training or deployment.">
          <ContractGrid items={guide.contract} />
        </GuideCard>
      </div>

      <GuideCard title="Training recipes" subtitle="Choose a recipe by data source and optimization objective.">
        <div className={`grid gap-3 ${guide.recipes.length >= 3 ? 'xl:grid-cols-3' : 'xl:grid-cols-2'}`}>
          {guide.recipes.map((recipe) => (
            <article key={recipe.name} className="rounded-xl border border-[#ded8cc] bg-white p-3.5">
              <div className="flex items-center justify-between gap-2">
                <h4 className="text-[11px] font-semibold text-[#3f3a33]">{recipe.name}</h4>
                <StatusPill tone={recipe.tone}>{recipe.status}</StatusPill>
              </div>
              <dl className="mt-3 space-y-2 text-[9px] leading-relaxed text-[#6f685d]">
                {[
                  ['When', recipe.when],
                  ['Data', recipe.data],
                  ['Configure', recipe.configure],
                  ['Output', recipe.output],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt className="font-bold uppercase tracking-[0.07em] text-[#90877b]">{label}</dt>
                    <dd className="mt-0.5">{value}</dd>
                  </div>
                ))}
              </dl>
            </article>
          ))}
        </div>
      </GuideCard>

      <GuideCard title="Run in PLAYGROUND" subtitle="Follow the controls in this order.">
        <TrainingSteps items={guide.steps} />
      </GuideCard>

      <div className="grid gap-4 xl:grid-cols-2">
        <GuideCard title="Monitor & validate" subtitle="Training loss is diagnostic; task performance is the decision metric.">
          <BulletList items={guide.monitor} />
        </GuideCard>
        <GuideCard title="Limitations & safety" subtitle="Current implementation boundary.">
          <BulletList items={guide.limitations} />
        </GuideCard>
      </div>

      {guide.reference && (
        <GuideCard title="External reference" subtitle="Related implementation documentation; not the Cyclo backend.">
          <a
            href={guide.reference.href}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-[11px] font-semibold text-[#4e6d56] underline decoration-[#9fb2a2] underline-offset-4 hover:text-[#344d3a]"
          >
            {guide.reference.label}
            <MdArrowForward size={14} aria-hidden="true" />
          </a>
          <p className="mt-2 text-[10px] leading-relaxed text-[#6f685d]">{guide.reference.detail}</p>
        </GuideCard>
      )}
    </div>
  );
}

export default function OfflineRLTrainingGuideModal({ open, onBack }) {
  const [activeTab, setActiveTab] = useState('quick-start');
  const backButtonRef = useRef(null);
  const dialogRef = useRef(null);
  const tabRefs = useRef([]);
  const previouslyFocusedRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;
    setActiveTab('quick-start');
    previouslyFocusedRef.current = document.activeElement;
    backButtonRef.current?.focus();
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    return () => {
      document.body.style.overflow = previousOverflow;
      previouslyFocusedRef.current?.focus?.();
      previouslyFocusedRef.current = null;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onBack?.();
        return;
      }
      if (event.key !== 'Tab') return;
      const focusable = Array.from(dialogRef.current?.querySelectorAll(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), '
        + 'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      ) || []);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onBack, open]);

  if (!open || typeof document === 'undefined') return null;

  const handleBackdropMouseDown = (event) => {
    if (event.target === event.currentTarget) onBack?.();
  };

  const handleTabKeyDown = (event, currentIndex) => {
    let nextIndex = null;
    if (['ArrowDown', 'ArrowRight'].includes(event.key)) {
      nextIndex = (currentIndex + 1) % GUIDE_TABS.length;
    } else if (['ArrowUp', 'ArrowLeft'].includes(event.key)) {
      nextIndex = (currentIndex - 1 + GUIDE_TABS.length) % GUIDE_TABS.length;
    } else if (event.key === 'Home') {
      nextIndex = 0;
    } else if (event.key === 'End') {
      nextIndex = GUIDE_TABS.length - 1;
    }
    if (nextIndex === null) return;
    event.preventDefault();
    setActiveTab(GUIDE_TABS[nextIndex].id);
    tabRefs.current[nextIndex]?.focus();
  };

  const activeTabDefinition = GUIDE_TABS.find((tab) => tab.id === activeTab) || GUIDE_TABS[0];
  const ActiveIcon = activeTabDefinition.icon;
  const activePanel = activeTab === 'quick-start'
    ? <QuickStartPanel />
    : <ModelGuidePanel guide={MODEL_GUIDES[activeTab]} />;

  return createPortal(
    <div
      className="fixed inset-0 z-[130] flex items-center justify-center bg-black/50 p-3"
      data-testid="offline-rl-training-guide-backdrop"
      onMouseDown={handleBackdropMouseDown}
    >
      <section
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="offline-rl-training-guide-title"
        aria-describedby="offline-rl-training-guide-description"
        className="flex h-[88vh] w-[92vw] max-w-[1480px] flex-col overflow-hidden rounded-2xl border border-[#d9d2c5] bg-[#f7f4ed] shadow-2xl"
      >
        <header className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-5 py-4">
          <button
            ref={backButtonRef}
            type="button"
            onClick={() => onBack?.()}
            aria-label="Back to Training"
            className="flex h-9 items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-white px-3 text-xs font-semibold text-[#514b42] transition-colors hover:bg-[#f1ede4] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2"
          >
            <MdArrowBack size={16} aria-hidden="true" />
            Back
          </button>
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]">
            <MdModelTraining size={19} aria-hidden="true" />
          </span>
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">Policy Workflow</p>
            <h2 id="offline-rl-training-guide-title" className="truncate text-sm font-semibold text-[#292720]">Training Guide</h2>
            <p id="offline-rl-training-guide-description" className="sr-only">
              Model-specific setup, training, validation, and deployment guidance for the current Training Pipeline.
            </p>
          </div>
        </header>

        <div className="flex min-h-0 flex-1">
          <aside className="w-[216px] shrink-0 border-r border-[#ded8cc] bg-[#f1ede5] p-3">
            <div className="px-2 pb-2 text-[9px] font-bold uppercase tracking-[0.13em] text-[#968e82]">
              Guide sections
            </div>
            <div
              role="tablist"
              aria-label="Training Guide sections"
              aria-orientation="vertical"
              className="space-y-1"
            >
              {GUIDE_TABS.map((tab, index) => {
                const selected = activeTab === tab.id;
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    ref={(node) => { tabRefs.current[index] = node; }}
                    id={`offline-rl-training-guide-tab-${tab.id}`}
                    type="button"
                    role="tab"
                    tabIndex={selected ? 0 : -1}
                    aria-label={tab.label}
                    aria-selected={selected}
                    aria-controls="offline-rl-training-guide-panel"
                    onClick={() => setActiveTab(tab.id)}
                    onKeyDown={(event) => handleTabKeyDown(event, index)}
                    className={selected
                      ? 'flex w-full items-center gap-2.5 rounded-xl border border-[#c9d5c8] bg-white px-3 py-2.5 text-left text-[#405d47] shadow-sm focus:outline-none focus:ring-2 focus:ring-[#879b83]'
                      : 'flex w-full items-center gap-2.5 rounded-xl border border-transparent px-3 py-2.5 text-left text-[#716a60] hover:bg-[#e8e2d8] focus:outline-none focus:ring-2 focus:ring-[#a3aaa0]'}
                  >
                    <span className={selected
                      ? 'grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-[#e5eee3] text-[#55705b]'
                      : 'grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-[#e6e0d6] text-[#81796e]'}
                    >
                      <Icon size={16} aria-hidden="true" />
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-[11px] font-semibold">{tab.label}</span>
                      <span className="mt-0.5 block truncate text-[8px] opacity-70">{tab.detail}</span>
                    </span>
                  </button>
                );
              })}
            </div>
          </aside>

          <div
            id="offline-rl-training-guide-panel"
            role="tabpanel"
            aria-labelledby={`offline-rl-training-guide-tab-${activeTab}`}
            className="min-h-0 min-w-0 flex-1 overflow-y-auto overscroll-contain p-5"
          >
            <div className="mb-4 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#91897d]">
              <ActiveIcon size={15} aria-hidden="true" />
              {activeTabDefinition.label}
            </div>
            {activePanel}
          </div>
        </div>
      </section>
    </div>,
    document.body
  );
}
