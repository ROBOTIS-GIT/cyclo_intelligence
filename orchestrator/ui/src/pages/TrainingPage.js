// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Kiwoong Park

import { useState } from 'react';
import clsx from 'clsx';
import {
  MdCheck,
  MdContentCopy,
  MdFolder,
  MdInfoOutline,
  MdOpenInNew,
  MdSchool,
} from 'react-icons/md';

const WORKFLOW_STEPS = [
  {
    title: 'Prepare dataset',
    body: 'Record and convert demonstrations in Data Tools. Use the converted LeRobot dataset as training input.',
  },
  {
    title: 'Use your GPU machine',
    body: 'Use the pinned LeRobot or Isaac-GR00T submodule from cyclo_intelligence, then install its training dependencies.',
  },
  {
    title: 'Run CLI training',
    body: 'Start from the template below, then adjust policy, steps, batch size, and workers for the hardware.',
  },
  {
    title: 'Load checkpoint',
    body: 'Copy the checkpoint into /workspace/model/<backend>/, then select it from the Inference page.',
  },
];

const TRAINING_GUIDES = [
  {
    key: 'lerobot',
    title: 'LeRobot',
    subtitle: 'ACT, Diffusion, SmolVLA, XVLA, Pi0, Pi0.5',
    badge: 'Inference backend: lerobot',
    setup:
      'Use the LeRobot submodule in cyclo_intelligence. Pass the converted dataset as <dataset_root>.',
    handoff:
      'Copy the run folder, or the checkpoint folder that contains pretrained_model, into the LeRobot model folder.',
    modelPath: '/workspace/model/lerobot/<run_name>',
    inferencePath:
      '/workspace/model/lerobot/<run_name>/checkpoints/<step>/pretrained_model',
    note:
      'Inference expects the selected path to include the exported LeRobot pretrained_model directory.',
    command: `lerobot-train \\
  --dataset.repo_id=<user>/<dataset_name> \\
  --dataset.root=<dataset_root> \\
  --policy.type=act \\
  --policy.device=cuda \\
  --output_dir=<output_dir>/<run_name> \\
  --job_name=<run_name> \\
  --batch_size=8 \\
  --steps=100000 \\
  --save_freq=20000 \\
  --wandb.enable=false`,
    links: [
      {
        label: 'LeRobot documentation',
        href: 'https://huggingface.co/docs/lerobot',
      },
      {
        label: 'Policy guides',
        href: 'https://huggingface.co/docs/lerobot/bring_your_own_policies',
      },
      {
        label: 'LeRobot GitHub',
        href: 'https://github.com/huggingface/lerobot',
      },
    ],
  },
  {
    key: 'groot',
    title: 'GR00T N1.7',
    subtitle: 'NVIDIA GR00T fine-tuning',
    badge: 'Inference backend: groot',
    setup:
      'Use the Isaac-GR00T submodule in cyclo_intelligence. Select the matching robot files from examples/CYCLO/<robot_name>/.',
    handoff:
      'Copy the fine-tuned GR00T checkpoint directory into the GR00T model folder.',
    modelPath: '/workspace/model/groot/<run_name>',
    inferencePath: '/workspace/model/groot/<run_name>/checkpoint-<step>',
    note:
      'The template below is a smoke test. Increase max_steps, global_batch_size, and workers after it runs successfully.',
    command: `CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \\
  --base_model_path nvidia/GR00T-N1.7-3B \\
  --dataset_path <dataset_path> \\
  --embodiment_tag NEW_EMBODIMENT \\
  --modality_config_path examples/CYCLO/<robot_name>/<robot_config.py> \\
  --num_gpus 1 \\
  --output_dir <output_dir>/<run_name> \\
  --max_steps 1 \\
  --global_batch_size 1 \\
  --dataloader_num_workers 0`,
    links: [
      {
        label: 'Cyclo GR00T configs',
        href: 'https://github.com/ROBOTIS-GIT/Isaac-GR00T-n1.7/tree/main/examples/CYCLO',
      },
      {
        label: 'GR00T fine-tuning guide',
        href: 'https://github.com/ROBOTIS-GIT/Isaac-GR00T-n1.7/blob/main/getting_started/finetune_new_embodiment.md',
      },
      {
        label: 'Isaac-GR00T GitHub',
        href: 'https://github.com/ROBOTIS-GIT/Isaac-GR00T-n1.7',
      },
    ],
  },
];

function PathRow({ label, value }) {
  return (
    <div className="flex items-start gap-2 text-sm">
      <MdFolder className="mt-0.5 shrink-0 text-gray-400" size={17} />
      <div className="min-w-0">
        <div className="font-medium text-gray-600">{label}</div>
        <code className="mt-1 block break-all rounded-md bg-gray-100 px-2 py-1 text-xs text-gray-800">
          {value}
        </code>
      </div>
    </div>
  );
}

function CopyButton({ commandKey, command, copiedKey, setCopiedKey }) {
  const copied = copiedKey === commandKey;

  const handleCopy = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(command);
      } else {
        const textarea = document.createElement('textarea');
        textarea.value = command;
        textarea.setAttribute('readonly', '');
        textarea.style.position = 'fixed';
        textarea.style.top = '-1000px';
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setCopiedKey(commandKey);
      window.setTimeout(() => setCopiedKey(null), 1600);
    } catch (error) {
      console.error('Failed to copy training command:', error);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={clsx(
        'inline-flex h-8 items-center gap-1.5 rounded-md px-2.5 text-xs font-semibold transition-colors',
        copied
          ? 'bg-emerald-100 text-emerald-700'
          : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
      )}
      aria-label={`Copy ${commandKey} training command`}
      title="Copy command"
    >
      {copied ? <MdCheck size={16} /> : <MdContentCopy size={16} />}
      {copied ? 'Copied' : 'Copy'}
    </button>
  );
}

function TrainingGuideCard({ guide, copiedKey, setCopiedKey }) {
  return (
    <section className="flex min-w-0 flex-col gap-4 rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="text-xl font-semibold text-gray-900">{guide.title}</div>
          <div className="mt-1 text-sm text-gray-500">{guide.subtitle}</div>
        </div>
        <span className="w-fit shrink-0 whitespace-nowrap rounded-md bg-blue-50 px-2.5 py-1 text-xs font-semibold text-blue-700 sm:ml-auto">
          {guide.badge}
        </span>
      </div>

      <div className="grid gap-3 text-sm text-gray-600 lg:grid-cols-2">
        <div className="border-l-2 border-gray-200 pl-3">
          <div className="font-medium text-gray-700">Before training</div>
          <div className="mt-1 leading-relaxed">{guide.setup}</div>
        </div>
        <div className="border-l-2 border-gray-200 pl-3">
          <div className="font-medium text-gray-700">After training</div>
          <div className="mt-1 leading-relaxed">{guide.handoff}</div>
        </div>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <PathRow label="Copy into Cyclo" value={guide.modelPath} />
        <PathRow label="Select in Inference" value={guide.inferencePath} />
      </div>

      <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
        <div className="flex gap-2">
          <MdInfoOutline className="mt-0.5 shrink-0" size={17} />
          <span>{guide.note}</span>
        </div>
      </div>

      <div className="min-w-0 overflow-hidden rounded-lg border border-gray-200 bg-slate-950">
        <div className="flex items-center justify-between gap-2 border-b border-slate-800 px-3 py-2">
          <span className="text-xs font-semibold uppercase text-slate-300">
            CLI template
          </span>
          <CopyButton
            commandKey={guide.key}
            command={guide.command}
            copiedKey={copiedKey}
            setCopiedKey={setCopiedKey}
          />
        </div>
        <pre className="max-h-72 overflow-auto p-4 text-xs leading-relaxed text-slate-100">
          <code>{guide.command}</code>
        </pre>
      </div>

      <div className="flex flex-wrap gap-2">
        {guide.links.map((link) => (
          <a
            key={link.href}
            href={link.href}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-9 items-center gap-1.5 rounded-md border border-gray-200 bg-white px-3 text-sm font-medium text-gray-700 no-underline transition-colors hover:border-blue-300 hover:text-blue-700"
          >
            {link.label}
            <MdOpenInNew size={15} />
          </a>
        ))}
      </div>
    </section>
  );
}

export default function TrainingPage() {
  const [copiedKey, setCopiedKey] = useState(null);

  return (
    <div className="h-full w-full overflow-y-auto bg-gray-50">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-5 px-6 py-6">
        <section className="pt-1">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-sm font-semibold text-blue-700">
                <MdSchool size={20} />
                Training Guide
              </div>
              <h1 className="mt-2 text-2xl font-semibold text-gray-950">
                Train outside Cyclo, deploy through Inference
              </h1>
              <p className="mt-2 max-w-3xl text-sm leading-relaxed text-gray-600">
                Train with the LeRobot and Isaac-GR00T submodules included in
                cyclo_intelligence. These pinned versions match Cyclo Inference;
                newer upstream releases may require an Inference update first.
              </p>
            </div>
          </div>
        </section>

        <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
          {WORKFLOW_STEPS.map((step, index) => (
            <div
              key={step.title}
              className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm"
            >
              <div className="mb-2 flex h-7 w-7 items-center justify-center rounded-md bg-gray-900 text-sm font-semibold text-white">
                {index + 1}
              </div>
              <div className="text-sm font-semibold text-gray-900">
                {step.title}
              </div>
              <div className="mt-1 text-sm leading-relaxed text-gray-600">
                {step.body}
              </div>
            </div>
          ))}
        </section>

        <div className="grid gap-5 xl:grid-cols-2">
          {TRAINING_GUIDES.map((guide) => (
            <TrainingGuideCard
              key={guide.key}
              guide={guide}
              copiedKey={copiedKey}
              setCopiedKey={setCopiedKey}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
