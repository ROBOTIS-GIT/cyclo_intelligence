// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import React from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import { tensorRtMode, isTensorRtMode } from '../utils/inferenceAcceleration';
import { defaultInferenceChunkSize } from '../utils/inferenceChunks';
import {
  markLocalTaskInfoEdited,
  selectInferenceTaskInfo,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';
import {
  supportsRltInference,
  supportsTtRtcInference,
} from '../constants/policyCapabilities';

// Inference models. Each option pairs a backend (orchestrator routing
// via TaskInfo.service_type) with a policy class (drives instruction
// visibility and future per-model UI knobs). LeRobot is the backend;
// ACT, SmolVLA, XVLA, Pi0, Pi0.5, and Diffusion are policy families that
// can be loaded by that backend when the selected checkpoint is compatible.
//
// Add an enabled entry once a policy is validated end-to-end. value is the
// composite key the dropdown stores; serviceType / policyType are the
// fields that get written into taskInfo on selection. comingSoon entries are
// displayed as disabled options only, so they do not affect runtime routing.
//
const MODEL_GROUPS = [
  {
    label: 'LeRobot',
    options: [
      { value: 'lerobot:act', label: 'ACT', serviceType: 'lerobot', policyType: 'act' },
      { value: 'lerobot:smolvla', label: 'SmolVLA', serviceType: 'lerobot', policyType: 'smolvla' },
      { value: 'lerobot:xvla', label: 'XVLA', serviceType: 'lerobot', policyType: 'xvla' },
      { value: 'lerobot:pi0', label: 'Pi0', serviceType: 'lerobot', policyType: 'pi0' },
      { value: 'lerobot:pi05', label: 'Pi0.5', serviceType: 'lerobot', policyType: 'pi05' },
      { value: 'lerobot:diffusion', label: 'Diffusion', serviceType: 'lerobot', policyType: 'diffusion' },
      {
        value: 'lerobot:multi_task_dit',
        label: 'Diffusion Transformer (Flow)',
        serviceType: 'lerobot',
        policyType: 'multi_task_dit',
      },
    ],
  },
  {
    label: 'GR00T',
    options: [
      { value: 'groot:n17', label: 'N1.7', serviceType: 'groot', policyType: 'n17' },
    ],
  },
  {
    label: 'Coming Soon',
    options: [
      {
        value: 'future:greenvla',
        label: 'GreenVLA',
        serviceType: 'future',
        policyType: 'greenvla',
        comingSoon: true,
      },
      {
        value: 'future:openpi',
        label: 'OpenPI',
        serviceType: 'future',
        policyType: 'openpi',
        comingSoon: true,
      },
      {
        value: 'future:rldx1',
        label: 'RLDX-1',
        serviceType: 'future',
        policyType: 'rldx1',
        comingSoon: true,
      },
    ],
  },
];

const OFFICIAL_MODEL_GROUPS = [
  {
    label: 'LeRobot',
    options: [
      { value: 'lerobot:act', label: 'ACT', serviceType: 'lerobot', policyType: 'act' },
      { value: 'lerobot:diffusion', label: 'Diffusion', serviceType: 'lerobot', policyType: 'diffusion' },
      { value: 'lerobot:smolvla', label: 'SmolVLA', serviceType: 'lerobot', policyType: 'smolvla' },
      { value: 'lerobot:xvla', label: 'XVLA', serviceType: 'lerobot', policyType: 'xvla' },
      { value: 'lerobot:pi0', label: 'Pi0', serviceType: 'lerobot', policyType: 'pi0' },
      { value: 'lerobot:pi05', label: 'Pi0.5', serviceType: 'lerobot', policyType: 'pi05' },
      { value: 'lerobot:molmoact2', label: 'MolmoAct2', serviceType: 'lerobot', policyType: 'molmoact2' },
      { value: 'lerobot:vla_jepa', label: 'VLA-JEPA', serviceType: 'lerobot', policyType: 'vla_jepa' },
      { value: 'lerobot:fastwam', label: 'FastWAM', serviceType: 'lerobot', policyType: 'fastwam' },
    ],
  },
  {
    label: 'GR00T',
    options: [
      { value: 'groot:n17', label: 'N1.7', serviceType: 'groot', policyType: 'n17' },
    ],
  },
  {
    label: 'Coming Soon',
    options: [
      {
        value: 'future:greenvla',
        label: 'GreenVLA',
        serviceType: 'future',
        policyType: 'greenvla',
        comingSoon: true,
      },
      {
        value: 'future:openpi',
        label: 'OpenPI',
        serviceType: 'future',
        policyType: 'openpi',
        comingSoon: true,
      },
      {
        value: 'future:rldx1',
        label: 'RLDX-1',
        serviceType: 'future',
        policyType: 'rldx1',
        comingSoon: true,
      },
    ],
  },
];

export const MODEL_OPTIONS = MODEL_GROUPS.flatMap((group) => group.options);
const AVAILABLE_MODEL_OPTIONS = MODEL_OPTIONS.filter((opt) => !opt.comingSoon);
const DEFAULT = AVAILABLE_MODEL_OPTIONS[0];

const InferenceModelSelector = ({
  readonly = false,
  label = 'Model',
  variant = 'default',
  official = false,
}) => {
  const dispatch = useDispatch();
  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const serviceType = info.serviceType || DEFAULT.serviceType;
  const policyType = info.policyType || DEFAULT.policyType;
  const value = `${serviceType}:${policyType}`;
  const modelGroups = official ? OFFICIAL_MODEL_GROUPS : MODEL_GROUPS;
  const availableOptions = official
    ? OFFICIAL_MODEL_GROUPS.flatMap((group) => group.options).filter((option) => !option.comingSoon)
    : AVAILABLE_MODEL_OPTIONS;
  const isOfflineRL = variant === 'offlineRL';
  const classLabel = clsx(
    'w-28 flex-shrink-0 font-medium',
    isOfflineRL ? 'text-[10px] text-[#6e675c]' : 'text-sm text-gray-600'
  );

  const handleChange = (e) => {
    const sel = availableOptions.find((o) => o.value === e.target.value);
    if (!sel) return;
    dispatch(
      setInferenceTaskInfo({
        serviceType: sel.serviceType,
        policyType: sel.policyType,
        actionChunkSize: defaultInferenceChunkSize(sel.serviceType, sel.policyType),
        accelerationMode: isTensorRtMode(info.accelerationMode)
          ? (tensorRtMode(sel.serviceType, sel.policyType) || 'pytorch') : 'pytorch',
        accelerationEnginePath: '',
        actionRequestMode: supportsTtRtcInference(
          sel.serviceType,
          sel.policyType
        )
          ? (info.actionRequestMode || 'async')
          : (info.actionRequestMode === 'tt_rtc' ? 'async' : info.actionRequestMode),
        // Keep the selected bundle for quick return to N1.7, but never leave
        // RLT logically enabled on a policy that cannot consume it.
        rltEnabled: supportsRltInference(sel.serviceType, sel.policyType)
          ? Boolean(info.rltEnabled)
          : false,
        rltRobotOverride: supportsRltInference(sel.serviceType, sel.policyType)
          ? Boolean(info.rltRobotOverride)
          : false,
      })
    );
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  };

  return (
    <div className={clsx('flex', 'items-center', 'mb-2.5')}>
      <span className={classLabel}>{label}</span>
      <select
        className={clsx(
          'flex-1',
          'h-8',
          'px-2',
          isOfflineRL ? 'text-[11px] text-[#403b34]' : 'text-sm',
          'border',
          isOfflineRL ? 'border-[#d9d2c5]' : 'border-gray-300',
          isOfflineRL ? 'rounded-lg' : 'rounded-md',
          'focus:outline-none',
          'focus:ring-2',
          isOfflineRL ? 'focus:ring-[#c7beb0]' : 'focus:ring-blue-500',
          'focus:border-transparent',
          {
            [isOfflineRL
              ? 'cursor-not-allowed bg-[#ece8df] text-[#8f877b]'
              : 'bg-gray-100 cursor-not-allowed text-gray-500']: readonly,
            [isOfflineRL ? 'bg-[#fffefa]' : 'bg-white']: !readonly,
          }
        )}
        value={value}
        onChange={handleChange}
        disabled={readonly}
      >
        {!modelGroups.some((group) => group.options.some((option) => option.value === value)) && (
          <option value={value}>{MODEL_OPTIONS.find((option) => option.value === value)?.label || policyType}</option>
        )}
        {modelGroups.map((group) => (
          <optgroup key={group.label} label={group.label}>
            {group.options.map((opt) => (
              <option
                key={opt.value}
                value={opt.value}
                disabled={Boolean(opt.comingSoon)}
              >
                {opt.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
};

export default InferenceModelSelector;
