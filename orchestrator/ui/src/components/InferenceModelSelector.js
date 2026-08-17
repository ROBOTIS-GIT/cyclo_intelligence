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
import {
  markLocalTaskInfoEdited,
  selectInferenceTaskInfo,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';

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

const classLabel = clsx(
  'text-sm', 'text-gray-600', 'w-28', 'flex-shrink-0', 'font-medium'
);

const InferenceModelSelector = ({ readonly = false }) => {
  const dispatch = useDispatch();
  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const serviceType = info.serviceType || DEFAULT.serviceType;
  const policyType = info.policyType || DEFAULT.policyType;
  const value = `${serviceType}:${policyType}`;

  const handleChange = (e) => {
    const sel = AVAILABLE_MODEL_OPTIONS.find((o) => o.value === e.target.value);
    if (!sel) return;
    dispatch(
      setInferenceTaskInfo({
        serviceType: sel.serviceType,
        policyType: sel.policyType,
        accelerationMode: sel.serviceType === 'groot'
          ? (info.accelerationMode || 'pytorch')
          : 'pytorch',
        accelerationEnginePath: sel.serviceType === 'groot'
          ? (info.accelerationEnginePath || '')
          : '',
      })
    );
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  };

  return (
    <div className={clsx('flex', 'items-center', 'mb-2.5')}>
      <span className={classLabel}>Model</span>
      <select
        className={clsx(
          'flex-1',
          'h-8',
          'px-2',
          'border',
          'border-gray-300',
          'rounded-md',
          'focus:outline-none',
          'focus:ring-2',
          'focus:ring-blue-500',
          'focus:border-transparent',
          {
            'bg-gray-100 cursor-not-allowed text-gray-500': readonly,
            'bg-white': !readonly,
          }
        )}
        value={value}
        onChange={handleChange}
        disabled={readonly}
      >
        {MODEL_GROUPS.map((group) => (
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
