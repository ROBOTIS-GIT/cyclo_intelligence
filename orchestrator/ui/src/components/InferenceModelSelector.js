// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import { setTaskInfo } from '../features/tasks/taskSlice';

// Inference models. Each option pairs a backend (orchestrator routing
// via TaskInfo.service_type) with a policy class (drives instruction
// visibility and future per-model UI knobs).
//
// Add a new entry once a policy is validated end-to-end. value is the
// composite key the dropdown stores; serviceType / policyType are the
// fields that get written into taskInfo on selection.
//
// Examples for follow-up PRs:
//   { value: 'lerobot:smolvla', label: 'LeRobot (SmolVLA)', serviceType: 'lerobot', policyType: 'smolvla' },
//   { value: 'lerobot:pi0',     label: 'LeRobot (Pi0)',     serviceType: 'lerobot', policyType: 'pi0' },
//   { value: 'groot:n17',       label: 'GR00T N1.7',        serviceType: 'groot',   policyType: 'n17' },
export const MODEL_OPTIONS = [
  { value: 'lerobot:act', label: 'LeRobot (ACT)', serviceType: 'lerobot', policyType: 'act' },
  { value: 'groot:n16',   label: 'GR00T N1.6',    serviceType: 'groot',   policyType: 'n16' },
];

const DEFAULT = MODEL_OPTIONS[0];

const classLabel = clsx(
  'text-sm', 'text-gray-600', 'w-28', 'flex-shrink-0', 'font-medium'
);

const InferenceModelSelector = ({ readonly = false }) => {
  const dispatch = useDispatch();
  const info = useSelector((state) => state.tasks.taskInfo);
  const serviceType = info.serviceType || DEFAULT.serviceType;
  const policyType = info.policyType || DEFAULT.policyType;
  const value = `${serviceType}:${policyType}`;

  const handleChange = (e) => {
    const sel = MODEL_OPTIONS.find((o) => o.value === e.target.value);
    if (!sel) return;
    dispatch(
      setTaskInfo({
        ...info,
        serviceType: sel.serviceType,
        policyType: sel.policyType,
      })
    );
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
        {MODEL_OPTIONS.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
};

export default InferenceModelSelector;
