// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

import React from 'react';
import clsx from 'clsx';

const visible = (parameter, values) => Object.entries(parameter.visible_when || {})
  .every(([key, expected]) => values[key] === expected);

const inputValue = (parameter, info) => {
  if (parameter.binding.startsWith('task_info.')) {
    return info[parameter.binding.slice('task_info.'.length)] ?? '';
  }
  return info.policyParameters?.[parameter.key] ?? '';
};

const convertedValue = (parameter, raw) => {
  if (parameter.control === 'toggle') return Boolean(raw);
  if (raw === '') return '';
  if (parameter.control === 'integer') return Number.parseInt(raw, 10);
  if (parameter.control === 'number') return Number(raw);
  return raw;
};

export default function PolicyParameterFields({
  model,
  info,
  disabled = false,
  labelClassName,
  onTaskInfoChange,
  onPolicyParametersChange,
  excludedBindings = [],
}) {
  const excluded = new Set(excludedBindings);
  const parameters = (model?.parameters || []).filter(
    (parameter) => !excluded.has(parameter.binding)
  );
  const values = Object.fromEntries(
    parameters.map((parameter) => [parameter.key, inputValue(parameter, info)])
  );

  const update = (parameter, raw) => {
    const value = convertedValue(parameter, raw);
    if (parameter.binding.startsWith('task_info.')) {
      onTaskInfoChange(parameter.binding.slice('task_info.'.length), value);
      return;
    }
    if (raw === '') {
      const next = { ...(info.policyParameters || {}) };
      delete next[parameter.key];
      onPolicyParametersChange(next);
      return;
    }
    onPolicyParametersChange({
      ...(info.policyParameters || {}),
      [parameter.key]: value,
    });
  };

  return parameters.filter((parameter) => visible(parameter, values)).map((parameter) => {
    const value = inputValue(parameter, info);
    const common = {
      disabled,
      'aria-label': parameter.label,
    };
    let control;
    if (parameter.control === 'select') {
      control = (
        <select
          {...common}
          value={value}
          onChange={(event) => update(parameter, event.target.value)}
          className="flex-1 h-8 px-2 border border-gray-300 rounded-md bg-white disabled:bg-gray-100"
        >
          {(parameter.options || []).map((option) => (
            <option key={String(option)} value={option}>{String(option)}</option>
          ))}
        </select>
      );
    } else if (parameter.control === 'toggle') {
      control = (
        <input
          {...common}
          type="checkbox"
          checked={Boolean(value)}
          onChange={(event) => update(parameter, event.target.checked)}
          className="w-4 h-4"
        />
      );
    } else {
      const numeric = parameter.control === 'number' || parameter.control === 'integer';
      control = (
        <input
          {...common}
          type={numeric ? 'number' : 'text'}
          value={value}
          min={parameter.min}
          max={parameter.max}
          step={parameter.step ?? (parameter.control === 'integer' ? 1 : undefined)}
          onChange={(event) => update(parameter, event.target.value)}
          className="flex-1 min-w-0 h-8 px-2 border border-gray-300 rounded-md disabled:bg-gray-100"
        />
      );
    }
    return (
      <div key={parameter.key} className={clsx('flex', 'items-center', 'mb-2.5')}>
        <span className={labelClassName}>{parameter.label}</span>
        {control}
      </div>
    );
  });
}
