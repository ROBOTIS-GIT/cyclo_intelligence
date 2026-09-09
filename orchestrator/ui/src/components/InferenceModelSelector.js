// Copyright 2025 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0

import React, { useEffect, useMemo } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import {
  markLocalTaskInfoEdited,
  selectInferenceTaskInfo,
  setInferenceTaskInfo,
} from '../features/tasks/taskSlice';
import {
  findPolicy,
  parameterDefaults,
  usePolicyCatalog,
} from '../contexts/PolicyCatalogContext';

const classLabel = clsx(
  'text-sm', 'text-gray-600', 'w-28', 'flex-shrink-0', 'font-medium'
);

const selectionPatch = (option, currentInfo, allOptions) => {
  const taskParameterKeys = new Set(
    allOptions.flatMap((model) => (
      (model.parameters || [])
        .filter((parameter) => parameter.binding.startsWith('task_info.'))
        .map((parameter) => parameter.binding.slice('task_info.'.length))
    ))
  );
  const taskDefaults = Object.fromEntries(
    (option.parameters || [])
      .filter((parameter) => parameter.binding.startsWith('task_info.'))
      .filter((parameter) => Object.prototype.hasOwnProperty.call(parameter, 'default'))
      .map((parameter) => [parameter.binding.slice('task_info.'.length), parameter.default])
  );
  const supportedModes = option.runtime.capabilities?.action_request_modes || ['async'];
  const actionRequestMode = supportedModes.includes(currentInfo.actionRequestMode)
    ? currentInfo.actionRequestMode
    : supportedModes[0];
  return {
    ...Object.fromEntries([...taskParameterKeys].map((key) => [key, ''])),
    serviceType: option.runtime.id,
    policyType: option.id,
    policyId: option.policy_id,
    policyParameters: parameterDefaults(option, 'policy_parameters.'),
    actionRequestMode,
    ...taskDefaults,
  };
};

const InferenceModelSelector = ({ readonly = false }) => {
  const dispatch = useDispatch();
  const info = useSelector(selectInferenceTaskInfo, shallowEqual);
  const { catalog, status } = usePolicyCatalog();
  const options = useMemo(() => (
    (catalog?.runtimes || []).flatMap((runtime) => (
      runtime.models.map((model) => ({ ...model, runtime }))
    ))
  ), [catalog]);
  const selected = findPolicy(
    catalog,
    info.policyId,
    info.serviceType,
    info.policyType
  );

  useEffect(() => {
    if (status !== 'ready' || options.length === 0) return;
    if (selected) {
      if (info.policyId === selected.policy_id &&
          info.serviceType === selected.runtime.id && info.policyType === selected.id) return;
      // Normalizing a restored/default selection is not a user edit.
      dispatch(setInferenceTaskInfo({
        policyId: selected.policy_id,
        serviceType: selected.runtime.id,
        policyType: selected.id,
      }));
      return;
    }
    const hasRequestedSelection = Boolean(
      String(info.policyId || '').trim() ||
      String(info.serviceType || '').trim() ||
      String(info.policyType || '').trim()
    );
    if (hasRequestedSelection) return;
    dispatch(setInferenceTaskInfo(selectionPatch(options[0], info, options)));
  }, [dispatch, info, options, selected, status]);

  const handleChange = (event) => {
    const option = options.find((model) => model.policy_id === event.target.value);
    if (!option) return;
    dispatch(setInferenceTaskInfo(selectionPatch(option, info, options)));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
  };

  const disabled = readonly || status !== 'ready';
  return (
    <div className={clsx('flex', 'items-center', 'mb-2.5')}>
      <span className={classLabel}>Model</span>
      <select
        className={clsx(
          'flex-1', 'h-8', 'px-2', 'border', 'border-gray-300', 'rounded-md',
          'focus:outline-none', 'focus:ring-2', 'focus:ring-blue-500',
          'focus:border-transparent',
          {
            'bg-gray-100 cursor-not-allowed text-gray-500': disabled,
            'bg-white': !disabled,
          }
        )}
        value={selected?.policy_id || ''}
        onChange={handleChange}
        disabled={disabled}
        aria-label="Policy model"
      >
        {!selected && (
          <option value="">
            {status === 'ready' ? 'Selected policy is unavailable' : 'Loading policy catalog...'}
          </option>
        )}
        {(catalog?.runtimes || []).map((runtime) => (
          <optgroup key={runtime.id} label={runtime.label}>
            {runtime.models.map((model) => (
              <option key={model.policy_id} value={model.policy_id}>
                {model.label}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </div>
  );
};

export default InferenceModelSelector;
