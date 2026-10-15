// Copyright 2026 ROBOTIS CO., LTD.
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
// Author: Seongwoo Kim

import React from 'react';
import { MdFolderOpen } from 'react-icons/md';
import { JOINT_SELECTOR_NODE_TYPES, TARGET_POSITION_PARAM } from './jointParams';
import { enumOptionsFor, BOOL_PARAMS, NUMBER_PARAMS } from './parameterPolicy';
import BtJointSelector from './BtJointSelector';

export default function BtParameterInput({ paramName: key, value, disabled = false,
  nodeType, localParams, robotType, handleChange, commitTargetJointSelection, setShowPolicyBrowser }) {
    const disabledCls = disabled
      ? ' !bg-[var(--mc-surface-hover)] !text-[var(--mc-text-subtle)] cursor-not-allowed'
      : '';

    if (JOINT_SELECTOR_NODE_TYPES.has(nodeType) && TARGET_POSITION_PARAM[key]) {
      return <BtJointSelector paramName={key} value={value} disabled={disabled}
        robotType={robotType} handleChange={handleChange} commitTargetJointSelection={commitTargetJointSelection} />;
    }

    const enumOptions = enumOptionsFor(nodeType, key, localParams);
    if (enumOptions) {
      return (
        <select
          value={value}
          disabled={disabled}
          onChange={(e) => handleChange(key, e.target.value)}
          className={`w-full px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] focus:outline-none focus:ring-1 focus:ring-[var(--mc-accent)]${disabledCls}`}
        >
          {enumOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
    }

    if (BOOL_PARAMS.has(key)) {
      return (
        <label className={`flex items-center gap-2 ${disabled ? 'cursor-not-allowed text-[var(--mc-text-subtle)]' : 'cursor-pointer'}`}>
          <input
            type="checkbox"
            aria-label={key}
            disabled={disabled}
            checked={value === 'true' || value === true}
            onChange={(e) => {
              const v = e.target.checked ? 'true' : 'false';
              handleChange(key, v);
            }}
            className="w-4 h-4 rounded border-[var(--mc-border-strong)] text-[var(--mc-accent)] focus:ring-[var(--mc-accent)]"
          />
          <span className="text-sm text-[var(--mc-text-muted)]">{value === 'true' || value === true ? 'true' : 'false'}</span>
        </label>
      );
    }

    if (nodeType === 'SendCommand' && key === 'policy_path') {
      return (
        <div className="flex flex-row items-start gap-2">
          <textarea
            value={value}
            disabled={disabled}
            onChange={(e) => handleChange(key, e.target.value)}
            rows={String(value).length > 60 ? 3 : 1}
            placeholder="Enter Policy Path or Repo ID"
            className={`flex-1 min-w-0 px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] focus:outline-none focus:ring-1 focus:ring-[var(--mc-accent)] resize-y${disabledCls}`}
          />
          <button
            type="button"
            onClick={() => !disabled && setShowPolicyBrowser(true)}
            disabled={disabled}
            className="flex items-center justify-center w-8 h-8 text-[var(--mc-accent)] bg-[var(--mc-surface-2)] border border-[var(--mc-border-strong)] rounded-lg hover:bg-[var(--mc-surface-hover)] disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            aria-label="Browse for policy model folder"
            title="Browse for policy model folder"
          >
            <MdFolderOpen size={18} />
          </button>
        </div>
      );
    }

    if (NUMBER_PARAMS.has(key)) {
      return (
        <input
          type="number"
          step="any"
          value={value}
          disabled={disabled}
          onChange={(e) => handleChange(key, e.target.value)}
          className={`w-full px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] focus:outline-none focus:ring-1 focus:ring-[var(--mc-accent)]${disabledCls}`}
        />
      );
    }

    return (
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(key, e.target.value)}
        rows={String(value).length > 60 ? 3 : 1}
        className={`w-full px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] focus:outline-none focus:ring-1 focus:ring-[var(--mc-accent)] resize-y${disabledCls}`}
      />
    );
}
