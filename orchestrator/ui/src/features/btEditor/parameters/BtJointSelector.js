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
import { getTargetJointOptions, csvParts } from './jointParams';

export default function BtJointSelector({ paramName: key, value, disabled = false,
  robotType, handleChange, commitTargetJointSelection }) {
    const options = getTargetJointOptions(robotType, key);
    const selected = csvParts(value);
    const selectedSet = new Set(selected);
    const disabledCls = disabled
      ? ' !bg-[var(--mc-surface-hover)] !text-[var(--mc-text-subtle)] cursor-not-allowed'
      : '';
    const jointTextarea = (
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(key, e.target.value)}
        rows={String(value).length > 60 ? 3 : 1}
        className={`w-full px-2 py-1.5 border border-[var(--mc-border-strong)] rounded-lg text-sm bg-[var(--mc-surface)] text-[var(--mc-text)] focus:outline-none focus:ring-1 focus:ring-[var(--mc-accent)] resize-y${disabledCls}`}
      />
    );

    if (options.length === 0) {
      return jointTextarea;
    }

    return (
      <div className="space-y-2">
        <div className="flex flex-wrap gap-1.5">
          {options.map((jointName) => {
            const isSelected = selectedSet.has(jointName);
            return (
              <button
                key={jointName}
                type="button"
                disabled={disabled}
                onClick={() => {
                  const nextJoints = isSelected
                    ? selected.filter((joint) => joint !== jointName)
                    : options.filter((joint) => (
                        joint === jointName || selectedSet.has(joint)
                      ));
                  commitTargetJointSelection(key, nextJoints);
                }}
                className={`px-2 py-1 border rounded-lg text-xs transition-colors ${
                  isSelected
                    ? 'border-[var(--mc-accent)] bg-[var(--mc-accent-soft)] text-[var(--mc-accent)]'
                    : 'border-[var(--mc-border-strong)] bg-[var(--mc-surface)] text-[var(--mc-text-muted)] hover:bg-[var(--mc-surface-hover)]'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {jointName}
              </button>
            );
          })}
        </div>
        {jointTextarea}
      </div>
    );
}
