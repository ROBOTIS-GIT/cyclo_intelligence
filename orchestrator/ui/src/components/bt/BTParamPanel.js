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
// Author: Claude (generated)

import React, { useState, useEffect } from 'react';
import { useDispatch } from 'react-redux';
import { MdClose } from 'react-icons/md';
import { setSelectedNodeId } from '../../features/btmanager/btmanagerSlice';

const NUMBER_PARAMS = new Set([
  'duration', 'angle_deg', 'lift_position', 'control_hz', 'inference_hz',
  'chunk_align_window_s', 'position_threshold',
]);

const BOOL_PARAMS = new Set([]);

// Enum params surface as <select> dropdowns. Keep value lists in sync with
// the Python action definitions (e.g. send_command.COMMAND_MAP,
// wait_until_pose.GRIPPER_CHECK_OPTIONS).
const ENUM_PARAMS = {
  command: ['LOAD', 'RESUME', 'STOP', 'CLEAR'],
  model: ['groot', 'lerobot'],
  gripper_check: ['none', 'left', 'right', 'both'],
};

// SendCommand inputs that are meaningful per command. Anything outside
// the set for the current command is rendered disabled — the value stays
// in params so flipping back to LOAD restores the user's earlier entries.
// 'command' itself is always editable.
const SEND_COMMAND_ACTIVE_FIELDS = {
  LOAD: new Set([
    'command', 'model', 'policy_path', 'task_instruction',
    'inference_hz', 'control_hz', 'chunk_align_window_s',
  ]),
  // Resume re-conditions language mid-run; the other LOAD inputs are
  // already baked into the loaded policy.
  RESUME: new Set(['command', 'task_instruction']),
  STOP: new Set(['command']),
  CLEAR: new Set(['command']),
};

function isSendCommandFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'SendCommand') return false;
  const cmd = String(params.command || 'LOAD').toUpperCase();
  const active = SEND_COMMAND_ACTIVE_FIELDS[cmd];
  if (!active) return false;
  return !active.has(key);
}

export default function BTParamPanel({ nodes, selectedNodeId, onParamChange }) {
  const dispatch = useDispatch();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Local param state — isolates keystrokes from parent re-renders (preserves cursor)
  const [localParams, setLocalParams] = useState({});

  // Reset local state only when switching to a different node
  useEffect(() => {
    if (selectedNode) {
      setLocalParams(selectedNode.data.params || {});
    }
  }, [selectedNodeId]); // intentionally excludes selectedNode to avoid resetting mid-edit

  if (!selectedNode) return null;

  const { label, nodeType } = selectedNode.data;
  const paramEntries = Object.entries(localParams);

  const handleChange = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
  };

  const handleBlur = (paramName) => {
    onParamChange(selectedNodeId, paramName, localParams[paramName]);
  };

  const renderInput = (key, value, disabled = false) => {
    const disabledCls = disabled
      ? ' bg-gray-100 text-gray-400 cursor-not-allowed'
      : '';

    if (ENUM_PARAMS[key]) {
      return (
        <select
          value={value}
          disabled={disabled}
          onChange={(e) => {
            handleChange(key, e.target.value);
            // select has no meaningful blur event for this; sync immediately
            onParamChange(selectedNodeId, key, e.target.value);
          }}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white focus:outline-none focus:ring-1 focus:ring-blue-400${disabledCls}`}
        >
          {ENUM_PARAMS[key].map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
    }

    if (BOOL_PARAMS.has(key)) {
      return (
        <label className={`flex items-center gap-2 ${disabled ? 'cursor-not-allowed text-gray-400' : 'cursor-pointer'}`}>
          <input
            type="checkbox"
            disabled={disabled}
            checked={value === 'true' || value === true}
            onChange={(e) => {
              const v = e.target.checked ? 'true' : 'false';
              handleChange(key, v);
              onParamChange(selectedNodeId, key, v);
            }}
            className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-blue-400"
          />
          <span className="text-sm text-gray-600">{value === 'true' || value === true ? 'true' : 'false'}</span>
        </label>
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
          onBlur={() => handleBlur(key)}
          className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400${disabledCls}`}
        />
      );
    }

    return (
      <textarea
        value={value}
        disabled={disabled}
        onChange={(e) => handleChange(key, e.target.value)}
        onBlur={() => handleBlur(key)}
        rows={String(value).length > 60 ? 3 : 1}
        className={`w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y${disabledCls}`}
      />
    );
  };

  return (
    <div className="absolute right-0 top-0 bottom-0 w-[320px] bg-white border-l border-gray-200 shadow-lg z-10 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200">
        <div>
          <div className="text-sm font-bold text-gray-800">{label}</div>
          <div className="text-xs text-gray-500">{nodeType}</div>
        </div>
        <button
          onClick={() => dispatch(setSelectedNodeId(null))}
          className="p-1 rounded hover:bg-gray-100 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <MdClose size={20} />
        </button>
      </div>

      {/* Params */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {paramEntries.length === 0 ? (
          <p className="text-sm text-gray-400">No parameters</p>
        ) : (
          paramEntries.map(([key, value]) => {
            const disabled = isSendCommandFieldDisabled(nodeType, key, localParams);
            return (
              <div key={key}>
                <label
                  className={`block text-xs font-medium mb-1 ${
                    disabled ? 'text-gray-400' : 'text-gray-600'
                  }`}
                >
                  {key}
                </label>
                {renderInput(key, value, disabled)}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
