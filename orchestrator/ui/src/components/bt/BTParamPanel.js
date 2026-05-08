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

import React from 'react';
import { useDispatch } from 'react-redux';
import { MdClose } from 'react-icons/md';
import { setSelectedNodeId, updateNodeParam } from '../../features/btmanager/btmanagerSlice';

const NUMBER_PARAMS = new Set([
  'duration', 'angle_deg', 'lift_position', 'control_hz', 'position_threshold',
]);

const BOOL_PARAMS = new Set(['wait_until_ready']);

const COMMAND_OPTIONS = ['START_INFERENCE', 'STOP_INFERENCE', 'RESUME_INFERENCE'];

export default function BTParamPanel({ nodes, selectedNodeId }) {
  const dispatch = useDispatch();

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);
  if (!selectedNode) return null;

  const { label, nodeType, params = {} } = selectedNode.data;
  const paramEntries = Object.entries(params);

  const handleChange = (paramName, value) => {
    dispatch(updateNodeParam({ nodeId: selectedNodeId, paramName, paramValue: value }));
  };

  const renderInput = (key, value) => {
    if (key === 'command') {
      return (
        <select
          value={value}
          onChange={(e) => handleChange(key, e.target.value)}
          className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm bg-white focus:outline-none focus:ring-1 focus:ring-blue-400"
        >
          {COMMAND_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );
    }

    if (BOOL_PARAMS.has(key)) {
      return (
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={value === 'true' || value === true}
            onChange={(e) => handleChange(key, e.target.checked ? 'true' : 'false')}
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
          onChange={(e) => handleChange(key, e.target.value)}
          className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400"
        />
      );
    }

    return (
      <textarea
        value={value}
        onChange={(e) => handleChange(key, e.target.value)}
        rows={String(value).length > 60 ? 3 : 1}
        className="w-full px-2 py-1.5 border border-gray-300 rounded text-sm focus:outline-none focus:ring-1 focus:ring-blue-400 resize-y"
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
          paramEntries.map(([key, value]) => (
            <div key={key}>
              <label className="block text-xs font-medium text-gray-600 mb-1">
                {key}
              </label>
              {renderInput(key, value)}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
