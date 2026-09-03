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

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { MdClose, MdFolderOpen } from 'react-icons/md';
import FileBrowserModal from '../FileBrowserModal';
import { setSelectedNodeId } from '../../features/actionCanvas/actionCanvasSlice';
import { DEFAULT_PATHS } from '../../constants/paths';

const NUMBER_PARAMS = new Set([
  'duration', 'angle_deg', 'lift_position', 'control_hz', 'inference_hz',
  'chunk_align_window_s', 'max_iterations', 'joint_threshold',
  'gripper_closed_value', 'gripper_open_value', 'gripper_threshold',
  'timeout_sec',
]);

const SG2_LEFT_JOINTS = [
  'arm_l_joint1',
  'arm_l_joint2',
  'arm_l_joint3',
  'arm_l_joint4',
  'arm_l_joint5',
  'arm_l_joint6',
  'arm_l_joint7',
  'gripper_l_joint1',
];

const SG2_RIGHT_JOINTS = [
  'arm_r_joint1',
  'arm_r_joint2',
  'arm_r_joint3',
  'arm_r_joint4',
  'arm_r_joint5',
  'arm_r_joint6',
  'arm_r_joint7',
  'gripper_r_joint1',
];

const SG2_TARGET_JOINTS = {
  // ArmStateGate observation targets
  left_target_joints: SG2_LEFT_JOINTS,
  right_target_joints: SG2_RIGHT_JOINTS,
  // JointControl trajectory joints
  left_joint_names: SG2_LEFT_JOINTS,
  right_joint_names: SG2_RIGHT_JOINTS,
};

// joint-selection param → its paired positions param. Toggling a joint chip
// keeps the two CSVs aligned (new joints get 0.0, removed joints drop theirs).
const TARGET_POSITION_PARAM = {
  left_target_joints: 'left_target_positions',
  right_target_joints: 'right_target_positions',
  left_joint_names: 'left_positions',
  right_joint_names: 'right_positions',
};

// Node types whose joint params render as the chip selector.
const JOINT_SELECTOR_NODE_TYPES = new Set(['ArmStateGate', 'JointControl']);

// Nodes created before the per-joint params existed (or from a stale bt_node
// catalog) lack left/right_joint_names entirely, so the selector would never
// render for them. Synthesize the params with the full joint list — exactly
// what the engine does when the names are omitted — inserting each one just
// before its positions field so the selector renders above it.
function withJointSelectionDefaults(params, nodeType) {
  if (nodeType !== 'JointControl') return params;
  const next = {};
  const insertBefore = {
    left_positions: ['left_joint_names', SG2_LEFT_JOINTS.join(', ')],
    right_positions: ['right_joint_names', SG2_RIGHT_JOINTS.join(', ')],
  };
  Object.entries(params).forEach(([key, value]) => {
    const missing = insertBefore[key];
    if (missing && !(missing[0] in params)) {
      next[missing[0]] = missing[1];
    }
    next[key] = value;
  });
  return next;
}

const csvParts = (value) =>
  String(value || '')
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

function isSg2LikeRobot(robotType) {
  const normalized = String(robotType || '').toLowerCase();
  return !normalized || normalized.includes('sg2') || normalized.includes('bg2');
}

function getTargetJointOptions(robotType, key) {
  if (!isSg2LikeRobot(robotType)) return [];
  return SG2_TARGET_JOINTS[key] || [];
}

// Per-param helper text shown beneath the input. Keep these short — they
// render directly under the field as a small gray hint.
const HELP_TEXT = {
  max_iterations: '0 = loop forever',
};

// Boolean params render as checkboxes.
const BOOL_PARAMS = new Set([
  'enable_head',
  'enable_arms',
  'enable_lift',
  'detect_left_gripper',
  'detect_right_gripper',
]);

const SEND_COMMAND_TARGETS = ['INFERENCE', 'DOCKER'];

const SEND_COMMAND_COMMANDS_BY_TARGET = {
  INFERENCE: ['LOAD', 'RESUME', 'STOP', 'CLEAR'],
  DOCKER: ['START', 'STOP', 'RESTART'],
};

function normalizeSendCommandTarget(target) {
  return String(target || 'INFERENCE').trim().toUpperCase() === 'DOCKER'
    ? 'DOCKER'
    : 'INFERENCE';
}

function normalizeSendCommandParams(params, nodeType) {
  const next = withJointSelectionDefaults(params, nodeType);
  if (nodeType !== 'SendCommand') return next;

  const target = normalizeSendCommandTarget(next.target);
  const commands = SEND_COMMAND_COMMANDS_BY_TARGET[target];
  const requestedCommand = String(next.command || commands[0]).trim().toUpperCase();
  const command = commands.includes(requestedCommand) ? requestedCommand : commands[0];
  const remaining = Object.fromEntries(
    Object.entries(next).filter(([key]) => key !== 'target' && key !== 'command'),
  );
  return { target, command, ...remaining };
}

// Enum params surface as <select> dropdowns. Keep value lists in sync with
// the Python action definitions (send_command.COMMAND_MAP).
const ENUM_PARAMS = {
  model: [
    'lerobot:act',
    'lerobot:diffusion',
    'lerobot:smolvla',
    'lerobot:xvla',
    'lerobot:pi0',
    'lerobot:pi05',
    'lerobot:molmoact2',
    'lerobot:vla_jepa',
    'lerobot:fastwam',
    'groot:n17',
    'groot',
    'lerobot',
  ],
  inference_mode: ['simulation', 'robot'],
  action_request_mode: ['async', 'sync'],
  acceleration_mode: ['pytorch', 'tensorrt_dit'],
};

function enumOptionsFor(nodeType, key, params) {
  if (nodeType === 'SendCommand') {
    if (key === 'target') return SEND_COMMAND_TARGETS;
    if (key === 'command') {
      return SEND_COMMAND_COMMANDS_BY_TARGET[
        normalizeSendCommandTarget(params.target)
      ];
    }
  }
  return ENUM_PARAMS[key];
}

// SendCommand inputs that are meaningful per command. Anything outside
// the set for the current command is rendered disabled — the value stays
// in params so flipping back to LOAD restores the user's earlier entries.
// 'command' itself is always editable.
const SEND_COMMAND_ACTIVE_FIELDS = {
  LOAD: new Set([
    'target', 'command', 'model', 'policy_path', 'task_instruction',
    'inference_mode', 'action_request_mode', 'inference_hz', 'control_hz',
    'chunk_align_window_s', 'acceleration_mode', 'acceleration_engine_path',
  ]),
  // Resume can re-condition language mid-run; output mode is fixed by LOAD.
  RESUME: new Set(['target', 'command', 'task_instruction']),
  STOP: new Set(['target', 'command']),
  CLEAR: new Set(['target', 'command']),
};

const SEND_COMMAND_DOCKER_ACTIVE_FIELDS = new Set(['target', 'command', 'model']);

// JointControl: each group's positions input is gated on its enable_*
// flag. enable_* toggles themselves + duration are always editable.
const truthy = (v) => v === true || v === 'true';

function isSendCommandFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'SendCommand') return false;
  if (normalizeSendCommandTarget(params.target) === 'DOCKER') {
    return !SEND_COMMAND_DOCKER_ACTIVE_FIELDS.has(key);
  }
  const cmd = String(params.command || 'LOAD').toUpperCase();
  const active = SEND_COMMAND_ACTIVE_FIELDS[cmd];
  if (!active) return false;
  return !active.has(key);
}

function isJointControlFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'JointControl') return false;
  if (key === 'head_positions') return !truthy(params.enable_head);
  if (
    key === 'left_positions' || key === 'right_positions'
    || key === 'left_joint_names' || key === 'right_joint_names'
  ) {
    return !truthy(params.enable_arms);
  }
  if (key === 'lift_position') return !truthy(params.enable_lift);
  return false;  // enable_*, duration stay editable
}

function isArmStateGateFieldDisabled(nodeType, key, params) {
  if (nodeType !== 'ArmStateGate') return false;
  if (key === 'left_gripper_joint') {
    return !truthy(params.detect_left_gripper);
  }
  if (key === 'right_gripper_joint') {
    return !truthy(params.detect_right_gripper);
  }
  return false;
}

function isFieldDisabled(nodeType, key, params) {
  return (
    isSendCommandFieldDisabled(nodeType, key, params) ||
    isJointControlFieldDisabled(nodeType, key, params) ||
    isArmStateGateFieldDisabled(nodeType, key, params)
  );
}

export default function BTParamPanel({
  nodes,
  selectedNodeId,
  onParamChange,
  onNameChange,
  onClose,
  variant = 'legacy',
}) {
  const dispatch = useDispatch();
  const robotType = useSelector((state) => state.tasks?.robotType || '');

  const selectedNode = nodes.find((n) => n.id === selectedNodeId);

  // Local param state — isolates keystrokes from parent re-renders (preserves cursor)
  const [localParams, setLocalParams] = useState({});
  // Local name buffer — same cursor-preservation trick as localParams.
  const [localName, setLocalName] = useState('');
  const nameAtFocusRef = useRef('');
  const suppressNextNameBlurRef = useRef(false);
  const [showPolicyBrowser, setShowPolicyBrowser] = useState(false);

  const policyBrowserPath = useMemo(() => {
    const model = String(localParams.model || '').toLowerCase();
    return model.startsWith('groot')
      ? DEFAULT_PATHS.GROOT_CHECKPOINTS_PATH
      : DEFAULT_PATHS.LEROBOT_CHECKPOINTS_PATH;
  }, [localParams.model]);

  // Reset local state only when switching to a different node
  useEffect(() => {
    if (selectedNode) {
      setLocalParams(normalizeSendCommandParams(
        selectedNode.data.params || {},
        selectedNode.data.nodeType,
      ));
      setLocalName(selectedNode.data.label || '');
    }
    setShowPolicyBrowser(false);
    // Keep mid-edit cursor position stable; reset only when the selection changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedNodeId]); // intentionally excludes selectedNode to avoid resetting mid-edit

  if (!selectedNode) return null;

  const { label, nodeType } = selectedNode.data;
  const paramEntries = Object.entries(localParams);

  const commitName = () => {
    if (suppressNextNameBlurRef.current) {
      suppressNextNameBlurRef.current = false;
      return;
    }
    const trimmed = localName.trim();
    if (!trimmed) {
      // Reject empty — snap input back to current label.
      setLocalName(label);
      return;
    }
    if (trimmed !== label) {
      onNameChange?.(selectedNodeId, trimmed);
    }
  };

  const handleChange = (paramName, value) => {
    if (nodeType === 'SendCommand' && paramName === 'target') {
      const target = normalizeSendCommandTarget(value);
      const commands = SEND_COMMAND_COMMANDS_BY_TARGET[target];
      const currentCommand = String(localParams.command || '').trim().toUpperCase();
      const command = commands.includes(currentCommand) ? currentCommand : commands[0];
      setLocalParams((prev) => ({ ...prev, target, command }));
      onParamChange(selectedNodeId, 'target', target);
      if (command !== currentCommand) {
        onParamChange(selectedNodeId, 'command', command);
      }
      return;
    }
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
    // A Design Save click can happen before a blur-driven graph update reaches
    // the mission state. Keep the graph current while the field is focused so
    // the visible value is always the value that gets serialized.
    onParamChange(selectedNodeId, paramName, value);
  };

  const commitParam = (paramName, value) => {
    setLocalParams((prev) => ({ ...prev, [paramName]: value }));
    onParamChange(selectedNodeId, paramName, value);
  };

  const handlePolicyFolderSelect = (item) => {
    const fullPath = item?.full_path || '';
    if (fullPath) {
      commitParam('policy_path', fullPath);
    }
    setShowPolicyBrowser(false);
  };

  const commitTargetJointSelection = (paramName, nextJoints) => {
    const positionsParam = TARGET_POSITION_PARAM[paramName];
    const currentJoints = csvParts(localParams[paramName]);
    const currentPositions = csvParts(localParams[positionsParam]);
    const positionByJoint = new Map(
      currentJoints.map((joint, idx) => [
        joint,
        currentPositions[idx] || '0.0',
      ])
    );
    const nextPositions = nextJoints.map(
      (joint) => positionByJoint.get(joint) || '0.0'
    );
    const jointsValue = nextJoints.join(', ');
    const positionsValue = nextPositions.join(', ');

    setLocalParams((prev) => ({
      ...prev,
      [paramName]: jointsValue,
      [positionsParam]: positionsValue,
    }));
    onParamChange(selectedNodeId, paramName, jointsValue);
    onParamChange(selectedNodeId, positionsParam, positionsValue);
  };

  const renderTargetJointSelector = (key, value, disabled = false) => {
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
  };

  const renderInput = (key, value, disabled = false) => {
    const disabledCls = disabled
      ? ' !bg-[var(--mc-surface-hover)] !text-[var(--mc-text-subtle)] cursor-not-allowed'
      : '';

    if (JOINT_SELECTOR_NODE_TYPES.has(nodeType) && TARGET_POSITION_PARAM[key]) {
      return renderTargetJointSelector(key, value, disabled);
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
  };

  // The --mc-* tokens are scoped to .autonomy-studio-page; outside it (BT
  // Manager) the vars are undefined, so the panel chrome needs opaque
  // fallbacks or the canvas nodes show through.
  return (
    <div className="absolute right-0 top-0 bottom-0 w-[320px] bg-[var(--mc-surface-2,#ffffff)] border-l border-[var(--mc-border,#e5e7eb)] shadow-lg z-10 flex flex-col">
      {/* Header */}
      <div className="flex items-start justify-between px-4 py-3 border-b border-[var(--mc-border,#e5e7eb)]">
        <div className="flex-1 min-w-0 pr-2">
          <div className="text-xs text-[var(--mc-text-muted)] mb-1 font-mono">{nodeType}</div>
          <input
            type="text"
            value={localName}
            onFocus={() => {
              nameAtFocusRef.current = label;
            }}
            onChange={(e) => {
              const value = e.target.value;
              setLocalName(value);
              const trimmed = value.trim();
              if (trimmed) onNameChange?.(selectedNodeId, trimmed);
            }}
            onBlur={commitName}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.currentTarget.blur();
              } else if (e.key === 'Escape') {
                const originalName = nameAtFocusRef.current || label;
                suppressNextNameBlurRef.current = true;
                setLocalName(originalName);
                onNameChange?.(selectedNodeId, originalName);
                e.currentTarget.blur();
              }
            }}
            className="w-full text-sm font-bold text-[var(--mc-text)] bg-transparent border-0 border-b border-transparent hover:border-[var(--mc-border-strong)] focus:border-[var(--mc-accent)] focus:outline-none px-0 py-0.5"
          />
        </div>
        <button
          onClick={() => (typeof onClose === 'function' ? onClose() : dispatch(setSelectedNodeId(null)))}
          className="p-1 rounded-lg hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-subtle)] hover:text-[var(--mc-text)] transition-colors"
        >
          <MdClose size={20} />
        </button>
      </div>

      {/* Params */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {paramEntries.length === 0 ? (
          <p className="text-sm text-[var(--mc-text-subtle)]">No parameters</p>
        ) : (
          paramEntries.map(([key, value]) => {
            const disabled = isFieldDisabled(nodeType, key, localParams);
            const help = HELP_TEXT[key];
            return (
              <div key={key}>
                <label
                  className={`block text-xs font-medium mb-1 font-mono ${
                    disabled ? 'text-[var(--mc-text-subtle)]' : 'text-[var(--mc-text-muted)]'
                  }`}
                >
                  {key}
                </label>
                {renderInput(key, value, disabled)}
                {help && !disabled && (
                  <div className="mt-1 text-xs text-[var(--mc-text-subtle)]">{help}</div>
                )}
              </div>
            );
          })
        )}
      </div>
      <FileBrowserModal
        isOpen={showPolicyBrowser}
        onClose={() => setShowPolicyBrowser(false)}
        onFileSelect={handlePolicyFolderSelect}
        title="Select policy model folder"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={policyBrowserPath}
        defaultPath={policyBrowserPath}
        homePath={DEFAULT_PATHS.POLICY_CHECKPOINTS_PATH}
        variant={variant}
      />
    </div>
  );
}
