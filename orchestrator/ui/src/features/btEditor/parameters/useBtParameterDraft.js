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

import { useState, useEffect, useMemo, useRef } from 'react';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { normalizeSendCommandParams, normalizeSendCommandTarget,
  SEND_COMMAND_COMMANDS_BY_TARGET, truthy } from './parameterPolicy';
import { TARGET_POSITION_PARAM, csvParts } from './jointParams';

// Preserve typing buffers when the parent graph re-renders, but emit edits
// immediately so Save includes the currently focused field.
export default function useBtParameterDraft({ nodes, selectedNodeId, onParamChange, onNameChange }) {
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


  const { label, nodeType } = selectedNode?.data || {};
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
    if (nodeType === 'SendCommand') {
      const clear = ((paramName === 'action_request_mode' && value !== 'async')
          || (paramName === 'model' && !String(value).startsWith('groot'))) ? ['vla_rail_enabled'] : [];
      for (const field of clear) {
        if (truthy(localParams[field])) {
          setLocalParams((prev) => ({ ...prev, [field]: 'false' }));
          onParamChange(selectedNodeId, field, 'false');
        }
      }
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

  return { selectedNode, nodeType, label, localName, setLocalName, localParams, setLocalParams,
    paramEntries, nameAtFocusRef, suppressNextNameBlurRef, showPolicyBrowser,
    setShowPolicyBrowser, policyBrowserPath, commitName, handleChange,
    handlePolicyFolderSelect, commitTargetJointSelection };
}
