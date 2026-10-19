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
import { MdClose } from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { isFieldDisabled, HELP_TEXT } from './parameterPolicy';
import useBtParameterDraft from './useBtParameterDraft';
import BtParameterInput from './BtParameterInput';
import ReturnWaypointFields from './ReturnWaypointFields';

export default function BtParameterPanel({ nodes, selectedNodeId, onParamChange,
  onNameChange, onClose, robotType = '', variant = 'legacy', allowNameEdit = true }) {
  const { selectedNode, nodeType, label, localName, setLocalName, localParams,
    paramEntries, nameAtFocusRef, suppressNextNameBlurRef, showPolicyBrowser,
    setShowPolicyBrowser, policyBrowserPath, commitName, handleChange,
    handlePolicyFolderSelect, commitTargetJointSelection } = useBtParameterDraft({
      nodes, selectedNodeId, onParamChange, onNameChange });
  if (!selectedNode) return null;
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
            readOnly={!allowNameEdit}
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
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-[var(--mc-surface-hover)] text-[var(--mc-text-subtle)] hover:text-[var(--mc-text)] transition-colors"
        >
          <MdClose size={20} />
        </button>
      </div>

      {/* Params */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
        {nodeType === 'ReturnToWaypoint' && <ReturnWaypointFields key={selectedNodeId} params={localParams} onChange={handleChange} />}
        {paramEntries.length === 0 ? (
          <p className="text-sm text-[var(--mc-text-subtle)]">No parameters</p>
        ) : (
          paramEntries.filter(([key]) => nodeType !== 'ReturnToWaypoint' || !['map_name', 'mission_name', 'waypoint', 'odom_topic'].includes(key)).map(([key, value]) => {
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
                <BtParameterInput paramName={key} value={value} disabled={disabled}
                  nodeType={nodeType} localParams={localParams} robotType={robotType}
                  handleChange={handleChange} commitTargetJointSelection={commitTargetJointSelection}
                  setShowPolicyBrowser={setShowPolicyBrowser} />
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
