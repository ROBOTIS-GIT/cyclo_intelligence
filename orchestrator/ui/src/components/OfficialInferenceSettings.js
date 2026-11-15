// Copyright 2025 ROBOTIS CO., LTD.
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
// Author: Kiwoong Park

// Settings markup from ROBOTIS-GIT/cyclo_intelligence main (cea736ad).
// Runtime state and command handlers remain in InferencePanel.
import React from 'react';
import clsx from 'clsx';
import { MdFolderOpen, MdHourglassEmpty, MdInfoOutline, MdPrecisionManufacturing, MdSync, MdViewInAr, MdWarningAmber } from 'react-icons/md';
import FileBrowserModal from './FileBrowserModal';
import InferenceModelSelector from './InferenceModelSelector';
import PolicyBackendControl from './PolicyBackendControl';
import TrtEngineControl from './TrtEngineControl';
import Tooltip from './Tooltip';
import { DEFAULT_PATHS } from '../constants/paths';
import { getInferenceTimingWarnings } from '../utils/inferenceTiming';

export default function OfficialInferenceSettings({
  classInfoPanel,
  classLabel,
  classTaskInstructionTextarea,
  classPolicyPathTextarea,
  classTextInput,
  deployButtonClass,
  actionModeButtonClass,
  info,
  isEditable,
  showInstruction,
  isRobotMode,
  isModeSwitchLocked,
  handleDeployModeChange,
  handleChange,
  handleUpdateInstruction,
  canUpdateInstruction,
  isInferencing,
  isUpdatingInstruction,
  setShowPolicyBrowser,
  isTensorRtEnabled,
  robotType,
  trtTaskInstruction,
  actionRequestMode,
  initialPoseSyncEnabled,
  showPolicyBrowser,
  handlePolicyFolderSelect,
  policyBrowserPath,
  handleActionRequestModeChange,
  supportsTensorRt
}) {
  const timingWarnings = getInferenceTimingWarnings(info);
  return (
    <div className={classInfoPanel}>
      <div className={clsx('text-lg', 'font-semibold', 'mb-3', 'text-gray-800')}>
        Task Information
      </div>

      <InferenceModelSelector readonly={!isEditable} official />

      <PolicyBackendControl
        serviceType={info.serviceType}
      />

      <div className="mb-3 rounded-lg border border-gray-200 bg-gray-50 p-2">
        <div className="flex items-center justify-between gap-2 mb-1.5">
          <span className="text-sm font-medium text-gray-600">Deploy Target</span>
          <span className={clsx(
            'inline-flex items-center gap-1 rounded-md px-2 py-0.5 text-xs font-semibold whitespace-nowrap',
            isRobotMode
              ? 'bg-orange-100 text-orange-700'
              : 'bg-emerald-100 text-emerald-700'
          )}>
            {isRobotMode ? <MdWarningAmber size={14} /> : <MdViewInAr size={14} />}
            {isRobotMode ? 'Commands Enabled' : 'Commands Blocked'}
          </span>
        </div>
        <div className="grid grid-cols-2 gap-1">
          <button
            type="button"
            onClick={() => handleDeployModeChange('simulation')}
            disabled={isModeSwitchLocked}
            className={deployButtonClass(!isRobotMode)}
            aria-label="Use 3D Sim Deploy"
            title="3D Sim Deploy"
          >
            <MdViewInAr size={17} className="shrink-0" />
            <span className="truncate">3D Sim Deploy</span>
          </button>
          <button
            type="button"
            onClick={() => handleDeployModeChange('robot')}
            disabled={isModeSwitchLocked}
            className={deployButtonClass(isRobotMode, true)}
            aria-label="Use Real Robot Deploy"
            title="Real Robot Deploy"
          >
            <MdPrecisionManufacturing size={17} className="shrink-0" />
            <span className="truncate">Real Robot Deploy</span>
          </button>
        </div>
      </div>

      {/* Edit mode indicator */}
      <div
        className={clsx('mb-3', 'p-2', 'rounded-md', 'text-sm', 'font-medium', {
          'bg-green-100 text-green-800': isEditable,
          'bg-gray-100 text-gray-600': !isEditable,
        })}
      >
        {isEditable ? (
          '✏️ Edit mode'
        ) : (
          <div className="leading-tight">
            <div>🔒 Read only</div>
            <div className="text-xs mt-1 opacity-80">task is running or robot is not connected</div>
          </div>
        )}
      </div>

      {/* Task Instruction — only shown for language-conditioned policies.
          Whitelist lives in constants/policyCapabilities.js. */}
      {showInstruction && (
        <>
          <div className={clsx('flex', 'items-start', 'mb-1')}>
            <span className={clsx(classLabel, 'pt-2')}>Task Instruction</span>
            <textarea
              className={classTaskInstructionTextarea}
              value={(info.taskInstruction && info.taskInstruction[0]) || ''}
              onChange={(e) => handleChange('taskInstruction', [e.target.value])}
              placeholder="Enter Task Instruction"
            />
          </div>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <span className={classLabel} />
            <button
              type="button"
              onClick={handleUpdateInstruction}
              disabled={!canUpdateInstruction}
              className={clsx(
                'px-3 py-1.5 text-sm rounded-md',
                'bg-blue-500 text-white hover:bg-blue-600',
                'disabled:bg-gray-300 disabled:text-gray-500 disabled:cursor-not-allowed',
                'transition-colors'
              )}
              title={
                isInferencing
                  ? 'Send the current Task Instruction to the running policy'
                  : 'Inference must be running to update the instruction'
              }
            >
              {isUpdatingInstruction ? 'Updating…' : 'Update Task Instruction'}
            </button>
          </div>
        </>
      )}

      {/* Policy Path */}
      <div className={clsx('flex', 'items-start', 'mb-2.5')}>
        <span className={clsx(classLabel, 'pt-2')}>Policy Path</span>
        <div className="flex flex-row items-start gap-2 flex-1 min-w-0">
          <textarea
            className={classPolicyPathTextarea}
            value={info.policyPath || ''}
            onChange={(e) => handleChange('policyPath', e.target.value)}
            disabled={!isEditable}
            placeholder="Enter Policy Path or Repo ID"
          />
          <button
            type="button"
            onClick={() => setShowPolicyBrowser(true)}
            disabled={!isEditable}
            className="flex items-center justify-center w-9 h-9 text-blue-500 bg-gray-200 rounded-md hover:text-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            aria-label="Browse for policy model folder"
          >
            <MdFolderOpen className="w-5 h-5" />
          </button>
        </div>
      </div>

      {supportsTensorRt && (
        <>
          <div className={clsx('flex', 'items-center', 'mb-2.5')}>
            <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
              <Tooltip content="Run GR00T with DiT TensorRT acceleration." position="bottom">
                <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
              </Tooltip>
              <span>TensorRT</span>
            </div>
            <label className={clsx('flex', 'items-center', 'gap-2', 'text-sm')}>
              <input
                type="checkbox"
                className={clsx('w-4 h-4', {
                  'cursor-not-allowed opacity-50': !isEditable,
                  'cursor-pointer': isEditable,
                })}
                checked={isTensorRtEnabled}
                onChange={(e) => handleChange(
                  'accelerationMode',
                  e.target.checked ? 'tensorrt_dit' : 'pytorch'
                )}
                disabled={!isEditable}
              />
              <span className="text-gray-500">Enable</span>
            </label>
          </div>
          {isTensorRtEnabled && (
            <TrtEngineControl
              modelPath={info.policyPath}
              enginePath={info.accelerationEnginePath}
              actionRequestMode={actionRequestMode}
              robotType={robotType}
              taskInstruction={trtTaskInstruction}
              disabled={!isEditable}
              labelClassName={classLabel}
            />
          )}
        </>
      )}

      <div className="w-full h-1 my-2 border-t border-gray-300"></div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="Choose when the next action chunk is requested." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Action Request</span>
        </div>
        <div className="grid grid-cols-2 gap-1 flex-1 min-w-0">
          <button
            type="button"
            onClick={() => handleActionRequestModeChange('async')}
            disabled={!isEditable}
            className={actionModeButtonClass(actionRequestMode !== 'sync')}
            aria-label="Use async action requests"
            title="Async"
          >
            <MdSync size={16} className="shrink-0" />
            <span className="truncate">{actionRequestMode === 'tt_rtc' ? 'TT-RTC' : 'Async'}</span>
          </button>
          <button
            type="button"
            onClick={() => handleActionRequestModeChange('sync')}
            disabled={!isEditable}
            className={actionModeButtonClass(actionRequestMode === 'sync')}
            aria-label="Use sync action requests"
            title="Sync"
          >
            <MdHourglassEmpty size={16} className="shrink-0" />
            <span className="truncate">Sync</span>
          </button>
        </div>
      </div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip
            content="Move slowly to the first predicted robot pose before inference."
            position="bottom"
          >
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Initial Pose Sync</span>
        </div>
        <label className={clsx('flex', 'items-center', 'gap-2', 'text-sm')}>
          <input
            type="checkbox"
            className={clsx('w-4 h-4', {
              'cursor-not-allowed opacity-50': !isEditable || !isRobotMode,
              'cursor-pointer': isEditable && isRobotMode,
            })}
            checked={initialPoseSyncEnabled}
            onChange={(e) => handleChange('initialPoseSync', e.target.checked)}
            disabled={!isEditable || !isRobotMode}
            aria-label="Initial Pose Sync"
          />
          <span className="text-gray-500">Enable</span>
        </label>
      </div>

      {initialPoseSyncEnabled && (
        <div className={clsx('flex', 'items-center', 'mb-2.5')}>
          <span className={classLabel}>Duration (s)</span>
          <input
            className={clsx(classTextInput, {
              'bg-gray-100 cursor-not-allowed': !isEditable || !isRobotMode,
            })}
            type="number"
            step="0.5"
            min="1"
            max="60"
            value={info.initialPoseSyncDurationS ?? 5.0}
            onChange={(e) => {
              const value = e.target.value;
              handleChange(
                'initialPoseSyncDurationS',
                value === '' ? '' : Number(value)
              );
            }}
            disabled={!isEditable || !isRobotMode}
            aria-label="Initial Pose Sync duration"
          />
        </div>
      )}

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip
            content="Frame rate used by the training dataset. This determines the time interval between predicted action waypoints."
            position="bottom"
          >
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Dataset FPS</span>
        </div>
        <input
          className={classTextInput}
          type="number"
          step="1"
          min="1"
          value={info.inferenceHz || ''}
          onChange={(e) => {
            const v = e.target.value;
            handleChange('inferenceHz', v === '' ? '' : Number(v));
          }}
          disabled={!isEditable || actionRequestMode === 'tt_rtc'}
          aria-label="Dataset FPS"
        />
      </div>

      <div className={clsx('flex', 'items-center', 'mb-2.5')}>
        <div className={clsx(classLabel, 'flex', 'items-center', 'gap-1')}>
          <Tooltip content="Rate of commands sent to the robot." position="bottom">
            <MdInfoOutline className="text-gray-400 hover:text-gray-600 cursor-help" size={14} />
          </Tooltip>
          <span>Control Hz</span>
        </div>
        <input
          className={classTextInput}
          type="number"
          step="5"
          min="1"
          value={info.controlHz || ''}
          onChange={(e) => {
            const v = e.target.value;
            handleChange('controlHz', v === '' ? '' : Number(v));
          }}
          disabled={!isEditable}
          aria-label="Control Hz"
        />
      </div>

      {timingWarnings.length > 0 && (
        <div className="mb-2.5 space-y-1" role="status" aria-label="Timing warnings">
          {timingWarnings.map((warning) => (
            <div
              key={warning}
              className="flex items-start gap-1.5 text-xs text-amber-700"
            >
              <MdWarningAmber size={14} className="mt-0.5 shrink-0" />
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}

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
      />
    </div>
  );
}
