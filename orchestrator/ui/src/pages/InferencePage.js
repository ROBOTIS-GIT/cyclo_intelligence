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

import React, { useState, useEffect } from 'react';
import { shallowEqual, useSelector, useDispatch } from 'react-redux';
import clsx from 'clsx';
import toast, { useToasterStore } from 'react-hot-toast';
import {
  MdKeyboardDoubleArrowLeft,
  MdKeyboardDoubleArrowRight,
  MdViewInAr,
} from 'react-icons/md';
import InferenceControlPanel from '../components/InferenceControlPanel';
import HeartbeatStatus from '../components/HeartbeatStatus';
import InlineSystemStatus from '../components/InlineSystemStatus';
import ImageGrid from '../components/ImageGrid';
import RobotViewer3D from '../components/RobotViewer3D';
import InferencePanel from '../components/InferencePanel';
import InferenceGripperPanel from '../components/InferenceGripperPanel';
import InferencePosePanel from '../components/InferencePosePanel';
import InferenceDaggerPanel from '../components/InferenceDaggerPanel';
import RecordTopicMonitor from '../components/RecordTopicMonitor';
import { selectInferenceTaskInfo } from '../features/tasks/taskSlice';
import { setIsFirstLoadFalse } from '../features/ui/uiSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

export default function InferencePage({ isActive = true }) {
  const dispatch = useDispatch();
  const { sendRecordCommand } = useRosServiceCaller();

  // Toast limit implementation using useToasterStore
  const { toasts } = useToasterStore();
  const TOAST_LIMIT = 3;

  const robotType = useSelector((state) => state.tasks.robotType);
  const joystickMode = useSelector((state) => state.tasks.joystickMode);
  const taskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);

  const [isRightPanelCollapsed, setIsRightPanelCollapsed] = useState(false);
  const [show3DViewer, setShow3DViewer] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState('settings');
  const [gripperBusy, setGripperBusy] = useState(false);
  const [poseBusy, setPoseBusy] = useState(false);
  const [daggerBusy, setDaggerBusy] = useState(false);

  const isFirstLoad = useSelector((state) => state.ui.isFirstLoad.inference);

  useEffect(() => {
    toasts
      .filter((t) => t.visible)
      .filter((_, i) => i >= TOAST_LIMIT)
      .forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  useEffect(() => {
    dispatch(setIsFirstLoadFalse('inference'));
  }, [dispatch, isFirstLoad]);

  // Refresh topic subscriptions when entering the page.
  useEffect(() => {
    if (isActive) {
      sendRecordCommand('refresh_topics').catch(() => {});
    }
  }, [isActive, sendRecordCommand]);

  const inferenceMode = taskInfo.inferenceMode || 'simulation';
  const isRobotMode = inferenceMode === 'robot';

  const classMainContainer = 'h-full flex flex-col overflow-hidden';
  const classContentsArea = 'flex-1 flex min-h-0 pt-0 px-0 justify-center items-start';
  const classLeftArea = clsx(
    'transition-all',
    'duration-300',
    'ease-in-out',
    'flex',
    'flex-col',
    'min-h-0',
    'h-full',
    'overflow-hidden',
    'm-2',
    {
      'flex-[12]': isRightPanelCollapsed,
      'flex-[10]': !isRightPanelCollapsed,
    }
  );

  const classRightPanelArea = clsx(
    'h-full',
    'w-full',
    'transition-all',
    'duration-300',
    'ease-in-out',
    'relative',
    'overflow-y-auto',
    {
      'flex-[0_0_40px]': isRightPanelCollapsed,
      'flex-[1]': !isRightPanelCollapsed,
      'min-w-[60px]': isRightPanelCollapsed,
      'min-w-[400px]': !isRightPanelCollapsed,
      'max-w-[60px]': isRightPanelCollapsed,
      'max-w-[400px]': !isRightPanelCollapsed,
    }
  );

  const classHideButton = clsx(
    'absolute',
    'top-3',
    'bg-white',
    'border',
    'border-gray-300',
    'rounded-full',
    'w-12',
    'h-12',
    'flex',
    'items-center',
    'justify-center',
    'shadow-md',
    'hover:bg-gray-50',
    'transition-all',
    'duration-200',
    'z-10',
    {
      'left-2': isRightPanelCollapsed,
      'left-[10px]': !isRightPanelCollapsed,
    }
  );

  const classRightPanel = clsx(
    'h-full',
    'flex',
    'flex-col',
    'items-center',
    'overflow-hidden',
    'transition-opacity',
    'duration-300',
    {
      'opacity-0': isRightPanelCollapsed,
      'opacity-100': !isRightPanelCollapsed,
      'pointer-events-none': isRightPanelCollapsed,
      'pointer-events-auto': !isRightPanelCollapsed,
    }
  );

  const classTopBar = clsx(
    'absolute', 'top-4', 'left-4', 'right-4', 'z-20',
    'flex', 'items-center', 'gap-4'
  );
  const classRobotTypeContainer = clsx(
    'flex', 'flex-row', 'items-center',
    'bg-white/90', 'backdrop-blur-sm',
    'rounded-full', 'px-3', 'py-1',
    'shadow-md', 'border', 'border-gray-100',
    'whitespace-nowrap', 'shrink-0'
  );
  const classRobotType = clsx('ml-1 mr-1 text-gray-600 text-sm');
  const classRobotTypeValue = clsx(
    'mx-0.5 px-2 py-0.5 text-sm text-blue-600 bg-blue-100 rounded-full',
    'whitespace-nowrap'
  );

  const classHeartbeatStatus = clsx('absolute', 'top-[4.5rem]', 'left-5', 'z-10');

  return (
    <div className={classMainContainer}>
      <div className={classContentsArea}>
        <div className={classLeftArea}>
          <div className="relative flex-[5] min-h-0 overflow-hidden pt-20">
            <div className={classTopBar}>
              <div className={classRobotTypeContainer}>
                <div className={classRobotType}>Robot Type</div>
                <div className={classRobotTypeValue}>{robotType}</div>
              </div>
              {joystickMode && (
                <div className={classRobotTypeContainer}>
                  <div className={classRobotType}>Mode</div>
                  <div className={classRobotTypeValue}>
                    {joystickMode}
                  </div>
                </div>
              )}
              <InlineSystemStatus />
              <div className="flex-grow" />
              <InferenceControlPanel manualControlBusy={gripperBusy || poseBusy || daggerBusy} />
            </div>
            <div className={classHeartbeatStatus}>
              <HeartbeatStatus />
            </div>
            <ImageGrid isActive={isActive} />
          </div>
          <div className="flex-[4] min-h-[120px] flex flex-row items-center justify-center mx-1 gap-2 h-full relative">
            {show3DViewer && (
              <div className="h-[85%] rounded-2xl overflow-hidden relative" style={{ aspectRatio: '4/3' }}>
                <RobotViewer3D
                  mode="live"
                  showSourceSelector
                  defaultVisualizationSource={isRobotMode ? 'state' : 'action'}
                />
              </div>
            )}
            <div className="h-[85%]" style={{ aspectRatio: '4/3' }}>
              <RecordTopicMonitor />
            </div>
            <button
              onClick={() => setShow3DViewer(!show3DViewer)}
              className={clsx(
                'absolute top-2 left-2 z-10 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-colors shadow-md border',
                show3DViewer
                  ? 'bg-indigo-500/90 text-white border-indigo-400 backdrop-blur-sm'
                  : 'bg-white/90 text-gray-600 border-gray-100 backdrop-blur-sm hover:bg-gray-50'
              )}
            >
              <MdViewInAr size={18} />
              3D
            </button>
          </div>
        </div>
        <div className={classRightPanelArea}>
          <button
            onClick={() => setIsRightPanelCollapsed(!isRightPanelCollapsed)}
            className={classHideButton}
            title="Hide"
          >
            <span className="text-gray-600 text-3xl transition-transform duration-200">
              {isRightPanelCollapsed ? (
                <MdKeyboardDoubleArrowLeft />
              ) : (
                <MdKeyboardDoubleArrowRight />
              )}
            </span>
          </button>
          <div className={classRightPanel}>
            <div className="w-full min-h-10"></div>
            <div role="tablist" aria-label="Inference tools" className="flex w-full shrink-0 border-b border-gray-200 px-5 pt-3"
              onKeyDown={(event) => { if (event.key === ' ' || event.key === 'Enter') event.stopPropagation(); }}
              onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') event.stopPropagation(); }}>
              {[['settings', 'Settings'], ['gripper', 'Gripper'], ['pose', 'Pose'], ['dagger', 'DAgger']].map(([key, label], index) => (
                <button key={key} id={`inference-${key}-tab`} type="button" role="tab"
                  aria-selected={rightPanelTab === key} aria-controls={`inference-${key}-panel`}
                  tabIndex={rightPanelTab === key ? 0 : -1}
                  className={clsx('flex-1 border-b-2 px-2 py-3 text-sm font-medium', rightPanelTab === key
                    ? 'border-blue-600 text-blue-700' : 'border-transparent text-gray-500 hover:text-gray-700')}
                  onClick={() => setRightPanelTab(key)}
                  onKeyDown={(event) => {
                    const keys = ['settings', 'gripper', 'pose', 'dagger'];
                    let next;
                    if (event.key === 'ArrowRight') next = keys[(index + 1) % keys.length];
                    else if (event.key === 'ArrowLeft') next = keys[(index + keys.length - 1) % keys.length];
                    else if (event.key === 'Home') next = keys[0];
                    else if (event.key === 'End') next = keys[keys.length - 1];
                    if (next) {
                      event.preventDefault();
                      setRightPanelTab(next);
                      document.getElementById(`inference-${next}-tab`)?.focus();
                    }
                  }}>{label}</button>
              ))}
            </div>
            <div id="inference-settings-panel" role="tabpanel" aria-labelledby="inference-settings-tab"
              hidden={rightPanelTab !== 'settings'} className="w-full min-h-0 overflow-y-auto">
              <InferencePanel />
            </div>
            <div id="inference-gripper-panel" role="tabpanel" aria-labelledby="inference-gripper-tab"
              hidden={rightPanelTab !== 'gripper'} className="w-full min-h-0 overflow-y-auto">
              <InferenceGripperPanel isActive={isActive && rightPanelTab === 'gripper'} externalBusy={poseBusy || daggerBusy} onBusyChange={setGripperBusy} />
            </div>
            <div id="inference-pose-panel" role="tabpanel" aria-labelledby="inference-pose-tab"
              hidden={rightPanelTab !== 'pose'} className="w-full min-h-0 overflow-y-auto">
              <InferencePosePanel isActive={isActive && rightPanelTab === 'pose'} externalBusy={gripperBusy || daggerBusy} onBusyChange={setPoseBusy} />
            </div>
            <div id="inference-dagger-panel" role="tabpanel" aria-labelledby="inference-dagger-tab"
              hidden={rightPanelTab !== 'dagger'} className="w-full min-h-0 overflow-y-auto">
              <InferenceDaggerPanel isActive={isActive} externalBusy={gripperBusy || poseBusy} onBusyChange={setDaggerBusy} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
