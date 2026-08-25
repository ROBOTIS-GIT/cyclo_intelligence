// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useState } from 'react';
import toast from 'react-hot-toast';
import { useDispatch, useSelector } from 'react-redux';
import {
  MdArrowForward,
  MdCloudUpload,
  MdKeyboardDoubleArrowLeft,
  MdKeyboardDoubleArrowRight,
  MdMemory,
  MdModelTraining,
  MdOutlineDataset,
  MdRestartAlt,
  MdStorage,
  MdSwapHoriz,
  MdUndo,
} from 'react-icons/md';
import OfflineRLInferenceWorkspace from './OfflineRLInferenceWorkspace';
import OfflineRLReplayBuffer from './OfflineRLReplayBuffer';
import OfflineRLDatasetConversion from './OfflineRLDatasetConversion';
import OfflineRLLeRobotDataset from './OfflineRLLeRobotDataset';
import OfflineRLTrainingSection from './OfflineRLTrainingSection';
import OfflineRLWorkspaceStatusModal from './OfflineRLWorkspaceStatusModal';
import { InferencePhase, RecordPhase } from '../../../constants/taskPhases';
import {
  getFlowSDEPPOStatus,
  startFlowSDEPPOTraining,
  stopFlowSDEPPOTraining,
  submitFlowSDEPPOOutcome,
} from '../../../utils/offlineRlApi';
import {
  markLocalTaskInfoEdited,
  selectInferenceRecordingControl,
  setInferenceTaskInfo,
} from '../../tasks/taskSlice';
import {
  advanceOfflineRLLineage,
  createOfflineRLLineage,
  persistOfflineRLLineageState,
  resolveOfflineRLLineageState,
} from '../../../utils/offlineRlLineageState';

const SURFACE_CLASS = [
  'rounded-xl border border-[#ded8cc] bg-[#fbfaf6]',
  'shadow-[0_5px_16px_rgba(69,61,47,0.055)]',
].join(' ');

const formatRLEpoch = (value) => {
  const parsed = Number(value);
  const epoch = Number.isInteger(parsed) && parsed >= 0 ? parsed : 0;
  return `E${String(epoch).padStart(4, '0')}`;
};

function SectionHeader({
  icon: Icon,
  eyebrow,
  title,
  badge,
  actions = null,
  onIconClick = null,
  iconButtonLabel = '',
}) {
  const iconClassName = [
    'grid h-8 w-8 shrink-0 place-items-center rounded-lg',
    'border border-[#ddd5c7] bg-[#f1ede4] text-[#555046]',
  ].join(' ');

  return (
    <div className="flex items-start justify-between gap-3">
      <div className="flex min-w-0 items-center gap-2.5">
        {onIconClick ? (
          <button
            type="button"
            onClick={onIconClick}
            aria-label={iconButtonLabel}
            title={iconButtonLabel}
            className={`${iconClassName} transition-colors hover:bg-[#e7e1d6] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-2`}
          >
            <Icon size={17} aria-hidden="true" />
          </button>
        ) : (
          <span className={iconClassName}>
            <Icon size={17} aria-hidden="true" />
          </span>
        )}
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
            {eyebrow}
          </p>
          <h2 className="truncate text-sm font-semibold text-[#292720]">{title}</h2>
        </div>
      </div>
      {actions}
      {!actions && badge && (
        <span className="shrink-0 rounded-full border border-[#d9d2c5] bg-white px-2.5 py-1 text-[10px] font-semibold text-[#6f685d]">
          {badge}
        </span>
      )}
    </div>
  );
}

function FlowArrow() {
  return (
    <div className="flex items-center justify-center py-1 text-[#aaa295] lg:py-0">
      <MdArrowForward className="rotate-90 lg:rotate-0" size={19} aria-hidden="true" />
    </div>
  );
}

function PipelineCard({ icon: Icon, step, title, children }) {
  return (
    <section className={`${SURFACE_CLASS} flex h-full min-h-[260px] flex-col p-4`}>
      <SectionHeader icon={Icon} eyebrow={`Step ${step}`} title={title} />
      <div className="mt-3 flex min-h-0 flex-1 flex-col">{children}</div>
    </section>
  );
}

export default function RLWorkflowLayout({ isActive = true }) {
  const dispatch = useDispatch();
  const inferencePhase = useSelector(
    (state) => state.tasks.inferenceStatus.inferencePhase
  );
  const recordPhase = useSelector(
    (state) => state.tasks.recordStatus.recordPhase
  );
  const recordingControl = useSelector(selectInferenceRecordingControl);
  const selectedPolicyPath = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.policyPath || '').trim()
  );
  const selectedServiceType = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.serviceType || '').trim()
  );
  const selectedPolicyType = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.policyType || '').trim()
  );
  const [showWorkspaceStatus, setShowWorkspaceStatus] = useState(false);
  const [isWorkflowPanelCollapsed, setIsWorkflowPanelCollapsed] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState('inference');
  const [trainingInteractionLocked, setTrainingInteractionLocked] = useState(true);
  const [rlLineage, setRLLineage] = useState(() => resolveOfflineRLLineageState());
  const [deploymentState, setDeploymentState] = useState({
    ready: false,
    modelPath: '',
    serviceType: 'lerobot',
    policyType: 'act',
    rlEpoch: 0,
  });
  const [activeDeployment, setActiveDeployment] = useState({
    modelPath: '',
    serviceType: '',
    policyType: '',
    previousLineage: null,
    previousTaskInfo: null,
  });

  useEffect(() => {
    persistOfflineRLLineageState(rlLineage);
  }, [rlLineage]);

  const handleDeploymentStateChange = useCallback((nextState) => {
    setDeploymentState({
      ready: Boolean(nextState?.ready),
      modelPath: String(nextState?.modelPath || '').trim(),
      serviceType: String(nextState?.serviceType || 'lerobot').trim(),
      policyType: String(nextState?.policyType || 'act').trim(),
      rlEpoch: Number.isInteger(Number(nextState?.rlEpoch)) && Number(nextState.rlEpoch) >= 0
        ? Number(nextState.rlEpoch)
        : rlLineage.policyEpoch,
    });
  }, [rlLineage.policyEpoch]);

  const handleDeployPolicy = useCallback(() => {
    if (!deploymentState.ready || !deploymentState.modelPath) return;
    setActiveDeployment({
      modelPath: deploymentState.modelPath,
      serviceType: deploymentState.serviceType,
      policyType: deploymentState.policyType,
      previousLineage: rlLineage,
      previousTaskInfo: {
        policyPath: selectedPolicyPath,
        serviceType: selectedServiceType,
        policyType: selectedPolicyType,
      },
    });
    dispatch(setInferenceTaskInfo({
      policyPath: deploymentState.modelPath,
      serviceType: deploymentState.serviceType,
      policyType: deploymentState.policyType,
    }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    setRLLineage((current) => advanceOfflineRLLineage(current, {
      policyEpoch: deploymentState.rlEpoch,
      policyPath: deploymentState.modelPath,
    }));
    toast.success('Trained policy selected for inference');
  }, [
    deploymentState,
    dispatch,
    rlLineage,
    selectedPolicyPath,
    selectedPolicyType,
    selectedServiceType,
  ]);

  const handleDiscardPolicy = useCallback(() => {
    if (
      !activeDeployment.modelPath ||
      selectedPolicyPath !== activeDeployment.modelPath
    ) {
      return;
    }
    dispatch(setInferenceTaskInfo(activeDeployment.previousTaskInfo));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    if (activeDeployment.previousLineage) {
      setRLLineage(activeDeployment.previousLineage);
    }
    setActiveDeployment({
      modelPath: '',
      serviceType: '',
      policyType: '',
      previousLineage: null,
      previousTaskInfo: null,
    });
    toast.success('Policy removed from inference; trained files retained');
  }, [activeDeployment, dispatch, selectedPolicyPath]);

  useEffect(() => {
    if (
      activeDeployment.modelPath &&
      (
        selectedPolicyPath !== activeDeployment.modelPath ||
        selectedServiceType !== activeDeployment.serviceType ||
        selectedPolicyType !== activeDeployment.policyType
      )
    ) {
      setActiveDeployment({
        modelPath: '',
        serviceType: '',
        policyType: '',
        previousLineage: null,
        previousTaskInfo: null,
      });
    }
  }, [
    activeDeployment,
    selectedPolicyPath,
    selectedPolicyType,
    selectedServiceType,
  ]);

  const deployedCandidateSelected = Boolean(
    deploymentState.modelPath &&
    selectedPolicyPath === deploymentState.modelPath &&
    selectedServiceType === deploymentState.serviceType &&
    selectedPolicyType === deploymentState.policyType
  );

  const workspaceModeSwitchLocked = (
    !isActive ||
    inferencePhase !== InferencePhase.READY ||
    recordPhase !== RecordPhase.READY ||
    recordingControl.lifecycleLocked
  );
  const isRecordingWorkspace = workspaceMode === 'recording';

  const lineageResetDisabled = (
    workspaceModeSwitchLocked ||
    trainingInteractionLocked
  );

  const handleNewRLLineage = useCallback(() => {
    if (lineageResetDisabled) return;
    const confirmed = window.confirm(
      'Start a new RL lineage at Epoch 0?\n\n' +
      'The previous ACT-TD3 training state will not be resumed. ' +
      'Saved MCAP, LeRobot datasets, models, and checkpoints will not be deleted.'
    );
    if (!confirmed) return;

    setRLLineage(createOfflineRLLineage(selectedPolicyPath));
    setDeploymentState({
      ready: false,
      modelPath: '',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 0,
    });
    setActiveDeployment({
      modelPath: '',
      serviceType: '',
      policyType: '',
      previousLineage: null,
      previousTaskInfo: null,
    });
    toast.success('New RL lineage started at Epoch 0; saved files were retained');
  }, [lineageResetDisabled, selectedPolicyPath]);

  const handleFreshLineageConsumed = useCallback(() => {
    setRLLineage((current) => ({
      ...current,
      forceFresh: false,
    }));
  }, []);

  const deployDisabled = (
    !isActive ||
    !deploymentState.ready ||
    inferencePhase !== InferencePhase.READY ||
    deployedCandidateSelected
  );
  const discardDisabled = (
    !isActive ||
    inferencePhase !== InferencePhase.READY ||
    !activeDeployment.modelPath ||
    selectedPolicyPath !== activeDeployment.modelPath ||
    selectedServiceType !== activeDeployment.serviceType ||
    selectedPolicyType !== activeDeployment.policyType
  );

  const closeWorkspaceStatus = useCallback(() => {
    setShowWorkspaceStatus(false);
  }, []);

  useEffect(() => {
    if (!isActive) closeWorkspaceStatus();
  }, [closeWorkspaceStatus, isActive]);

  return (
    <div
      className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-[#f3f0e8] text-[#302d27]"
      style={{ fontFamily: "'Pretendard Variable', sans-serif" }}
    >
      <main className="min-h-0 w-full min-w-0 flex-1 overflow-y-auto p-4 lg:p-5 2xl:p-6">
        <div className="flex min-h-full rounded-2xl border border-[#ded9cf] bg-[#eeebe3] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]">
          <div
            className={`grid w-full flex-1 gap-4 ${
              isWorkflowPanelCollapsed
                ? 'xl:grid-cols-[minmax(0,1fr)]'
                : 'xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]'
            }`}
            data-testid="offline-rl-workflow-grid"
            data-workflow-panel-collapsed={String(isWorkflowPanelCollapsed)}
          >
            <section className={`${SURFACE_CLASS} flex min-h-[570px] min-w-0 flex-col p-4`}>
              <SectionHeader
                icon={MdMemory}
                eyebrow="Environment"
                title={isRecordingWorkspace
                  ? 'Recording Workspace'
                  : 'Inference Workspace'}
                onIconClick={() => setShowWorkspaceStatus(true)}
                iconButtonLabel="Open inference workspace status"
                actions={(
                  <div className="flex shrink-0 items-center gap-2">
                    <div
                      className="flex items-center rounded-lg border border-[#d4ccbf] bg-[#f1ede4] p-0.5"
                      role="group"
                      aria-label="Workspace mode"
                    >
                      {['inference', 'recording'].map((mode) => {
                        const selected = workspaceMode === mode;
                        const label = mode === 'inference' ? 'Inference' : 'Recording';
                        const selectedClass = mode === 'recording'
                          ? 'bg-[#a86b68] text-white shadow-sm'
                          : 'bg-[#69866f] text-white shadow-sm';
                        return (
                          <button
                            key={mode}
                            type="button"
                            onClick={() => setWorkspaceMode(mode)}
                            disabled={workspaceModeSwitchLocked}
                            aria-pressed={selected}
                            title={workspaceModeSwitchLocked
                              ? 'Stop inference or recording before switching workspace'
                              : `Switch to ${label.toLowerCase()} workspace`}
                            className={`h-7 rounded-md px-2.5 text-[10px] font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1 ${
                              selected
                                ? selectedClass
                                : 'text-[#6f685d] hover:bg-[#e5dfd4]'
                            } disabled:cursor-not-allowed disabled:opacity-50`}
                          >
                            {label}
                          </button>
                        );
                      })}
                    </div>
                    <button
                      type="button"
                      onClick={() => setIsWorkflowPanelCollapsed((collapsed) => !collapsed)}
                      aria-label={isWorkflowPanelCollapsed
                        ? 'Show workflow panel'
                        : 'Hide workflow panel'}
                      aria-expanded={!isWorkflowPanelCollapsed}
                      title={isWorkflowPanelCollapsed
                        ? 'Show workflow panel'
                        : 'Hide workflow panel'}
                      className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-[#d4ccbf] bg-[#f5f2eb] text-[#6f685d] shadow-sm transition-colors hover:bg-[#e7e1d6] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1"
                    >
                      {isWorkflowPanelCollapsed ? (
                        <MdKeyboardDoubleArrowLeft size={19} aria-hidden="true" />
                      ) : (
                        <MdKeyboardDoubleArrowRight size={19} aria-hidden="true" />
                      )}
                    </button>
                  </div>
                )}
              />
              <OfflineRLInferenceWorkspace
                isActive={isActive}
                workspaceMode={workspaceMode}
                policyEpoch={rlLineage.policyEpoch}
              />
            </section>

            <div
              className={`${
                isWorkflowPanelCollapsed ? 'hidden' : 'grid'
              } min-h-0 min-w-0 content-start gap-4 xl:grid-rows-[minmax(260px,2fr)_minmax(390px,3fr)_auto]`}
              data-testid="offline-rl-workflow-steps"
              aria-hidden={isWorkflowPanelCollapsed}
            >
              <div
                className="grid h-full min-h-[260px] items-stretch gap-2 lg:grid-cols-[minmax(0,1fr)_28px_minmax(0,1fr)_28px_minmax(0,1fr)]"
                data-testid="offline-rl-dataset-pipeline"
              >
                <PipelineCard icon={MdStorage} step="01" title="Replay Buffer">
                  <OfflineRLReplayBuffer isActive={isActive} />
                </PipelineCard>

                <FlowArrow />

                <PipelineCard icon={MdSwapHoriz} step="02" title="Dataset Conversion">
                  <OfflineRLDatasetConversion isActive={isActive} />
                </PipelineCard>

                <FlowArrow />

                <PipelineCard icon={MdOutlineDataset} step="03" title="LeRobot Dataset">
                  <OfflineRLLeRobotDataset isActive={isActive} />
                </PipelineCard>
              </div>

              <section className={`${SURFACE_CLASS} flex min-h-0 min-w-0 flex-col p-3.5`}>
                <SectionHeader
                  icon={MdModelTraining}
                  eyebrow="Step 04"
                  title="Training"
                  actions={(
                    <div className="flex shrink-0 items-center gap-1.5">
                      <span
                        className="rounded-full border border-[#cfd8cd] bg-[#e8eee6] px-2.5 py-1 font-mono text-[9px] font-bold text-[#58705d]"
                        aria-label={`Training policy RL Epoch ${rlLineage.policyEpoch} to ${rlLineage.policyEpoch + 1}`}
                      >
                        RL Epoch {formatRLEpoch(rlLineage.policyEpoch)} → {formatRLEpoch(rlLineage.policyEpoch + 1)}
                      </span>
                      <button
                        type="button"
                        onClick={handleNewRLLineage}
                        disabled={lineageResetDisabled}
                        className="flex h-7 items-center gap-1 rounded-md border border-[#d4ccbf] bg-[#f5f2eb] px-2 text-[9px] font-semibold text-[#6f685d] transition-colors hover:bg-[#e7e1d6] disabled:cursor-not-allowed disabled:opacity-45"
                        title={lineageResetDisabled
                          ? 'Stop inference, recording, or training before starting a new RL lineage'
                          : 'Start a new RL lineage without deleting saved files'}
                        aria-label="New RL Lineage"
                      >
                        <MdRestartAlt size={13} aria-hidden="true" />
                        New Lineage
                      </button>
                    </div>
                  )}
                />
                <OfflineRLTrainingSection
                  isActive={isActive}
                  variant="workflow"
                  inferencePhase={inferencePhase}
                  currentPolicyEpoch={rlLineage.policyEpoch}
                  forceFreshLineage={rlLineage.forceFresh}
                  onFreshLineageConsumed={handleFreshLineageConsumed}
                  onRunningChange={setTrainingInteractionLocked}
                  flowSdePpoReady
                  getFlowSDEPPOStatus={getFlowSDEPPOStatus}
                  onStartFlowSDEPPO={startFlowSDEPPOTraining}
                  onStopFlowSDEPPO={stopFlowSDEPPOTraining}
                  onSubmitFlowSDEPPOOutcome={submitFlowSDEPPOOutcome}
                  onDeploymentStateChange={handleDeploymentStateChange}
                />
              </section>

              <section className={`${SURFACE_CLASS} flex items-center justify-between gap-4 p-4`}>
                <div className="flex min-w-0 items-center gap-3">
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[#eeebe3] text-[#6f685d]">
                    <MdCloudUpload size={18} />
                  </span>
                  <div className="min-w-0">
                    <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#91897d]">
                      Step 05 · Deployment
                    </div>
                    <div className="truncate text-[11px] font-semibold text-[#403b34]">
                      Policy Deploy
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={handleDiscardPolicy}
                    disabled={discardDisabled}
                    className={discardDisabled
                      ? 'flex h-9 shrink-0 cursor-not-allowed items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-[#f0ede6] px-4 text-[10px] font-semibold text-[#9a9286]'
                      : 'flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[#a86b68] bg-[#a86b68] px-4 text-[10px] font-semibold text-white hover:bg-[#965d5a]'}
                  >
                    <MdUndo size={14} /> Discard Policy
                  </button>
                  <button
                    type="button"
                    onClick={handleDeployPolicy}
                    disabled={deployDisabled}
                    className={deployDisabled
                      ? 'flex h-9 shrink-0 cursor-not-allowed items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-[#f0ede6] px-4 text-[10px] font-semibold text-[#9a9286]'
                      : 'flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[#5f7965] bg-[#69866f] px-4 text-[10px] font-semibold text-white hover:bg-[#5f7965]'}
                  >
                    <MdCloudUpload size={14} /> Deploy Policy
                  </button>
                </div>
              </section>
            </div>
          </div>
        </div>
      </main>
      <OfflineRLWorkspaceStatusModal
        isOpen={isActive && showWorkspaceStatus}
        onClose={closeWorkspaceStatus}
      />
    </div>
  );
}
