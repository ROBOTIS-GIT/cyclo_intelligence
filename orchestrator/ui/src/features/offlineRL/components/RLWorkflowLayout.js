// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useRef, useState } from 'react';
import toast from 'react-hot-toast';
import { useDispatch, useSelector } from 'react-redux';
import {
  MdCloudUpload,
  MdDns,
  MdModelTraining,
  MdOutlineDataset,
  MdRestartAlt,
  MdSwapHoriz,
  MdUndo,
} from 'react-icons/md';
import OfflineRLInferenceWorkspace from './OfflineRLInferenceWorkspace';
import OfflineRLReplayBuffer from './OfflineRLReplayBuffer';
import OfflineRLDatasetConversion from './OfflineRLDatasetConversion';
import OfflineRLLeRobotDataset from './OfflineRLLeRobotDataset';
import OfflineRLTrainingSection from './OfflineRLTrainingSection';
import OfflineRLDataConversionGuideModal from './OfflineRLDataConversionGuideModal';
import OfflineRLTrainingGuideModal from './OfflineRLTrainingGuideModal';
import RLFrameworkRail from './RLFrameworkRail';
import PanelToggleGlyph from './PanelToggleGlyph';
import RobotLabIcon from './RobotLabIcon';
import { InferencePhase, RecordPhase } from '../../../constants/taskPhases';
import PageType from '../../../constants/pageType';
import {
  getFlowSDEPPOPolicyRolloutStatus,
  getFlowSDEPPOUpdateStatus,
  getFlowSDEPPOValueWarmupStatus,
  startFlowSDEPPOPolicyRollout,
  startFlowSDEPPOUpdate,
  stopFlowSDEPPOPolicyRollout,
  stopFlowSDEPPOUpdate,
  submitFlowSDEPPOPolicyRolloutOutcome,
} from '../../../utils/offlineRlApi';
import {
  markLocalTaskInfoEdited,
  selectInferenceRecordingControl,
  setInferenceTaskInfo,
} from '../../tasks/taskSlice';
import { moveToPage } from '../../ui/uiSlice';
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

function PipelineCard({
  icon: Icon,
  step,
  title,
  children,
  onIconClick = null,
  iconButtonLabel = '',
}) {
  return (
    <section
      className={`${SURFACE_CLASS} flex min-h-0 w-full shrink-0 flex-col p-3`}
      data-testid={`offline-rl-pipeline-step-${step}`}
    >
      <SectionHeader
        icon={Icon}
        eyebrow={`Step ${step}`}
        title={title}
        onIconClick={onIconClick}
        iconButtonLabel={iconButtonLabel}
      />
      <div className="mt-2 flex min-h-0 flex-1 flex-col">{children}</div>
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
  const selectedAccelerationMode = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.accelerationMode || 'pytorch').trim()
  );
  const selectedAccelerationEnginePath = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.accelerationEnginePath || '').trim()
  );
  const selectedRltEnabled = useSelector(
    (state) => Boolean(state.tasks.inferenceTaskInfo.rltEnabled)
  );
  const selectedRltBundlePath = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.rltBundlePath || '').trim()
  );
  const selectedRltRobotOverride = useSelector(
    (state) => Boolean(state.tasks.inferenceTaskInfo.rltRobotOverride)
  );
  const selectedActionPolicyMode = useSelector(
    (state) => String(state.tasks.inferenceTaskInfo.actionPolicyMode || 'base').trim()
  );
  const [showWorkspaceStatus, setShowWorkspaceStatus] = useState(false);
  const [showDataConversionGuide, setShowDataConversionGuide] = useState(false);
  const [showTrainingGuide, setShowTrainingGuide] = useState(false);
  const [frameworkPanels, setFrameworkPanels] = useState({
    replay: false,
    training: false,
    lastActive: null,
  });
  const [isFrameworkRailCollapsed, setIsFrameworkRailCollapsed] = useState(false);
  const replayDrawerCloseButtonRef = useRef(null);
  const trainingDrawerCloseButtonRef = useRef(null);
  const [workspaceMode, setWorkspaceMode] = useState('inference');
  const [trainingInteractionLocked, setTrainingInteractionLocked] = useState(true);
  const [trainingMethod, setTrainingMethod] = useState('reinforcement');
  const [isTrainingCompact, setIsTrainingCompact] = useState(false);
  const [flowSdeRolloutBundle, setFlowSdeRolloutBundle] = useState('');
  const [rlLineage, setRLLineage] = useState(() => resolveOfflineRLLineageState());
  const [deploymentState, setDeploymentState] = useState({
    ready: false,
    artifactKind: 'policy',
    modelPath: '',
    rltBundlePath: '',
    serviceType: 'lerobot',
    policyType: 'act',
    rlEpoch: 0,
    lineageMode: 'unchanged',
  });
  const [activeDeployment, setActiveDeployment] = useState({
    artifactKind: '',
    modelPath: '',
    rltBundlePath: '',
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
      artifactKind: nextState?.artifactKind === 'rlt_bundle'
        ? 'rlt_bundle'
        : 'policy',
      modelPath: String(nextState?.modelPath || '').trim(),
      rltBundlePath: String(nextState?.rltBundlePath || '').trim(),
      serviceType: String(nextState?.serviceType || 'lerobot').trim(),
      policyType: String(nextState?.policyType || 'act').trim(),
      rlEpoch: Number.isInteger(Number(nextState?.rlEpoch)) && Number(nextState.rlEpoch) >= 0
        ? Number(nextState.rlEpoch)
        : rlLineage.policyEpoch,
      lineageMode: ['new', 'advance'].includes(nextState?.lineageMode)
        ? nextState.lineageMode
        : 'unchanged',
    });
  }, [rlLineage.policyEpoch]);

  const handleDeployPolicy = useCallback(() => {
    const isRltBundle = deploymentState.artifactKind === 'rlt_bundle';
    const candidatePath = isRltBundle
      ? deploymentState.rltBundlePath
      : deploymentState.modelPath;
    if (!deploymentState.ready || !candidatePath) return;
    if (
      isRltBundle &&
      (
        !deploymentState.modelPath ||
        selectedPolicyPath !== deploymentState.modelPath ||
        selectedServiceType !== deploymentState.serviceType ||
        selectedPolicyType !== deploymentState.policyType
      )
    ) return;
    setActiveDeployment({
      artifactKind: deploymentState.artifactKind,
      modelPath: deploymentState.modelPath,
      rltBundlePath: deploymentState.rltBundlePath,
      serviceType: deploymentState.serviceType,
      policyType: deploymentState.policyType,
      previousLineage: rlLineage,
      previousTaskInfo: {
        policyPath: selectedPolicyPath,
        serviceType: selectedServiceType,
        policyType: selectedPolicyType,
        accelerationMode: selectedAccelerationMode,
        accelerationEnginePath: selectedAccelerationEnginePath,
        rltEnabled: selectedRltEnabled,
        rltBundlePath: selectedRltBundlePath,
        rltRobotOverride: selectedRltRobotOverride,
        actionPolicyMode: selectedActionPolicyMode,
      },
    });
    dispatch(setInferenceTaskInfo(isRltBundle ? {
      serviceType: deploymentState.serviceType,
      policyType: deploymentState.policyType,
      rltEnabled: true,
      rltBundlePath: deploymentState.rltBundlePath,
      rltRobotOverride: false,
      actionPolicyMode: 'base',
    } : {
      policyPath: deploymentState.modelPath,
      serviceType: deploymentState.serviceType,
      policyType: deploymentState.policyType,
      accelerationMode: 'pytorch',
      accelerationEnginePath: '',
    }));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    setRLLineage((current) => (
      deploymentState.lineageMode === 'new'
        ? createOfflineRLLineage(deploymentState.modelPath)
        : advanceOfflineRLLineage(current, {
          policyEpoch: deploymentState.rlEpoch,
          policyPath: isRltBundle ? selectedPolicyPath : deploymentState.modelPath,
        })
    ));
    toast.success(isRltBundle
      ? 'Trained RLT Bundle selected for inference'
      : 'Trained policy selected for inference');
  }, [
    deploymentState,
    dispatch,
    rlLineage,
    selectedAccelerationEnginePath,
    selectedAccelerationMode,
    selectedActionPolicyMode,
    selectedPolicyPath,
    selectedPolicyType,
    selectedRltBundlePath,
    selectedRltEnabled,
    selectedRltRobotOverride,
    selectedServiceType,
  ]);

  const handleDiscardPolicy = useCallback(() => {
    const isRltBundle = activeDeployment.artifactKind === 'rlt_bundle';
    const deploymentStillSelected = isRltBundle
      ? Boolean(
        activeDeployment.rltBundlePath &&
        selectedRltEnabled &&
        selectedRltBundlePath === activeDeployment.rltBundlePath &&
        selectedPolicyPath === activeDeployment.modelPath &&
        selectedServiceType === activeDeployment.serviceType &&
        selectedPolicyType === activeDeployment.policyType
      )
      : Boolean(
        activeDeployment.modelPath &&
        selectedPolicyPath === activeDeployment.modelPath
      );
    if (!deploymentStillSelected) return;
    dispatch(setInferenceTaskInfo(activeDeployment.previousTaskInfo));
    dispatch(markLocalTaskInfoEdited({ source: 'inference' }));
    if (activeDeployment.previousLineage) {
      setRLLineage(activeDeployment.previousLineage);
    }
    setActiveDeployment({
      artifactKind: '',
      modelPath: '',
      rltBundlePath: '',
      serviceType: '',
      policyType: '',
      previousLineage: null,
      previousTaskInfo: null,
    });
    toast.success(isRltBundle
      ? 'RLT Bundle removed from inference; trained files retained'
      : 'Policy removed from inference; trained files retained');
  }, [
    activeDeployment,
    dispatch,
    selectedPolicyPath,
    selectedPolicyType,
    selectedRltBundlePath,
    selectedRltEnabled,
    selectedServiceType,
  ]);

  const activeDeploymentSelected = activeDeployment.artifactKind === 'rlt_bundle'
    ? Boolean(
      activeDeployment.rltBundlePath &&
      selectedRltEnabled &&
      selectedRltBundlePath === activeDeployment.rltBundlePath &&
      selectedPolicyPath === activeDeployment.modelPath &&
      selectedServiceType === activeDeployment.serviceType &&
      selectedPolicyType === activeDeployment.policyType
    )
    : Boolean(
      activeDeployment.modelPath &&
      selectedPolicyPath === activeDeployment.modelPath &&
      selectedServiceType === activeDeployment.serviceType &&
      selectedPolicyType === activeDeployment.policyType
    );

  useEffect(() => {
    const hasActiveDeployment = Boolean(
      activeDeployment.modelPath || activeDeployment.rltBundlePath
    );
    if (hasActiveDeployment && !activeDeploymentSelected) {
      setActiveDeployment({
        artifactKind: '',
        modelPath: '',
        rltBundlePath: '',
        serviceType: '',
        policyType: '',
        previousLineage: null,
        previousTaskInfo: null,
      });
    }
  }, [
    activeDeployment,
    activeDeploymentSelected,
    selectedPolicyPath,
    selectedPolicyType,
    selectedServiceType,
  ]);

  const deployedCandidateSelected = Boolean(
    deploymentState.artifactKind === 'rlt_bundle'
      ? (
        deploymentState.rltBundlePath &&
        selectedRltEnabled &&
        selectedRltBundlePath === deploymentState.rltBundlePath &&
        selectedPolicyPath === deploymentState.modelPath &&
        selectedServiceType === deploymentState.serviceType &&
        selectedPolicyType === deploymentState.policyType
      )
      : (
        deploymentState.modelPath &&
        selectedPolicyPath === deploymentState.modelPath &&
        selectedServiceType === deploymentState.serviceType &&
        selectedPolicyType === deploymentState.policyType
      )
  );

  const workspaceModeSwitchLocked = (
    !isActive ||
    inferencePhase !== InferencePhase.READY ||
    recordPhase !== RecordPhase.READY ||
    recordingControl.lifecycleLocked
  );
  const isRecordingWorkspace = workspaceMode === 'recording';
  const isReplayDrawerOpen = isActive && frameworkPanels.replay;
  const isTrainingDrawerOpen = isActive && frameworkPanels.training;
  const activeFrameworkSection = isReplayDrawerOpen || isTrainingDrawerOpen
    ? (
      frameworkPanels.lastActive === 'replay' && isReplayDrawerOpen
        ? 'replay'
        : frameworkPanels.lastActive === 'training' && isTrainingDrawerOpen
          ? 'training'
          : isTrainingDrawerOpen ? 'training' : 'replay'
    )
    : 'environment';
  const frameworkPanelState = isReplayDrawerOpen && isTrainingDrawerOpen
    ? 'both'
    : activeFrameworkSection;
  const drawerWidthClass = isReplayDrawerOpen && isTrainingDrawerOpen
    ? 'lg:w-[calc(50%_-_1.5rem)]'
    : 'lg:w-1/2';

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
      artifactKind: 'policy',
      modelPath: '',
      rltBundlePath: '',
      serviceType: 'lerobot',
      policyType: 'act',
      rlEpoch: 0,
      lineageMode: 'unchanged',
    });
    setActiveDeployment({
      artifactKind: '',
      modelPath: '',
      rltBundlePath: '',
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
    (
      deploymentState.artifactKind === 'rlt_bundle' &&
      (
        !deploymentState.rltBundlePath ||
        !deploymentState.modelPath ||
        selectedPolicyPath !== deploymentState.modelPath ||
        selectedServiceType !== deploymentState.serviceType ||
        selectedPolicyType !== deploymentState.policyType
      )
    ) ||
    deployedCandidateSelected
  );
  const discardDisabled = (
    !isActive ||
    inferencePhase !== InferencePhase.READY ||
    !activeDeploymentSelected
  );
  const isRltBundleCandidate = deploymentState.artifactKind === 'rlt_bundle';
  const isRltBundleActive = activeDeployment.artifactKind === 'rlt_bundle';

  const closeWorkspaceStatus = useCallback(() => {
    setShowWorkspaceStatus(false);
  }, []);

  const closeTrainingGuide = useCallback(() => {
    setShowTrainingGuide(false);
  }, []);

  const closeDataConversionGuide = useCallback(() => {
    setShowDataConversionGuide(false);
  }, []);

  useEffect(() => {
    if (!isActive) {
      closeWorkspaceStatus();
      closeDataConversionGuide();
      closeTrainingGuide();
    }
  }, [closeDataConversionGuide, closeTrainingGuide, closeWorkspaceStatus, isActive]);

  const closeReplayDrawer = useCallback(() => {
    setFrameworkPanels((current) => ({
      ...current,
      replay: false,
      lastActive: current.training ? 'training' : null,
    }));
    if (typeof document !== 'undefined') {
      document.getElementById('rl-framework-section-replay')?.focus();
    }
  }, []);

  useEffect(() => {
    if (isReplayDrawerOpen) replayDrawerCloseButtonRef.current?.focus();
  }, [isReplayDrawerOpen]);

  useEffect(() => {
    if (!isReplayDrawerOpen) closeDataConversionGuide();
  }, [closeDataConversionGuide, isReplayDrawerOpen]);

  const closeTrainingDrawer = useCallback(() => {
    setFrameworkPanels((current) => ({
      ...current,
      training: false,
      lastActive: current.replay ? 'replay' : null,
    }));
    if (typeof document !== 'undefined') {
      document.getElementById('rl-framework-section-training')?.focus();
    }
  }, []);

  const handleFrameworkSectionChange = useCallback((nextSection) => {
    if (nextSection === 'environment') {
      setFrameworkPanels({ replay: false, training: false, lastActive: null });
      return;
    }
    if (nextSection !== 'replay' && nextSection !== 'training') return;
    setFrameworkPanels((current) => {
      const otherSection = nextSection === 'replay' ? 'training' : 'replay';
      const nextOpen = !current[nextSection];
      return {
        ...current,
        [nextSection]: nextOpen,
        lastActive: nextOpen
          ? nextSection
          : current[otherSection] ? otherSection : null,
      };
    });
    if (frameworkPanels[nextSection] && typeof document !== 'undefined') {
      document.getElementById(`rl-framework-section-${nextSection}`)?.focus();
    }
  }, [frameworkPanels]);

  useEffect(() => {
    if (isTrainingDrawerOpen) trainingDrawerCloseButtonRef.current?.focus();
  }, [isTrainingDrawerOpen]);

  useEffect(() => {
    if (
      (!isReplayDrawerOpen && !isTrainingDrawerOpen) ||
      showWorkspaceStatus ||
      showDataConversionGuide ||
      showTrainingGuide ||
      typeof document === 'undefined'
    ) return undefined;
    const handleKeyDown = (event) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      if (activeFrameworkSection === 'training') {
        closeTrainingDrawer();
      } else {
        closeReplayDrawer();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [
    activeFrameworkSection,
    closeReplayDrawer,
    closeTrainingDrawer,
    isReplayDrawerOpen,
    isTrainingDrawerOpen,
    showDataConversionGuide,
    showTrainingGuide,
    showWorkspaceStatus,
  ]);

  useEffect(() => {
    if (!isTrainingDrawerOpen) closeTrainingGuide();
  }, [closeTrainingGuide, isTrainingDrawerOpen]);

  return (
    <div
      className="flex h-full min-h-0 w-full overflow-hidden bg-[#f3f0e8] text-[#302d27]"
      style={{ fontFamily: "'Pretendard Variable', sans-serif" }}
    >
      <RLFrameworkRail
        activeSection={activeFrameworkSection}
        openSections={{
          replay: isReplayDrawerOpen,
          training: isTrainingDrawerOpen,
        }}
        collapsed={isFrameworkRailCollapsed}
        onBack={() => dispatch(moveToPage(PageType.HOME))}
        onSectionChange={handleFrameworkSectionChange}
        onToggleCollapsed={() => setIsFrameworkRailCollapsed((collapsed) => !collapsed)}
        sectionControls={{
          replay: 'offline-rl-replay-drawer',
          training: 'offline-rl-training-drawer',
        }}
      />
      <main
        className="relative min-h-0 min-w-0 flex-1 overflow-hidden p-4 lg:p-5 2xl:p-6"
        data-testid="offline-rl-main"
      >
        <div
          className="relative flex h-full min-h-0 w-full rounded-2xl border border-[#ded9cf] bg-[#eeebe3] p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.75)]"
          data-testid="offline-rl-workflow-grid"
          data-layout="environment-canvas"
          data-active-section={activeFrameworkSection}
        >
          <div
            className="flex h-full min-h-0 w-full min-w-0 flex-1 overflow-y-auto overscroll-contain"
            data-testid="offline-rl-environment-canvas"
          >
            <section
              className={`${SURFACE_CLASS} flex min-h-[570px] w-full min-w-0 flex-col p-4`}
              data-testid="offline-rl-environment-stage"
            >
              <SectionHeader
                icon={RobotLabIcon}
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
                  </div>
                )}
              />
              <OfflineRLInferenceWorkspace
                isActive={isActive}
                workspaceMode={workspaceMode}
                policyEpoch={rlLineage.policyEpoch}
                workspaceStatusOpen={showWorkspaceStatus}
                onCloseWorkspaceStatus={closeWorkspaceStatus}
                getFlowSDEPPOPolicyRolloutStatus={getFlowSDEPPOPolicyRolloutStatus}
                onStartFlowSDEPPOPolicyRollout={startFlowSDEPPOPolicyRollout}
                onStopFlowSDEPPOPolicyRollout={stopFlowSDEPPOPolicyRollout}
                onSubmitFlowSDEPPOPolicyRolloutOutcome={submitFlowSDEPPOPolicyRolloutOutcome}
                getFlowSDEPPOValueWarmupStatus={getFlowSDEPPOValueWarmupStatus}
                onFlowSDEPPOPolicyRolloutBundleChange={setFlowSdeRolloutBundle}
              />
            </section>
          </div>

          <div
            className="pointer-events-none absolute inset-0 z-20 overflow-hidden"
            data-testid="offline-rl-workflow-steps"
            data-panel-state={frameworkPanelState}
          >
            <aside
              id="offline-rl-replay-drawer"
              aria-labelledby="offline-rl-replay-drawer-title"
              aria-hidden={!isReplayDrawerOpen}
              className={`absolute inset-y-4 left-4 flex min-h-0 min-w-0 w-[calc(100%_-_2rem)] max-w-[calc(100%_-_2rem)] flex-col overflow-hidden rounded-2xl border border-[#d8d1c5] bg-[#f3f0e8] shadow-[0_18px_45px_rgba(55,49,39,0.2)] transition-[transform,opacity,visibility] duration-300 ease-out motion-reduce:transition-none ${drawerWidthClass} ${
                isReplayDrawerOpen
                  ? 'visible translate-x-0 opacity-100 pointer-events-auto'
                  : 'invisible -translate-x-[calc(100%_+_2rem)] opacity-0 pointer-events-none'
              }`}
              data-testid="offline-rl-replay-drawer"
              data-panel-state={isReplayDrawerOpen ? 'open' : 'closed'}
              inert={!isReplayDrawerOpen}
            >
              <div
                className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-4 py-3"
                data-testid="offline-rl-replay-drawer-header"
              >
                <button
                  ref={replayDrawerCloseButtonRef}
                  type="button"
                  onClick={closeReplayDrawer}
                  aria-label="Close Replay Buffer panel"
                  title="Close Replay Buffer panel"
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-transparent bg-[#f3f0e8] text-[#6f685d] transition-colors hover:bg-[#e7e1d6] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1"
                >
                  <PanelToggleGlyph
                    glyphTestId="replay-drawer-toggle-glyph"
                    accentTestId="replay-drawer-toggle-accent"
                  />
                </button>
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-[#ddd5c7] bg-[#ece8df] text-[#555046]">
                    <MdDns size={17} aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
                      Data Workflow
                    </p>
                    <h2
                      id="offline-rl-replay-drawer-title"
                      className="truncate text-sm font-semibold text-[#292720]"
                    >
                      Replay Buffer
                    </h2>
                  </div>
                </div>
              </div>
              <div
                className="flex min-h-0 w-full flex-1 flex-col justify-between gap-3 overflow-y-auto overscroll-contain p-4"
                data-testid="offline-rl-dataset-pipeline"
              >
                <PipelineCard
                  icon={MdDns}
                  step="01"
                  title="Data Collection"
                  onIconClick={() => setShowDataConversionGuide(true)}
                  iconButtonLabel="Open Data Conversion Guide"
                >
                  <OfflineRLReplayBuffer isActive={isActive} />
                </PipelineCard>

                <PipelineCard icon={MdSwapHoriz} step="02" title="Dataset Conversion">
                  <OfflineRLDatasetConversion isActive={isActive} />
                </PipelineCard>

                <PipelineCard icon={MdOutlineDataset} step="03" title="LeRobot Dataset">
                  <OfflineRLLeRobotDataset isActive={isActive} />
                </PipelineCard>
              </div>
            </aside>

            <aside
              id="offline-rl-training-drawer"
              aria-labelledby="offline-rl-training-drawer-title"
              aria-hidden={!isTrainingDrawerOpen}
              className={`absolute inset-y-4 right-4 flex min-h-0 min-w-0 w-[calc(100%_-_2rem)] max-w-[calc(100%_-_2rem)] flex-col overflow-hidden rounded-2xl border border-[#d8d1c5] bg-[#f3f0e8] shadow-[0_18px_45px_rgba(55,49,39,0.2)] transition-[transform,opacity,visibility] duration-300 ease-out motion-reduce:transition-none ${drawerWidthClass} ${
                isTrainingDrawerOpen
                  ? 'visible translate-x-0 opacity-100 pointer-events-auto'
                  : 'invisible translate-x-[calc(100%_+_2rem)] opacity-0 pointer-events-none'
              }`}
              data-testid="offline-rl-training-drawer"
              data-panel-state={isTrainingDrawerOpen ? 'open' : 'closed'}
              inert={!isTrainingDrawerOpen}
            >
              <div
                className="flex shrink-0 items-center gap-3 border-b border-[#ded8cc] bg-[#fbfaf6] px-4 py-3"
                data-testid="offline-rl-training-drawer-header"
              >
                <button
                  ref={trainingDrawerCloseButtonRef}
                  type="button"
                  onClick={closeTrainingDrawer}
                  aria-label="Close Training panel"
                  title="Close Training panel"
                  className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-transparent bg-[#f3f0e8] text-[#6f685d] transition-colors hover:bg-[#e7e1d6] focus:outline-none focus:ring-2 focus:ring-[#879b83] focus:ring-offset-1"
                >
                  <PanelToggleGlyph
                    glyphTestId="training-drawer-toggle-glyph"
                    accentTestId="training-drawer-toggle-accent"
                    accentSide="right"
                  />
                </button>
                <div className="flex min-w-0 items-center gap-2.5">
                  <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg border border-[#ddd5c7] bg-[#ece8df] text-[#555046]">
                    <MdModelTraining size={17} aria-hidden="true" />
                  </span>
                  <div className="min-w-0">
                    <p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-[#989083]">
                      Policy Workflow
                    </p>
                    <h2
                      id="offline-rl-training-drawer-title"
                      className="truncate text-sm font-semibold text-[#292720]"
                    >
                      Training Pipeline
                    </h2>
                  </div>
                </div>
              </div>

              <div
                className="flex min-h-0 w-full flex-1 flex-col gap-3 overflow-y-auto overscroll-contain p-4"
                data-testid="offline-rl-training-content"
              >
                <section
                  className={`${SURFACE_CLASS} flex w-full min-w-0 shrink-0 flex-col p-4 ${
                    isTrainingCompact ? 'min-h-0' : 'h-full min-h-[640px]'
                  }`}
                  data-testid="offline-rl-training-stage"
                  data-compact-layout={isTrainingCompact ? 'true' : 'false'}
                >
                  <SectionHeader
                  icon={MdModelTraining}
                  eyebrow="Step 04"
                  title="Training"
                  onIconClick={() => setShowTrainingGuide(true)}
                  iconButtonLabel="Open Training Guide"
                  actions={(
                    <div className="flex shrink-0 items-center gap-1.5">
                      {trainingMethod === 'imitation' ? (
                        <span
                          className="rounded-full border border-[#d8d0c3] bg-[#f2eee6] px-2.5 py-1 font-mono text-[9px] font-bold text-[#746c61]"
                          aria-label="Imitation Learning base policy RL Epoch 0"
                        >
                          Base Policy {formatRLEpoch(0)}
                        </span>
                      ) : trainingMethod === 'critic' ? (
                        <span
                          className="rounded-full border border-[#d8d0c3] bg-[#f2eee6] px-2.5 py-1 font-mono text-[9px] font-bold text-[#746c61]"
                          aria-label={`Critic Warm-up policy RL Epoch ${rlLineage.policyEpoch} unchanged`}
                        >
                          Critic · {formatRLEpoch(rlLineage.policyEpoch)}
                        </span>
                      ) : (
                        <span
                          className="rounded-full border border-[#cfd8cd] bg-[#e8eee6] px-2.5 py-1 font-mono text-[9px] font-bold text-[#58705d]"
                          aria-label={`Training policy RL Epoch ${rlLineage.policyEpoch} to ${rlLineage.policyEpoch + 1}`}
                        >
                          RL Epoch {formatRLEpoch(rlLineage.policyEpoch)} → {formatRLEpoch(rlLineage.policyEpoch + 1)}
                        </span>
                      )}
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
                  onTrainingMethodStateChange={setTrainingMethod}
                  flowSdePpoReady
                  flowSdeRolloutBundle={flowSdeRolloutBundle}
                  getFlowSDEPPOStatus={getFlowSDEPPOUpdateStatus}
                  onStartFlowSDEPPO={startFlowSDEPPOUpdate}
                  onStopFlowSDEPPO={stopFlowSDEPPOUpdate}
                  onDeploymentStateChange={handleDeploymentStateChange}
                  onCompactLayoutChange={setIsTrainingCompact}
                  />
                </section>

                <section
                  className={`${SURFACE_CLASS} flex w-full shrink-0 items-center justify-between gap-4 p-4`}
                  data-testid="offline-rl-deployment"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[#eeebe3] text-[#6f685d]">
                      <MdCloudUpload size={18} />
                    </span>
                    <div className="min-w-0">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#91897d]">
                        Step 05 · Deployment
                      </div>
                      <div className="truncate text-[11px] font-semibold text-[#403b34]">
                        {isRltBundleCandidate ? 'RLT Bundle Deploy' : 'Policy Deploy'}
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
                      <MdUndo size={14} /> {isRltBundleActive ? 'Discard RLT Bundle' : 'Discard Policy'}
                    </button>
                    <button
                      type="button"
                      onClick={handleDeployPolicy}
                      disabled={deployDisabled}
                      className={deployDisabled
                        ? 'flex h-9 shrink-0 cursor-not-allowed items-center gap-1.5 rounded-lg border border-[#d9d2c5] bg-[#f0ede6] px-4 text-[10px] font-semibold text-[#9a9286]'
                        : 'flex h-9 shrink-0 items-center gap-1.5 rounded-lg border border-[#5f7965] bg-[#69866f] px-4 text-[10px] font-semibold text-white hover:bg-[#5f7965]'}
                    >
                      <MdCloudUpload size={14} /> {isRltBundleCandidate ? 'Deploy RLT Bundle' : 'Deploy Policy'}
                    </button>
                  </div>
                </section>
              </div>
            </aside>
          </div>
        </div>
      </main>
      <OfflineRLTrainingGuideModal
        open={showTrainingGuide}
        onBack={closeTrainingGuide}
      />
      <OfflineRLDataConversionGuideModal
        open={showDataConversionGuide}
        onBack={closeDataConversionGuide}
      />
    </div>
  );
}
