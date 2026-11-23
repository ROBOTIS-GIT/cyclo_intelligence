import React, { useEffect, useState, useSyncExternalStore } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { MdClose, MdViewInAr, MdMonitorHeart, MdKeyboardDoubleArrowLeft, MdKeyboardDoubleArrowRight, MdVolumeOff, MdVolumeUp } from 'react-icons/md';
import toast, { useToasterStore } from 'react-hot-toast';
import PageType from '../constants/pageType';
import { RecordPhase } from '../constants/taskPhases';
import { moveToPage, setIsFirstLoadFalse } from '../features/ui/uiSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import useDocumentVisible from '../hooks/useDocumentVisible';
import HeartbeatStatus from '../components/HeartbeatStatus';
import InlineSystemStatus from '../components/InlineSystemStatus';
import ImageGrid from '../components/ImageGrid';
import OperateSplitter from '../components/layout/OperateSplitter';
import RobotViewer3D from '../components/RobotViewer3D';
import RecordTopicMonitor from '../components/RecordTopicMonitor';
import RecordControlPanel from '../components/RecordControlPanel';
import InferenceControlPanel from '../components/InferenceControlPanel';
import SegmentPanel from '../components/SegmentPanel';
import InferencePanel from '../components/InferencePanel';
import {
  disableRecordingAudio, enableRecordingAudio, recordingSoundState, subscribeRecordingSound,
} from '../utils/recordingAudio';
import './OperationWorkspace.css';

const MONITORS_STORAGE_KEY = 'cyclo_intelligence.operate.monitors';

const collapseButtonClass = 'w-9 h-9 shrink-0 inline-flex items-center justify-center rounded border border-gray-300 bg-white text-gray-600 hover:bg-gray-100';

function readMonitorVisibility() {
  try {
    const saved = JSON.parse(localStorage.getItem(MONITORS_STORAGE_KEY));
    return { viewer: saved?.viewer !== false, topics: saved?.topics !== false };
  } catch {
    return { viewer: true, topics: true };
  }
}

export default function OperationWorkspace() {
  const dispatch = useDispatch();
  const documentVisible = useDocumentVisible();
  const page = useSelector((state) => state.ui.currentPage);
  const robotType = useSelector((state) => state.tasks.robotType);
  const joystickMode = useSelector((state) => state.tasks.joystickMode);
  const inferenceStatus = useSelector((state) => state.tasks.inferenceStatus);
  const isRecording = useSelector((state) => state.tasks.recordStatus.topicReceived &&
    state.tasks.recordStatus.recordPhase === RecordPhase.RECORDING);
  const recordingBusy = useSelector((state) => state.tasks.recordStatus.running ||
    state.tasks.recordStatus.recordPhase !== RecordPhase.READY);
  const { sendRecordCommand } = useRosServiceCaller('record');
  const { toasts } = useToasterStore();
  const [actionsTarget, setActionsTarget] = useState(null);
  const [showSettings, setShowSettings] = useState(true);
  const [monitors, setMonitors] = useState(readMonitorVisibility);
  const sound = useSyncExternalStore(subscribeRecordingSound, recordingSoundState);
  const soundLabel = sound === 'on' ? 'Mute recording announcements' : 'Enable recording announcements';
  const show3D = monitors.viewer;
  const showTopics = monitors.topics;
  const inferenceSelected = page === PageType.INFERENCE;
  const monitorsExpanded = show3D || showTopics;

  useEffect(() => {
    try {
      localStorage.setItem(MONITORS_STORAGE_KEY, JSON.stringify(monitors));
    } catch {
      // Browsers that block storage still allow in-session visibility changes.
    }
  }, [monitors]);

  useEffect(() => {
    toasts.filter((t) => t.visible).slice(3).forEach((t) => toast.dismiss(t.id));
  }, [toasts]);

  useEffect(() => {
    dispatch(setIsFirstLoadFalse('record'));
    dispatch(setIsFirstLoadFalse('inference'));
  }, [dispatch]);

  useEffect(() => {
    if (!recordingBusy) sendRecordCommand('refresh_topics').catch(() => {});
  }, [recordingBusy, sendRecordCommand]);

  return (
    <div className="operation-workspace bg-gray-100 text-gray-900 h-full min-w-0 flex flex-col overflow-y-auto lg:overflow-hidden">
      <header className="operation-header bg-white border-gray-300 shrink-0 border-b px-3 py-2 space-y-2">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 min-w-0">
          <span aria-label="Robot type" className="max-w-full min-w-0 rounded border border-gray-300 bg-gray-50 px-3 py-1.5 text-sm font-medium text-gray-800 break-all">{robotType}</span>
          {joystickMode && <span className="text-xs text-gray-500">{joystickMode}</span>}
          <HeartbeatStatus autoHideConnectedMs={3000} size="small" className="operation-connection" />
          <button type="button" className={collapseButtonClass} title={soundLabel} aria-label={soundLabel}
            aria-pressed={sound === 'on'} onClick={() => {
              if (sound === 'on') disableRecordingAudio();
              else enableRecordingAudio().catch(() => toast.error('Could not enable recording sounds. Try again.'));
            }}>
            {sound === 'on' ? <MdVolumeUp size={20} /> : <MdVolumeOff size={20} />}
          </button>
          <div aria-label="System metrics" className="hidden lg:block"><InlineSystemStatus /></div>
        </div>
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-2">
          <section aria-label="Recording" className="flex flex-wrap items-center gap-1 min-w-0">
            <RecordControlPanel />
          </section>
          <div aria-label="Inference status" className="w-fit max-w-full min-w-0 self-start flex flex-wrap items-center gap-x-2 gap-y-1 rounded border border-gray-300 bg-gray-50 px-3 py-1.5 text-sm text-gray-800" title={inferenceStatus.error || inferenceStatus.loadedModelPath || ''}>
            <span className="font-semibold">Inference</span>
            <span className="min-w-0 break-words">{inferenceStatus.topicReceived ? (inferenceStatus.runtimeState || 'unknown') : 'Checking status...'}</span>
          </div>
        </div>
      </header>
      <div className="flex-1 min-h-0 flex flex-col lg:flex-row">
        <section aria-label="Live monitoring" data-split={monitorsExpanded} className="operation-monitoring min-w-0 flex-1 flex flex-col gap-3 p-3 shrink-0 lg:shrink">
          <div id="operate-cameras" className="h-[360px] lg:h-auto min-h-0 overflow-hidden"><ImageGrid isActive responsive isRecording={isRecording} headerActions={<>
            {!show3D && <button type="button" title="Expand Robot Viewer" aria-label="Expand Robot Viewer"
              aria-expanded={false} aria-controls="operate-robot-viewer" className="image-grid-tool"
              onClick={() => setMonitors((value) => ({ ...value, viewer: true }))}><MdViewInAr size={20} /></button>}
            {!showTopics && <button type="button" title="Expand Topic Monitor" aria-label="Expand Topic Monitor"
              aria-expanded={false} aria-controls="operate-topic-monitor" className="image-grid-tool"
              onClick={() => setMonitors((value) => ({ ...value, topics: true }))}><MdMonitorHeart size={20} /></button>}
          </>} /></div>
          {monitorsExpanded && <OperateSplitter orientation="horizontal" initialPercent={60}
            controls="operate-cameras" label="Resize cameras and observation monitors" />}
          <div aria-label="Observation monitors" hidden={!monitorsExpanded} data-split={show3D && showTopics}
            className="operation-observations grid grid-cols-1 gap-3 min-h-[280px] lg:min-h-0">
            <section aria-label="Robot Viewer" hidden={!show3D} className="operation-monitor flex flex-col bg-white border border-gray-300 min-w-0 min-h-0 overflow-hidden">
              <div className="flex items-center justify-between gap-2 px-3 py-1 bg-gray-50 border-b border-gray-300">
                <h2 className="text-sm font-semibold text-gray-800">Robot Viewer</h2>
                <button type="button" title="Close Robot Viewer" aria-label="Close Robot Viewer"
                  aria-expanded={show3D} aria-controls="operate-robot-viewer" className={collapseButtonClass}
                  onClick={() => setMonitors((value) => ({ ...value, viewer: !value.viewer }))}>
                  <MdClose size={20} />
                </button>
              </div>
              <div id="operate-robot-viewer" hidden={!show3D} className="min-h-[220px] h-[260px] lg:min-h-0 lg:h-auto lg:flex-1">
                <RobotViewer3D mode="live" isActive={show3D && documentVisible} showSourceSelector defaultVisualizationSource="state" />
              </div>
            </section>
            {show3D && showTopics && <OperateSplitter orientation="vertical" initialPercent={50}
              controls="operate-robot-viewer" label="Resize Robot Viewer and Topic Monitor" />}
            <div id="operate-topic-monitor" hidden={!showTopics} className="operation-topic-monitor bg-white border border-gray-300 min-w-0 min-h-0 overflow-hidden">
              <RecordTopicMonitor embedded collapsed={!showTopics || !documentVisible} actions={
                <button type="button" title="Close Topic Monitor" aria-label="Close Topic Monitor"
                  aria-expanded={showTopics} aria-controls="operate-topic-monitor" className={collapseButtonClass}
                  onClick={() => setMonitors((value) => ({ ...value, topics: !value.topics }))}>
                  <MdClose size={20} />
                </button>
              } />
            </div>
          </div>
        </section>
        <aside aria-label="Operation settings" className={`operation-settings bg-white border-gray-300 w-full ${showSettings ? 'lg:w-[380px]' : 'lg:w-12'} shrink-0 border-t lg:border-t-0 lg:border-l lg:overflow-y-auto`}>
          <div hidden={showSettings} className="flex justify-end lg:justify-center p-1 border-b border-gray-300">
            <button type="button" title="Expand operation settings" aria-label="Expand operation settings" aria-expanded={false} aria-controls="operation-settings-content"
              onClick={() => setShowSettings(true)} className={collapseButtonClass}><MdKeyboardDoubleArrowLeft size={20} /></button>
          </div>
          <div id="operation-settings-content" hidden={!showSettings}>
          <div role="tablist" aria-label="Operation" className="operation-tabs h-12 bg-gray-50 border-b border-gray-300 sticky top-0 z-20 flex items-stretch">
            {[['Record', PageType.RECORD], ['Inference', PageType.INFERENCE]].map(([label, id]) => {
              const selected = (id === PageType.INFERENCE) === inferenceSelected;
              return <button key={id} type="button" role="tab" id={`${id}-tab`} aria-selected={selected} aria-controls={`${id}-settings`}
                onClick={() => dispatch(moveToPage(id))}
                className={`flex-1 px-3 text-sm font-semibold border-r border-gray-300 border-b-2 ${selected ? 'border-b-blue-500 text-blue-600 bg-blue-50' : 'border-b-transparent text-gray-600 hover:bg-gray-100'}`}>{label}</button>;
            })}
            <button type="button" title="Collapse operation settings" aria-label="Collapse operation settings" aria-expanded={true} aria-controls="operation-settings-content"
              onClick={() => setShowSettings(false)} className={`${collapseButtonClass} self-center mx-1`}><MdKeyboardDoubleArrowRight size={20} /></button>
          </div>
          {/* Visibility changes only: controllers retain in-flight operations and episode identity. */}
          <div role="tabpanel" id="record-settings" aria-labelledby="record-tab" hidden={inferenceSelected}>
            <div ref={setActionsTarget} className="operation-record-actions bg-white border-b border-gray-300 p-3 sticky top-12 z-10" />
            <SegmentPanel embedded actionsTarget={actionsTarget} shortcutsEnabled={!inferenceSelected && showSettings} />
          </div>
          <div role="tabpanel" id="inference-settings" aria-labelledby="inference-tab" hidden={!inferenceSelected}>
            <div className="operation-inference-actions bg-white border-b border-gray-300 p-3 sticky top-12 z-10">
              <InferenceControlPanel isActive={inferenceSelected && showSettings && documentVisible} shortcutsEnabled={inferenceSelected && showSettings} />
            </div>
            <InferencePanel embedded isActive={inferenceSelected && showSettings && documentVisible} />
          </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
