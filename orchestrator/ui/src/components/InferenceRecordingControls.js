// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import toast from 'react-hot-toast';
import { MdFiberManualRecord, MdSave, MdDeleteOutline, MdFolderOpen, MdCreateNewFolder, MdClose } from 'react-icons/md';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import FileBrowserModal from './FileBrowserModal';
import { DEFAULT_PATHS } from '../constants/paths';
import { getInferenceRecordingSessionId, inferenceRecordingPath } from '../utils/inferenceRecordingFolder';

export default function InferenceRecordingControls() {
  const record = useSelector((state) => state.tasks.recordStatus);
  const inference = useSelector((state) => state.tasks.inferenceStatus);
  const robotType = useSelector((state) => state.tasks.robotType);
  const returning = useSelector((state) => Boolean(state.tasks.robotPoseStatus?.returning));
  const rosbridgeUrl = useSelector((state) => state.ros?.rosbridgeUrl);
  const { sendRecordCommand } = useRosServiceCaller();
  const [pending, setPending] = useState(null);
  const [folderOpen, setFolderOpen] = useState(false);
  const [browserOpen, setBrowserOpen] = useState(false);
  const requestRef = useRef(null);

  useEffect(() => {
    requestRef.current = null;
    setPending(null);
    setFolderOpen(false);
    setBrowserOpen(false);
    return () => { requestRef.current = null; };
  }, [robotType, rosbridgeUrl, inference.topicReceived, inference.sourceId]);

  const recording = record.taskType === 'inference' &&
    record.recordPhase === RecordPhase.RECORDING;
  const ready = record.topicReceived && inference.topicReceived;
  const canRecord = ready && !pending && !returning &&
    inference.inferencePhase === InferencePhase.INFERENCING &&
    inference.runtimeState === 'running' && inference.publishToRobot &&
    record.recordPhase === RecordPhase.READY && !record.running;
  const canFinish = ready && recording && !pending;
  const canSelectFolder = ready && !pending && !record.running &&
    record.recordPhase === RecordPhase.READY;
  const folderPath = inferenceRecordingPath(inference.recordingSessionId);

  const execute = async (command, options) => {
    const allowed = command === 'set_inference_record_folder' ? canSelectFolder
      : command === 'start_inference_record' ? canRecord : canFinish;
    if (requestRef.current || !allowed) return;
    const request = {};
    requestRef.current = request;
    setPending(command);
    try {
      const result = await (options ? sendRecordCommand(command, options) : sendRecordCommand(command));
      if (requestRef.current !== request) return;
      if (!result?.success) throw new Error(result?.message || 'Recording command failed');
    } catch (error) {
      if (requestRef.current === request) {
        toast.error(error.message || 'Recording command failed');
      }
    } finally {
      if (requestRef.current === request) {
        requestRef.current = null;
        setPending(null);
      }
    }
  };

  const selectFolder = (item) => {
    const sessionId = getInferenceRecordingSessionId(item?.full_path);
    if (!sessionId) {
      toast.error('Select a Task_*_inference_MCAP folder directly under the recording root');
      return;
    }
    execute('set_inference_record_folder', { recordingSessionId: sessionId });
  };

  const buttons = recording ? [
    { label: 'Save', icon: MdSave, command: 'stop_inference_record', enabled: canFinish,
      title: 'Save inference recording', color: 'bg-green-100 text-green-700 hover:bg-green-200' },
    { label: 'Discard', icon: MdDeleteOutline, command: 'cancel_inference_record', enabled: canFinish,
      title: 'Discard inference recording', color: 'bg-gray-100 text-gray-700 hover:bg-gray-200' },
  ] : [
    { label: 'Record', icon: MdFiberManualRecord, command: 'start_inference_record', enabled: canRecord,
      title: 'Start inference recording', color: 'bg-red-100 text-red-700 hover:bg-red-200' },
  ];

  return (
    <div className="relative flex items-center mb-2.5" role="group" aria-label="Inference recording" aria-busy={Boolean(pending)}>
      <span className="w-28 shrink-0 text-sm font-medium text-gray-600">Recording</span>
      <div className="flex flex-1 min-w-0 gap-1">
        {buttons.map(({ label, icon: Icon, command, enabled, title, color }) => (
          <button
            key={command}
            type="button"
            aria-label={title}
            title={title}
            disabled={!enabled}
            onClick={() => execute(command)}
            className={`h-8 px-1.5 rounded-md inline-flex items-center justify-center gap-0.5 text-xs font-semibold whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed ${color}`}
          >
            <Icon size={14} className="shrink-0" />{label}
          </button>
        ))}
        <button type="button" aria-label="Recording folder" title="Recording folder"
          aria-expanded={folderOpen} onClick={() => setFolderOpen(value => !value)}
          className="ml-auto h-8 w-7 shrink-0 rounded-md flex items-center justify-center text-gray-500 hover:bg-gray-100">
          <MdFolderOpen size={18} />
        </button>
      </div>
      {folderOpen && (
        <div role="dialog" aria-label="Recording folder settings"
          className="absolute top-full right-0 left-0 z-30 mt-1 border border-gray-200 bg-white rounded-md shadow-lg p-2">
          <div className="flex items-center justify-between text-xs font-medium text-gray-600">
            <span>Save folder</span>
            <button type="button" aria-label="Close recording folder" title="Close" onClick={() => setFolderOpen(false)}
              className="h-7 w-7 shrink-0 flex items-center justify-center rounded-md hover:bg-gray-100"><MdClose size={16} /></button>
          </div>
          <div className="text-xs break-all my-2">{folderPath || 'New on next Record'}</div>
          <div className="flex gap-2">
            <button type="button" disabled={!canSelectFolder} onClick={() => setBrowserOpen(true)}
              className="h-8 px-2 rounded-md bg-gray-100 text-xs inline-flex items-center gap-1 disabled:opacity-40">
              <MdFolderOpen size={16} />Use existing
            </button>
            <button type="button" disabled={!canSelectFolder} onClick={() => execute('set_inference_record_folder', { recordingSessionId: '' })}
              className="h-8 px-2 rounded-md bg-gray-100 text-xs inline-flex items-center gap-1 disabled:opacity-40">
              <MdCreateNewFolder size={16} />Use new
            </button>
          </div>
        </div>
      )}
      <FileBrowserModal isOpen={browserOpen} onClose={() => setBrowserOpen(false)}
        onFileSelect={selectFolder} title="Select inference recording folder" selectButtonText="Use Folder"
        allowDirectorySelect allowFileSelect={false} initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
        defaultPath={DEFAULT_PATHS.ROSBAG2_PATH} homePath={DEFAULT_PATHS.ROSBAG2_PATH} />
    </div>
  );
}
