// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import { MdFolderOpen, MdPlayArrow } from 'react-icons/md';
import FileBrowserModal from '../../../components/FileBrowserModal';
import { DEFAULT_PATHS } from '../../../constants/paths';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import {
  selectInferenceTaskInfo,
  selectRecordTaskInfo,
  setRecordTaskInfo,
} from '../../tasks/taskSlice';
import { reserveOfflineRLDataEpoch } from '../../../utils/offlineRlApi';
import {
  selectOfflineRLConversionFormats,
  selectOfflineRLConversionFps,
  selectOfflineRLConversionDestinationPath,
  selectOfflineRLReplayBufferPath,
  setOfflineRLConversionFormats,
  setOfflineRLConversionFps,
  setOfflineRLConversionDestinationPath,
  setOfflineRLConvertedDatasetPaths,
  setOfflineRLDatasetSelection,
} from '../offlineRLSlice';

const ROSBAG_ROOT = DEFAULT_PATHS.ROSBAG2_PATH.replace(/\/+$/, '');
const LEROBOT_ROOT = DEFAULT_PATHS.LEROBOT_DATASETS_PATH.replace(/\/+$/, '');

export function normalizeConversionDestinationPath(item) {
  const path = String(item?.full_path || item?.path || item?.name || '').trim();
  if (path === '/') return path;
  return path.replace(/\/+$/, '');
}

export function isAllowedConversionDestinationPath(path) {
  const normalized = normalizeConversionDestinationPath({ full_path: path });
  if (!normalized.startsWith('/')) return false;
  if (normalized.split('/').some((part) => part === '.' || part === '..')) return false;
  return normalized === LEROBOT_ROOT || normalized.startsWith(`${LEROBOT_ROOT}/`);
}

export function conversionTaskName(sourcePath) {
  const path = String(sourcePath || '').trim().replace(/\/+$/, '');
  if (!path.startsWith(`${ROSBAG_ROOT}/`)) return '';
  return path.slice(ROSBAG_ROOT.length + 1);
}

export function deriveConvertedDatasetPaths(sourcePath, destinationPath = LEROBOT_ROOT) {
  const sourceName = String(sourcePath || '').trim().replace(/\/+$/, '').split('/').pop();
  const destination = normalizeConversionDestinationPath({
    full_path: destinationPath,
  });
  if (!sourceName || !destination) return { v21: '', v30: '' };
  return {
    v21: `${destination}/${sourceName}_lerobot_v21`,
    v30: `${destination}/${sourceName}_lerobot_v30`,
  };
}

export default function OfflineRLDatasetConversion({ isActive = true }) {
  const dispatch = useDispatch();
  const { sendRecordCommand } = useRosServiceCaller();
  const inferenceTaskInfo = useSelector(selectInferenceTaskInfo, shallowEqual);
  const recordTaskInfo = useSelector(selectRecordTaskInfo, shallowEqual);
  const sourcePath = useSelector(selectOfflineRLReplayBufferPath);
  const destinationPath = useSelector(selectOfflineRLConversionDestinationPath);
  const fps = useSelector(selectOfflineRLConversionFps);
  const formats = useSelector(selectOfflineRLConversionFormats, shallowEqual);
  const conversionStatus = useSelector(
    (state) => state.editDataset?.conversionStatus || {
      status: 'idle', progress: 0, message: '', jobId: '',
    },
    shallowEqual
  );
  const [showBrowser, setShowBrowser] = useState(false);
  const [pendingStart, setPendingStart] = useState(false);
  const [requestActive, setRequestActive] = useState(false);
  const [startAccepted, setStartAccepted] = useState(false);
  const [acceptedJobId, setAcceptedJobId] = useState('');
  const [feedback, setFeedback] = useState('');
  const [reservedEpoch, setReservedEpoch] = useState(null);
  const baselineJobIdRef = useRef('');
  const reservationInFlightRef = useRef(false);

  const taskName = useMemo(() => conversionTaskName(sourcePath), [sourcePath]);
  const fpsValid = Number.isInteger(fps) && fps >= 1 && fps <= 120;
  const formatsValid = Boolean(formats.v21 || formats.v30);
  const destinationValid = isAllowedConversionDestinationPath(destinationPath);
  const isRunning = requestActive || conversionStatus.status === 'running';
  const canConvert = isActive && !isRunning && Boolean(taskName) && destinationValid &&
    fpsValid && formatsValid;

  const handleConvert = useCallback(async () => {
    if (!canConvert || reservationInFlightRef.current) return;
    if (!window.confirm(
      `Convert ${sourcePath} to LeRobot?\n\n` +
      `Collection root: ${destinationPath}\n` +
      'The next Data Epoch will be reserved automatically.\n\n' +
      'After every selected output is verified, the source MCAP episode folders will be deleted.'
    )) return;

    baselineJobIdRef.current = conversionStatus.jobId || '';
    reservationInFlightRef.current = true;
    setFeedback('Reserving the next Data Epoch…');
    setRequestActive(true);
    setStartAccepted(false);
    setAcceptedJobId('');
    try {
      const reservation = await reserveOfflineRLDataEpoch({
        destination_root: destinationPath,
        source_mcap: sourcePath,
        behavior_policy_path: String(inferenceTaskInfo.policyPath || '').trim(),
        boundary_reason: 'manual_conversion',
        fps,
        formats: [
          ...(formats.v21 ? ['v2.1'] : []),
          ...(formats.v30 ? ['v3.0'] : []),
        ],
      });
      if (
        !reservation?.output_root ||
        (formats.v21 && !reservation?.expected_outputs?.v21) ||
        (formats.v30 && !reservation?.expected_outputs?.v30)
      ) {
        throw new Error('Data Epoch reservation returned an incomplete output contract');
      }
      setReservedEpoch(reservation);
      setFeedback(`${reservation.epoch_name} reserved · starting conversion…`);
      dispatch(setRecordTaskInfo({
        taskName,
        taskInstruction: [taskName],
      }));
      setPendingStart(true);
    } catch (error) {
      setRequestActive(false);
      setStartAccepted(false);
      setFeedback(error?.message || 'Failed to reserve a Data Epoch');
    } finally {
      reservationInFlightRef.current = false;
    }
  }, [
    canConvert,
    conversionStatus.jobId,
    destinationPath,
    dispatch,
    formats.v21,
    formats.v30,
    fps,
    inferenceTaskInfo.policyPath,
    sourcePath,
    taskName,
  ]);

  useEffect(() => {
    if (
      !pendingStart ||
      recordTaskInfo.taskName !== taskName ||
      !reservedEpoch?.output_root
    ) return;
    setPendingStart(false);

    sendRecordCommand('convert_mp4', {
      taskSource: 'record',
      conversionFps: fps,
      convertV21: Boolean(formats.v21),
      convertV30: Boolean(formats.v30),
      lerobotOutputRoot: reservedEpoch.output_root,
      deleteSourceAfterSuccess: true,
      cameraRotations: {},
      imageResize: null,
    }).then((result) => {
      if (!result?.success) {
        throw new Error(result?.message || 'Conversion request was rejected');
      }
      const responseJobId = String(result?.job_id || result?.jobId || '').trim();
      setAcceptedJobId(responseJobId);
      setStartAccepted(true);
      setFeedback(responseJobId
        ? `Conversion queued · ${responseJobId}`
        : 'Conversion queued · waiting for job ID');
    }).catch((error) => {
      setRequestActive(false);
      setStartAccepted(false);
      setAcceptedJobId('');
      setFeedback(error?.message || 'Failed to start conversion');
    });
  }, [
    formats.v21,
    formats.v30,
    fps,
    pendingStart,
    recordTaskInfo.taskName,
    reservedEpoch,
    sendRecordCommand,
    taskName,
  ]);

  useEffect(() => {
    if (!requestActive || !startAccepted) return;
    const statusJobId = String(conversionStatus.jobId || '').trim();
    if (!statusJobId) return;

    let correlatedJobId = acceptedJobId;
    if (!correlatedJobId) {
      if (statusJobId === baselineJobIdRef.current) return;
      // SendCommand.srv currently omits StartConversion.job_id. Since the
      // conversion worker accepts only one job at a time, the first new
      // /data/status ID observed after the successful start response is the
      // accepted job. Future forwarders can return job_id directly above.
      correlatedJobId = statusJobId;
      setAcceptedJobId(statusJobId);
    }
    if (statusJobId !== correlatedJobId) return;

    if (conversionStatus.status === 'running') {
      setFeedback(conversionStatus.message || 'Converting dataset…');
      return;
    }

    if (conversionStatus.status === 'completed') {
      const outputs = {
        v21: reservedEpoch?.expected_outputs?.v21 || '',
        v30: reservedEpoch?.expected_outputs?.v30 || '',
      };
      dispatch(setOfflineRLConvertedDatasetPaths({
        v21: formats.v21 ? outputs.v21 : '',
        v30: formats.v30 ? outputs.v30 : '',
      }));
      dispatch(setOfflineRLDatasetSelection(formats.v30
        ? { path: outputs.v30, version: 'v3.0' }
        : { path: outputs.v21, version: 'v2.1' }));
      setFeedback(
        `Complete · ${reservedEpoch?.epoch_name || 'Data Epoch'} · source MCAP episodes removed`
      );
      setRequestActive(false);
      setStartAccepted(false);
    } else if (
      conversionStatus.status === 'failed' ||
      conversionStatus.status === 'cancelled'
    ) {
      setFeedback(conversionStatus.message || `Conversion ${conversionStatus.status}`);
      setRequestActive(false);
      setStartAccepted(false);
    }
  }, [
    acceptedJobId,
    conversionStatus.jobId,
    conversionStatus.message,
    conversionStatus.status,
    dispatch,
    formats.v21,
    formats.v30,
    requestActive,
    reservedEpoch,
    sourcePath,
    startAccepted,
  ]);

  const progress = Math.max(0, Math.min(100, Number(conversionStatus.progress) || 0));

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex flex-col gap-1 text-[9px] font-medium text-[#756e63]">
        MCAP source · Step 1 Replay Buffer
        <div
          className="h-7 truncate rounded-md border border-[#e0d9cd] bg-[#efebe3] px-2 py-1.5 text-[9px] font-normal text-[#6f685d]"
          title={sourcePath || undefined}
        >
          {sourcePath || 'Waiting for an inference recording'}
        </div>
      </div>

      <label className="flex flex-col gap-1 text-[9px] font-medium text-[#756e63]">
        LeRobot collection root
        <div className="flex min-w-0 gap-1">
          <input
            aria-label="LeRobot collection root"
            value={destinationPath}
            onChange={(event) => dispatch(
              setOfflineRLConversionDestinationPath(event.target.value)
            )}
            disabled={isRunning || !isActive}
            placeholder="/workspace/lerobot"
            className="h-7 min-w-0 flex-1 rounded-md border border-[#d9d2c5] bg-white px-2 text-[9px] text-[#4c473f] outline-none focus:border-[#879b89] disabled:bg-[#efebe3]"
          />
          <button
            type="button"
            onClick={() => setShowBrowser(true)}
            disabled={isRunning || !isActive}
            className="grid h-7 w-7 shrink-0 place-items-center rounded-md border border-[#d9d2c5] bg-[#f1ede4] text-[#70695e] hover:bg-[#e9e4da] disabled:opacity-45"
            aria-label="Browse LeRobot collection root"
          >
            <MdFolderOpen size={13} />
          </button>
        </div>
      </label>

      <div
        className="rounded-md border border-[#ded7ca] bg-[#f7f4ed] px-2 py-1.5 text-[8px] text-[#756e63]"
        data-testid="offline-rl-data-epoch-output"
        title={reservedEpoch?.output_root || 'Assigned atomically when conversion starts'}
      >
        <div className="flex items-center justify-between gap-2">
          <span className="font-semibold uppercase tracking-[0.08em]">Data Epoch</span>
          <span className="font-semibold text-[#58705d]">
            {reservedEpoch?.epoch_name || 'Next available'}
          </span>
        </div>
        <div className="mt-0.5 truncate text-[#948c80]">
          {reservedEpoch?.output_root || 'An immutable output folder is reserved at start'}
        </div>
      </div>

      <div className="grid grid-cols-[56px_minmax(0,1fr)] gap-2">
        <label className="flex flex-col gap-1 text-[9px] font-medium text-[#756e63]">
          FPS
          <input
            aria-label="Conversion FPS"
            type="number"
            min="1"
            max="120"
            step="1"
            value={fps}
            onChange={(event) => dispatch(setOfflineRLConversionFps(event.target.value))}
            disabled={isRunning || !isActive}
            className="h-7 rounded-md border border-[#d9d2c5] bg-white px-2 text-[9px] text-[#4c473f] outline-none focus:border-[#879b89] disabled:bg-[#efebe3]"
          />
        </label>
        <div className="flex flex-col gap-1 text-[9px] font-medium text-[#756e63]">
          Output format
          <div className="grid h-7 grid-cols-2 gap-1">
            {[
              ['v21', 'v2.1'],
              ['v30', 'v3.0'],
            ].map(([key, label]) => (
              <button
                key={key}
                type="button"
                aria-pressed={Boolean(formats[key])}
                onClick={() => dispatch(setOfflineRLConversionFormats({
                  [key]: !formats[key],
                }))}
                disabled={isRunning || !isActive}
                className={clsx(
                  'rounded-md border text-[9px] font-semibold transition-colors disabled:opacity-45',
                  formats[key]
                    ? 'border-[#7f9a84] bg-[#e3ece3] text-[#58705d]'
                    : 'border-[#ddd6ca] bg-white text-[#938b7f]'
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {!destinationValid && (
        <div className="text-[8px] text-[#a06b61]">
          Destination must be inside {LEROBOT_ROOT}
        </div>
      )}

      {isRunning && (
        <div>
          <div className="mb-1 flex items-center justify-between text-[8px] text-[#857d71]">
            <span className="truncate">{conversionStatus.stage || 'queued'}</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-[#e8e3da]">
            <div
              className="h-full rounded-full bg-[#72917a] transition-[width] duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      <button
        type="button"
        onClick={handleConvert}
        disabled={!canConvert}
        className="mt-auto flex h-8 items-center justify-center gap-1 rounded-lg border border-[#66836c] bg-[#69866f] text-[9px] font-semibold text-white hover:bg-[#5e7b64] disabled:cursor-not-allowed disabled:border-[#d7d0c4] disabled:bg-[#e8e3da] disabled:text-[#9d9589]"
      >
        <MdPlayArrow size={13} />
        {isRunning ? 'Converting…' : 'Convert Dataset'}
      </button>

      <div
        className={clsx(
          'min-h-[11px] truncate text-[8px]',
          feedback.toLowerCase().includes('fail') || feedback.toLowerCase().includes('reject')
            ? 'text-[#a06b61]'
            : 'text-[#8c857a]'
        )}
        title={feedback}
      >
        {feedback || 'Ready · source is removed only after verified conversion'}
      </div>

      <FileBrowserModal
        isOpen={showBrowser}
        onClose={() => setShowBrowser(false)}
        onFileSelect={(item) => {
          dispatch(setOfflineRLConversionDestinationPath(
            normalizeConversionDestinationPath(item)
          ));
          setShowBrowser(false);
        }}
        title="Select LeRobot collection root"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={isAllowedConversionDestinationPath(destinationPath)
          ? destinationPath
          : DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
        defaultPath={DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
        homePath={DEFAULT_PATHS.LEROBOT_DATASETS_PATH}
      />
    </div>
  );
}
