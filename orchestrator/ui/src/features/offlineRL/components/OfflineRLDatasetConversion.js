// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import clsx from 'clsx';
import { shallowEqual, useDispatch, useSelector } from 'react-redux';
import {
  MdCheckCircle,
  MdFolderOpen,
  MdPlayArrow,
  MdStorage,
  MdWarningAmber,
} from 'react-icons/md';
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
  const validationItems = [
    { label: 'MCAP linked', valid: Boolean(taskName) },
    { label: 'Destination valid', valid: destinationValid },
    { label: 'Output selected', valid: formatsValid },
  ];

  return (
    <div className="flex min-h-0 flex-col gap-2" data-testid="offline-rl-conversion-engine">
      <div
        className="rounded-xl border border-[#e2dbcf] bg-[#f8f5ef] p-2.5"
        data-testid="conversion-setup-surface"
      >
        <div
          className="grid grid-cols-2 gap-2"
          data-testid="conversion-path-row"
        >
          <div className="min-w-0">
            <div className="mb-1 text-[10px] font-semibold text-[#5e584f]">
              Conversion setup
            </div>
            <div className="rounded-lg border border-[#e2dbcf] bg-white px-2.5 py-1.5">
              <div className="mb-0.5 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.08em] text-[#8a8276]">
                <MdStorage size={12} aria-hidden="true" />
                MCAP source
              </div>
              <div
                className="truncate text-[10px] font-medium text-[#5f594f]"
                title={sourcePath || undefined}
              >
                {sourcePath || 'No MCAP selected'}
              </div>
            </div>
          </div>

          <label className="flex min-w-0 flex-col gap-1 text-[10px] font-medium text-[#756e63]">
            LeRobot collection root
            <div className="flex min-w-0 gap-1.5">
              <input
                aria-label="LeRobot collection root"
                value={destinationPath}
                onChange={(event) => dispatch(
                  setOfflineRLConversionDestinationPath(event.target.value)
                )}
                disabled={isRunning || !isActive}
                placeholder="/workspace/lerobot"
                className="h-9 min-w-0 flex-1 rounded-lg border border-[#d9d2c5] bg-white px-2.5 text-[10px] text-[#4c473f] outline-none transition-colors focus:border-[#879b89] focus:ring-2 focus:ring-[#879b89]/15 disabled:bg-[#efebe3]"
              />
              <button
                type="button"
                onClick={() => setShowBrowser(true)}
                disabled={isRunning || !isActive}
                className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-[#d9d2c5] bg-[#f1ede4] text-[#70695e] transition-colors hover:bg-[#e9e4da] disabled:opacity-45"
                aria-label="Browse LeRobot collection root"
              >
                <MdFolderOpen size={14} />
              </button>
            </div>
          </label>
        </div>

        <div
          className="mt-2 grid grid-cols-[64px_minmax(112px,0.7fr)_minmax(190px,1.2fr)_minmax(104px,0.7fr)_minmax(116px,0.65fr)] items-start gap-2"
          data-testid="conversion-options-row"
        >
          <label className="flex flex-col gap-1 text-[10px] font-medium text-[#756e63]">
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
              className="h-9 rounded-lg border border-[#d9d2c5] bg-white px-2 text-[10px] font-semibold text-[#4c473f] outline-none focus:border-[#879b89] disabled:bg-[#efebe3]"
            />
          </label>
          <div
            className="flex flex-col gap-1 text-[10px] font-medium text-[#756e63]"
            data-testid="conversion-output-format"
          >
            Output format
            <div className="grid h-9 grid-cols-2 gap-1">
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
                    'rounded-lg border text-[10px] font-semibold transition-colors disabled:opacity-45',
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
          <div
            className="flex min-w-0 flex-col gap-1 text-[10px] font-medium text-[#756e63]"
            aria-label="Conversion validation"
          >
            Validation
            <div
              className="flex min-h-9 flex-wrap items-center gap-x-2 gap-y-0.5 rounded-lg border border-[#e0d9cd] bg-white px-2"
              data-testid="conversion-validation-summary"
            >
              {validationItems.map(({ label, valid }) => (
                <span
                  key={label}
                  className={clsx(
                    'inline-flex items-center gap-1 text-[9px] font-medium',
                    valid ? 'text-[#5e7a64]' : 'text-[#9a7250]'
                  )}
                >
                  {valid ? <MdCheckCircle size={12} /> : <MdWarningAmber size={12} />}
                  {label}
                </span>
              ))}
            </div>
            {!destinationValid && (
              <div className="text-[9px] text-[#a06b61]">
                Destination must be inside {LEROBOT_ROOT}
              </div>
            )}
          </div>
          <div
            className="flex min-w-0 flex-col gap-1 text-[10px] font-medium text-[#756e63]"
            data-testid="offline-rl-data-epoch-output"
            title={reservedEpoch?.output_root || undefined}
          >
            Data Epoch
            <div className="flex h-9 min-w-0 items-center rounded-lg border border-[#ded7ca] bg-white px-2.5">
              <span className="truncate text-[10px] font-semibold text-[#58705d]">
                {reservedEpoch?.epoch_name || 'Next available'}
              </span>
            </div>
          </div>
          <button
            type="button"
            onClick={handleConvert}
            disabled={!canConvert}
            className="flex h-9 self-end items-center justify-center gap-1.5 rounded-lg border border-[#66836c] bg-[#69866f] px-2 text-[10px] font-semibold text-white shadow-[0_4px_10px_rgba(74,101,80,0.15)] transition-colors hover:bg-[#5e7b64] disabled:cursor-not-allowed disabled:border-[#d7d0c4] disabled:bg-[#e8e3da] disabled:text-[#9d9589] disabled:shadow-none"
          >
            <MdPlayArrow size={14} />
            {isRunning ? 'Converting…' : 'Convert Dataset'}
          </button>
        </div>
        {reservedEpoch?.output_root && (
          <div
            className="mt-1 truncate text-[9px] text-[#948c80]"
            title={reservedEpoch.output_root}
          >
            {reservedEpoch.output_root}
          </div>
        )}
      </div>

      {(feedback || isRunning) && (
        <div className="rounded-xl border border-[#e2dbcf] bg-[#f8f5ef] px-2.5 py-2">
          <div className={clsx(
            'flex items-center justify-between gap-3 text-[10px]',
            isRunning && 'mb-1.5'
          )}>
            <span
              className={clsx(
                'min-w-0 truncate font-medium',
                feedback.toLowerCase().includes('fail') || feedback.toLowerCase().includes('reject')
                  ? 'text-[#a06b61]'
                  : 'text-[#756e63]'
              )}
              title={feedback}
            >
              {feedback || 'Converting dataset…'}
            </span>
            {isRunning && (
              <span className="shrink-0 font-semibold text-[#58705d]">
                {Math.round(progress)}%
              </span>
            )}
          </div>
          {isRunning && (
            <>
              <div
                className="h-1.5 overflow-hidden rounded-full bg-[#e8e3da]"
                role="progressbar"
                aria-label="Dataset conversion progress"
                aria-valuemin="0"
                aria-valuemax="100"
                aria-valuenow={progress}
              >
                <div
                  className="h-full rounded-full bg-[#72917a] transition-[width] duration-300"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <div className="mt-1 truncate text-[9px] text-[#91897d]">
                {conversionStatus.stage || 'queued'}
              </div>
            </>
          )}
        </div>
      )}

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
