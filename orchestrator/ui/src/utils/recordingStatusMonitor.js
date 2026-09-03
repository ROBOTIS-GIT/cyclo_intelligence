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

import { RecordPhase } from '../constants/taskPhases';
import PageType from '../constants/pageType';

export const shouldShowCameraMonitorAlert = (page) => (
  page !== PageType.AUTONOMY_STUDIO
);

export const decodeRosUint8Array = (value) => {
  if (typeof value === 'string') {
    const bin = atob(value);
    return Array.from(bin, (ch) => ch.charCodeAt(0));
  }
  if (ArrayBuffer.isView(value)) {
    return Array.from(value);
  }
  return Array.isArray(value) ? value : [];
};

export const buildCameraMonitorTopics = (msg) => {
  const cameraStatusArr = decodeRosUint8Array(msg.camera_monitor_status);
  const cameraTimestampStatusArr = decodeRosUint8Array(
    msg.camera_monitor_timestamp_status
  );
  const cameraMonitorNames = msg.camera_monitor_names || [];
  return (msg.camera_monitor_topics || []).map((name, i) => {
    const rawStatus = cameraStatusArr[i] ?? 0;
    const timestampStatus = cameraTimestampStatusArr[i] ?? 0;
    const status = Math.max(rawStatus, timestampStatus);
    let statusLabel = 'OK';
    if (timestampStatus === 2) {
      statusLabel = 'Timestamp skew';
    } else if (status === 2) {
      statusLabel = 'Stalled';
    } else if (status === 1) {
      statusLabel = 'Slow';
    }
    return {
      name,
      cameraName: cameraMonitorNames[i] || name,
      rateHz: msg.camera_monitor_rates_hz?.[i] ?? 0,
      baselineHz: msg.camera_monitor_baseline_hz?.[i] ?? 0,
      secondsSinceLast: msg.camera_monitor_seconds_since_last?.[i] ?? -1,
      status,
      statusLabel,
      timestampSkewS: msg.camera_monitor_timestamp_skew_s?.[i] ?? 0,
      timestampStatus,
      source: 'camera',
    };
  });
};

export const hasRecordSessionContext = (msg) => Boolean(
  msg.task_info?.task_name ||
  msg.current_task_instruction ||
  (msg.subtask_count || 0) > 0
);

const hasNonZeroSystemMetrics = (msg) => [
  msg.used_storage_size,
  msg.total_storage_size,
  msg.used_cpu,
  msg.used_ram_size,
  msg.total_ram_size,
].some((value) => Number(value || 0) !== 0);

export const isMonitorOnlyStatusMessage = (msg, cameraTopics) => (
  cameraTopics.length > 0 &&
  !hasRecordSessionContext(msg) &&
  !hasNonZeroSystemMetrics(msg)
);

export const shouldAnnounceRecordingStart = ({
  hasSeenRecordPhase,
  previousPhase,
  currentPhase,
  proceedTime,
}) => (
  hasSeenRecordPhase &&
  currentPhase === RecordPhase.RECORDING &&
  previousPhase !== RecordPhase.RECORDING &&
  Number(proceedTime || 0) <= 2
);
