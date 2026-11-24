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

import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import { useSelector } from 'react-redux';
import { MdRefresh } from 'react-icons/md';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { RecordPhase } from '../constants/taskPhases';
import { requestSupervisorApi } from '../features/missionCanvas/lib/btNodeApi';

const STATUS_OK = 0;
const STATUS_SLOW = 1;
const STATUS_STALLED = 2;

const statusColor = {
  [STATUS_OK]: 'bg-green-500',
  [STATUS_SLOW]: 'bg-yellow-400',
  [STATUS_STALLED]: 'bg-red-500',
};

const statusLabel = {
  [STATUS_OK]: 'OK',
  [STATUS_SLOW]: 'Slow',
  [STATUS_STALLED]: 'Stalled',
};

const formatRate = (value) => Number(value || 0).toFixed(1);

const rowLabel = (topic, topicKeys) => {
  const keys = topicKeys?.[topic.name];
  if (Array.isArray(keys) && keys.length) return [...new Set(keys)].sort().join(', ');
  return (topic.source === 'camera' && topic.cameraName) || topic.name;
};

const rowStatusTitle = (topic) => {
  const statusText = topic.source === 'camera' && topic.timestampStatus === STATUS_STALLED
    ? 'Timestamp skew'
    : statusLabel[topic.status] || '?';
  const lastText = topic.secondsSinceLast >= 0
    ? `${topic.secondsSinceLast.toFixed(1)}s ago`
    : 'never';
  const cameraText = topic.source === 'camera'
    ? `, timestamp ${topic.timestampStatus === STATUS_STALLED ? 'skew' : 'OK'}, skew ${(topic.timestampSkewS || 0).toFixed(3)}s`
    : '';
  return `${statusText} (baseline ${formatRate(topic.baselineHz)} Hz, last ${lastText}${cameraText})`;
};

export default function RecordTopicMonitor({ embedded = false, actions, collapsed = false }) {
  const monitor = useSelector((state) => state.tasks.recordingMonitor);
  const robotType = useSelector((state) => state.tasks.robotType);
  const recordingBusy = useSelector((state) => state.tasks.recordStatus.running ||
    state.tasks.recordStatus.recordPhase !== RecordPhase.READY);
  const { sendRecordCommand } = useRosServiceCaller();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [metadata, setMetadata] = useState(null);

  useEffect(() => {
    setMetadata(null);
    if (!robotType) return;
    const controller = new AbortController();
    let active = true;
    requestSupervisorApi(`/robots/${encodeURIComponent(robotType)}/topic-keys`, {
      signal: controller.signal,
    }).then((data) => {
      if (active) setMetadata({ robotType, topicKeys: data.topic_keys, topicRoles: data.topic_roles });
    }).catch(() => {
      // Labels are optional; monitoring and recording work without this API.
    });
    return () => {
      active = false;
      controller.abort();
    };
  }, [robotType]);

  // Never render the previous robot's labels while the new fetch is pending.
  const topicKeys = metadata?.robotType === robotType ? metadata?.topicKeys : null;
  const topicRoles = metadata?.robotType === robotType ? metadata?.topicRoles : null;

  const handleRefresh = async () => {
    if (isRefreshing || recordingBusy) return;
    setIsRefreshing(true);
    try {
      await sendRecordCommand('refresh_topics');
    } catch (_) {
      // ignore; spinner will clear
    } finally {
      // Small delay so the spinner is visible and users see the reset.
      setTimeout(() => setIsRefreshing(false), 400);
    }
  };

  const rows = [
    ...(monitor?.topics || []).map((topic) => ({ ...topic, source: topic.source || 'rosbag' })),
    ...(monitor?.cameraTopics || []).map((topic) => ({ ...topic, source: 'camera' })),
  ];

  const sorted = [...rows].sort((a, b) => b.status - a.status);
  const displayRows = sorted.flatMap((topic) => {
    const roles = topicRoles?.[topic.name];
    const configured = ['observation', 'action'].flatMap((role) => {
      const keys = roles?.[role];
      return Array.isArray(keys) && keys.length
        ? [{ ...topic, role, label: [...new Set(keys)].sort().join(', ') }]
        : [];
    });
    return configured.length ? configured : [{ ...topic, role: 'unknown', label: rowLabel(topic, topicKeys) }];
  });
  const problemCount = rows.filter((t) => t.status !== STATUS_OK).length;
  const worstStatus = rows.reduce((max, t) => Math.max(max, t.status), STATUS_OK);
  const overallColor = statusColor[worstStatus] || 'bg-green-500';

  return (
    <div className={clsx(
      'w-full h-full min-w-0 min-h-0 flex flex-col overflow-hidden',
      !embedded && 'bg-white border border-gray-200 rounded-2xl shadow-md p-3'
    )}>
      <div className="flex shrink-0 items-center gap-2 min-h-[46px] border-b border-gray-200 px-3">
        <span className={clsx('inline-block w-3 h-3 shrink-0 rounded-full', overallColor)} />
        <span className="text-sm font-semibold text-gray-800 shrink-0">Topic Monitor</span>
        <span className="text-xs text-gray-400 ml-auto min-w-0 truncate">
          {!rows.length ? null : problemCount > 0
            ? `${problemCount} issue(s)`
            : `All ${rows.length} OK`}
        </span>
        <button
          type="button"
          onClick={handleRefresh}
          disabled={isRefreshing || recordingBusy}
          aria-label="Reset topic subscriptions"
          title="Reset topic subscriptions"
          className={clsx(
            'p-1 shrink-0 rounded-md text-gray-500 hover:bg-gray-100 hover:text-gray-700',
            'disabled:text-gray-300 disabled:cursor-not-allowed'
          )}
        >
          <MdRefresh className={clsx('w-4 h-4', isRefreshing && 'animate-spin')} />
        </button>
        {actions}
      </div>
      <div hidden={collapsed} className={clsx(
        'flex-1 min-h-0 overflow-y-auto', embedded ? 'p-3' : 'pt-2'
      )}>
        {!rows.length ? <p className="text-xs text-gray-400">No topics</p> : (
          <table className="w-full text-xs table-fixed">
            <thead>
              <tr className="text-gray-500 border-b border-gray-100">
                <th className="text-left py-1 font-medium w-20">Type</th>
                <th className="text-left py-1 font-medium">Key</th>
                <th className="text-right py-1 font-medium w-12">Hz</th>
                <th className="text-center py-1 font-medium w-10">Status</th>
              </tr>
            </thead>
            <tbody>
              {displayRows.map((t) => (
                <tr key={`${t.source}:${t.name}:${t.role}`} className="border-b border-gray-50 last:border-0">
                  <td className="py-1 pr-2 align-top text-gray-500 break-words">
                    {t.role === 'observation' ? 'Observation' : t.role === 'action' ? 'Action' : 'Unknown'}
                  </td>
                  <td
                    className="py-1 break-all"
                    title={t.source === 'camera' && t.cameraName
                      ? `${t.name} (${t.cameraName})`
                      : t.name}
                  >
                    {t.label}
                  </td>
                  <td className="py-1 text-right font-mono text-gray-700">
                    {formatRate(t.rateHz)}
                  </td>
                  <td className="py-1 text-center">
                    <span
                      className={clsx(
                        'inline-block w-2.5 h-2.5 rounded-full',
                        statusColor[t.status] || 'bg-gray-300'
                      )}
                      title={rowStatusTitle(t)}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
