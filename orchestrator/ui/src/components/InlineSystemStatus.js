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

import React from 'react';
import { useSelector } from 'react-redux';
import clsx from 'clsx';

function formatBytes(bytes) {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function getBarColor(pct) {
  if (pct >= 90) return 'bg-red-500';
  if (pct >= 75) return 'bg-orange-500';
  if (pct >= 50) return 'bg-yellow-500';
  return 'bg-green-500';
}

function getRingColor(pct) {
  if (pct >= 90) return 'text-red-500';
  if (pct >= 75) return 'text-orange-500';
  if (pct >= 50) return 'text-yellow-500';
  return 'text-green-500';
}

function StatusItem({ label, percentage, detail }) {
  const radius = 12;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (percentage / 100) * circumference;

  return (
    <div className="flex items-center gap-1.5">
      <div className="relative w-8 h-8 shrink-0">
        <svg className="w-8 h-8 transform -rotate-90" viewBox="0 0 28 28">
          <circle
            cx="14" cy="14" r={radius}
            stroke="currentColor" strokeWidth="2.5"
            fill="transparent" className="text-gray-200"
          />
          <circle
            cx="14" cy="14" r={radius}
            stroke="currentColor" strokeWidth="2.5"
            fill="transparent"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className={clsx('transition-all duration-300', getRingColor(percentage))}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-[11px] font-bold text-gray-600">
            {Math.round(percentage)}%
          </span>
        </div>
      </div>
      <div className="flex flex-col min-w-0">
        <span className="text-lg font-semibold text-gray-600 leading-none">{label}</span>
        <div className="flex items-center gap-1 mt-0.5">
          <div className="w-12 bg-gray-200 rounded-full h-1.5">
            <div
              className={clsx('h-1.5 rounded-full transition-all duration-300', getBarColor(percentage))}
              style={{ width: `${Math.min(percentage, 100)}%` }}
            />
          </div>
          <span className="text-sm text-gray-500 whitespace-nowrap">{detail}</span>
        </div>
      </div>
    </div>
  );
}

export default function InlineSystemStatus() {
  const cpuPercentage = useSelector((state) => state.tasks.recordStatus.usedCpu) || 0;
  const totalRamSize = useSelector((state) => state.tasks.recordStatus.totalRamSize) || 0;
  const usedRamSize = useSelector((state) => state.tasks.recordStatus.usedRamSize) || 0;
  const totalStorageSize = useSelector((state) => state.tasks.recordStatus.totalStorageSize) || 0;
  const usedStorageSize = useSelector((state) => state.tasks.recordStatus.usedStorageSize) || 0;

  const ramBytes = usedRamSize * 1024 * 1024 * 1024;
  const totalRamBytes = totalRamSize * 1024 * 1024 * 1024;
  const storageBytes = usedStorageSize * 1024 * 1024 * 1024;
  const totalStorageBytes = totalStorageSize * 1024 * 1024 * 1024;

  const ramPercent = totalRamSize > 0 ? (usedRamSize / totalRamSize) * 100 : 0;
  const storagePercent = totalStorageSize > 0 ? (usedStorageSize / totalStorageSize) * 100 : 0;

  const cpuDetail = `${Math.round(cpuPercentage)}%`;
  const ramDetail = `${formatBytes(ramBytes)} / ${formatBytes(totalRamBytes)}`;
  const storageDetail = `${formatBytes(storageBytes)} / ${formatBytes(totalStorageBytes)}`;

  return (
    <div
      className={clsx(
        'flex', 'items-center', 'gap-3',
        'bg-white/90', 'backdrop-blur-sm',
        'rounded-xl', 'px-3', 'py-1.5',
        'shadow-md', 'border', 'border-gray-100'
      )}
    >
      <StatusItem label="CPU" percentage={cpuPercentage} detail={cpuDetail} />
      <StatusItem label="RAM" percentage={ramPercent} detail={ramDetail} />
      <StatusItem label="Storage" percentage={storagePercent} detail={storageDetail} />
    </div>
  );
}
