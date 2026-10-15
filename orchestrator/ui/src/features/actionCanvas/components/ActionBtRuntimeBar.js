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

import React from 'react';
import clsx from "clsx";
import { MdPlayArrow, MdStop, MdPowerSettingsNew } from "react-icons/md";

export default function ActionBtRuntimeBar({ view, runtime }) {
  const { autonomyStudioVariant } = view;
  const {
    btNodeStatusColor,
    btNodeStatusLabel,
    handleBtNodeOn,
    canStartBtNode,
    handleBtNodeOff,
    canStopBtNode,
    handleStart,
    canStartBt,
    handleStop,
    canStopBt,
    statusColor,
    statusLabel,
  } = runtime;
  return (<>
      {/* Bottom Control Bar */}
      <div className={clsx(
        'flex items-center justify-between px-6 border-t',
        autonomyStudioVariant
          ? 'h-14 border-[var(--mc-border)] bg-[var(--mc-surface-2)]'
          : 'py-3 border-black bg-white',
      )}>
        <div className="flex items-center gap-3">
          <div className={clsx(
            'flex items-center gap-2 pr-3 mr-1 border-r',
            autonomyStudioVariant ? 'border-[var(--mc-border)]' : 'border-gray-200',
          )}>
            <div className={clsx('w-3 h-3 rounded-full', btNodeStatusColor)} />
            <span className={clsx(
              'text-sm',
              autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
            )}>
              Task Engine {btNodeStatusLabel}
            </span>
            <button
              onClick={handleBtNodeOn}
              disabled={!canStartBtNode}
              title="Turn on Task Engine"
              className={clsx(
                'flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                autonomyStudioVariant
                  ? !canStartBtNode
                    ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                    : 'border border-[var(--mc-success)] bg-[var(--mc-success)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                  : !canStartBtNode
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              )}
            >
              <MdPowerSettingsNew size={18} />
              Turn On
            </button>
            <button
              onClick={handleBtNodeOff}
              disabled={!canStopBtNode}
              title="Turn off Task Engine"
              className={clsx(
                'flex items-center gap-1 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                autonomyStudioVariant
                  ? !canStopBtNode
                    ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                    : 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                  : !canStopBtNode
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-red-600 hover:bg-red-700 text-white'
              )}
            >
              <MdStop size={18} />
              Turn Off
            </button>
          </div>
          <button
            onClick={handleStart}
            disabled={!canStartBt}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              autonomyStudioVariant
                ? !canStartBt
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : 'border border-[var(--mc-success)] bg-[var(--mc-success)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                : !canStartBt
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-green-600 hover:bg-green-700 text-white'
            )}
          >
            <MdPlayArrow size={20} />
            Run Task
          </button>
          <button
            onClick={handleStop}
            disabled={!canStopBt}
            className={clsx(
              'flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-medium transition-colors',
              autonomyStudioVariant
                ? !canStopBt
                  ? 'border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                  : 'border border-[var(--mc-danger)] bg-[var(--mc-danger)] text-[var(--mc-accent-fg)] shadow-[var(--mc-shadow)] hover:opacity-90'
                : !canStopBt
                ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                : 'bg-red-600 hover:bg-red-700 text-white'
            )}
          >
            <MdStop size={20} />
            Stop Task
          </button>
        </div>

        <div className="flex items-center gap-2">
          <div className={clsx('w-3 h-3 rounded-full', statusColor)} />
          <span className={clsx(
            'text-sm',
            autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
          )}>
            {statusLabel}
          </span>
        </div>
      </div>


  </>);
}
