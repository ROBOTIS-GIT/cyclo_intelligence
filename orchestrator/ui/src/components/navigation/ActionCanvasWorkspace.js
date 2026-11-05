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
import clsx from 'clsx';
import { useSelector } from 'react-redux';
import { MdAccountTree } from 'react-icons/md';

import BTEditorSurface from '../../features/actionCanvas/components/BTEditorSurface';
import {
  btUnsupportedRobotMessage,
  isBtRobotSupported,
} from '../../constants/btSupport';
import { selectBtSupportedRobotTypes } from '../../features/actionCanvas/btSupportSlice';

export default function ActionCanvasWorkspace({
  isActive = true,
  title = 'Action Canvas',
  subtitle = '',
  className = 'w-full h-full',
  variant = 'legacy',
  onExitStateChange,
}) {
  const robotType = useSelector((state) => state.tasks.robotType);
  const supportedRobotTypes = useSelector(selectBtSupportedRobotTypes);
  const btRobotSupported = isBtRobotSupported(robotType, supportedRobotTypes);
  const autonomyStudioVariant = variant === 'autonomy-studio';

  if (!btRobotSupported) {
    return (
      <div
        data-variant={variant}
        className={clsx(
          className,
          'flex flex-col',
          autonomyStudioVariant && 'bg-[var(--mc-bg)] text-[var(--mc-text)]',
        )}
      >
        <div className={clsx(
          'flex items-center justify-between px-6 py-4 border-b',
          autonomyStudioVariant
            ? 'border-[var(--mc-border)] bg-[var(--mc-surface)]'
            : 'border-black bg-white',
        )}>
          <div className="min-w-0">
            <h1 className={clsx(
              'font-bold',
              autonomyStudioVariant ? 'text-[15px] text-[var(--mc-text)]' : 'text-xl text-gray-800',
            )}>
              {title}
            </h1>
            {subtitle && (
              <p className={clsx(
                'mt-0.5 truncate text-[10px] font-mono',
                autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
              )}>
                {subtitle}
              </p>
            )}
          </div>
        </div>

        <div className={clsx(
          'flex-1 flex items-center justify-center px-6',
          autonomyStudioVariant ? 'bg-[var(--mc-bg)]' : 'bg-gray-50',
        )}>
          <div className={clsx(
            'w-full max-w-xl rounded-2xl border px-8 py-7 text-center shadow-sm',
            autonomyStudioVariant
              ? 'border-[var(--mc-border)] bg-[var(--mc-surface)]'
              : 'border-gray-200 bg-white',
          )}>
            <MdAccountTree
              size={40}
              className={clsx(
                'mx-auto mb-4',
                autonomyStudioVariant ? 'text-[var(--mc-accent)]' : 'text-gray-400',
              )}
            />
            <h2 className={clsx(
              'text-lg font-semibold',
              autonomyStudioVariant ? 'text-[var(--mc-text)]' : 'text-gray-900',
            )}>
              {btUnsupportedRobotMessage(supportedRobotTypes)}
            </h2>
            <p className={clsx(
              'mt-3 text-sm',
              autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
            )}>
              Current robot type: {robotType || 'Not selected'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <BTEditorSurface
      isActive={isActive}
      title={title}
      subtitle={subtitle}
      className={className}
      variant={variant}
      onExitStateChange={onExitStateChange}
    />
  );
}
