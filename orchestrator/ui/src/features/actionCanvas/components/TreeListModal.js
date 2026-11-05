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

import React, { useState, useEffect, useCallback } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdClose, MdDescription, MdRefresh } from 'react-icons/md';

import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { listBtTrees } from '../btTreesApi';

export default function TreeListModal({
  isOpen,
  onClose,
  onSelect,
  variant = 'legacy',
}) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const autonomyStudioVariant = variant === 'autonomy-studio';

  const fetchTrees = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const result = await listBtTrees();
      const next = (result.trees || []).map((tree) => ({
        name: tree.name,
        full_path: tree.path || tree.name,
      }));
      setItems(next);
    } catch (err) {
      setItems([]);
      const msg = formatTaskDisplayMessage(err.message || err);
      setErrorMsg(msg);
      toast.error(`Failed to load tasks: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      fetchTrees();
    }
  }, [isOpen, fetchTrees]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="fixed inset-0 bg-black bg-opacity-50 transition-opacity" />

      <div className="flex min-h-full items-center justify-center p-4">
        <div className={clsx(
          'relative max-w-lg w-full max-h-[80vh] flex flex-col',
          autonomyStudioVariant
            ? 'rounded-2xl border border-[var(--mc-border)] bg-[var(--mc-surface)] text-[var(--mc-text)] shadow-[var(--mc-shadow)]'
            : 'bg-white rounded-lg shadow-xl',
        )}>
          <div className={clsx(
            'flex items-center justify-between px-6 py-4 border-b',
            autonomyStudioVariant ? 'border-[var(--mc-border)]' : 'border-gray-200',
          )}>
            <h2 className={clsx(
              'font-semibold',
              autonomyStudioVariant ? 'text-[15px] text-[var(--mc-text)]' : 'text-xl text-gray-900',
            )}>
              Open Task
            </h2>
            <div className="flex items-center gap-1">
              <button
                onClick={fetchTrees}
                disabled={loading}
                className={clsx(
                  'p-2 rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? loading
                      ? 'text-[var(--mc-text-subtle)] cursor-not-allowed opacity-50'
                      : 'text-[var(--mc-text-muted)] hover:text-[var(--mc-text)] hover:bg-[var(--mc-surface-hover)]'
                    : loading
                    ? 'text-gray-300 cursor-not-allowed'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                )}
                title="Refresh"
              >
                <MdRefresh size={20} />
              </button>
              <button
                onClick={onClose}
                className={clsx(
                  'p-2 rounded-lg transition-colors',
                  autonomyStudioVariant
                    ? 'text-[var(--mc-text-subtle)] hover:text-[var(--mc-text)] hover:bg-[var(--mc-surface-hover)]'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100',
                )}
              >
                <MdClose size={24} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 py-2 min-h-[200px]">
            {loading ? (
              <div className={clsx(
                'flex items-center justify-center h-full py-12 text-sm',
                autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
              )}>
                Loading tasks…
              </div>
            ) : errorMsg ? (
              <div className="flex flex-col items-center justify-center h-full py-12 text-center">
                <p className={clsx(
                  'text-sm font-medium',
                  autonomyStudioVariant ? 'text-[var(--mc-danger)]' : 'text-red-500',
                )}>
                  Failed to load tasks
                </p>
                <p className={clsx(
                  'text-xs mt-1 break-all px-4',
                  autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
                )}>
                  {errorMsg}
                </p>
              </div>
            ) : items.length === 0 ? (
              <div className={clsx(
                'flex items-center justify-center h-full py-12 text-sm',
                autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
              )}>
                No saved tasks found
              </div>
            ) : (
              <ul className={clsx(
                'divide-y',
                autonomyStudioVariant ? 'divide-[var(--mc-border)]' : 'divide-gray-100',
              )}>
                {items.map((item) => (
                  <li key={item.full_path}>
                    <button
                      onClick={() => {
                        onSelect(item);
                        onClose();
                      }}
                      className={clsx(
                        'w-full flex items-center gap-3 px-4 py-3',
                        'text-left text-sm',
                        autonomyStudioVariant
                          ? 'text-[var(--mc-text)] hover:bg-[var(--mc-accent-soft)] hover:text-[var(--mc-accent-hover)]'
                          : 'text-gray-800 hover:bg-blue-50 hover:text-blue-700',
                        'transition-colors rounded-md'
                      )}
                    >
                      <MdDescription
                        size={18}
                        className={clsx(
                          'flex-shrink-0',
                          autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
                        )}
                      />
                      <span className="font-mono break-all">{item.name}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
