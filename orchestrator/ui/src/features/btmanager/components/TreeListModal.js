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

import React, { useState, useEffect, useCallback } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import { MdClose, MdDescription, MdRefresh } from 'react-icons/md';

import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

export default function TreeListModal({ isOpen, onClose, onSelect }) {
  const { getTreeList } = useRosServiceCaller();
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const fetchTrees = useCallback(async () => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const result = await getTreeList();
      if (!result || !result.success) {
        const msg = (result && result.message) || 'Unknown error';
        setItems([]);
        setErrorMsg(msg);
        toast.error(`Failed to list trees: ${msg}`);
        return;
      }
      const names = result.tree_names || [];
      const paths = result.tree_full_paths || [];
      const next = names.map((name, i) => ({
        name,
        full_path: paths[i] || name,
      }));
      setItems(next);
    } catch (err) {
      setItems([]);
      const msg = err.message || String(err);
      setErrorMsg(msg);
      toast.error(`Failed to list trees: ${msg}`);
    } finally {
      setLoading(false);
    }
  }, [getTreeList]);

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
        <div className="relative bg-white rounded-lg shadow-xl max-w-lg w-full max-h-[80vh] flex flex-col">
          <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900">
              Select BT XML
            </h2>
            <div className="flex items-center gap-1">
              <button
                onClick={fetchTrees}
                disabled={loading}
                className={clsx(
                  'p-2 rounded-lg transition-colors',
                  loading
                    ? 'text-gray-300 cursor-not-allowed'
                    : 'text-gray-400 hover:text-gray-600 hover:bg-gray-100'
                )}
                title="Refresh"
              >
                <MdRefresh size={20} />
              </button>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <MdClose size={24} />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto px-2 py-2 min-h-[200px]">
            {loading ? (
              <div className="flex items-center justify-center h-full py-12 text-gray-500 text-sm">
                Loading trees…
              </div>
            ) : errorMsg ? (
              <div className="flex flex-col items-center justify-center h-full py-12 text-center">
                <p className="text-red-500 text-sm font-medium">
                  Failed to load trees
                </p>
                <p className="text-gray-500 text-xs mt-1 break-all px-4">
                  {errorMsg}
                </p>
              </div>
            ) : items.length === 0 ? (
              <div className="flex items-center justify-center h-full py-12 text-gray-500 text-sm">
                No tree XMLs found in bt/trees/
              </div>
            ) : (
              <ul className="divide-y divide-gray-100">
                {items.map((item) => (
                  <li key={item.full_path}>
                    <button
                      onClick={() => {
                        onSelect(item);
                        onClose();
                      }}
                      className={clsx(
                        'w-full flex items-center gap-3 px-4 py-3',
                        'text-left text-sm text-gray-800',
                        'hover:bg-blue-50 hover:text-blue-700',
                        'transition-colors rounded-md'
                      )}
                    >
                      <MdDescription
                        size={18}
                        className="text-gray-400 flex-shrink-0"
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
