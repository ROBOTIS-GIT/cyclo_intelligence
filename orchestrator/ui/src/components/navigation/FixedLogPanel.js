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
//
// Author: Howon Kim

import { useCallback, useEffect, useRef, useState } from 'react';
import { clearServiceLogs, getServiceLogs } from '../../utils/navigationApi';
import LogViewer from './LogViewer';

const POLL_INTERVAL_MS = 200;
const MAX_LOG_LINES = 1000;

export default function FixedLogPanel({ service }) {
  const [lines, setLines] = useState([]);
  const [error, setError] = useState(null);
  const [clearLoading, setClearLoading] = useState(false);
  const cursorRef = useRef(null);
  const loadingRef = useRef(false);

  const appendLogs = useCallback((logs) => {
    if (!logs) return;
    const nextLines = logs.split('\n');
    if (nextLines[nextLines.length - 1] === '') nextLines.pop();
    if (nextLines.length === 0) return;
    setLines((current) => {
      const combined = [...current, ...nextLines];
      return combined.length > MAX_LOG_LINES
        ? combined.slice(-MAX_LOG_LINES)
        : combined;
    });
  }, []);

  const pollLogs = useCallback(async (initial = false) => {
    if (loadingRef.current) return;
    loadingRef.current = true;
    try {
      const result = await getServiceLogs({
        tail: initial ? 300 : 1,
        cursor: initial ? undefined : cursorRef.current,
      });
      appendLogs(result.logs || '');
      cursorRef.current = result.cursor ?? 0;
      setError(null);
    } catch (pollError) {
      setError(pollError instanceof Error ? pollError.message : 'Failed to load logs');
    } finally {
      loadingRef.current = false;
    }
  }, [appendLogs]);

  useEffect(() => {
    setLines([]);
    cursorRef.current = null;
    pollLogs(true);
    const interval = window.setInterval(() => pollLogs(false), POLL_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [pollLogs]);

  const handleClear = async () => {
    setClearLoading(true);
    try {
      await clearServiceLogs();
      setLines([]);
      cursorRef.current = 0;
      setError(null);
    } catch (clearError) {
      setError(clearError instanceof Error ? clearError.message : 'Failed to clear logs');
    } finally {
      setClearLoading(false);
    }
  };

  return (
    <div
      className="h-full flex flex-col overflow-hidden border rounded"
      style={{
        backgroundColor: 'var(--vscode-editor-background)',
        borderColor: 'var(--vscode-panel-border)',
      }}
    >
      <div
        className="px-3 py-2 flex items-center justify-between border-b shrink-0"
        style={{
          backgroundColor: 'var(--vscode-titleBar-activeBackground)',
          borderColor: 'var(--vscode-panel-border)',
        }}
      >
        <div className="text-xs font-medium">{service} Logs</div>
        <div className="flex items-center gap-2 text-xs">
          <button type="button" disabled={clearLoading} onClick={handleClear}>
            {clearLoading ? 'Clearing...' : 'Clear'}
          </button>
        </div>
      </div>
      {error && (
        <div className="px-3 py-2 text-xs text-red-500 border-b" style={{ borderColor: 'var(--vscode-panel-border)' }}>
          {error}
        </div>
      )}
      <div className="flex-1 min-h-0 relative">
        <LogViewer lines={lines} autoScroll className="h-full" />
      </div>
    </div>
  );
}
