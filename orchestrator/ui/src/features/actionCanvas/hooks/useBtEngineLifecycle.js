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

import { useState, useCallback, useEffect } from 'react';
import toast from 'react-hot-toast';
import { setBtStatus, setActiveNodeNames } from '../actionCanvasSlice';
import { formatBtSupportedRobotTypes, isBtRobotSupported } from '../../../constants/btSupport';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { normalizeBtStatus } from '../lib/btRuntimeStatus';
const API_BASE = '/api';
async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export default function useBtEngineLifecycle({ dispatch, isActive, robotType,
  supportedRobotTypes, btStatus, callService, refreshCatalog }) {
  const [btNodeStatus, setBtNodeStatus] = useState({ state: 'unknown', raw: 'not checked' });
  const [btNodePendingAction, setBtNodePendingAction] = useState(null);
  // ── BT node process lifecycle via supervisor API ─────────────────────────
  const refreshBtNodeStatus = useCallback(async ({ quiet = false } = {}) => {
    try {
      const response = await fetch(`${API_BASE}/services/bt_node/status`);
      const data = await readJsonResponse(response);
      if (!response.ok) {
        throw new Error(data.detail || `status failed (${response.status})`);
      }
      setBtNodeStatus(data);
      if (data.state === 'down') {
        dispatch(setBtStatus('stopped'));
        dispatch(setActiveNodeNames([]));
      }
      return data;
    } catch (err) {
      const next = {
        state: 'unknown',
        raw: err.message,
      };
      setBtNodeStatus(next);
      if (!quiet) toast.error(`Task Engine status unavailable: ${formatTaskDisplayMessage(err.message)}`);
      return next;
    }
  }, [dispatch]);

  useEffect(() => {
    if (!isActive) return undefined;
    refreshBtNodeStatus({ quiet: true });
    const id = setInterval(
      () => refreshBtNodeStatus({ quiet: true }),
      5000,
    );
    return () => clearInterval(id);
  }, [isActive, refreshBtNodeStatus]);

  const callBtNodeService = useCallback(async (action) => {
    setBtNodePendingAction(action);
    try {
      const init = { method: 'POST' };
      if (action === 'start') {
        init.headers = { 'Content-Type': 'application/json' };
        init.body = JSON.stringify({ robot_type: robotType || '' });
      }
      const response = await fetch(`${API_BASE}/services/bt_node/${action}`, init);
      const data = await readJsonResponse(response);
      if (!response.ok || data.ok === false) {
        throw new Error(data.detail || data.message || `${action} failed`);
      }
      return data;
    } finally {
      setBtNodePendingAction(null);
    }
  }, [robotType]);

  const handleBtNodeOn = useCallback(async () => {
    if (robotType && !isBtRobotSupported(robotType, supportedRobotTypes)) {
      toast.error(`Action Canvas currently supports only ${formatBtSupportedRobotTypes(supportedRobotTypes)}`);
      return;
    }

    try {
      await callBtNodeService('start');
      toast.success('Task Engine started');
      await refreshBtNodeStatus({ quiet: true });
      try {
        await refreshCatalog({ force: true });
      } catch (err) {
        console.debug('BT catalog refresh after node start failed:', err.message);
      }
    } catch (err) {
      toast.error(`Failed to start Task Engine: ${formatTaskDisplayMessage(err.message)}`);
      await refreshBtNodeStatus({ quiet: true });
    }
  }, [callBtNodeService, refreshBtNodeStatus, refreshCatalog, robotType, supportedRobotTypes]);

  const handleBtNodeOff = useCallback(async () => {
    try {
      const executionStatus = normalizeBtStatus(btStatus);
      if (['completed', 'failed', 'failure'].includes(executionStatus)) {
        const result = await callService(
          '/bt/set_running',
          'std_srvs/srv/SetBool',
          { data: false },
        );
        if (!result?.success) {
          throw new Error(formatTaskDisplayMessage(result?.message) || 'Failed to reset the completed task');
        }
      }
      await callBtNodeService('stop');
      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('Task Engine stopped');
      await refreshBtNodeStatus({ quiet: true });
    } catch (err) {
      toast.error(`Failed to stop Task Engine: ${formatTaskDisplayMessage(err.message)}`);
      await refreshBtNodeStatus({ quiet: true });
    }
  }, [btStatus, callBtNodeService, callService, dispatch, refreshBtNodeStatus]);

  return { btNodeStatus, btNodePendingAction, handleBtNodeOn, handleBtNodeOff };
}
