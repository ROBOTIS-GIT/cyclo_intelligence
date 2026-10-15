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

import { useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import { setBtStatus, setSelectedNodeId, setActiveNodeNames } from '../actionCanvasSlice';
import { formatTaskDisplayMessage } from '../../../utils/taskTerminology';
import { normalizeBtStatus, getSimulationInferenceNodeNames } from '../lib/btRuntimeStatus';

export default function useActionBtExecution({ nodes, nodeDataMap, btNodeStatus,
  btStatus, dispatch, callService, getSerializedXml }) {
  const [btExecutionPending, setBtExecutionPending] = useState(null);
  // ── BT Start ──────────────────────────────────────────────────────────────
  const handleStart = useCallback(async () => {
    if (nodes.length === 0) {
      toast.error('No task to run');
      return;
    }
    if (btNodeStatus.state !== 'up') {
      toast.error('Task Engine is off');
      return;
    }
    if (btExecutionPending) return;
    setBtExecutionPending('start');
    try {
      const executionStatus = normalizeBtStatus(btStatus);
      if (['completed', 'failed', 'failure'].includes(executionStatus)) {
        const cleanupResult = await callService(
          '/bt/set_running',
          'std_srvs/srv/SetBool',
          { data: false },
        );
        if (!cleanupResult?.success) {
          throw new Error(formatTaskDisplayMessage(cleanupResult?.message) || 'Failed to reset the completed task');
        }
      }

      const currentXml = getSerializedXml();
      const simulationInferenceNodes = getSimulationInferenceNodeNames(nodeDataMap);
      if (simulationInferenceNodes.length > 0) {
        toast(
          `SendCommand is in simulation mode: ${simulationInferenceNodes.join(', ')}. Robot command topics will not be published.`,
          { duration: 7000 },
        );
      }

      const result = await callService(
        '/bt/load_and_run',
        'interfaces/srv/LoadAndRunTree',
        { tree_xml: currentXml },
        30000
      );
      if (result.success) {
        dispatch(setBtStatus('running'));
        dispatch(setSelectedNodeId(null));
        toast.success('Task started');
      } else {
        toast.error(`Failed: ${formatTaskDisplayMessage(result.message)}`);
      }
    } catch (err) {
      toast.error(`Failed to start task: ${formatTaskDisplayMessage(err.message)}`);
    } finally {
      setBtExecutionPending(null);
    }
  }, [
    btExecutionPending,
    btNodeStatus.state,
    btStatus,
    callService,
    dispatch,
    getSerializedXml,
    nodeDataMap,
    nodes.length,
  ]);

  // ── BT Stop ───────────────────────────────────────────────────────────────
  const handleStop = useCallback(async () => {
    if (btExecutionPending) return;
    setBtExecutionPending('stop');
    try {
      const result = await callService('/bt/set_running', 'std_srvs/srv/SetBool', { data: false });
      if (!result.success) {
        toast.error(`Failed: ${formatTaskDisplayMessage(result.message)}`);
        return;
      }
      dispatch(setBtStatus('stopped'));
      dispatch(setActiveNodeNames([]));
      toast.success('Task stopped');
    } catch (err) {
      toast.error(`Failed to stop task: ${formatTaskDisplayMessage(err.message)}`);
    } finally {
      setBtExecutionPending(null);
    }
  }, [btExecutionPending, callService, dispatch]);

  return { btExecutionPending, handleStart, handleStop };
}
