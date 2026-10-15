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

import { useCallback, useEffect } from 'react';
import toast from 'react-hot-toast';
import { setTreeXml, setTreeFileName } from '../actionCanvasSlice';
import { normalizeBtStatus } from '../lib/btRuntimeStatus';

export default function useActionBtClear({ graph, history, confirmation, runtime, dispatch, selectNode, btStatus }) {
  const { nodesRef, setNodes, setEdges, setNodeDataMap, setParseError } = graph;
  const { captureHistory, getHistorySnapshot } = history;
  const { clearTreeArmed, armClearTree, clearTreeTargetRef, disarmClearTree } = confirmation;
  const { btNodePendingAction, btExecutionPending } = runtime;
  useEffect(() => {
    if (!clearTreeArmed) return;
    if (
      btExecutionPending ||
      btNodePendingAction ||
      ['running', 'stopping'].includes(normalizeBtStatus(btStatus))
    ) {
      disarmClearTree();
    }
  }, [
    btExecutionPending,
    btNodePendingAction,
    btStatus,
    clearTreeArmed,
    disarmClearTree,
  ]);

  const handleClearTree = useCallback(() => {
    if (nodesRef.current.length === 0) return;
    if (
      ['running', 'stopping'].includes(normalizeBtStatus(btStatus)) ||
      btNodePendingAction ||
      btExecutionPending
    ) {
      return;
    }
    const currentSnapshot = getHistorySnapshot();
    if (!clearTreeArmed) {
      armClearTree(currentSnapshot);
      return;
    }
    if (clearTreeTargetRef.current !== currentSnapshot) {
      armClearTree(currentSnapshot);
      return;
    }

    captureHistory();
    setNodes([]);
    setEdges([]);
    setNodeDataMap(new Map());
    setParseError(null);
    selectNode(null);
    dispatch(setTreeXml(''));
    dispatch(setTreeFileName(''));
    disarmClearTree();
    toast.success('Task cleared');
  }, [
    nodesRef,
    btStatus,
    btNodePendingAction,
    btExecutionPending,
    getHistorySnapshot,
    clearTreeArmed,
    clearTreeTargetRef,
    captureHistory,
    setNodes,
    setEdges,
    setNodeDataMap,
    setParseError,
    selectNode,
    dispatch,
    disarmClearTree,
    armClearTree,
  ]);

  return handleClearTree;
}
