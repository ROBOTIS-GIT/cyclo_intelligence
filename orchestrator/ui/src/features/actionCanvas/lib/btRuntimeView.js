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

import { normalizeBtStatus, getBtStatusLabel } from './btRuntimeStatus';

export default function btRuntimeView({ hasTree, btStatus, btNodeStatus, btNodePendingAction, btExecutionPending, autonomyStudioVariant }) {
  const normalizedBtStatus = normalizeBtStatus(btStatus);
  const isBtNodeUp = btNodeStatus.state === 'up';
  const isBtNodeBusy = Boolean(btNodePendingAction);
  const isBtExecutionBusy = Boolean(btExecutionPending);
  const isBtRunning = normalizedBtStatus === 'running';
  const isBtBusy = isBtRunning || normalizedBtStatus === 'stopping';
  const isBtTerminal = ['completed', 'failed', 'failure'].includes(normalizedBtStatus);
  const canStartBt = hasTree && isBtNodeUp && !isBtBusy && !isBtNodeBusy && !isBtExecutionBusy;
  const canStopBt = isBtNodeUp && isBtRunning && !isBtNodeBusy && !isBtExecutionBusy;
  const canClearTree = hasTree && !isBtBusy && !isBtNodeBusy && !isBtExecutionBusy;
  const canStartBtNode = !isBtNodeUp && !isBtNodeBusy && !isBtExecutionBusy;
  const canStopBtNode = isBtNodeUp &&
    (normalizedBtStatus === 'stopped' || isBtTerminal) &&
    !isBtNodeBusy &&
    !isBtExecutionBusy;
  const statusColor = autonomyStudioVariant
    ? isBtRunning ? 'bg-[var(--mc-success)]' :
      normalizedBtStatus === 'completed' ? 'bg-[var(--mc-warning)]' :
      ['failed', 'failure'].includes(normalizedBtStatus) ? 'bg-[var(--mc-danger)]' :
      normalizedBtStatus === 'stopping' ? 'bg-[var(--mc-warning)]' :
      'bg-[var(--mc-text-subtle)]'
    : isBtRunning ? 'bg-green-500' :
      normalizedBtStatus === 'completed' ? 'bg-yellow-400' :
      ['failed', 'failure'].includes(normalizedBtStatus) ? 'bg-red-500' :
      normalizedBtStatus === 'stopping' ? 'bg-orange-400' :
      'bg-gray-400';
  const statusLabel = getBtStatusLabel(btStatus);
  const btNodeStatusColor = autonomyStudioVariant
    ? isBtNodeUp ? 'bg-[var(--mc-success)]' :
      btNodeStatus.state === 'down' ? 'bg-[var(--mc-text-subtle)]' :
      'bg-[var(--mc-warning)]'
    : isBtNodeUp ? 'bg-green-500' :
      btNodeStatus.state === 'down' ? 'bg-gray-400' :
      'bg-yellow-400';
  const btNodeStatusLabel =
    isBtNodeUp ? 'On' :
    btNodeStatus.state === 'down' ? 'Off' :
    'Unknown';

  return { isBtBusy, canClearTree, isBtNodeUp, canStartBtNode, canStopBtNode,
    canStartBt, canStopBt, statusColor, statusLabel, btNodeStatusColor, btNodeStatusLabel };
}
