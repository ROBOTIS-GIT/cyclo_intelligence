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

export function normalizeBtStatus(status) {
  return String(status || 'stopped').trim().toLowerCase();
}

export function getBtStatusLabel(status) {
  switch (normalizeBtStatus(status)) {
    case 'running':
      return 'Running';
    case 'completed':
      return 'Completed';
    case 'failed':
    case 'failure':
      return 'Failed';
    case 'stopping':
      return 'Stopping';
    default:
      return 'Stopped';
  }
}

export function getSimulationInferenceNodeNames(nodeDataMap) {
  return Array.from(nodeDataMap.values())
    .filter(({ tag, params = {} }) => {
      if (tag !== 'SendCommand') return false;
      const command = String(params.command || 'LOAD').toUpperCase();
      if (command !== 'LOAD') return false;
      const mode = String(params.inference_mode || 'simulation').toLowerCase();
      return mode !== 'robot';
    })
    .map(({ name }) => name)
    .filter(Boolean);
}
