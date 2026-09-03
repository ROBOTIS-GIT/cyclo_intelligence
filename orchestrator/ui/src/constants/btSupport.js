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

// The supervisor API (backed by shared.robot_configs.schema) owns the list of
// robots the behavior-tree engine supports; see features/actionCanvas/
// btSupportSlice.js for the fetch. This default only covers the moment before
// /api/bt/support has answered.
export const DEFAULT_BT_SUPPORTED_ROBOT_TYPES = Object.freeze(['ffw_sg2_rev1']);

export function normalizeBtSupportedRobotTypes(value) {
  const list = Array.isArray(value)
    ? value.map((item) => String(item || '').trim()).filter(Boolean)
    : [];
  return list.length ? list : [...DEFAULT_BT_SUPPORTED_ROBOT_TYPES];
}

export function isBtRobotSupported(robotType, supportedRobotTypes = DEFAULT_BT_SUPPORTED_ROBOT_TYPES) {
  const normalized = String(robotType || '').trim();
  return Boolean(normalized) && supportedRobotTypes.includes(normalized);
}

export function formatBtSupportedRobotTypes(supportedRobotTypes = DEFAULT_BT_SUPPORTED_ROBOT_TYPES) {
  return supportedRobotTypes.join(', ');
}

export function btUnsupportedRobotMessage(
  supportedRobotTypes = DEFAULT_BT_SUPPORTED_ROBOT_TYPES,
  product = 'Action Canvas',
) {
  return (
    `${product} currently supports only ${formatBtSupportedRobotTypes(supportedRobotTypes)}. `
    + 'Support for other robot types is coming soon.'
  );
}
