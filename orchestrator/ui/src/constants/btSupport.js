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

export const BT_SUPPORTED_ROBOT_TYPE = 'ffw_sg2_rev1';

export const BT_UNSUPPORTED_ROBOT_MESSAGE =
  `BT Manager currently supports only ${BT_SUPPORTED_ROBOT_TYPE}. ` +
  'Support for other robot types is coming soon.';

export function isBtRobotSupported(robotType) {
  return String(robotType || '').trim() === BT_SUPPORTED_ROBOT_TYPE;
}
