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

const runtimeConfig = (
  typeof window !== 'undefined' && window.__CYCLO_CONFIG__
) ? window.__CYCLO_CONFIG__ : {};

function getPort(name, defaultValue) {
  const port = Number(runtimeConfig[name]);
  return Number.isInteger(port) && port > 0 ? port : defaultValue;
}

export const CYCLO_UI_PORT = getPort('uiPort', 7080);
export const CYCLO_ROSBRIDGE_PORT = getPort('rosbridgePort', 7090);
export const CYCLO_VIDEO_SERVER_PORT = getPort('videoServerPort', 7082);
export const CYCLO_WEB_VIDEO_SERVER_PORT = getPort('webVideoServerPort', 7085);
export const CYCLO_SUPERVISOR_API_PORT = getPort('supervisorApiPort', 7100);
