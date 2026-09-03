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

import { ROBOT_TYPE_STORAGE_KEY } from "../../tasks/taskSlice";

export const SUPERVISOR_API_BASE = "/api";

// The robot the operator selected on the Home page (persisted by taskSlice).
function readPersistedRobotType() {
  try {
    return String(window.sessionStorage.getItem(ROBOT_TYPE_STORAGE_KEY) || "").trim();
  } catch {
    return "";
  }
}

export async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

export async function requestSupervisorApi(path, init) {
  if (typeof fetch !== "function") {
    throw new Error("Supervisor API is not available");
  }
  const response = await fetch(`${SUPERVISOR_API_BASE}${path}`, init);
  const data = await readJsonResponse(response);
  if (!response.ok || data.ok === false) {
    throw new Error(data.detail || data.message || `Request failed (${response.status})`);
  }
  return data;
}

export function getBtNodeServiceStatus() {
  return requestSupervisorApi("/services/bt_node/status");
}

// The supervisor validates the robot type against shared.robot_configs.schema
// and falls back to its default when none is given.
export function setBtNodeServiceActive(active, robotType = "") {
  const init = { method: "POST" };
  if (active) {
    const resolved = String(robotType || "").trim() || readPersistedRobotType();
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify({ robot_type: resolved });
  }
  return requestSupervisorApi(`/services/bt_node/${active ? "start" : "stop"}`, init);
}
