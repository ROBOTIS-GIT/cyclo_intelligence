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

export function formatTaskDisplayMessage(value, taskLabel = "Task") {
  const label = String(taskLabel || "Task");
  const pluralLabel = `${label}s`;
  const inlineLabel = label.toLowerCase();

  return String(value || "")
    .replace(/\bunknown node type\b/gi, (match) => (
      match.charAt(0) === "U" ? "Unknown step type" : "unknown step type"
    ))
    .replace(/\bBT node (?=(?:class|type)\b)/gi, "step ")
    .replace(/\bBT runtime\b/gi, "Task Engine")
    .replace(/\bBT node\b/gi, "Task Engine")
    .replace(/(^|[\s:(])bt_node\b/gi, (_match, prefix) => `${prefix}Task Engine`)
    .replace(/(^|[\s:(])BehaviorTree\b(?![./_-])/g, (_match, prefix) => `${prefix}${label}`)
    .replace(/\bbehavior trees\b/gi, pluralLabel)
    .replace(/\bbehavior tree\b/gi, label)
    .replace(/\blocal BT\b/gi, "Waypoint Task")
    .replace(/\bwaypoint BT\b/gi, "Waypoint Task")
    .replace(/\bBT XML\b/gi, `${label} file`)
    .replace(/(^|[\s:(])BT\b(?![/._-])/gi, (_match, prefix) => `${prefix}${label}`)
    .replace(/\bno tree loaded\b/gi, `No ${inlineLabel} loaded`)
    .replace(/\bfailed to (load|save) tree\b(?![./_-])/gi, (_match, action) => (
      `Failed to ${action.toLowerCase()} ${inlineLabel}`
    ))
    .replace(/\btrees directory\b/gi, `${pluralLabel} directory`)
    .replace(/\btree directory\b/gi, `${label} directory`)
    .replace(/\btree already exists\b/gi, `${label} already exists`)
    .replace(/\btree (loaded|saved|rejected|failed)\b(?![./_-])/gi, (_match, state) => (
      `${label} ${state.toLowerCase()}`
    ));
}
