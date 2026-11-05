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

import { FALLBACK_CATALOG } from "../../../constants/btNodeCatalogFallback";

export const MISSION_DESIGN_STORAGE_KEY = "mission_canvas_designs";

export function behaviorNodeDefinition(tag) {
  return FALLBACK_CATALOG.find((node) => node.tag === tag) || {
    tag,
    category: "action",
  };
}

export function behaviorNodeId(tag, index) {
  return `behavior_${index}_${tag.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`;
}

export function readDesignStore() {
  if (typeof window === "undefined" || !window.localStorage) return {};
  try {
    const raw = window.localStorage.getItem(MISSION_DESIGN_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function savedBehaviorNodesForMap(mapName) {
  const saved = readDesignStore()[mapName]?.behaviorNodes;
  return Array.isArray(saved) ? saved : null;
}

export function saveBehaviorNodesForMap(mapName, nodes) {
  if (typeof window === "undefined" || !window.localStorage) return;
  const store = readDesignStore();
  store[mapName] = {
    ...store[mapName],
    behaviorNodes: nodes,
    savedAt: new Date().toISOString(),
  };
  window.localStorage.setItem(MISSION_DESIGN_STORAGE_KEY, JSON.stringify(store));
}

export function behaviorNodeSerialFromNodes(nodes) {
  return nodes.reduce((max, node) => {
    const match = String(node.id || "").match(/^behavior_(\d+)_/);
    return match ? Math.max(max, Number(match[1])) : max;
  }, 0);
}
