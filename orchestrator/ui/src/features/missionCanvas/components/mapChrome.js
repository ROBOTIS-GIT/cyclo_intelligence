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

import { MISSION_GLASS } from "../lib/theme";
import { LayerToggle, Panel } from "./primitives";

// Layers as a glass popover over the map (replaces the docked LayersPanel).
export function LayersPopover({ layerToggles }) {
  return (
    <div
      className="absolute top-5 right-5 z-10 p-3.5"
      style={{ width: 190, borderRadius: 14, backgroundColor: MISSION_GLASS, border: "1px solid var(--mc-border)", boxShadow: "var(--mc-shadow)", backdropFilter: "blur(8px)" }}
    >
      <div className="text-[12.5px] font-bold mb-2.5">Layers</div>
      <div className="grid gap-2.5">
        {layerToggles.map((layer) => (
          <LayerToggle key={layer.id} label={layer.label} checked={layer.checked} compact onChange={layer.onChange} />
        ))}
      </div>
    </div>
  );
}

export function TopicStatusPanel({ topicRows }) {
  return (
    <Panel title="Topics" className="grid gap-2 text-xs min-h-0 overflow-auto content-start">
      {topicRows.map(({ topic, isLive }) => (
        <div key={topic} className="min-h-6 flex items-center justify-between gap-2 min-w-0">
          <div className="font-mono truncate min-w-0 text-[11.5px]" style={{ color: "var(--mc-text-muted)" }}>{topic}</div>
          <span
            className="shrink-0 text-[11px] font-semibold px-2 py-0.5"
            style={{ borderRadius: 6, color: isLive ? "var(--mc-success)" : "var(--mc-text-subtle)", backgroundColor: isLive ? "color-mix(in srgb, var(--mc-success) 16%, transparent)" : "var(--mc-surface-2)" }}
          >
            {isLive ? "live" : "wait"}
          </span>
        </div>
      ))}
    </Panel>
  );
}
