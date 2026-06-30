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
// Author: Howon Kim

"use client";
function Toggle({ label, checked, onChange, }) {
    return (<label className="h-8 px-2 border flex items-center gap-2 text-xs font-medium" style={{
            color: "var(--vscode-foreground)",
            borderColor: "var(--vscode-panel-border)",
            backgroundColor: "var(--vscode-sidebar-background)",
        }}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.currentTarget.checked)}/>
      {label}
    </label>);
}
function TopicName({ topic }) {
    if (topic === "/local_costmap/published_footprint") {
        return (<span className="font-mono min-w-0 leading-4">
        <span className="block">/local_costmap</span>
        <span className="block">/published_footprint</span>
      </span>);
    }
    return <span className="font-mono truncate min-w-0">{topic}</span>;
}
export function NavigationSidePanel({ layerToggles, mapName, status, topicRows, }) {
    var _a;
    return (<aside className="min-h-0 flex flex-col gap-3">
      <div className="border p-3 grid gap-2" style={{
            color: "var(--vscode-foreground)",
            borderColor: "var(--vscode-panel-border)",
            backgroundColor: "var(--vscode-sidebar-background)",
        }}>
        <div className="text-xs font-semibold">Layers</div>
        <div className="flex flex-wrap gap-2">
          {layerToggles.map((layer) => (<Toggle key={layer.id} label={layer.label} checked={layer.checked} onChange={layer.onChange}/>))}
        </div>
      </div>
      <div className="border p-3 grid gap-2 text-xs" style={{
            color: "var(--vscode-foreground)",
            borderColor: "var(--vscode-panel-border)",
            backgroundColor: "var(--vscode-sidebar-background)",
        }}>
        <div className="font-semibold">Topics</div>
        {topicRows.map(({ topic, isLive }) => (<div key={topic} className="flex items-center justify-between gap-3 min-w-0">
            <TopicName topic={topic}/>
            <span className="shrink-0" style={{ color: "var(--vscode-descriptionForeground)" }}>
              {isLive ? "live" : "wait"}
            </span>
          </div>))}
      </div>
      <div className="border p-3 grid gap-2 text-xs shrink-0" style={{
            color: "var(--vscode-descriptionForeground)",
            borderColor: "var(--vscode-panel-border)",
            backgroundColor: "var(--vscode-sidebar-background)",
        }}>
        <div>
          Map name: <span className="font-mono">{mapName || "-"}</span>
        </div>
        <div>
          PID: <span className="font-mono">{(_a = status === null || status === void 0 ? void 0 : status.pid) !== null && _a !== void 0 ? _a : "-"}</span>
        </div>
      </div>
    </aside>);
}

