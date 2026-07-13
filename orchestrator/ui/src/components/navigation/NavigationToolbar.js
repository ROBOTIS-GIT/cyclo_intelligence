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
function ToolbarSeparator() {
    return (<div className="h-6 w-px" aria-hidden="true" style={{ backgroundColor: "var(--vscode-panel-border)" }}/>);
}
function ViewModeIcon() {
    return (<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M12 2v20"/>
      <path d="M2 12h20"/>
      <path d="m5 9-3 3 3 3"/>
      <path d="m19 9 3 3-3 3"/>
      <path d="m9 5 3-3 3 3"/>
      <path d="m9 19 3 3 3-3"/>
    </svg>);
}
function LogIcon({ className }) {
    return (<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
      <line x1="16" y1="13" x2="8" y2="13"/>
      <line x1="16" y1="17" x2="8" y2="17"/>
      <polyline points="10 9 9 9 8 9"/>
    </svg>);
}
export function NavigationToolbar({ busy, hasMapName, clickMode, mapName, mode, running, showLogs, showPgmFix, onCancel, onFixToggle, onMapping, onNavigation, onSaveMap, onStop, setClickMode, setMapName, setShowLogs, }) {
    return (<div className="flex flex-wrap items-center gap-2">
      <div className="h-8 px-2.5 border flex items-center gap-2 text-sm" style={{
            color: "var(--vscode-foreground)",
            backgroundColor: "var(--vscode-sidebar-background)",
            borderColor: "var(--vscode-panel-border)",
        }}>
        <span
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: running ? "#22c55e" : "#ef4444" }}
          title={running ? "Running" : "Idle"}
          aria-label={running ? "Running" : "Idle"}
        />
        <span className="font-medium">{mode}</span>
      </div>
      <button type="button" disabled={busy !== null || running} onClick={onMapping} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: "var(--vscode-button-foreground)",
            backgroundColor: "var(--vscode-button-background)",
            borderColor: "var(--vscode-focusBorder)",
        }}>
        Mapping
      </button>
      <button type="button" disabled={busy !== null || running} onClick={onNavigation} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: "var(--vscode-button-foreground)",
            backgroundColor: "var(--vscode-button-background)",
            borderColor: "var(--vscode-focusBorder)",
        }}>
        Navigation
      </button>
      <button type="button" onClick={() => setShowLogs((value) => !value)} title="Log" aria-label="Log" className="h-8 w-8 border cursor-pointer inline-flex items-center justify-center" style={{
            color: showLogs
                ? "var(--vscode-button-secondaryForeground)"
                : "var(--vscode-button-foreground)",
            backgroundColor: showLogs
                ? "var(--vscode-button-secondaryBackground)"
                : "var(--vscode-button-background)",
            borderColor: showLogs
                ? "var(--vscode-panel-border)"
                : "var(--vscode-focusBorder)",
        }}>
        <LogIcon />
      </button>
      <ToolbarSeparator />
      <button type="button" disabled={busy !== null || !running} onClick={onCancel} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: "var(--vscode-button-secondaryForeground)",
            backgroundColor: "var(--vscode-button-secondaryBackground)",
            borderColor: "var(--vscode-panel-border)",
        }}>
        Cancel
      </button>
      <button type="button" disabled={busy !== null || !running} onClick={onStop} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: "#000000",
            backgroundColor: "var(--vscode-inputValidation-errorBackground, #b91c1c)",
            borderColor: "var(--vscode-inputValidation-errorBorder, #ef4444)",
        }}>
        Stop
      </button>
      <ToolbarSeparator />
      <input value={mapName} onChange={(event) => setMapName(event.currentTarget.value)} className="h-8 w-28 px-2 border text-sm" style={{
            color: "var(--vscode-input-foreground)",
            backgroundColor: "var(--vscode-input-background)",
            borderColor: "var(--vscode-input-border, var(--vscode-panel-border))",
        }}/>
      <button type="button" disabled={busy !== null || !hasMapName} onClick={onSaveMap} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: "var(--vscode-button-foreground)",
            backgroundColor: "var(--vscode-button-background)",
            borderColor: "var(--vscode-focusBorder)",
        }}>
        Save Map
      </button>
      <button type="button" disabled={busy !== null || !hasMapName} onClick={onFixToggle} className="h-8 px-3 border text-sm font-semibold disabled:opacity-50" style={{
            color: showPgmFix
                ? "var(--vscode-button-secondaryForeground)"
                : "var(--vscode-button-foreground)",
            backgroundColor: showPgmFix
                ? "var(--vscode-button-secondaryBackground)"
                : "var(--vscode-button-background)",
            borderColor: showPgmFix
                ? "var(--vscode-panel-border)"
                : "var(--vscode-focusBorder)",
        }}>
        Fix
      </button>
      <ToolbarSeparator />
      <div className="h-8 border grid grid-cols-3 overflow-hidden" style={{ borderColor: "var(--vscode-panel-border)" }}>
        {["view", "goal", "initial"].map((modeValue) => (<button key={modeValue} type="button" onClick={() => setClickMode(modeValue)} className="px-2 text-xs font-semibold inline-flex items-center justify-center" title={modeValue === "view" ? "View" : modeValue === "goal" ? "Goal" : "Initial"} aria-label={modeValue === "view" ? "View" : modeValue === "goal" ? "Goal" : "Initial"} style={{
                color: clickMode === modeValue
                    ? "var(--vscode-button-foreground)"
                    : "var(--vscode-foreground)",
                backgroundColor: clickMode === modeValue
                    ? "var(--vscode-button-background)"
                    : "var(--vscode-button-secondaryBackground)",
            }}>
            {modeValue === "view" ? <ViewModeIcon /> : modeValue === "goal" ? "Goal" : "Initial"}
          </button>))}
      </div>
    </div>);
}
