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
// Author: Howon Kim, Seongwoo Kim

"use client";

export function BtCanvasNode({
  className = "",
  children,
  tone = "default"
}) {
  const styles = {
    default: {
      color: "var(--mc-text)",
      backgroundColor: "var(--mc-surface)",
      borderColor: "var(--mc-text)"
    },
    active: {
      color: "var(--mc-accent-fg)",
      backgroundColor: "var(--mc-accent)",
      borderColor: "var(--mc-accent)"
    },
    muted: {
      color: "var(--mc-text-muted)",
      backgroundColor: "var(--mc-surface-2)",
      borderColor: "var(--mc-success)"
    }
  };
  return <div className={`absolute min-w-[112px] h-10 px-3 border rounded-md flex items-center justify-center text-xs font-semibold shadow-sm ${className}`} style={styles[tone] || styles.default}>
      {children}
    </div>;
}
export function WaypointBtFocusLayer({
  layer,
  onClose
}) {
  const spot = (layer === null) || (layer === void 0) ? void 0 : layer.spot;
  if (!spot) return null;
  return <section className="absolute inset-0 z-20 pointer-events-none" aria-label="Waypoint Task focus canvas">
      {onClose ? <button type="button" className="absolute inset-y-0 left-0 w-[25%] pointer-events-auto cursor-pointer border-0 p-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-[-2px]" style={{
      background: "linear-gradient(90deg, rgba(28,26,23,0.08), rgba(28,26,23,0))",
      outlineColor: "var(--mc-accent)"
    }} aria-label="Back to Map from waypoint context" title="Back to Map" onClick={event => {
      event.stopPropagation();
      onClose();
    }} /> : <div className="absolute inset-y-0 left-0 w-[25%] pointer-events-none" style={{
      background: "linear-gradient(90deg, rgba(28,26,23,0.08), rgba(28,26,23,0))"
    }} />}
      <div className="absolute inset-y-0 right-0 w-[75%] pointer-events-auto overflow-hidden" style={{
      color: "var(--mc-text)",
      backgroundColor: "var(--mc-surface)",
      borderLeft: "1px solid var(--mc-border-strong)"
    }}>
        <div className="h-full min-h-0 p-4">
          <div className="relative min-h-0 overflow-hidden rounded-md border" style={{
          borderColor: "var(--mc-border)",
          height: "100%"
        }}>
            {layer.editor ? layer.editor : <>
            <div className="absolute inset-0 opacity-80" style={{
              backgroundImage: "linear-gradient(rgba(28,26,23,0.13) 1px, transparent 1px), linear-gradient(90deg, rgba(28,26,23,0.13) 1px, transparent 1px)",
              backgroundSize: "28px 28px"
            }} />
            <svg className="absolute inset-0 h-full w-full" aria-hidden="true">
              <line x1="20%" y1="49%" x2="42%" y2="49%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round" />
              <line x1="56%" y1="38%" x2="76%" y2="28%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round" />
              <line x1="56%" y1="60%" x2="76%" y2="72%" stroke="var(--mc-accent)" strokeWidth="2.5" strokeLinecap="round" />
            </svg>
            <BtCanvasNode className="left-[8%] top-[43%]" tone={spot.linked_bt_tree ? "active" : "muted"}>
              {spot.linked_bt_tree ? "Task" : "New Task"}
            </BtCanvasNode>
            <BtCanvasNode className="left-[38%] top-[43%]">
              Sequence
            </BtCanvasNode>
            <BtCanvasNode className="left-[68%] top-[22%]" tone="active">
              Navigate
            </BtCanvasNode>
            <BtCanvasNode className="left-[68%] top-[66%]" tone="default">
              Task
            </BtCanvasNode>
            </>}
          </div>
        </div>
      </div>
    </section>;
}
