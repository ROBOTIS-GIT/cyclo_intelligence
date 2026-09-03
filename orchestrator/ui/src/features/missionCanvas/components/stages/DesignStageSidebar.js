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

import { MdDelete } from "react-icons/md";
import { localBtPathForSpot } from "../../lib/missionBtFiles";

export default function DesignStageSidebar({
  busy,
  designDocumentReady,
  designPanelBehaviorNodes,
  designPanelRouteClosed,
  designPanelRouteSpots,
  designPanelSpots,
  editingSpotId,
  editingSpotLabel,
  missionFlowEdges,
  missionRouteMode,
  onCancelSpotRename,
  onClearMissionRoute,
  onCommitSpotRename,
  onDeleteBehaviorNode,
  onDeleteSpot,
  onEditingSpotLabelChange,
  onMissionRouteSpotClick,
  onMoveRouteSpot,
  onOpenMissionRouteLoop,
  onOpenWaypointBt,
  onRemoveRouteSpot,
  onSelectBehaviorNode,
  onSelectSpot,
  onStartRenameSpot,
  selectedBehaviorNodeId,
  selectedSpotId,
}) {
  return (
    <aside className="min-h-0 grid grid-rows-[minmax(0,1fr)_minmax(220px,0.7fr)] gap-4 overflow-hidden p-4">
      {/* Waypoints — LIST ONLY (Create moved to the map HUD) */}
      <div className="min-h-0 overflow-auto" style={{ backgroundColor: "var(--mc-surface)", border: "1px solid var(--mc-border)", borderRadius: 16, boxShadow: "var(--mc-shadow)", padding: 18 }}>
        <div className="flex items-center justify-between mb-3.5">
          <span className="text-[13.5px] font-bold">Waypoints</span>
          <span className="text-[11px] font-mono" style={{ color: "var(--mc-text-subtle)" }}>{designPanelSpots.length}</span>
        </div>
        <div className="grid gap-2">
          {designPanelSpots.map((spot) => {
            const selected = spot.id === selectedSpotId;
            const editing = editingSpotId === spot.id;
            return (
              <div key={spot.id} className="grid gap-1.5 min-w-0" style={{ padding: 8, borderRadius: 12, border: `1px solid ${selected ? "var(--mc-accent)" : "var(--mc-border)"}`, backgroundColor: selected ? "var(--mc-accent-soft)" : "var(--mc-surface-2)" }}>
                <div className="flex items-center gap-1.5 min-w-0">
                  {editing ? (
                    <input aria-label="Waypoint name" value={editingSpotLabel} autoFocus
                      onChange={(event) => onEditingSpotLabelChange(event.currentTarget.value)}
                      onBlur={() => { void onCommitSpotRename(spot); }}
                      onKeyDown={(event) => { if (event.key === "Enter") { event.preventDefault(); void onCommitSpotRename(spot); } if (event.key === "Escape") { event.preventDefault(); onCancelSpotRename(); } }}
                      className="h-8 flex-1 px-2 text-[13px] min-w-0" style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }} />
                  ) : (
                    <button
                      type="button"
                      onClick={() => (
                        missionRouteMode
                          ? onMissionRouteSpotClick(spot.id)
                          : onSelectSpot(spot.id)
                      )}
                      onDoubleClick={missionRouteMode ? undefined : () => onStartRenameSpot(spot)}
                      className="h-8 flex-1 px-2.5 text-left text-[12.5px] font-semibold min-w-0"
                      style={{ borderRadius: 8, border: "1px solid var(--mc-border)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text)" }}>
                      <span className="block truncate">{spot.label || spot.id}</span>
                    </button>
                  )}
                  <button
                    type="button"
                    aria-label={`Edit Task for ${spot.label || spot.id}`}
                    title={`Edit ${spot.label || spot.id} Waypoint Task`}
                    onClick={() => onOpenWaypointBt(spot.id)}
                    className="h-8 shrink-0 px-2.5 text-[11.5px] font-semibold active:translate-y-px"
                    style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text-muted)" }}
                  >
                    Edit Task
                  </button>
                  <button type="button" aria-label={`Delete Waypoint ${spot.label || spot.id}`} title={`Delete ${spot.label || spot.id}`} disabled={!!busy} onClick={() => { void onDeleteSpot(spot); }}
                    className="h-8 w-8 shrink-0 inline-flex items-center justify-center active:translate-y-px disabled:opacity-45"
                    style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-danger)" }}>
                    <MdDelete size={15} />
                  </button>
                </div>
              </div>
            );
          })}
          {designPanelSpots.length === 0 && <div className="text-[12px]" style={{ color: "var(--mc-text-muted)" }}>No waypoints for this map yet.</div>}
          {designPanelBehaviorNodes.map((node) => {
            const selected = node.id === selectedBehaviorNodeId;
            return (
              <div key={node.id} className="flex items-center gap-1.5 min-w-0">
                <button type="button" onClick={() => onSelectBehaviorNode(node.id)} className="h-8 flex-1 px-2.5 text-left text-[12px] font-semibold min-w-0"
                  style={{ borderRadius: 8, border: `1px solid ${selected ? "var(--mc-accent)" : "var(--mc-border)"}`, backgroundColor: selected ? "var(--mc-accent-soft)" : "var(--mc-surface-2)", color: "var(--mc-text)" }}>
                  <span className="block truncate">{node.tag}</span>
                </button>
                <button type="button" aria-label={`Delete Node ${node.tag}`} title={`Delete ${node.tag}`} onClick={() => onDeleteBehaviorNode(node)}
                  className="h-8 w-8 shrink-0 inline-flex items-center justify-center active:translate-y-px"
                  style={{ borderRadius: 8, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-danger)" }}>
                  <MdDelete size={15} />
                </button>
              </div>
            );
          })}
        </div>
      </div>

      {/* Mission Route — LIST ONLY (Edit/Clear moved to the map HUD) */}
      {/* Route-edit mode is signalled by the card itself (accent border +
          title dot) instead of a third header chip, so the header stays
          one clean row: title · closed-loop chip · Clear Route. */}
      <div
        className="min-h-0 overflow-hidden"
        aria-label={missionRouteMode ? "Mission Route (editing on map)" : "Mission Route"}
        style={{ backgroundColor: "var(--mc-surface)", border: `1px solid ${missionRouteMode ? "var(--mc-accent)" : "var(--mc-border)"}`, borderRadius: 16, boxShadow: "var(--mc-shadow)", padding: 18 }}
      >
        <div className="h-full min-h-0 grid grid-rows-[auto_minmax(0,1fr)] gap-2.5">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-[13.5px] font-bold" style={{ color: missionRouteMode ? "var(--mc-accent-hover)" : "var(--mc-text)" }}>
              {missionRouteMode && (
                <span aria-hidden="true" className="inline-block shrink-0" style={{ width: 8, height: 8, borderRadius: 999, backgroundColor: "var(--mc-accent)" }} />
              )}
              Mission Route
            </span>
            <div className="flex items-center gap-1.5">
              {designPanelRouteClosed && <span className="text-[10.5px] font-mono px-2 py-1" style={{ borderRadius: 6, backgroundColor: "color-mix(in srgb, var(--mc-success) 14%, transparent)", color: "var(--mc-success)" }}>closed loop</span>}
              {/* Only while a route exists — map clicks can only ADD
                  edges, so this is the sole way to discard a route (or a
                  closed loop) without deleting waypoints. */}
              {missionRouteMode && designDocumentReady && missionFlowEdges.length > 0 && (
                <button
                  type="button"
                  onClick={onClearMissionRoute}
                  disabled={!!busy}
                  aria-label="Clear Route"
                  title="Remove all route connections (waypoints stay)"
                  className="h-7 px-2.5 text-[11px] font-semibold disabled:opacity-45 active:translate-y-px"
                  style={{ borderRadius: 7, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-danger)" }}
                >
                  Clear Route
                </button>
              )}
            </div>
          </div>
          <div className="min-h-0 overflow-auto pr-1">
            <div className="grid gap-0">
              {designPanelRouteSpots.map((spot, index) => {
                const selected = spot.id === selectedSpotId;
                const routeEnd = index === designPanelRouteSpots.length - 1;
                const last = routeEnd && !designPanelRouteClosed;
                return (
                  <div key={spot.id}>
                    <div className="flex gap-3 items-stretch">
                      <div className="flex flex-col items-center" style={{ width: 26 }}>
                        <span className="h-[26px] w-[26px] shrink-0 rounded-full inline-flex items-center justify-center text-[11px] font-semibold font-mono" style={{ color: "var(--mc-accent-fg)", backgroundColor: "var(--mc-accent)" }}>{index + 1}</span>
                        {!last && <span className="flex-1 my-0.5" style={{ width: 2, backgroundColor: "var(--mc-border)" }} />}
                      </div>
                      <div className="flex-1 mb-2 grid grid-cols-[1fr_auto] items-center gap-2 min-w-0" style={{ padding: 10, borderRadius: 11, border: `1px solid ${selected ? "var(--mc-accent)" : "var(--mc-border)"}`, backgroundColor: selected ? "var(--mc-accent-soft)" : "var(--mc-surface-2)" }}>
                        <button
                          type="button"
                          onClick={() => (
                            missionRouteMode
                              ? onMissionRouteSpotClick(spot.id)
                              : onSelectSpot(spot.id)
                          )}
                          className="min-w-0 text-left"
                        >
                          <span className="block truncate text-[12.5px] font-semibold" style={{ color: "var(--mc-text)" }}>{spot.label || spot.id}</span>
                          <span className="block truncate text-[10px] font-mono" style={{ color: "var(--mc-text-subtle)" }}>{localBtPathForSpot(spot)}</span>
                        </button>
                        {missionRouteMode && <div className="flex items-center gap-1">
                          <button type="button" aria-label={`Move ${spot.label || spot.id} up`} disabled={!!busy || index === 0} onClick={() => onMoveRouteSpot(spot.id, -1)} className="h-7 w-7 text-[12px] font-semibold disabled:opacity-40" style={{ borderRadius: 7, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text-muted)" }}>↑</button>
                          <button type="button" aria-label={`Move ${spot.label || spot.id} down`} disabled={!!busy || routeEnd} onClick={() => onMoveRouteSpot(spot.id, 1)} className="h-7 w-7 text-[12px] font-semibold disabled:opacity-40" style={{ borderRadius: 7, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-text-muted)" }}>↓</button>
                          {/* Route membership only — deleting the spot
                              itself lives in the Waypoints panel. */}
                          <button type="button" aria-label={`Remove ${spot.label || spot.id} from route`} title="Remove from route (waypoint stays)" disabled={!!busy} onClick={() => onRemoveRouteSpot(spot.id)}
                            className="h-7 w-7 shrink-0 inline-flex items-center justify-center text-[13px] leading-none active:translate-y-px disabled:opacity-40"
                            style={{ borderRadius: 7, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-danger)" }}>
                            ×
                          </button>
                        </div>}
                      </div>
                    </div>
                  </div>
                );
              })}
              {designPanelRouteClosed && designPanelRouteSpots.length > 1 && (
                <div className="flex gap-3 items-stretch" aria-label={`Return to ${designPanelRouteSpots[0].label || designPanelRouteSpots[0].id}`}>
                  <div className="flex flex-col items-center" style={{ width: 26 }}>
                    <span className="h-[26px] w-[26px] shrink-0 rounded-full inline-flex items-center justify-center text-[13px] font-semibold" style={{ color: "var(--mc-success)", backgroundColor: "color-mix(in srgb, var(--mc-success) 14%, transparent)", border: "1px solid var(--mc-success)" }}>↻</span>
                  </div>
                  <div className="flex-1 mb-2 grid grid-cols-[1fr_auto] items-center gap-2 min-w-0" style={{ padding: 10, borderRadius: 11, border: "1px solid var(--mc-success)", backgroundColor: "color-mix(in srgb, var(--mc-success) 10%, transparent)" }}>
                    <div className="min-w-0">
                      <span className="block truncate text-[12.5px] font-semibold" style={{ color: "var(--mc-success)" }}>
                        Return to {designPanelRouteSpots[0].label || designPanelRouteSpots[0].id}
                      </span>
                      <span className="block truncate text-[10px] font-mono" style={{ color: "var(--mc-text-subtle)" }}>Loop closure</span>
                    </div>
                    {missionRouteMode && (
                      <button type="button" aria-label="Open loop" title="Remove the loop closure so the route can be edited again"
                        disabled={!!busy}
                        onClick={onOpenMissionRouteLoop}
                        className="h-7 w-7 shrink-0 inline-flex items-center justify-center text-[13px] leading-none active:translate-y-px disabled:opacity-40"
                        style={{ borderRadius: 7, border: "1px solid var(--mc-border-strong)", backgroundColor: "var(--mc-surface)", color: "var(--mc-danger)" }}>
                        ×
                      </button>
                    )}
                  </div>
                </div>
              )}
              {designPanelRouteSpots.length === 0 && (
                <div className="text-[12px]" style={{ color: "var(--mc-text-muted)" }}>
                  {missionRouteMode
                    ? "Click waypoints on the map or in the list to build the route."
                    : "Turn on Edit Route to build the mission route."}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
