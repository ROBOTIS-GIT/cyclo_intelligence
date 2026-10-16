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

import { useCallback, useEffect, useRef, useState } from "react";
import { observationSpots } from "../lib/observationSpots";
import { autonomyRequest, OBSERVATION_PLAN_CHANGED } from "./autonomyApi";
import useObservationPlan from "./useObservationPlan";
import useRecordRouteEditor from "./useRecordRouteEditor";
import useRecordEditHistory from "./useRecordEditHistory";
import useRecordMapEditSession from "./useRecordMapEditSession";

export const OBSERVATION_SPOTS_CHANGED = OBSERVATION_PLAN_CHANGED;
export { observationSpots } from "../lib/observationSpots";

export function missionsAtObservation(configuration, spotId) {
  const point = configuration?.plan?.observations?.find(p => p.spot_id === spotId);
  const excluded = new Set(configuration?.plan?.excluded_mission_ids || []);
  const ids = new Set((point?.mission_ids || []).filter(id => !excluded.has(id)));
  return (configuration?.missions || []).filter(m => ids.has(m.mission_id));
}

// Coordinates, route and mission bindings share the same map document as Operate.
export default function useRecordObservationSpots({ enabled, mapName, currentPose, localized, blocked, onMapMode, missionId, isBlocked }) {
  const [mode, setMode] = useState("observation");
  const planState = useObservationPlan({ enabled, mapName, editable: true, locked: blocked });
  const configuration = planState.view, refreshPlan = planState.refresh;
  const history = useRecordEditHistory(`${enabled}:${mapName}`);
  const clearHistory = history.clear;
  const [selectedId, select] = useState("");
  const [patrolChoice, setPatrolChoice] = useState(null);
  const plannedIds = (configuration?.plan?.observations || []).map(p => p.spot_id);
  const patrolSpotIds = patrolChoice || plannedIds;
  const setPatrolSpotIds = setPatrolChoice;
  const [draft, setDraft] = useState(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState(null);
  const epoch = useRef(0), pending = useRef(false), latest = useRef({});
  latest.current = { enabled, mapName, configuration, draft, blocked, onMapMode, isBlocked };
  const blockedNow = () => latest.current.blocked || !!latest.current.isBlocked?.();
  const refresh = useCallback(async () => {
    const generation = epoch.current;
    if (!enabled || !mapName) return { spots: [] };
    const data = await refreshPlan();
    if (generation !== epoch.current) throw new Error("The active map changed. Select the spot again.");
    setError(null);
    const spots = observationSpots(data);
    select(id => spots.some(s => s.id === id) ? id : "");
    setPatrolChoice(ids => ids?.filter(id => spots.some(s => s.id === id)) ?? null);
    return { spots, plan: data.plan, missions: data.missions };
  }, [enabled, mapName, refreshPlan]);
  useEffect(() => {
    epoch.current += 1;
    setDraft(null); select(""); setPatrolChoice(null); setMode("observation"); setError(null);
    return () => { epoch.current += 1; };
  }, [enabled, mapName]);
  const mapEditor = useRecordMapEditSession({ enabled, mapName, configuration, refresh: refreshPlan, isBlocked: blockedNow, history });
  latest.current.mapEditor = mapEditor;
  const spots = mapEditor.snapshot?.spots || observationSpots(configuration);
  const selected = spots.find(s => s.id === selectedId);
  const savedSelected = observationSpots(configuration).find(s => s.id === selectedId);
  const linkedMissions = missionsAtObservation(configuration, selectedId);
  const returnSpotId = savedSelected && linkedMissions.some(m => m.mission_id === missionId) ? selectedId : "";
  async function resolveReturnSpot(id, forMissionId = missionId) {
    const view = await refresh();
    const allowed = missionsAtObservation({ plan: view.plan, missions: view.missions }, id);
    const spot = view.spots.find(value => value.id === id);
    if (!spot || !allowed.some(m => m.mission_id === forMissionId))
      throw new Error("The mission is not linked to this observation. Update the observation route and linked missions.");
    const point = view.plan?.observations?.find(value => value.spot_id === id);
    return { ...spot, pose: { ...spot.pose }, arrival_bt: point?.arrival_bt || "" };
  }
  function cancel() { setDraft(null); onMapMode("view"); }
  function changeMode(value) { cancel(); setMode(value); }
  function begin(editing = false, placement = "") {
    if (blockedNow() || working || mapEditor.working || mapEditor.dirty || latest.current.route?.routeMode) return;
    let number = 1;
    const names = new Set(spots.map(s => s.label));
    while (names.has(`Observation ${number}`)) number += 1;
    const next = editing && selected
      ? { id: selected.id, label: selected.label, pose: { ...selected.pose } }
      : { label: `Observation ${number}`, pose: null };
    if (placement === "robot") next.pose = robotPose();
    setDraft(next);
    if (placement === "map") onMapMode("spot");
    setError(null);
  }
  function place(pose) {
    if (blockedNow()) return;
    setDraft(value => value ? { ...value, pose: { frame_id: "map", ...pose } } : value);
    onMapMode("view");
  }
  function robotPose() {
    if (!localized || !currentPose) return null;
    const q = currentPose.orientation || currentPose.pose?.orientation || {};
    const p = currentPose.position || currentPose.pose?.position;
    if (!p || !Number.isFinite(p.x) || !Number.isFinite(p.y)) return null;
    return { frame_id: "map", x: p.x, y: p.y, yaw: Math.atan2(2 * ((q.w || 0) * (q.z || 0) + (q.x || 0) * (q.y || 0)),
      1 - 2 * ((q.y || 0) ** 2 + (q.z || 0) ** 2)) };
  }
  function atRobot() { const pose = robotPose(); if (pose) place(pose); }
  async function mutate(action, record = true) {
    const v = latest.current, generation = epoch.current;
    if (!v.enabled || !v.mapName || !v.configuration || blockedNow() || v.mapEditor?.dirty || v.mapEditor?.working || v.route?.routeMode || pending.current) return false;
    pending.current = true; setWorking(true); setError(null);
    try {
      const id = await action(v);
      if (generation !== epoch.current) return false;
      const after = await refresh();
      if (generation !== epoch.current) return false;
      if (record) {
        const beforeSpot = observationSpots(v.configuration).find(spot => spot.id === id);
        const afterSpot = after.spots.find(spot => spot.id === id);
        if (beforeSpot && afterSpot) history.push({ type: "waypoint", before: beforeSpot, after: afterSpot });
        else history.clear(); // Creation/deletion starts a new edit history.
      }
      setDraft(null); select(id || ""); onMapMode("view");
      window.dispatchEvent(new CustomEvent(OBSERVATION_SPOTS_CHANGED, { detail: { mapName: v.mapName } }));
      return true;
    } catch (failure) {
      if (generation === epoch.current) setError(failure.message);
      return false;
    } finally { pending.current = false; setWorking(false); }
  }
  function save() {
    return mutate(async v => {
      const value = v.draft;
      if (!value?.label.trim() || !value.pose || ![value.pose.x, value.pose.y, value.pose.yaw].every(Number.isFinite)) {
        throw new Error("Enter a name and place the observation on the map or at the robot.");
      }
      const base = `/autonomy/maps/${encodeURIComponent(v.mapName)}/spots`;
      const result = await autonomyRequest(value.id ? `${base}/${encodeURIComponent(value.id)}` : base,
        value.id ? "PATCH" : "POST", { label: value.label, pose: value.pose, plan_revision: v.configuration.plan.revision });
      return result.id;
    });
  }
  async function move(id, x, y, yaw) {
    const v = latest.current;
    if (!v.enabled || !v.configuration || blockedNow() || pending.current || working
      || v.mapEditor?.working || ![x, y, yaw].every(Number.isFinite)) return false;
    const pose = { frame_id: "map", x, y, yaw };
    if (v.route?.orderMode) return false;
    if (v.route?.routeMode) return v.mapEditor.updateSpot(id, { pose });
    if (v.mapEditor?.dirty) return false;
    if (v.draft) {
      if (id !== (v.draft.id || "observation-preview")) return false;
      setDraft({ ...v.draft, pose });
      onMapMode("view");
      return true;
    }
    const spot = observationSpots(v.configuration).find(value => value.id === id);
    if (!spot) return false;
    return mutate(async current => {
      const result = await autonomyRequest(
        `/autonomy/maps/${encodeURIComponent(current.mapName)}/spots/${encodeURIComponent(id)}`,
        "PATCH", { label: spot.label, pose, plan_revision: current.configuration.plan.revision });
      return result.id;
    });
  }
  const remove = id => mutate(async v => {
    await autonomyRequest(`/autonomy/maps/${encodeURIComponent(v.mapName)}/spots/${encodeURIComponent(id)}?plan_revision=${v.configuration.plan.revision}`, "DELETE");
    return "";
  });
  const editablePlan = {
    draft: mapEditor.snapshot?.plan || configuration?.plan,
    missions: configuration?.missions || [], dirty: mapEditor.dirty, working: mapEditor.working,
    error: mapEditor.error || planState.error,
    edit: mapEditor.editPlan, save: mapEditor.save, discard: mapEditor.discard, reset: mapEditor.reset,
    updateSpot: mapEditor.updateSpot, removeWaypoint: mapEditor.removeWaypoint,
  };
  const route = useRecordRouteEditor({ planState: editablePlan, spots, blocked: blocked || working || !!draft, isBlocked: blockedNow,
    onPrepare: () => { cancel(); return mapEditor.begin(); }, onSelect: select, onMessage: () => {} });
  latest.current.route = route;
  const resetMapEditor = mapEditor.reset;
  const resetRouteEditing = route.resetEditing;
  const resetSession = useCallback(() => {
    const generation = ++epoch.current;
    clearHistory();
    setDraft(null); select(""); setPatrolChoice(null); setMode("observation"); setError(null);
    resetRouteEditing(); resetMapEditor();
    // Explicit map loading starts a fresh editor, including same-map reloads.
    void refreshPlan().catch(failure => { if (epoch.current === generation) setError(failure.message); });
  }, [refreshPlan, resetRouteEditing, resetMapEditor, clearHistory]);
  useEffect(() => { resetRouteEditing(); }, [enabled, mapName, resetRouteEditing]);
  const applyHistory = async (entry, direction) => {
    const value = direction === "undo" ? entry.before : entry.after;
    const expected = direction === "undo" ? entry.after : entry.before;
    const existing = spots.find(spot => spot.id === value.id);
    const matches = entry.type === "map-edit"
      ? JSON.stringify(mapEditor.snapshot) === JSON.stringify(expected)
      : existing?.label === expected.label && JSON.stringify(existing?.pose) === JSON.stringify(expected.pose);
    if (!matches) {
      history.clear(); setError("Observation settings changed. Edit history has been cleared."); return false;
    }
    if (entry.type === "map-edit") return mapEditor.editSnapshot(value, false);
    if (mapEditor.dirty || route.routeMode) return false;
    return mutate(async current => {
      const result = await autonomyRequest(`/autonomy/maps/${encodeURIComponent(current.mapName)}/spots/${encodeURIComponent(value.id)}`,
        "PATCH", { label: value.label, pose: value.pose, plan_revision: current.configuration.plan.revision });
      return result.id;
    }, false);
  };
  const historyLocked = blockedNow() || working || mapEditor.working || !!draft || history.busy;
  const allowedEntry = entry => !!entry && !historyLocked &&
    (entry.type === "map-edit" ? route.routeMode : (!mapEditor.dirty && !route.routeMode));
  const travel = direction => !allowedEntry(direction === "undo" ? history.undoEntry : history.redoEntry)
    ? false : history.travel(direction, applyHistory);
  return { mode, setMode: changeMode, spots, selectedId, select, patrolSpotIds, setPatrolSpotIds, draft, setDraft,
    canUndo: allowedEntry(history.undoEntry), canRedo: allowedEntry(history.redoEntry),
    undo: () => travel("undo"), redo: () => travel("redo"),
    route, routeEditor: route.editor, blocked: !!blocked, createOnMap: () => begin(false, "map"), createAtRobot: () => begin(false, "robot"),
    resetSession, reload: () => { history.clear(); return refresh().catch(failure => setError(failure.message)); },
    returnSpotId, returnSpots: savedSelected ? [savedSelected] : [], resolveReturnSpot,
    linkedMissions, configuration, selectedInPlan: plannedIds.includes(selectedId), planDirty: mapEditor.dirty,
    planRevision: configuration?.plan?.revision,
    useSavedPatrol: () => setPatrolChoice(null),
    working: working || mapEditor.working || planState.working, error: error || mapEditor.error || planState.error, ready: !!configuration, begin, cancel, place, move, save, remove, refresh, atRobot,
    canUseRobot: localized && !!currentPose, onMap: () => { if (!blockedNow()) onMapMode("spot"); } };
}
