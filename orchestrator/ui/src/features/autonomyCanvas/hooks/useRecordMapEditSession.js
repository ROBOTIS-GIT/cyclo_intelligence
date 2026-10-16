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
import { autonomyRequest, OBSERVATION_PLAN_CHANGED } from "./autonomyApi";
import { observationSpots } from "../lib/observationSpots";

const canonical = value => Array.isArray(value) ? value.map(canonical)
  : value && typeof value === "object" ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
const same = (a, b) => JSON.stringify(canonical(a)) === JSON.stringify(canonical(b));
const planValue = plan => ({ ...plan, revision: 0, route_closed: plan?.route_closed ?? true });
const spotValue = spot => ({ id: spot.id, label: spot.label, pose: {
  frame_id: spot.pose?.frame_id ?? "map", x: spot.pose?.x, y: spot.pose?.y, yaw: spot.pose?.yaw,
} });
const snapshotOf = view => ({ plan: view.plan, spots: observationSpots(view) });
const sameSnapshot = (a, b) => !!a && !!b && same(planValue(a.plan), planValue(b.plan))
  && same(a.spots.map(spotValue), b.spots.map(spotValue));
const sameSaved = (a, b) => a?.plan?.revision === b?.plan?.revision
  && sameSnapshot(snapshotOf(a), snapshotOf(b));

function validateDraft(snapshot, saved) {
  for (const spot of snapshot.spots) {
    if (typeof spot.label !== "string" || !spot.label.trim() || spot.label.length > 128
      || !spot.pose || (spot.pose.frame_id ?? "map") !== "map"
      || ![spot.pose.x, spot.pose.y, spot.pose.yaw].every(Number.isFinite))
      throw new Error("Each waypoint needs a name of 1–128 characters and a valid map pose.");
  }
  const points = snapshot.plan.observations || [];
  const spotIds = new Set(snapshot.spots.map(spot => spot.id));
  const pointIds = new Set(points.map(point => point.spot_id));
  const missionIds = new Set((saved.missions || []).map(mission => mission.mission_id));
  if (points.length > 100 || pointIds.size !== points.length || points.some(point => !spotIds.has(point.spot_id)))
    throw new Error("The route must contain at most 100 distinct saved waypoints.");
  for (const point of points) {
    const dwell = point.dwell_seconds ?? 15;
    if (!["reasoning", "timed"].includes(point.departure_policy || "reasoning"))
      throw new Error("Select a valid observation departure policy.");
    if (!Number.isFinite(dwell) || dwell < 2 || dwell > 3600)
      throw new Error("Waypoint dwell time must be between 2 and 3600 seconds.");
    if (!Array.isArray(point.mission_ids ?? []) || (point.mission_ids || []).length > 100
      || (point.mission_ids || []).some(id => !missionIds.has(id)))
      throw new Error("A waypoint references an unavailable mission. Update its mission selection before saving.");
    const arrival = point.arrival_bt || "";
    const original = saved.plan.observations.find(value => value.spot_id === point.spot_id)?.arrival_bt || "";
    if (typeof arrival !== "string" || arrival.length > 50000 || (arrival && arrival !== original
      && new DOMParser().parseFromString(arrival, "application/xml").getElementsByTagName("parsererror").length))
      throw new Error("A waypoint has invalid arrival posture XML.");
  }
}

// The API saves waypoints and the route separately. Keep an independent editor
// snapshot and a confirmed server checkpoint so a failed later request never
// destroys the remaining draft or retries with an obsolete revision.
export default function useRecordMapEditSession({ enabled, mapName, configuration, refresh, isBlocked, history }) {
  const [snapshot, setSnapshot] = useState(null);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState(null);
  const [saveFailed, setSaveFailed] = useState(false);
  const latest = useRef({}), checkpoint = useRef(null), pending = useRef(false), epoch = useRef(0);
  const persisted = useRef(false), uncertain = useRef(null);
  latest.current = { enabled, mapName, configuration, snapshot, refresh, isBlocked, history };
  const reset = useCallback(() => {
    ++epoch.current;
    checkpoint.current = null; uncertain.current = null; persisted.current = false;
    latest.current.snapshot = null;
    setSnapshot(null); setWorking(false); setError(null); setSaveFailed(false);
  }, []);
  useEffect(reset, [enabled, mapName, reset]);
  const setValue = value => { latest.current.snapshot = value; setSnapshot(value); };
  const begin = () => {
    const v = latest.current;
    if (!v.configuration || pending.current || v.isBlocked()) return false;
    checkpoint.current = v.configuration; persisted.current = false; uncertain.current = null;
    setValue(snapshotOf(v.configuration)); setError(null); setSaveFailed(false);
    v.history.clear();
    return true;
  };
  const editSnapshot = (next, record = true) => {
    const v = latest.current;
    if (!v.snapshot || pending.current || v.isBlocked()) return false;
    if (sameSnapshot(next, v.snapshot)) return false;
    if (record) v.history.push({ type: "map-edit", before: v.snapshot, after: next });
    setValue(next); setError(null);
    return true;
  };
  const editPlan = changes => {
    const value = latest.current.snapshot;
    return value && editSnapshot({ ...value, plan: { ...value.plan, ...changes } });
  };
  const updateSpot = (id, changes) => {
    const value = latest.current.snapshot;
    if (!value?.spots.some(spot => spot.id === id)) return false;
    return editSnapshot({ ...value, spots: value.spots.map(spot => spot.id === id ? {
      ...spot, ...(changes.label !== undefined ? { label: changes.label } : {}),
      ...(changes.pose !== undefined ? { pose: { ...spot.pose, ...changes.pose } } : {}),
    } : spot) });
  };
  const removeWaypoint = id => {
    const value = latest.current.snapshot;
    if (!value?.spots.some(spot => spot.id === id)) return false;
    const observations = value.plan.observations.filter(point => point.spot_id !== id);
    return editSnapshot({ spots: value.spots.filter(spot => spot.id !== id), plan: {
      ...value.plan, observations, route_closed: observations.length > 1 && (value.plan.route_closed ?? true),
    } });
  };
  const save = async () => {
    const v = latest.current, token = epoch.current;
    if (!v.snapshot || !v.enabled || !v.mapName || pending.current || v.isBlocked()) return false;
    const current = () => token === epoch.current && latest.current.enabled && latest.current.mapName === v.mapName;
    const check = () => {
      if (!current()) throw new Error("The active map changed. Select the waypoint again.");
      if (latest.current.isBlocked()) throw new Error("Stop the active episode or navigation before saving map edits.");
    };
    pending.current = true; setWorking(true); setError(null);
    const base = `/autonomy/maps/${encodeURIComponent(v.mapName)}`;
    try {
      const saved = await v.refresh(); check();
      if (uncertain.current) {
        const { before, after } = uncertain.current;
        if (sameSaved(saved, after)) { checkpoint.current = saved; persisted.current = true; }
        else if (!sameSaved(saved, before)) throw new Error("Observation settings changed elsewhere. Exit Edit observation route to reload before editing again.");
        uncertain.current = null;
      }
      if (!sameSaved(saved, checkpoint.current))
        throw new Error("Observation settings changed elsewhere. Exit Edit observation route to reload before editing again.");
      checkpoint.current = saved;
      // Validate the entire edit against the refreshed mission catalog before
      // any PATCH/DELETE, so invalid route fields cannot cause a partial save.
      validateDraft(v.snapshot, saved);
      for (const spot of v.snapshot.spots) {
        const original = observationSpots(checkpoint.current).find(value => value.id === spot.id);
        if (!original) throw new Error("A waypoint was deleted elsewhere. Exit Edit observation route to reload before editing again.");
        if (same(spotValue(original), spotValue(spot))) continue;
        check();
        const before = checkpoint.current;
        const after = { ...before, plan: { ...before.plan, revision: before.plan.revision + 1 },
          spots: before.spots.map(value => value.id === spot.id ? { ...value, label: spot.label.trim(), pose: spot.pose } : value) };
        uncertain.current = { before, after };
        const written = await autonomyRequest(`${base}/spots/${encodeURIComponent(spot.id)}`, "PATCH", {
          label: spot.label, pose: spot.pose, plan_revision: before.plan.revision,
        });
        if (!current()) return false;
        persisted.current = true;
        const expected = { ...after, spots: after.spots.map(value => value.id === spot.id ? written : value) };
        checkpoint.current = expected; uncertain.current = { before, after: expected };
        check();
        const confirmed = await v.refresh(); check();
        if (!sameSaved(confirmed, expected)) throw new Error("Observation settings changed during Save. Exit Edit observation route to reload before editing again.");
        checkpoint.current = confirmed; uncertain.current = null;
      }
      for (const spot of observationSpots(checkpoint.current)) {
        if (v.snapshot.spots.some(value => value.id === spot.id)) continue;
        check();
        const before = checkpoint.current;
        const observations = before.plan.observations.filter(point => point.spot_id !== spot.id);
        const after = { ...before, spots: before.spots.filter(value => value.id !== spot.id), plan: {
          ...before.plan, observations, revision: before.plan.revision + (observations.length !== before.plan.observations.length ? 1 : 0),
        } };
        uncertain.current = { before, after };
        const written = await autonomyRequest(`${base}/spots/${encodeURIComponent(spot.id)}?plan_revision=${before.plan.revision}`, "DELETE");
        if (!current()) return false;
        persisted.current = true; checkpoint.current = written;
        uncertain.current = { before, after: written };
        check();
        const confirmed = await v.refresh(); check();
        if (!sameSaved(confirmed, written)) throw new Error("Observation settings changed during Save. Exit Edit observation route to reload before editing again.");
        checkpoint.current = confirmed; uncertain.current = null;
      }
      if (!same(planValue(v.snapshot.plan), planValue(checkpoint.current.plan))) {
        check();
        const before = checkpoint.current;
        const plan = { ...v.snapshot.plan, revision: before.plan.revision };
        uncertain.current = { before, after: { ...before, plan: { ...plan, revision: plan.revision + 1 } } };
        const written = await autonomyRequest(base, "PUT", plan);
        if (!current()) return false;
        persisted.current = true; checkpoint.current = written;
        uncertain.current = { before, after: written };
        check();
        const confirmed = await v.refresh(); check();
        if (!sameSaved(confirmed, written)) throw new Error("Observation settings changed during Save. Exit Edit observation route to reload before editing again.");
        checkpoint.current = confirmed; uncertain.current = null;
      }
      setValue(null); setSaveFailed(false); v.history.clear();
      window.dispatchEvent(new CustomEvent(OBSERVATION_PLAN_CHANGED, { detail: { mapName: v.mapName } }));
      return true;
    } catch (failure) {
      if (current()) {
        setSaveFailed(true);
        if (persisted.current) v.history.clear();
        setError(`${failure.message}${persisted.current
          ? " Some changes were saved. Your remaining draft is kept; retry Save, or exit Edit observation route to discard only unsaved changes."
          : uncertain.current ? " Save could not be confirmed. Your draft is kept; retry to check the saved state." : " Your draft is kept."}`);
      }
      return false;
    } finally { pending.current = false; if (current()) setWorking(false); }
  };
  const discard = async () => {
    const v = latest.current, token = epoch.current;
    if (pending.current || v.isBlocked()) return false;
    pending.current = true; setWorking(true);
    try {
      await v.refresh();
      if (token !== epoch.current || latest.current.isBlocked()) return false;
      const message = persisted.current || uncertain.current
        ? "Unsaved edits discarded. Any changes already saved by the earlier Save remain." : null;
      reset(); latest.current.snapshot = null; v.history.clear(); setError(message);
      return true;
    } catch (failure) { if (token === epoch.current) setError(`${failure.message} Your draft is kept.`); return false; }
    finally { pending.current = false; setWorking(false); }
  };
  return { snapshot, begin, reset, editPlan, editSnapshot, updateSpot, removeWaypoint, save, discard, working, error,
    dirty: !!snapshot && (saveFailed || !sameSnapshot(snapshot, configuration && snapshotOf(configuration))),
  };
}
