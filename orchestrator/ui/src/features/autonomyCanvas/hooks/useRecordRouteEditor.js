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

import { useCallback, useMemo, useRef, useState } from "react";
import { missionFlowEdgesForRouteOrder, syncMissionFlowNodesWithSpots } from "../../mapWorkspace/lib/missionFlow";
import { deriveMissionRouteView } from "../../mapWorkspace/lib/missionRouteView";

const newPoint = id => ({ spot_id: id, departure_policy: "reasoning", dwell_seconds: 15, mission_ids: [], arrival_bt: "" });

// Map editing selects waypoint settings. Its separate order mode connects
// clicked waypoints, staging the new route until Save.
export default function useRecordRouteEditor({ planState, spots, blocked, isBlocked, onPrepare, onSelect, onMessage }) {
  const [routeMode, setRouteMode] = useState(false);
  const [orderMode, setOrderMode] = useState(false);
  const [orderError, setOrderError] = useState(null);
  const latest = useRef({}), pointCache = useRef(new Map());
  const plan = planState.draft;
  const points = plan?.observations || [];
  const ids = points.map(point => point.spot_id);
  const closed = ids.length > 1 && (plan?.route_closed ?? true);
  const locked = blocked || planState.working;
  latest.current = { routeMode, orderMode, plan, dirty: planState.dirty, planState, spots, blocked: locked, isBlocked, onPrepare, onSelect, onMessage };
  points.forEach(point => pointCache.current.set(point.spot_id, point));
  const lockedNow = () => latest.current.blocked || !!latest.current.isBlocked?.();
  const nodes = useMemo(() => syncMissionFlowNodesWithSpots([], spots), [spots]);
  const sharedView = deriveMissionRouteView({ spots, flowNodes: nodes, flowEdges: missionFlowEdgesForRouteOrder(ids, closed) });
  const routeSpots = ids.map(id => spots.find(spot => spot.id === id)).filter(Boolean);
  const routeView = { ...sharedView, treeSpots: routeSpots, orderedSpots: routeSpots,
    executionSpots: closed && routeSpots.length ? [...routeSpots, routeSpots[0]] : routeSpots,
    closed, order: ids.map((id, index) => ({ id, order: index + 1 })) };
  const clearSource = useCallback(() => {}, []);
  const resetEditing = useCallback(() => {
    latest.current.routeMode = false; latest.current.orderMode = false;
    pointCache.current.clear(); setRouteMode(false); setOrderMode(false); setOrderError(null);
  }, []);
  const editPlan = changes => {
    const v = latest.current;
    const changed = v.planState.edit(changes);
    if (changed) {
      // React may batch rapid canvas clicks. Keep the last staged plan available
      // synchronously so an earlier click is never replaced by a later one.
      v.plan = { ...v.plan, ...changes }; v.dirty = true;
      v.plan.observations.forEach(point => pointCache.current.set(point.spot_id, point));
      setOrderError(null);
    }
    return changed;
  };
  const finishOrder = () => { latest.current.orderMode = false; setOrderMode(false); };
  const updateOrder = (nextIds, loop) => {
    const v = latest.current;
    if (!v.routeMode || lockedNow() || !v.plan) return false;
    const current = v.plan;
    const valid = [...new Set(nextIds)].filter(id => v.spots.some(spot => spot.id === id));
    const keepLoop = loop ?? (current.observations.length > 1 && (current.route_closed ?? true));
    const existing = new Map([...pointCache.current, ...(current.observations || []).map(point => [point.spot_id, point])]);
    current.observations.forEach(point => pointCache.current.set(point.spot_id, point));
    return editPlan({ observations: valid.map(id => existing.get(id) || newPoint(id)), route_closed: keepLoop && valid.length > 1 });
  };
  const setRouteOrder = nextIds => updateOrder(nextIds);
  const setIncluded = (id, included) => {
    const v = latest.current;
    const current = (v.plan?.observations || []).map(point => point.spot_id);
    if (included === current.includes(id)) return false;
    return updateOrder(included ? [...current, id] : current.filter(value => value !== id));
  };
  return {
    routeMode, orderMode, routeView, routeSourceId: "", clearSource, resetEditing,
    handleMapClick: () => latest.current.routeMode,
    handleSpotClick: id => {
      const v = latest.current;
      if (!v.routeMode || lockedNow() || !v.spots.some(spot => spot.id === id)) return false;
      if (!v.orderMode) { v.onSelect(id); return true; }
      const order = v.plan.observations.map(point => point.spot_id);
      if (order.length > 1 && order[0] === id) {
        updateOrder(order, true); finishOrder();
      } else if (!order.includes(id)) updateOrder([...order, id], false);
      return true;
    },
    toggleOrderMode: () => {
      const v = latest.current;
      if (!v.routeMode || lockedNow() || !v.plan) return false;
      if (v.orderMode) { finishOrder(); return true; }
      updateOrder([], false);
      v.onSelect(""); setOrderError(null);
      latest.current.orderMode = true; setOrderMode(true); return true;
    },
    toggleMode: () => {
      const v = latest.current;
      if (lockedNow() || !v.plan) return false;
      if (v.routeMode && v.dirty) {
        v.onMessage?.("Save your changes or turn off Edit observation route to discard them."); return false;
      }
      if (v.routeMode) { resetEditing(); v.planState.reset?.(); return true; }
      pointCache.current.clear();
      if (v.onPrepare?.() === false) return false;
      setOrderError(null); latest.current.routeMode = true; setRouteMode(true); return true;
    },
    setRouteOrder, setIncluded,
    appendSpot: id => setIncluded(id, true),
    removeSpot: id => setIncluded(id, false),
    clearRoute: () => updateOrder([], false),
    openLoop: () => updateOrder((latest.current.plan?.observations || []).map(point => point.spot_id), false),
    closeLoop: () => {
      const order = (latest.current.plan?.observations || []).map(point => point.spot_id);
      return order.length > 1 && updateOrder(order, true);
    },
    moveSpot: (id, delta) => {
      const order = (latest.current.plan?.observations || []).map(point => point.spot_id);
      const index = order.indexOf(id), target = index + delta;
      if (index < 0 || target < 0 || target >= order.length) return false;
      [order[index], order[target]] = [order[target], order[index]];
      return setRouteOrder(order);
    },
    editor: {
      plan, locked, dirty: planState.dirty, error: orderError || planState.error, missions: planState.missions || [],
      pointFor: id => pointCache.current.get(id) || newPoint(id),
      save: async () => {
        if (lockedNow()) return false;
        if (!latest.current.plan?.observations.length) {
          setOrderError("Select at least one waypoint for the observation route before saving."); return false;
        }
        const saved = await latest.current.planState.save();
        if (saved) resetEditing();
        return saved;
      },
      discard: async () => {
        if (lockedNow()) return false;
        const discarded = await latest.current.planState.discard();
        if (discarded) resetEditing();
        return discarded;
      },
      updateSpot: (id, changes) => !lockedNow() && latest.current.routeMode && !latest.current.orderMode && latest.current.planState.updateSpot(id, changes),
      removeWaypoint: id => !lockedNow() && latest.current.routeMode && !latest.current.orderMode && latest.current.planState.removeWaypoint(id),
      updatePoint: (id, changes) => {
        const v = latest.current;
        if (lockedNow() || !v.routeMode || v.orderMode) return false;
        return editPlan({ observations: v.plan.observations.map(point => point.spot_id === id ? { ...point, ...changes } : point) });
      },
    },
  };
}
