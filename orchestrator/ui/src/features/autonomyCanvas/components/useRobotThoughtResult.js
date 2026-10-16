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

import { useEffect, useRef, useState } from "react";
import { displayedReasoningResult } from "./autonomyPresentation";
import { thoughtRunning } from "./robotThought";

const DEPARTURE_DISPLAY_MS = 3200;

// A next_observation result is cleared by the server's phase transition before
// the next status poll. Recover only a confirmed departure from this session's
// activity, never an unaccepted model proposal. This is display state only.
function acceptedDeparture(events = []) {
  const complete = events.find(event => event.title === "Observation complete");
  if (!complete) return null;
  const decision = events.find(event => event.title === "AI · next_observation"
    && event.time_ms <= complete.time_ms && complete.time_ms - event.time_ms < 2000
    && event.detail && complete.detail?.endsWith(` · ${event.detail}`));
  if (!decision) return null;
  return { time: complete.time_ms, reason: decision.detail };
}

export default function useRobotThoughtResult(state) {
  const current = displayedReasoningResult(state);
  const departure = acceptedDeparture(state.events);
  const session = state.session_id || state.execution?.session_id || "current";
  const target = state.execution?.target_spot_id || state.observation?.spot_id;
  const moving = state.phase === "positioning" || state.execution?.activity === "patrol_navigation";
  const urgent = !!(state.error || state.paused || state.execution?.control === "paused"
    || current?.error || current?.action_allowed === false
    || ["pause", "cancel", "unknown", "start"].includes(current?.decision?.decision));
  const eligible = thoughtRunning(state) && moving && !urgent;
  const seen = useRef("");
  const [held, setHeld] = useState(null);
  const departureTime = departure?.time;
  const departureReason = departure?.reason;
  useEffect(() => {
    if (!eligible) { setHeld(null); return; }
    if (departureTime == null) return;
    const key = `${session}:${departureTime}`;
    if (seen.current === key) return;
    seen.current = key;
    // Do not replay old departure explanations when opening a running session.
    if (Math.abs(Date.now() - departureTime) > 10000) return;
    setHeld({ session, target, result: {
      reasoning_task: "mission_selection",
      decision: { decision: "next_observation", mission_id: null, reason: departureReason },
    } });
  }, [eligible, departureTime, departureReason, session, target]);
  useEffect(() => {
    if (!held) return;
    const timer = setTimeout(() => setHeld(null), DEPARTURE_DISPLAY_MS);
    return () => clearTimeout(timer);
  }, [held]);
  return eligible && held?.session === session && held?.target === target ? held.result : current;
}
