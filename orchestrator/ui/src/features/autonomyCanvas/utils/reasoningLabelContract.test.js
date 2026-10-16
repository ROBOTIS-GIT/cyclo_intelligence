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

import { frameReasoningContract, labelMatchesFrame } from "./reasoningLabelContract";

test("patrol labels only allow pause/continue with no mission", () => {
  const frame = { context: { reasoning_task: "patrol_supervision", active_mission: null, candidate_mission_ids: [] } };
  expect(frameReasoningContract(frame).decisions).toEqual(["pause", "continue"]);
  expect(labelMatchesFrame(frame, { decision: "continue", mission_id: null })).toBe(true);
  expect(labelMatchesFrame(frame, { decision: "pause", mission_id: "task" })).toBe(false);
  expect(labelMatchesFrame(frame, { decision: "start", mission_id: "task" })).toBe(false);
});


test.each(["continue", "pause", "cancel"])("mission %s requires the captured active mission", (decision) => {
  const frame = { context: { reasoning_task: "mission_supervision", active_mission: { mission_id: "clean", status: "running" } } };
  expect(frameReasoningContract(frame).decisions).toEqual(["continue", "pause", "cancel"]);
  expect(labelMatchesFrame(frame, { decision, mission_id: "clean" })).toBe(true);
  expect(labelMatchesFrame(frame, { decision, mission_id: "other" })).toBe(false);
  expect(labelMatchesFrame(frame, { decision, mission_id: null })).toBe(false);
});

test("cancel is unavailable while returning to observation and while patrolling", () => {
  const returning = { context: { reasoning_task: "mission_supervision", active_mission: { mission_id: "clean", status: "returning" } } };
  expect(frameReasoningContract(returning).decisions).toEqual(["pause", "continue"]);
  expect(labelMatchesFrame(returning, { decision: "cancel", mission_id: "clean" })).toBe(false);
  expect(labelMatchesFrame(returning, { decision: "continue", mission_id: "clean" })).toBe(true);
  const patrol = { context: { reasoning_task: "patrol_supervision", active_mission: null } };
  expect(labelMatchesFrame(patrol, { decision: "cancel", mission_id: null })).toBe(false);
});

test("legacy selection context does not turn a cancel label into mission supervision", () => {
  const frame = { context: { active_mission: { mission_id: "clean" } } };
  expect(labelMatchesFrame(frame, { decision: "cancel", mission_id: "clean" })).toBe(false);
});


test.each([
  { active_mission: { mission_id: "clean", status: "paused", action: "return-navigation" } },
  { active_mission: { mission_id: "clean", status: "paused", action: "return-posture" } },
  { active_mission: { mission_id: "clean", status: "paused" }, observation: { phase: "returning" } },
])("a paused historical return still cannot cancel a finished mission: %j", (context) => {
  const frame = { context: { reasoning_task: "mission_supervision", ...context } };
  expect(frameReasoningContract(frame).decisions).toEqual(["pause", "continue"]);
  expect(labelMatchesFrame(frame, { decision: "cancel", mission_id: "clean" })).toBe(false);
  expect(labelMatchesFrame(frame, { decision: "continue", mission_id: "clean" })).toBe(true);
});

test("next observation is available only for selection with a next spot and no mission",()=>{
 const frame={context:{reasoning_task:"mission_selection",observation:{can_advance:true}}};
 expect(labelMatchesFrame(frame,{decision:"next_observation",mission_id:null})).toBe(true);
 expect(labelMatchesFrame(frame,{decision:"next_observation",mission_id:"task"})).toBe(false);
 frame.context.observation.can_advance=false;
 expect(frameReasoningContract(frame).decisions).not.toContain("next_observation");
 expect(labelMatchesFrame(frame,{decision:"next_observation",mission_id:null})).toBe(false);
});
