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

import { executionPhaseLabel, executionOwnsRobot, latestExecutionStatus } from "./executionState";
import { displayedReasoningResult } from "../components/autonomyPresentation";
const live = { lifecycle: "active", activity: "mission_action", control: "running", owns_execution: true };
test("pending stop and pause are not reported as acknowledged idle or paused", () => {
  expect(executionPhaseLabel({ ...live, control: "pausing" })).toBe("Pausing");
  expect(executionPhaseLabel({ ...live, control: "resuming" })).toBe("Resuming");
  expect(executionPhaseLabel({ ...live, lifecycle: "stopping", control: "paused" })).toBe("Stopping");
  expect(executionPhaseLabel({ ...live, lifecycle: "failed", control: "paused" })).toBe("Failed");
  expect(executionOwnsRobot({ execution: { ...live, lifecycle: "failed" } })).toBe(true);
  expect(executionOwnsRobot({ execution: { ...live, lifecycle: "stopped", owns_execution: false } })).toBe(false);
});
test("Record observation readiness can exist without owning a patrol session", () => {
  expect(executionPhaseLabel({ ...live, lifecycle: "idle", activity: "observing", owns_execution: false })).toBe("Observing");
  expect(executionPhaseLabel({ ...live, activity: "return_navigation" }, { compact: true })).toBe("Returning");
});
test("status revisions reject old snapshots but accept a restarted controller", () => {
  const old = { execution: { ...live, session_id: "s", epoch: "boot-a", revision: 8 } };
  const late = { execution: { ...old.execution, revision: 7 } };
  expect(latestExecutionStatus(old, late)).toBe(old);
  const restarted = { execution: { ...late.execution, epoch: "boot-b", revision: 1, lifecycle: "interrupted" } };
  expect(latestExecutionStatus(old, restarted)).toBe(restarted);
  const newSession = { execution: { ...late.execution, session_id: "new", revision: 1 } };
  expect(latestExecutionStatus(old, newSession)).toBe(newSession);
});
test("a previous observation result cannot be shown in a newer context of the same task", () => {
  const result = { reasoning_task: "mission_selection", context_version: 4, decision: { decision: "start" } };
  const state = { phase: "observing", reasoning_task: "mission_selection", context_version: 5, result };
  expect(displayedReasoningResult(state)).toBeNull();
  expect(displayedReasoningResult({ ...state, context_version: 4 })).toBe(result);
  expect(displayedReasoningResult({ ...state, context_version: 4, execution: { ...live, lifecycle: "interrupted" } })).toBeNull();
});
