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

import { act, renderHook } from "@testing-library/react";
import { requestNoMotionUpdate } from "../../../utils/navigationApi";
import useAutoLocalizedPose from "./useAutoLocalizedPose";

jest.mock("../../../utils/navigationApi", () => ({ requestNoMotionUpdate: jest.fn() }));

function poseMessage(x, variance = 0.1) {
  const covariance = Array(36).fill(0);
  covariance[0] = covariance[7] = covariance[35] = variance;
  return { pose: { covariance, pose: { position: { x, y: 0 }, orientation: { w: 1 } } } };
}

beforeEach(() => {
  jest.useFakeTimers();
  requestNoMotionUpdate.mockReset().mockResolvedValue({});
});
afterEach(() => jest.useRealTimers());

async function begin(amclPoseRef) {
  const view = renderHook(() => useAutoLocalizedPose(amclPoseRef));
  const finished = jest.fn();
  let outcome;
  await act(async () => {
    // Handle rejection immediately so fake-timer advancement cannot cause an
    // unhandled rejection before the final assertions inspect the outcome.
    outcome = view.result.current().then(
      pose => { finished(); return { pose }; },
      error => { finished(); return { error }; },
    );
  });
  return { view, outcome, finished };
}
async function poll(count) {
  for (let step = 0; step < count; step += 1) {
    await act(async () => { jest.advanceTimersByTime(700); });
  }
}

test("requires three updates even when covariance is already low", async () => {
  const amclPoseRef = { current: null };
  requestNoMotionUpdate.mockImplementation(async () => {
    amclPoseRef.current = poseMessage(requestNoMotionUpdate.mock.calls.length);
  });
  const { outcome, finished } = await begin(amclPoseRef);
  await poll(2);
  expect(finished).not.toHaveBeenCalled();
  expect(requestNoMotionUpdate).toHaveBeenCalledTimes(3);
  await poll(1);
  expect(await outcome).toEqual({ pose: amclPoseRef.current.pose.pose });
  expect(requestNoMotionUpdate).toHaveBeenCalledTimes(3);
});

test("falls back to the latest available pose after ten unconverged updates", async () => {
  const amclPoseRef = { current: null };
  const lastAvailable = poseMessage(8, 1);
  requestNoMotionUpdate.mockImplementation(async () => {
    const attempt = requestNoMotionUpdate.mock.calls.length;
    amclPoseRef.current = attempt < 8 ? poseMessage(attempt, 1)
      : attempt === 8 ? lastAvailable : null;
  });
  const { outcome, finished } = await begin(amclPoseRef);
  await poll(9);
  expect(finished).not.toHaveBeenCalled();
  await poll(1);
  expect(await outcome).toEqual({ pose: lastAvailable.pose.pose });
  expect(requestNoMotionUpdate).toHaveBeenCalledTimes(10);
});

test("reports a missing robot pose after all ten updates", async () => {
  const { outcome } = await begin({ current: null });
  await poll(10);
  expect((await outcome).error.message).toBe("Robot pose unavailable after automatic localization");
  expect(requestNoMotionUpdate).toHaveBeenCalledTimes(10);
});
