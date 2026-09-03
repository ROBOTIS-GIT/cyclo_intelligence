// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useNavigateGoalController from "./useNavigateGoalController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function renderController(sendGoalAndWait = jest.fn(async () => ({
  ok: true,
  status: "SUCCEEDED",
}))) {
  const onDisarm = jest.fn();
  const onMessage = jest.fn();
  const view = renderHook(() => useNavigateGoalController({
    sendGoalAndWait,
    onDisarm,
    onMessage,
  }));
  return { view, sendGoalAndWait, onDisarm, onMessage };
}

test("publishes a map pose, disarms once and keeps the reached pose", async () => {
  const { view, sendGoalAndWait, onDisarm, onMessage } = renderController();
  await act(async () => view.result.current.sendGoal(1.25, -2.5, Math.PI / 2));

  expect(sendGoalAndWait).toHaveBeenCalledWith({
    pose: {
      header: expect.objectContaining({ frame_id: "map", stamp: expect.any(Object) }),
      pose: {
        position: { x: 1.25, y: -2.5, z: 0 },
        orientation: expect.objectContaining({
          z: expect.closeTo(Math.SQRT1_2),
          w: expect.closeTo(Math.SQRT1_2),
        }),
      },
    },
  });
  expect(onDisarm).toHaveBeenCalledTimes(1);
  expect(view.result.current.goalStatus).toBe("reached");
  expect(view.result.current.goalPose.pose.position.x).toBe(1.25);
  expect(onMessage).toHaveBeenLastCalledWith("Goal reached");
});

test.each([
  [{ ok: false, status: "ABORTED", message: "aborted" }, "aborted"],
  [{ ok: true, status: "REJECTED" }, "Navigation goal REJECTED"],
])("treats a non-success result as failed", async (result, message) => {
  const { view, onMessage } = renderController(jest.fn(async () => result));
  await act(async () => view.result.current.sendGoal(1, 2, 0));
  expect(view.result.current.goalStatus).toBe("failed");
  expect(onMessage).toHaveBeenLastCalledWith(message);
});

test("reports a thrown goal error as failed", async () => {
  const { view, onMessage } = renderController(jest.fn(async () => {
    throw new Error("goal transport failed");
  }));
  await act(async () => view.result.current.sendGoal(1, 2, 0));
  expect(view.result.current.goalStatus).toBe("failed");
  expect(onMessage).toHaveBeenLastCalledWith("goal transport failed");
});

test("only the latest out-of-order request may settle the state", async () => {
  const first = deferred();
  const second = deferred();
  const sendGoalAndWait = jest.fn()
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise);
  const { view } = renderController(sendGoalAndWait);
  let firstRun;
  let secondRun;
  await act(async () => {
    firstRun = view.result.current.sendGoal(1, 1, 0);
    secondRun = view.result.current.sendGoal(2, 2, 0);
  });
  await act(async () => {
    first.resolve({ ok: true, status: "SUCCEEDED" });
    await firstRun;
  });
  expect(view.result.current.goalStatus).toBe("driving");
  expect(view.result.current.goalPose.pose.position.x).toBe(2);
  await act(async () => {
    second.resolve({ ok: false, status: "ABORTED" });
    await secondRun;
  });
  expect(view.result.current.goalStatus).toBe("failed");
});

test("an earlier rejection cannot overwrite a newer success or message", async () => {
  const first = deferred();
  const second = deferred();
  const { view, onMessage } = renderController(jest.fn()
    .mockImplementationOnce(() => first.promise)
    .mockImplementationOnce(() => second.promise));
  let firstRun;
  let secondRun;
  await act(async () => {
    firstRun = view.result.current.sendGoal(1, 1, 0);
    secondRun = view.result.current.sendGoal(2, 2, 0);
  });
  await act(async () => {
    second.resolve({ ok: true, status: "SUCCEEDED" });
    await secondRun;
  });
  const messageCount = onMessage.mock.calls.length;
  await act(async () => {
    first.reject(new Error("stale failure"));
    await firstRun;
  });
  expect(view.result.current.goalStatus).toBe("reached");
  expect(view.result.current.goalPose.pose.position.x).toBe(2);
  expect(onMessage).toHaveBeenCalledTimes(messageCount);
  expect(onMessage).toHaveBeenLastCalledWith("Goal reached");
});

test.each(["resolve", "reject"])(
  "invalidate clears immediately and ignores a late %s",
  async (settle) => {
    const request = deferred();
    const { view, onMessage } = renderController(jest.fn(() => request.promise));
    let running;
    await act(async () => { running = view.result.current.sendGoal(3, 4, 0); });
    act(() => view.result.current.invalidateGoal());
    expect(view.result.current.goalStatus).toBe("idle");
    expect(view.result.current.goalPose).toBeNull();
    const messageCount = onMessage.mock.calls.length;
    await act(async () => {
      if (settle === "resolve") request.resolve({ ok: true, status: "SUCCEEDED" });
      else request.reject(new Error("late failure"));
      await running;
    });
    expect(view.result.current.goalStatus).toBe("idle");
    expect(view.result.current.goalPose).toBeNull();
    expect(onMessage).toHaveBeenCalledTimes(messageCount);
  },
);
