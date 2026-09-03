// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import { STAGE_AUTHORING } from "../lib/stages";
import useDesignAtRobotWaypointController from "./useDesignAtRobotWaypointController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(overrides = {}) {
  let generation = 1;
  let state = {
    stage: STAGE_AUTHORING, interactionMode: "view", documentReady: true,
    mapName: "factory", mapPath: "factory.pgm", mappingActive: false,
    runActive: false, runnerActive: false, designLocalizationActive: true,
  };
  const order = [];
  const record = (name, implementation) => jest.fn((...args) => {
    order.push(name);
    return implementation?.(...args);
  });
  const ui = {
    prepareBegin: record("prepare"), setRuntimeMode: record("runtimeMode"),
    setRuntimeOwned: jest.fn(), setShutdownPending: jest.fn(),
    setInteractionMode: record("interaction"), setWaypointOptionsOpen: jest.fn(),
    setDesignPoseReady: jest.fn(),
  };
  const runtime = {
    startLocalization: record("start", async () => ({})),
    configureAmcl: record("configure", async () => ({})),
    cleanupStartedRuntime: jest.fn(async () => ({})), clearPoseCache: jest.fn(),
    publishInitialPose: jest.fn(async () => ({})),
    waitForConvergence: jest.fn(async () => ({ position: { x: 4, y: 5 }, yaw: 0.7 })),
    poseCoordinates: (pose) => ({ x: pose.position.x, y: pose.position.y, yaw: pose.yaw }),
    invalidateStatus: jest.fn(), stop: jest.fn(async () => ({})),
    stopShared: jest.fn(async () => ({})),
  };
  const waypoint = {
    createAtRobot: jest.fn(async ({ resolvePose, finalize, resultMessage }) => {
      const resolved = await resolvePose();
      const initialized = { ...resolved, label: "Waypoint 2" };
      await finalize({ documentCurrent: true });
      return { initialized, message: resultMessage(initialized) };
    }),
  };
  const ports = {
    getState: () => state,
    document: {
      captureLease: () => ({ generation }),
      isLeaseCurrent: (lease) => lease.generation === generation,
    },
    runtime, waypoint, ui,
    session: { save: record("session") },
    runCommand: jest.fn(async (_label, action) => action()),
    onMessage: jest.fn(),
    ...overrides,
  };
  const view = renderHook(() => useDesignAtRobotWaypointController(ports));
  return {
    view, ports, runtime, waypoint, ui, order,
    patchState: (patch) => { state = { ...state, ...patch }; },
    changeDocument: () => { generation += 1; },
  };
}

test.each([
  [{ mappingActive: true }, "runtime-active"],
  [{ documentReady: false }, "document-not-ready"],
])("begin rejects unsafe state without starting localization", async (patch, reason) => {
  const { view, runtime, patchState } = setup();
  patchState(patch);
  let result;
  await act(async () => { result = await view.result.current.begin(); });
  expect(result).toEqual(expect.objectContaining({ skipped: true, reason }));
  expect(runtime.startLocalization).not.toHaveBeenCalled();
});

test("begin preserves preparation, start, configure, session, and arm order", async () => {
  const { view, runtime, order, ports } = setup();
  await act(async () => view.result.current.begin());
  expect(runtime.startLocalization).toHaveBeenCalledWith("factory");
  expect(order).toEqual(["prepare", "start", "configure", "runtimeMode", "session", "interaction"]);
  expect(ports.session.save).toHaveBeenCalledWith(expect.objectContaining({
    mapName: "factory", designMapPath: "factory.pgm",
    navigationRuntimeMode: "localization",
  }));
});

test("an armed transaction cannot start localization twice", async () => {
  const { view, runtime } = setup();
  await act(async () => view.result.current.begin());
  let second;
  await act(async () => { second = await view.result.current.begin(); });
  expect(second).toEqual({ skipped: true, reason: "operation-active" });
  expect(runtime.startLocalization).toHaveBeenCalledTimes(1);
});

test("begin failure does not arm pose mode and releases for retry", async () => {
  const { view, runtime, ui } = setup();
  runtime.startLocalization.mockRejectedValueOnce(new Error("start failed"));
  await act(async () => { await expect(view.result.current.begin()).rejects.toThrow("start failed"); });
  expect(ui.setInteractionMode).not.toHaveBeenCalledWith("initial");
  await act(async () => view.result.current.begin());
  expect(runtime.startLocalization).toHaveBeenCalledTimes(2);
});

test("AMCL configuration failure cleans the started runtime and releases for retry", async () => {
  const { view, runtime, ui } = setup();
  runtime.configureAmcl.mockRejectedValueOnce(new Error("configure failed"));
  await act(async () => {
    await expect(view.result.current.begin()).rejects.toThrow("configure failed");
  });
  expect(runtime.cleanupStartedRuntime).toHaveBeenCalledTimes(1);
  expect(ui.setInteractionMode).not.toHaveBeenCalledWith("initial");
  await act(async () => view.result.current.begin());
  expect(runtime.startLocalization).toHaveBeenCalledTimes(2);
});

test("late begin after document change cannot commit and cleans stale start", async () => {
  const pending = deferred();
  const { view, runtime, ui, changeDocument } = setup();
  runtime.configureAmcl.mockReturnValue(pending.promise);
  let request;
  act(() => { request = view.result.current.begin(); });
  changeDocument();
  await act(async () => { pending.resolve({}); await request; });
  expect(runtime.cleanupStartedRuntime).toHaveBeenCalled();
  expect(ui.setRuntimeMode).not.toHaveBeenCalled();
  expect(ui.setInteractionMode).not.toHaveBeenCalledWith("initial");
});

test("complete publishes the captured map pose and creates from convergence", async () => {
  const { view, runtime, waypoint } = setup();
  await act(async () => view.result.current.begin());
  let result;
  await act(async () => { result = await view.result.current.completeAtPose(1, 2, 0.3); });
  expect(runtime.publishInitialPose).toHaveBeenCalledWith({
    x: 1, y: 2, yaw: 0.3, frameId: "map", mapName: "factory",
  });
  expect(result.initialized).toEqual(expect.objectContaining({ x: 4, y: 5, yaw: 0.7 }));
  expect(result.message).toBe("Created Waypoint 2 at robot");
  expect(waypoint.createAtRobot).toHaveBeenCalledTimes(1);
});

test("finalize resets ownership even when runtime Stop rejects", async () => {
  const { view, runtime, ui, ports } = setup();
  runtime.stop.mockRejectedValue(new Error("lost response"));
  await act(async () => view.result.current.begin());
  await act(async () => { await expect(view.result.current.completeAtPose(1, 2, 0)).rejects.toThrow("lost response"); });
  expect(ui.setRuntimeMode).toHaveBeenLastCalledWith("idle");
  expect(ui.setDesignPoseReady).toHaveBeenLastCalledWith(false);
  expect(ports.session.save).toHaveBeenLastCalledWith(expect.objectContaining({
    navigationRuntimeMode: "idle", runRuntimeOwned: false,
  }));
});

test("stale document during pose convergence cannot create or stop a new session", async () => {
  const convergence = deferred();
  const { view, runtime, changeDocument } = setup();
  runtime.waitForConvergence.mockReturnValue(convergence.promise);
  await act(async () => view.result.current.begin());
  let request;
  act(() => { request = view.result.current.completeAtPose(1, 2, 0); });
  changeDocument();
  await act(async () => { convergence.resolve({ position: { x: 1, y: 2 } }); await expect(request).rejects.toMatchObject({ code: "STALE_DESIGN_DOCUMENT" }); });
  expect(runtime.stop).not.toHaveBeenCalled();
});

test("duplicate pose completion is rejected while the first is pending", async () => {
  const convergence = deferred();
  const { view, runtime, waypoint } = setup();
  runtime.waitForConvergence.mockReturnValue(convergence.promise);
  await act(async () => view.result.current.begin());
  let first;
  act(() => { first = view.result.current.completeAtPose(1, 2, 0); });
  let second;
  await act(async () => { second = await view.result.current.completeAtPose(3, 4, 0); });
  expect(second).toEqual({ skipped: true, reason: "operation-active" });
  expect(waypoint.createAtRobot).toHaveBeenCalledTimes(1);
  await act(async () => { convergence.resolve({ position: { x: 1, y: 2 }, yaw: 0 }); await first; });
});

test("pose completion disarms and refuses while another runtime is active", async () => {
  const { view, runtime, waypoint, ui, patchState } = setup();
  await act(async () => view.result.current.begin());
  patchState({ runActive: true });
  let result;
  await act(async () => { result = await view.result.current.completeAtPose(1, 2, 0); });
  expect(result).toEqual({ skipped: true, reason: "runtime-active" });
  expect(ui.setInteractionMode).toHaveBeenLastCalledWith("view");
  expect(waypoint.createAtRobot).not.toHaveBeenCalled();
  expect(runtime.publishInitialPose).not.toHaveBeenCalled();
});

test("cancelPending stops only an armed Design localization", () => {
  const { view, runtime, patchState } = setup();
  patchState({ interactionMode: "initial" });
  expect(view.result.current.cancelPending()).toBe(true);
  expect(runtime.stopShared).toHaveBeenCalledTimes(1);
  patchState({ stage: "run" });
  expect(view.result.current.cancelPending()).toBe(false);
  expect(runtime.stopShared).toHaveBeenCalledTimes(1);
});
