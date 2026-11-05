// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import { STAGE_NAVIGATE, STAGE_RUN } from "../lib/stages";
import useRunNavigationLocalizationController from "./useRunNavigationLocalizationController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(overrides = {}) {
  let state = {
    stage: STAGE_RUN, runStage: STAGE_RUN, mapBusy: false,
    snapshotInvalid: false, interactionMode: "view", running: false,
    runtimeMode: "idle", targetMapName: "factory", sessionMapName: "design-map",
  };
  let pageExit = false;
  const commits = {
    setStage: jest.fn(), setRuntimeMode: jest.fn(), setDesignPoseReady: jest.fn(),
    setRunPoseReady: jest.fn(), setRuntimeOwned: jest.fn(),
    setShutdownPending: jest.fn(), setInteractionMode: jest.fn(),
    persistClaim: jest.fn(), persistRollback: jest.fn(),
  };
  const ports = {
    getState: () => state,
    runtime: {
      start: jest.fn(async () => ({})), stop: jest.fn(async () => ({})),
      pageExitStopSent: () => pageExit,
    },
    commits,
    pose: {
      clearCache: jest.fn(), publish: jest.fn(async () => ({})),
      waitForConvergence: jest.fn(async () => ({})),
    },
    runCommand: jest.fn(async (_label, action) => action()),
    onMessage: jest.fn(),
    ...overrides,
  };
  const view = renderHook(() => useRunNavigationLocalizationController(ports));
  return {
    view, ports, commits,
    patchState: (patch) => { state = { ...state, ...patch }; },
    setPageExit: (value) => { pageExit = value; },
  };
}

test("starts navigation, claims ownership, and arms initial pose", async () => {
  const { view, ports, commits } = setup();
  await act(async () => view.result.current.localize());
  expect(ports.runtime.start).toHaveBeenCalledWith("nav", "factory");
  expect(commits.persistClaim).toHaveBeenCalledWith(expect.objectContaining({
    workspaceStage: STAGE_RUN, runRuntimeOwned: true,
  }));
  expect(commits.setRunPoseReady).toHaveBeenCalledWith(false);
  expect(commits.setInteractionMode).toHaveBeenLastCalledWith("initial");
});

test("Navigate uses its map-only gate and a second Localize disarms", async () => {
  const { view, ports, commits, patchState } = setup();
  patchState({ stage: STAGE_NAVIGATE, snapshotInvalid: true, interactionMode: "initial" });
  await act(async () => view.result.current.localize());
  expect(ports.runtime.start).not.toHaveBeenCalled();
  expect(commits.setInteractionMode).toHaveBeenCalledWith("view");
});

test("an already-up Run runtime preserves the accepted pose", async () => {
  const { view, ports, commits, patchState } = setup();
  patchState({ running: true, runtimeMode: "run" });
  await act(async () => view.result.current.localize());
  expect(ports.runtime.start).not.toHaveBeenCalled();
  expect(commits.setRunPoseReady).not.toHaveBeenCalledWith(false);
});

test("start failure rolls ownership back and preserves the captured target map", async () => {
  const start = deferred();
  const { view, ports, commits, patchState } = setup();
  ports.runtime.start.mockReturnValue(start.promise);
  let request;
  act(() => { request = view.result.current.localize(); });
  patchState({ targetMapName: "other" });
  await act(async () => {
    start.reject(new Error("start failed"));
    await expect(request).rejects.toThrow("start failed");
  });
  expect(ports.runtime.start).toHaveBeenCalledWith("nav", "factory");
  expect(commits.setRuntimeMode).toHaveBeenLastCalledWith("idle");
  expect(commits.setRuntimeOwned).toHaveBeenLastCalledWith(false);
  expect(commits.persistRollback).toHaveBeenCalled();
});

test("pagehide during start orders a keepalive Stop after start settles", async () => {
  const start = deferred();
  const { view, ports, commits, setPageExit } = setup();
  ports.runtime.start.mockReturnValue(start.promise);
  let request;
  act(() => { request = view.result.current.localize(); });
  setPageExit(true);
  await act(async () => { start.resolve({}); await request; });
  expect(ports.runtime.stop).toHaveBeenCalledWith({ keepalive: true });
  expect(commits.persistRollback).not.toHaveBeenCalled();
  expect(commits.setInteractionMode).not.toHaveBeenCalledWith("initial");
});

test("only the latest pose convergence may mark localization ready", async () => {
  const first = deferred();
  const second = deferred();
  const { view, ports, commits } = setup();
  ports.pose.waitForConvergence
    .mockReturnValueOnce(first.promise)
    .mockReturnValueOnce(second.promise);
  let a;
  await act(async () => { a = view.result.current.estimatePose(1, 2, 0.1); await Promise.resolve(); });
  let b;
  await act(async () => { b = view.result.current.estimatePose(3, 4, 0.2); await Promise.resolve(); });
  await act(async () => { first.resolve({}); await a; });
  expect(commits.setRunPoseReady).not.toHaveBeenCalledWith(true);
  await act(async () => { second.resolve({}); await b; });
  expect(commits.setRunPoseReady).toHaveBeenCalledTimes(1);
  expect(ports.pose.publish).toHaveBeenNthCalledWith(2, {
    x: 3, y: 4, yaw: 0.2, frameId: "map",
  });
});

test("explicit invalidation makes a late pose response inert", async () => {
  const convergence = deferred();
  const { view, ports, commits } = setup();
  ports.pose.waitForConvergence.mockReturnValue(convergence.promise);
  let request;
  await act(async () => { request = view.result.current.estimatePose(1, 2, 0); await Promise.resolve(); });
  act(() => view.result.current.invalidatePoseRequest());
  await act(async () => { convergence.resolve({}); await request; });
  expect(commits.setRunPoseReady).not.toHaveBeenCalledWith(true);
});

test("start success after unmount preserves the durable claim without arming interaction", async () => {
  const start = deferred();
  const { view, ports, commits } = setup();
  ports.runtime.start.mockReturnValue(start.promise);
  let request;
  act(() => { request = view.result.current.localize(); });

  expect(commits.persistClaim).toHaveBeenCalledTimes(1);
  view.unmount();
  await act(async () => {
    start.resolve({ ok: true });
    await request;
  });

  expect(commits.persistClaim).toHaveBeenCalledTimes(1);
  expect(commits.persistRollback).not.toHaveBeenCalled();
  expect(commits.setInteractionMode).not.toHaveBeenCalledWith("initial");
});

test("start failure after unmount persists rollback without mutating unmounted UI state", async () => {
  const start = deferred();
  const { view, ports, commits } = setup();
  ports.runtime.start.mockReturnValue(start.promise);
  let request;
  act(() => { request = view.result.current.localize(); });

  view.unmount();
  await act(async () => {
    start.reject(new Error("late start failure"));
    await expect(request).rejects.toThrow("late start failure");
  });

  expect(commits.persistClaim).toHaveBeenCalledTimes(1);
  expect(commits.persistRollback).toHaveBeenCalledWith({
    navigationRuntimeMode: "idle",
    runRuntimeOwned: false,
    runShutdownPending: false,
    runShutdownRequestedAt: null,
  });
  expect(commits.setRuntimeMode).not.toHaveBeenCalledWith("idle");
  expect(commits.setRuntimeOwned).not.toHaveBeenCalledWith(false);
  expect(commits.setInteractionMode).not.toHaveBeenCalledWith("initial");
});

test("pose convergence after unmount cannot mark localization ready", async () => {
  const convergence = deferred();
  const { view, ports, commits } = setup();
  ports.pose.waitForConvergence.mockReturnValue(convergence.promise);
  let request;
  await act(async () => {
    request = view.result.current.estimatePose(1, 2, 0.25);
    await Promise.resolve();
  });
  expect(ports.pose.publish).toHaveBeenCalledWith({
    x: 1, y: 2, yaw: 0.25, frameId: "map",
  });

  view.unmount();
  await act(async () => {
    convergence.resolve({});
    await request;
  });

  expect(commits.setRunPoseReady).not.toHaveBeenCalledWith(true);
});
