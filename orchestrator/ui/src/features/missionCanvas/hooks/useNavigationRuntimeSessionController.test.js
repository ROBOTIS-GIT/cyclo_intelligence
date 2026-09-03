// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

import { act, renderHook } from "@testing-library/react";
import useNavigationRuntimeSessionController from "./useNavigationRuntimeSessionController";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function setup(overrides = {}) {
  let saved = { navigationRuntimeMode: "run", runRuntimeOwned: true };
  const state = {
    mapName: "factory", stage: "run", designMapPath: "factory.pgm",
    runtimeMode: "run", designPoseReady: true, runtimeOwned: true,
    shutdownPending: false, missionName: "edit", runMissionName: "delivery",
    status: { is_up: true, mode: "run" },
  };
  const ports = {
    enabled: false,
    pollIntervalMs: 5000,
    state,
    status: { get: jest.fn(async () => ({ is_up: true, mode: "run" })), modeOf: (value) => value.mode },
    session: {
      read: jest.fn(() => saved),
      save: jest.fn((patch) => { saved = { ...saved, ...patch }; }),
      recentShutdownMarker: jest.fn(() => false),
    },
    runtime: { stop: jest.fn(async () => ({})) },
    commits: {
      setStatus: jest.fn(), setRuntimeMode: jest.fn(), setDesignPoseReady: jest.fn(),
      setRuntimeOwned: jest.fn(), confirmStopped: jest.fn(),
    },
    onMessage: jest.fn(),
    ...overrides,
  };
  const view = renderHook((props) => useNavigationRuntimeSessionController(props), {
    initialProps: ports,
  });
  return { view, ports, state, getSaved: () => saved };
}

test("poll commits canonical modes and clears incompatible ownership", async () => {
  const status = {
    get: jest.fn(async () => ({ is_up: true, mode: "localization" })),
    modeOf: (value) => value.mode,
  };
  const { view, ports } = setup({ enabled: true, status });
  await act(async () => view.result.current.refreshStatus());
  expect(ports.commits.setRuntimeMode).toHaveBeenCalledWith("localization");
  expect(ports.commits.setRuntimeOwned).toHaveBeenCalledWith(false);
});

test("a stale poll cannot commit after shutdown becomes pending", async () => {
  const pending = deferred();
  const status = {
    get: jest.fn(() => pending.promise),
    modeOf: (value) => value.mode,
  };
  const { view, ports, state } = setup({ enabled: true, status });
  expect(status.get).toHaveBeenCalledTimes(1);
  view.rerender({ ...ports, state: { ...state, shutdownPending: true } });
  await act(async () => {
    pending.resolve({ is_up: true, mode: "run" });
    await Promise.resolve();
  });
  expect(ports.commits.setStatus).not.toHaveBeenCalled();
});

test("polling resumes after a stale request is invalidated by shutdown", async () => {
  const pending = deferred();
  const status = {
    get: jest.fn()
      .mockReturnValueOnce(pending.promise)
      .mockResolvedValueOnce({ is_up: false, mode: "idle" }),
    modeOf: (value) => value.mode,
  };
  const { view, ports, state } = setup({ enabled: true, status });
  expect(status.get).toHaveBeenCalledTimes(1);
  view.rerender({ ...ports, state: { ...state, shutdownPending: true } });
  await act(async () => {
    pending.resolve({ is_up: true, mode: "run" });
    await Promise.resolve();
  });
  view.rerender({ ...ports, state: { ...state, shutdownPending: false } });
  await act(async () => view.result.current.refreshStatus());
  expect(status.get).toHaveBeenCalledTimes(2);
  expect(ports.commits.setStatus).toHaveBeenCalledWith({ is_up: false, mode: "idle" });
});

test("restored shutdown confirms once and clears ownership markers", async () => {
  const stop = deferred();
  const { view, ports, state, getSaved } = setup({
    enabled: true,
    runtime: { stop: jest.fn(() => stop.promise) },
  });
  view.rerender({ ...ports, enabled: true, state: { ...state, shutdownPending: true } });
  view.rerender({ ...ports, enabled: true, state: { ...state, shutdownPending: true } });
  expect(ports.runtime.stop).toHaveBeenCalledTimes(1);
  await act(async () => { stop.resolve({}); await stop.promise; });
  expect(ports.commits.confirmStopped).toHaveBeenCalledTimes(1);
  expect(getSaved()).toEqual(expect.objectContaining({
    navigationRuntimeMode: "idle", runRuntimeOwned: false, runShutdownPending: false,
  }));
});

test("pagehide stops an owned Run once and leaves a retry marker", () => {
  const { view, ports, getSaved } = setup({ enabled: true });
  act(() => window.dispatchEvent(new Event("pagehide")));
  act(() => window.dispatchEvent(new Event("pagehide")));
  expect(ports.runtime.stop).toHaveBeenCalledTimes(1);
  expect(ports.runtime.stop).toHaveBeenCalledWith({ keepalive: true });
  expect(view.result.current.isPageExitStopSent()).toBe(true);
  expect(getSaved()).toEqual(expect.objectContaining({
    runRuntimeOwned: true, runShutdownPending: true,
  }));
});

test("pagehide does not stop a known Mapping runtime", () => {
  const { ports } = setup({
    enabled: true,
    state: {
      mapName: "factory", stage: "mapping", designMapPath: "",
      runtimeMode: "mapping", designPoseReady: false, runtimeOwned: true,
      shutdownPending: false, missionName: "", runMissionName: "",
      status: { is_up: true, mode: "mapping" },
    },
    session: {
      read: jest.fn(() => ({ navigationRuntimeMode: "mapping", runRuntimeOwned: true })),
      save: jest.fn(), recentShutdownMarker: jest.fn(() => false),
    },
  });
  act(() => window.dispatchEvent(new Event("pagehide")));
  expect(ports.runtime.stop).not.toHaveBeenCalled();
});

test("session mirror preserves an existing pending timestamp", () => {
  const { ports, state } = setup();
  ports.session.read.mockReturnValue({ runShutdownRequestedAt: 1234 });
  renderHook(() => useNavigationRuntimeSessionController({
    ...ports, enabled: true, state: { ...state, shutdownPending: true },
  }));
  expect(ports.session.save).toHaveBeenCalledWith(expect.objectContaining({
    runShutdownPending: true, runShutdownRequestedAt: 1234,
  }));
});

test("a poll that settles after unmount cannot commit runtime state", async () => {
  const pending = deferred();
  const status = {
    get: jest.fn(() => pending.promise),
    modeOf: (value) => value.mode,
  };
  const { view, ports } = setup({ status });
  let poll;
  act(() => { poll = view.result.current.refreshStatus(); });
  view.unmount();
  await act(async () => {
    pending.resolve({ is_up: true, mode: "run" });
    await poll;
  });
  expect(ports.commits.setStatus).not.toHaveBeenCalled();
  expect(ports.commits.setRuntimeMode).not.toHaveBeenCalled();
  expect(ports.commits.setRuntimeOwned).not.toHaveBeenCalled();
});

test("pagehide retry ownership marker survives later state mirroring", () => {
  const { view, ports, state, getSaved } = setup({ enabled: true });
  act(() => window.dispatchEvent(new Event("pagehide")));
  const marked = getSaved();
  expect(marked).toEqual(expect.objectContaining({
    navigationRuntimeMode: "idle",
    runRuntimeOwned: true,
    runShutdownPending: true,
  }));
  expect(Number(marked.runShutdownRequestedAt)).toBeGreaterThan(0);

  view.rerender({
    ...ports,
    enabled: true,
    state: {
      ...state,
      runtimeMode: "idle",
      runtimeOwned: false,
      shutdownPending: false,
    },
  });
  expect(getSaved()).toEqual(marked);
  expect(ports.runtime.stop).toHaveBeenCalledTimes(1);
});

test("a failed restored shutdown reports the error without clearing its durable marker", async () => {
  const runtime = { stop: jest.fn(() => Promise.reject(new Error("stop failed"))) };
  const { view, ports, state, getSaved } = setup({ runtime });
  ports.session.save.mockClear();
  view.rerender({ ...ports, enabled: true, state: { ...state, shutdownPending: true } });

  await act(async () => { await Promise.resolve(); await Promise.resolve(); });

  expect(runtime.stop).toHaveBeenCalledTimes(1);
  expect(ports.commits.confirmStopped).not.toHaveBeenCalled();
  expect(ports.onMessage).toHaveBeenCalledWith(
    "Failed to stop the previous Run session: stop failed",
  );
  expect(getSaved()).toEqual(expect.objectContaining({
    runRuntimeOwned: true,
    runShutdownPending: true,
  }));
  expect(ports.session.save).not.toHaveBeenCalledWith(expect.objectContaining({
    runRuntimeOwned: false,
    runShutdownPending: false,
  }));
});

test("an unmounted restored shutdown cannot confirm or clear ownership", async () => {
  const stop = deferred();
  const runtime = { stop: jest.fn(() => stop.promise) };
  const { view, ports, state } = setup({ runtime });
  view.rerender({ ...ports, enabled: true, state: { ...state, shutdownPending: true } });
  expect(runtime.stop).toHaveBeenCalledTimes(1);
  ports.commits.confirmStopped.mockClear();
  ports.session.save.mockClear();
  ports.onMessage.mockClear();

  view.unmount();
  await act(async () => {
    stop.resolve({ ok: true });
    await stop.promise;
  });

  expect(ports.commits.confirmStopped).not.toHaveBeenCalled();
  expect(ports.session.save).not.toHaveBeenCalled();
  expect(ports.onMessage).not.toHaveBeenCalled();
});
