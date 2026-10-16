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
import useNavigationStopCoordinator from "./useNavigationStopCoordinator";

function deferred() {
  let reject;
  let resolve;
  const promise = new Promise((promiseResolve, promiseReject) => {
    reject = promiseReject;
    resolve = promiseResolve;
  });
  return { promise, reject, resolve };
}

function createPorts({ defaultClear = true } = {}) {
  const calls = [];
  const mark = (name, result) => jest.fn(() => {
    calls.push(name);
    return result;
  });
  const ports = {
    getDefaultClearRunSnapshot: mark("default-clear", defaultClear),
    invalidate: {
      mapSelection: mark("invalidate-map"),
      poseRequest: mark("invalidate-pose"),
      navGoal: mark("invalidate-goal"),
    },
    runner: { stop: mark("runner-stop", Promise.resolve("runner stopped")) },
    navigation: { stop: mark("navigation-stop", Promise.resolve("navigation stopped")) },
    snapshot: { cancelAndClear: mark("clear-snapshot") },
    pose: {
      clearCache: mark("clear-pose-cache"),
      resetMappingSync: mark("reset-mapping-sync"),
    },
    commits: {
      setInteractionView: mark("interaction-view"),
      hideMap: mark("hide-map"),
      setStatusIdle: mark("status-idle"),
      setRuntimeModeIdle: mark("runtime-idle"),
      setDesignPoseReady: mark("design-pose"),
      setRunPoseReady: mark("run-pose"),
      setRuntimeOwned: mark("runtime-owned"),
      setShutdownPending: mark("shutdown-pending"),
    },
    session: { save: jest.fn((patch) => calls.push(["session", patch])) },
  };
  return { calls, ports };
}

test("invalidates first, clears the Run snapshot eagerly, and commits idle after both stops settle", async () => {
  const runnerStop = deferred();
  const navigationStop = deferred();
  const { calls, ports } = createPorts();
  ports.runner.stop.mockImplementation(() => {
    calls.push("runner-stop");
    return runnerStop.promise;
  });
  ports.navigation.stop.mockImplementation(() => {
    calls.push("navigation-stop");
    return navigationStop.promise;
  });
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  let stopping;
  act(() => { stopping = result.current(); });

  expect(calls).toEqual([
    "default-clear",
    "invalidate-map",
    "invalidate-pose",
    "runner-stop",
    "invalidate-goal",
    "interaction-view",
    "hide-map",
    "clear-snapshot",
    "run-pose",
    ["session", { runMissionName: "" }],
    "navigation-stop",
  ]);

  await act(async () => { navigationStop.resolve("stopped"); await Promise.resolve(); });
  expect(ports.pose.clearCache).not.toHaveBeenCalled();
  await act(async () => { runnerStop.resolve("runner done"); await stopping; });

  expect(calls.slice(-9, -1)).toEqual([
    "clear-pose-cache",
    "reset-mapping-sync",
    "status-idle",
    "runtime-idle",
    "design-pose",
    "run-pose",
    "runtime-owned",
    "shutdown-pending",
  ]);
  expect(ports.session.save).toHaveBeenLastCalledWith({
    navigationRuntimeMode: "idle",
    designPoseInitialized: false,
    runRuntimeOwned: false,
    runShutdownPending: false,
    runShutdownRequestedAt: null,
    runMissionName: "",
  });
  await expect(stopping).resolves.toBe("stopped");
});

test("ignores runner rejection but still returns a successful navigation result", async () => {
  const { ports } = createPorts({ defaultClear: false });
  ports.runner.stop.mockRejectedValue(new Error("runner failed"));
  ports.navigation.stop.mockResolvedValue("nav stopped");
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  await expect(result.current()).resolves.toBe("nav stopped");
  expect(ports.commits.setStatusIdle).toHaveBeenCalledTimes(1);
});

test("navigation rejection waits for runner settlement and skips every final idle commit", async () => {
  const runnerStop = deferred();
  const navigationError = new Error("navigation failed");
  const { ports } = createPorts({ defaultClear: false });
  ports.runner.stop.mockReturnValue(runnerStop.promise);
  ports.navigation.stop.mockRejectedValue(navigationError);
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  let settled = false;
  const stopping = result.current().finally(() => { settled = true; });
  await act(async () => { await Promise.resolve(); });
  expect(settled).toBe(false);
  expect(ports.pose.clearCache).not.toHaveBeenCalled();

  runnerStop.resolve();
  await expect(stopping).rejects.toBe(navigationError);
  expect(ports.pose.clearCache).not.toHaveBeenCalled();
  expect(ports.pose.resetMappingSync).not.toHaveBeenCalled();
  expect(ports.commits.setStatusIdle).not.toHaveBeenCalled();
  expect(ports.commits.setRuntimeModeIdle).not.toHaveBeenCalled();
  expect(ports.session.save).not.toHaveBeenCalled();
});

test("explicit false bypasses the default and preserves the loaded Run snapshot", async () => {
  const { ports } = createPorts({ defaultClear: true });
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  await act(async () => { await result.current({ clearRunSnapshot: false }); });

  expect(ports.getDefaultClearRunSnapshot).not.toHaveBeenCalled();
  expect(ports.commits.hideMap).not.toHaveBeenCalled();
  expect(ports.snapshot.cancelAndClear).not.toHaveBeenCalled();
  expect(ports.session.save).toHaveBeenCalledTimes(1);
  expect(ports.session.save).toHaveBeenLastCalledWith(expect.not.objectContaining({
    runMissionName: expect.anything(),
  }));
});

test("synchronous runner failure is ignored and navigation stop still starts", async () => {
  const { ports } = createPorts({ defaultClear: false });
  ports.runner.stop.mockImplementation(() => { throw new Error("sync runner failure"); });
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  await expect(result.current()).resolves.toBe("navigation stopped");
  expect(ports.navigation.stop).toHaveBeenCalledTimes(1);
});

test("explicit true eagerly clears Run state even when the default is false and navigation fails", async () => {
  const navigationError = new Error("navigation failed after eager clear");
  const { ports } = createPorts({ defaultClear: false });
  ports.navigation.stop.mockRejectedValue(navigationError);
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));

  const stopping = result.current({ clearRunSnapshot: true });

  expect(ports.getDefaultClearRunSnapshot).not.toHaveBeenCalled();
  expect(ports.commits.hideMap).toHaveBeenCalledTimes(1);
  expect(ports.snapshot.cancelAndClear).toHaveBeenCalledTimes(1);
  expect(ports.commits.setRunPoseReady).toHaveBeenCalledWith(false);
  expect(ports.session.save).toHaveBeenCalledTimes(1);
  expect(ports.session.save).toHaveBeenCalledWith({ runMissionName: "" });

  await expect(stopping).rejects.toBe(navigationError);
  expect(ports.commits.setStatusIdle).not.toHaveBeenCalled();
  expect(ports.commits.setRuntimeModeIdle).not.toHaveBeenCalled();
  expect(ports.pose.clearCache).not.toHaveBeenCalled();
  expect(ports.session.save).toHaveBeenCalledTimes(1);
});

test("stable stop callback reads replacement ports after rerender", async () => {
  const first = createPorts({ defaultClear: true }).ports;
  const second = createPorts({ defaultClear: false }).ports;
  second.navigation.stop.mockResolvedValue("replacement stopped");
  const view = renderHook((ports) => useNavigationStopCoordinator(ports), {
    initialProps: first,
  });
  const stableStop = view.result.current;

  view.rerender(second);

  expect(view.result.current).toBe(stableStop);
  await expect(stableStop()).resolves.toBe("replacement stopped");
  expect(first.getDefaultClearRunSnapshot).not.toHaveBeenCalled();
  expect(first.runner.stop).not.toHaveBeenCalled();
  expect(first.navigation.stop).not.toHaveBeenCalled();
  expect(first.commits.setStatusIdle).not.toHaveBeenCalled();
  expect(second.getDefaultClearRunSnapshot).toHaveBeenCalledTimes(1);
  expect(second.runner.stop).toHaveBeenCalledTimes(1);
  expect(second.navigation.stop).toHaveBeenCalledTimes(1);
  expect(second.commits.setStatusIdle).toHaveBeenCalledTimes(1);
  expect(second.snapshot.cancelAndClear).not.toHaveBeenCalled();
});

test('confirmed Collect motion stop skips cancel-all while shutting down Nav2 and clearing map state', async () => {
  const {ports} = createPorts();
  ports.runner.stop.mockImplementation(() => new Promise(() => {}));
  const {result} = renderHook(() => useNavigationStopCoordinator(ports));
  await act(async () => {
    await expect(result.current({runnerAlreadyStopped:true})).resolves.toBe('navigation stopped');
  });
  expect(ports.runner.stop).not.toHaveBeenCalled();
  expect(ports.navigation.stop).toHaveBeenCalledTimes(1);
  expect(ports.snapshot.cancelAndClear).toHaveBeenCalledTimes(1);
  expect(ports.commits.setRuntimeModeIdle).toHaveBeenCalledTimes(1);
  expect(ports.commits.setRunPoseReady).toHaveBeenLastCalledWith(false);
});

test('server autonomy Stop completes before browser actions, map cleanup, and navigation teardown', async () => {
  const serverStop = deferred();
  const { calls, ports } = createPorts();
  ports.autonomy = { needsStop: true, stop: jest.fn(() => serverStop.promise) };
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));
  let activeTab = 'operate';
  const changeTab = async () => { await result.current(); activeTab = 'record'; };
  const pending = changeTab();
  expect(ports.autonomy.stop).toHaveBeenCalledTimes(1);
  expect(calls).toEqual([]);
  expect(activeTab).toBe('operate');
  await act(async () => { serverStop.resolve(true); await pending; });
  expect(ports.runner.stop).toHaveBeenCalledTimes(1);
  expect(ports.navigation.stop).toHaveBeenCalledTimes(1);
  expect(activeTab).toBe('record');
});

test.each([false, new Error('ROS cancellation was not acknowledged')])(
  'unconfirmed server Stop preserves the current tab, loaded map, and navigation (%s)', async failure => {
    const { calls, ports } = createPorts();
    ports.autonomy = { needsStop: true, stop: jest.fn(() => failure instanceof Error
      ? Promise.reject(failure) : Promise.resolve(failure)) };
    const { result } = renderHook(() => useNavigationStopCoordinator(ports));
    let activeTab = 'operate';
    const changeTab = async () => { await result.current(); activeTab = 'record'; };
    await expect(changeTab()).rejects.toThrow();
    expect(activeTab).toBe('operate');
    expect(calls).toEqual([]);
    expect(ports.navigation.stop).not.toHaveBeenCalled();
    expect(ports.commits.hideMap).not.toHaveBeenCalled();
  },
);

test('ordinary Run teardown does not stop idle server autonomy', async () => {
  const { ports } = createPorts();
  ports.autonomy = { needsStop: false, stop: jest.fn() };
  const { result } = renderHook(() => useNavigationStopCoordinator(ports));
  await result.current();
  expect(ports.autonomy.stop).not.toHaveBeenCalled();
  expect(ports.runner.stop).toHaveBeenCalledTimes(1);
});
