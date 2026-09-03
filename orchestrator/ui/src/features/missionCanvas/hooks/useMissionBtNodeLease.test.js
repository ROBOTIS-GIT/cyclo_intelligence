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

import { act, renderHook, waitFor } from "@testing-library/react";
import { getBtNodeServiceStatus, setBtNodeServiceActive } from "../lib/btNodeApi";
import useMissionBtNodeLease from "./useMissionBtNodeLease";

jest.mock("../lib/btNodeApi", () => ({
  getBtNodeServiceStatus: jest.fn(),
  setBtNodeServiceActive: jest.fn(),
}));

const callService = jest.fn();
const onMessage = jest.fn();

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

function renderLease(options = {}) {
  const { btStatusText, ...hookOptions } = options;
  const initialProps = {
    needsBtTopics: false,
    ...hookOptions,
  };
  const view = renderHook((props) => useMissionBtNodeLease({
    callService,
    onMessage,
    ...props,
  }), { initialProps });
  if (btStatusText !== undefined) {
    act(() => {
      view.result.current.setBtStatusText(btStatusText);
    });
  }
  return view;
}

function setDocumentVisibility(value) {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value,
  });
}

beforeEach(() => {
  jest.clearAllMocks();
  jest.useRealTimers();
  setDocumentVisibility("visible");
  callService.mockResolvedValue({ success: true });
  setBtNodeServiceActive.mockResolvedValue({ ok: true });
});

afterEach(() => {
  jest.restoreAllMocks();
  jest.useRealTimers();
});

test("borrows an idle node and does not stop it when the lease is released", async () => {
  getBtNodeServiceStatus.mockResolvedValue({ state: "up", raw: "up" });
  const view = renderLease();

  let ready;
  await act(async () => {
    ready = await view.result.current.ensureMissionBtActive();
  });

  expect(ready).toBe(true);
  expect(callService).toHaveBeenCalledWith(
    "/bt/nodes/catalog",
    "interfaces/srv/GetNodeCatalog",
    {},
    1000,
  );
  expect(setBtNodeServiceActive).not.toHaveBeenCalled();

  await act(async () => {
    await view.result.current.releaseMissionBt();
  });

  expect(setBtNodeServiceActive).not.toHaveBeenCalled();
  expect(view.result.current.btNodeStatus).toEqual({ state: "up", raw: "up" });
});

test.each(["running", "stopping"])(
  "fails closed when a borrowed node reports execution status %s",
  async (btStatusText) => {
    getBtNodeServiceStatus.mockResolvedValue({ state: "up", raw: "up" });
    const view = renderLease({ btStatusText });

    let ready;
    await act(async () => {
      ready = await view.result.current.ensureMissionBtActive();
    });

    expect(ready).toBe(false);
    expect(callService).not.toHaveBeenCalled();
    expect(setBtNodeServiceActive).not.toHaveBeenCalled();
    expect(onMessage).toHaveBeenCalledWith(
      "Task Engine is already running another task. Stop it before running this mission.",
    );
  },
);

test("fails closed when a borrowed node has no recognized idle execution status", async () => {
  getBtNodeServiceStatus.mockResolvedValue({ state: "up", raw: "up" });
  const view = renderLease({ btStatusText: "unknown" });

  let ready;
  await act(async () => {
    ready = await view.result.current.ensureMissionBtActive();
  });

  expect(ready).toBe(false);
  expect(callService).not.toHaveBeenCalled();
  expect(onMessage).toHaveBeenCalledWith(
    "Unable to verify that the Task Engine is idle. Wait for its status and try again.",
  );
});

test("rejects activation when the supervisor state cannot be verified", async () => {
  getBtNodeServiceStatus.mockRejectedValue(new Error("offline"));
  const view = renderLease();

  let ready;
  await act(async () => {
    ready = await view.result.current.ensureMissionBtActive();
  });

  expect(ready).toBe(false);
  expect(setBtNodeServiceActive).not.toHaveBeenCalled();
  expect(onMessage).toHaveBeenCalledWith(
    "Unable to verify the Task Engine state. Try again after its status is available.",
  );
});

test("starts a down node, waits for ROS readiness, and stops the owned node", async () => {
  getBtNodeServiceStatus
    .mockResolvedValueOnce({ state: "down", raw: "down" })
    .mockResolvedValueOnce({ state: "up", raw: "up" })
    .mockResolvedValueOnce({ state: "down", raw: "down" });
  const view = renderLease();

  let ready;
  await act(async () => {
    ready = await view.result.current.ensureMissionBtActive();
  });

  expect(ready).toBe(true);
  expect(setBtNodeServiceActive).toHaveBeenNthCalledWith(1, true);
  expect(view.result.current.btNodeIsUp).toBe(true);

  await act(async () => {
    await view.result.current.releaseMissionBt();
  });

  expect(setBtNodeServiceActive).toHaveBeenNthCalledWith(2, false);
  expect(view.result.current.btNodeBusy).toBe("");
  expect(view.result.current.btNodeStatus).toEqual({ state: "down", raw: "down" });
});

test("requires both an up process and a successful ROS readiness probe", async () => {
  getBtNodeServiceStatus.mockResolvedValue({ state: "up", raw: "up" });
  callService.mockResolvedValue({ success: false });
  const timeout = jest.spyOn(window, "setTimeout").mockImplementation((callback) => {
    callback();
    return 1;
  });
  const view = renderLease();

  let ready;
  await act(async () => {
    ready = await view.result.current.ensureMissionBtActive();
  });

  expect(ready).toBe(false);
  expect(callService).toHaveBeenCalledTimes(10);
  expect(timeout).toHaveBeenCalledTimes(10);
  expect(setBtNodeServiceActive).not.toHaveBeenCalled();
});

test("serializes a new activation behind an owned-node release", async () => {
  let serviceState = "down";
  const stopRequest = deferred();
  getBtNodeServiceStatus.mockImplementation(async () => ({
    state: serviceState,
    raw: serviceState,
  }));
  setBtNodeServiceActive.mockImplementation(async (active) => {
    if (active) {
      serviceState = "up";
      return { ok: true };
    }
    await stopRequest.promise;
    serviceState = "down";
    return { ok: true };
  });
  const view = renderLease();

  await act(async () => {
    expect(await view.result.current.ensureMissionBtActive()).toBe(true);
  });

  let release;
  act(() => {
    release = view.result.current.releaseMissionBt();
  });
  await waitFor(() => {
    expect(setBtNodeServiceActive).toHaveBeenLastCalledWith(false);
  });
  const readsBeforeRetry = getBtNodeServiceStatus.mock.calls.length;

  let retry;
  act(() => {
    retry = view.result.current.ensureMissionBtActive();
  });
  await act(async () => {
    await Promise.resolve();
  });
  expect(getBtNodeServiceStatus).toHaveBeenCalledTimes(readsBeforeRetry);

  await act(async () => {
    stopRequest.resolve();
    await release;
    expect(await retry).toBe(true);
  });

  expect(setBtNodeServiceActive.mock.calls.map(([active]) => active)).toEqual([
    true,
    false,
    true,
  ]);
});

test("treats owned-node stop as best effort and still refreshes status", async () => {
  getBtNodeServiceStatus
    .mockResolvedValueOnce({ state: "down", raw: "down" })
    .mockResolvedValueOnce({ state: "up", raw: "up" })
    .mockRejectedValueOnce(new Error("status unavailable"));
  setBtNodeServiceActive
    .mockResolvedValueOnce({ ok: true })
    .mockRejectedValueOnce(new Error("already stopped"));
  const view = renderLease();

  await act(async () => {
    expect(await view.result.current.ensureMissionBtActive()).toBe(true);
    await view.result.current.releaseMissionBt();
  });

  expect(setBtNodeServiceActive).toHaveBeenLastCalledWith(false);
  expect(view.result.current.btNodeStatus).toEqual({
    state: "unknown",
    raw: "status failed",
  });
});

test("polls supervisor status immediately and every five seconds while BT topics are needed", async () => {
  jest.useFakeTimers();
  getBtNodeServiceStatus.mockResolvedValue({ state: "up", raw: "up" });
  const view = renderLease({ needsBtTopics: true });

  await act(async () => {
    await Promise.resolve();
  });
  expect(getBtNodeServiceStatus).toHaveBeenCalledTimes(1);

  act(() => {
    jest.advanceTimersByTime(5000);
  });
  await act(async () => {
    await Promise.resolve();
  });
  expect(getBtNodeServiceStatus).toHaveBeenCalledTimes(2);

  view.unmount();
  act(() => {
    jest.advanceTimersByTime(10000);
  });
  expect(getBtNodeServiceStatus).toHaveBeenCalledTimes(2);
});

test("does not poll while the document is hidden", async () => {
  setDocumentVisibility("hidden");
  const view = renderLease({ needsBtTopics: true });

  await act(async () => {
    await Promise.resolve();
  });

  expect(getBtNodeServiceStatus).not.toHaveBeenCalled();
  view.unmount();
});

test("reports a non-quiet status failure using Task Engine terminology", async () => {
  getBtNodeServiceStatus.mockRejectedValue(new Error("BT node unavailable"));
  const view = renderLease();

  let status;
  await act(async () => {
    status = await view.result.current.refreshBtNodeStatus();
  });

  expect(status).toEqual({ state: "unknown", raw: "BT node unavailable" });
  expect(onMessage).toHaveBeenCalledWith(
    "Task Engine status check failed: Task Engine unavailable",
  );
});

test("does not stop an owned node merely because the hook unmounts", async () => {
  getBtNodeServiceStatus
    .mockResolvedValueOnce({ state: "down", raw: "down" })
    .mockResolvedValueOnce({ state: "up", raw: "up" });
  const view = renderLease();

  await act(async () => {
    expect(await view.result.current.ensureMissionBtActive()).toBe(true);
  });
  view.unmount();

  expect(setBtNodeServiceActive.mock.calls.map(([active]) => active)).toEqual([true]);
});
