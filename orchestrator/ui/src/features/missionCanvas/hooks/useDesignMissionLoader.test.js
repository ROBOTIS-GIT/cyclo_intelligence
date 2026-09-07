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
import { getNavigationMissionBtFile } from "../../../utils/navigationMissionsApi";
import useDesignMissionLoader, {
  loadDesignMissionBtFileOrDefault,
  loadDesignMissionSnapshot,
} from "./useDesignMissionLoader";

jest.mock("../../../utils/navigationMissionsApi", () => ({
  getNavigationMission: jest.fn(),
  getNavigationMissionBtFile: jest.fn(),
}));

jest.mock("../../../utils/navigationSpotsApi", () => ({
  getNavigationSpots: jest.fn(),
}));

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

function waypoint(id, options = {}) {
  const defaultPath = options.defaultPath ?? `locals/${id}/main.xml`;
  return {
    id,
    label: options.label ?? id,
    pose: options.pose ?? { frame_id: "map", x: 1, y: 2, yaw: 0.5 },
    local_bt: defaultPath,
    local_bt_files: options.localBtFiles ?? [defaultPath],
    metadata: options.metadata ?? {},
  };
}

function manifest(id, options = {}) {
  return {
    exists: true,
    revision: options.revision ?? 7,
    global_bt: options.globalBt ?? "global.xml",
    waypoints: options.waypoints ?? [waypoint(id)],
    metadata: options.metadata ?? {},
  };
}

function loaderDependencies(overrides = {}) {
  return {
    getMission: jest.fn().mockResolvedValue(manifest("wp_1")),
    loadBtFile: jest.fn(async (_mapName, _missionName, path) => `server:${path}`),
    loadLegacySpots: jest.fn().mockResolvedValue([]),
    loadLegacyBehaviorNodes: jest.fn().mockReturnValue(null),
    ...overrides,
  };
}

function renderLoader(dependencies = loaderDependencies()) {
  return {
    dependencies,
    view: renderHook(() => useDesignMissionLoader(dependencies)),
  };
}

beforeEach(() => {
  jest.clearAllMocks();
});

test("assembles an existing Design mission and every authoring BT as one snapshot", async () => {
  const getMission = jest.fn().mockResolvedValue(manifest("wp_1", {
    globalBt: "mission.xml",
    waypoints: [waypoint("wp_1", {
      localBtFiles: ["locals/wp_1/main.xml", "locals/wp_1/alternate.xml"],
    })],
    metadata: {
      mission_flow: {
        nodes: [{ id: "wp_1", position: { x: 42, y: 84 } }],
        edges: [],
      },
    },
  }));
  const loadBtFile = jest.fn(async (_mapName, _missionName, path) => `server:${path}`);

  const result = await loadDesignMissionSnapshot(
    { mapName: " warehouse ", missionName: " inspection " },
    { getMission, loadBtFile },
  );

  expect(getMission).toHaveBeenCalledWith("warehouse", "inspection");
  expect(loadBtFile.mock.calls.map((call) => call.slice(0, 3))).toEqual([
    ["warehouse", "inspection", "global.xml"],
    ["warehouse", "inspection", "locals/wp_1/main.xml"],
    ["warehouse", "inspection", "locals/wp_1/alternate.xml"],
    ["warehouse", "inspection", "mission.xml"],
  ]);
  expect(loadBtFile.mock.calls.every((call) => call[4] === 7)).toBe(true);
  expect(result).toMatchObject({
    exists: true,
    loadedDesign: false,
    spotCount: 1,
    snapshot: {
      identity: { mapName: "warehouse", missionName: "inspection" },
      behaviorNodesPatch: null,
      flowNodes: [{ id: "wp_1", position: { x: 42, y: 84 } }],
      btFiles: {
        "global.xml": "server:global.xml",
        "mission.xml": "server:mission.xml",
        "locals/wp_1/main.xml": "server:locals/wp_1/main.xml",
        "locals/wp_1/alternate.xml": "server:locals/wp_1/alternate.xml",
      },
      baseline: {
        revision: 7,
        persistedLocalBtPaths: [
          "locals/wp_1/main.xml",
          "locals/wp_1/alternate.xml",
        ],
      },
    },
  });
  expect(result.snapshot.baseline.persistedBtFiles).toEqual(result.snapshot.btFiles);
  expect(result.snapshot.baseline.persistedBtFiles).not.toBe(result.snapshot.btFiles);
});

test("uses the default mission request name and falls back only for a missing BT file", async () => {
  getNavigationMissionBtFile.mockResolvedValue({
    exists: false,
    revision: 3,
  });

  await expect(loadDesignMissionBtFileOrDefault(
    "warehouse",
    "default",
    "locals/wp/main.xml",
    "fallback xml",
    3,
  )).resolves.toBe("fallback xml");

  expect(getNavigationMissionBtFile).toHaveBeenCalledWith(
    "warehouse",
    "locals/wp/main.xml",
    "",
  );
});

test("rejects a BT response from a different manifest revision", async () => {
  getNavigationMissionBtFile.mockResolvedValue({
    exists: true,
    content: "newer xml",
    revision: 8,
  });

  await expect(loadDesignMissionBtFileOrDefault(
    "warehouse",
    "inspection",
    "global.xml",
    "fallback xml",
    7,
  )).rejects.toThrow(
    "Mission changed while loading global.xml; reload the mission before editing or running it",
  );
});

test("assembles a missing mission from legacy spots without marking defaults as persisted", async () => {
  const legacySpots = [
    {
      id: "legacy_b",
      map_name: "warehouse",
      label: "B",
      pose: { frame_id: "map", x: 2, y: 0, yaw: 0 },
      linked_bt_tree: "legacy_b.xml",
    },
    {
      id: "legacy_a",
      map_name: "warehouse",
      label: "A",
      pose: { frame_id: "map", x: 1, y: 0, yaw: 0 },
      linked_bt_tree: "legacy_a.xml",
    },
  ];
  const savedNodes = [{ id: "behavior_1", map_name: "warehouse", tag: "Rotate" }];
  const dependencies = loaderDependencies({
    getMission: jest.fn().mockResolvedValue({ exists: false, revision: 4 }),
    loadLegacySpots: jest.fn().mockResolvedValue({ spots: legacySpots }),
    loadLegacyBehaviorNodes: jest.fn().mockReturnValue(savedNodes),
  });

  const result = await loadDesignMissionSnapshot(
    { mapName: "warehouse", missionName: "default", loadLegacyDesign: true },
    dependencies,
  );

  expect(dependencies.getMission).toHaveBeenCalledWith("warehouse", "");
  expect(dependencies.loadLegacySpots).toHaveBeenCalledWith("warehouse", { apply: false });
  expect(dependencies.loadBtFile).not.toHaveBeenCalled();
  expect(result.loadedDesign).toBe(true);
  expect(result.snapshot.spots).toEqual(legacySpots);
  expect(result.snapshot.flowNodes.map((node) => node.id)).toEqual([
    "legacy_a",
    "legacy_b",
  ]);
  expect(result.snapshot.behaviorNodesPatch).toBe(savedNodes);
  expect(result.snapshot.btFiles).toEqual(expect.objectContaining({
    "global.xml": expect.stringContaining("GlobalMission"),
    "legacy_a.xml": expect.stringContaining("BehaviorTree"),
    "legacy_b.xml": expect.stringContaining("BehaviorTree"),
  }));
  expect(result.snapshot.baseline).toEqual({
    revision: 4,
    persistedBtFiles: {},
    persistedLocalBtPaths: [],
  });
});

test("does not read or clear legacy behavior nodes unless legacy Design loading is requested", async () => {
  const dependencies = loaderDependencies({
    getMission: jest.fn().mockResolvedValue({ exists: false }),
    loadLegacyBehaviorNodes: jest.fn().mockReturnValue([]),
  });

  const result = await loadDesignMissionSnapshot(
    { mapName: "warehouse", missionName: "draft" },
    dependencies,
  );

  expect(dependencies.loadLegacyBehaviorNodes).not.toHaveBeenCalled();
  expect(result.loadedDesign).toBe(false);
  expect(result.snapshot.behaviorNodesPatch).toBeNull();
});

test("treats an asynchronously loaded empty legacy behavior list as an intentional patch", async () => {
  const dependencies = loaderDependencies({
    getMission: jest.fn().mockResolvedValue({ exists: false }),
    loadLegacyBehaviorNodes: jest.fn().mockResolvedValue([]),
  });

  const result = await loadDesignMissionSnapshot(
    { mapName: "warehouse", missionName: "draft", loadLegacyDesign: true },
    dependencies,
  );

  expect(result.loadedDesign).toBe(true);
  expect(result.snapshot.behaviorNodesPatch).toEqual([]);
});

test("does not expose a partial snapshot when one required BT fails", async () => {
  const callerDocument = { id: "previous" };
  const dependencies = loaderDependencies({
    loadBtFile: jest.fn(async (_mapName, _missionName, path) => {
      if (path.endsWith("main.xml")) throw new Error("BT read failed");
      return `server:${path}`;
    }),
  });

  let nextDocument = callerDocument;
  const loadPromise = loadDesignMissionSnapshot(
      { mapName: "warehouse", missionName: "inspection" },
      dependencies,
    ).then((result) => {
    nextDocument = result.snapshot;
    return result;
  });

  await expect(loadPromise).rejects.toThrow("BT read failed");
  expect(nextDocument).toBe(callerDocument);
});

test("the latest Design load wins when an older map resolves later", async () => {
  const slow = deferred();
  const dependencies = loaderDependencies({
    getMission: jest.fn((mapName) => (
      mapName === "slow_map" ? slow.promise : Promise.resolve(manifest("fast_wp"))
    )),
  });
  const { view } = renderLoader(dependencies);

  let slowPromise;
  act(() => {
    slowPromise = view.result.current.load({ mapName: "slow_map", missionName: "slow" });
  });
  await act(async () => {
    await view.result.current.load({ mapName: "fast_map", missionName: "fast" });
  });
  slow.resolve(manifest("slow_wp"));

  let slowResult;
  await act(async () => {
    slowResult = await slowPromise;
  });

  expect(slowResult).toEqual({
    exists: true,
    loadedDesign: false,
    spotCount: 1,
    stale: true,
  });
  expect(slowResult).not.toHaveProperty("snapshot");
  expect(view.result.current.identity).toEqual({ mapName: "fast_map", missionName: "fast" });
  expect(view.result.current.phase).toBe("idle");
});

test("a same-identity reload invalidates the older request", async () => {
  const first = deferred();
  const getMission = jest.fn()
    .mockReturnValueOnce(first.promise)
    .mockResolvedValueOnce(manifest("fresh_wp"));
  const { view } = renderLoader(loaderDependencies({ getMission }));

  let firstPromise;
  act(() => {
    firstPromise = view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
  });
  let secondResult;
  await act(async () => {
    secondResult = await view.result.current.load({
      mapName: "warehouse",
      missionName: "inspection",
    });
  });
  first.resolve(manifest("old_wp"));

  let firstResult;
  await act(async () => {
    firstResult = await firstPromise;
  });

  expect(secondResult.snapshot.spots[0].id).toBe("fresh_wp");
  expect(firstResult).toMatchObject({ stale: true });
  expect(firstResult).not.toHaveProperty("snapshot");
});

test("a rejection from an invalidated request is stale and cannot replace the current error state", async () => {
  const first = deferred();
  const getMission = jest.fn()
    .mockReturnValueOnce(first.promise)
    .mockResolvedValueOnce(manifest("fresh_wp"));
  const { view } = renderLoader(loaderDependencies({ getMission }));

  let firstPromise;
  act(() => {
    firstPromise = view.result.current.load({ mapName: "warehouse", missionName: "first" });
  });
  await act(async () => {
    await view.result.current.load({ mapName: "warehouse", missionName: "second" });
  });
  first.reject(new Error("late failure"));

  let firstResult;
  await act(async () => {
    firstResult = await firstPromise;
  });

  expect(firstResult).toEqual({
    exists: false,
    loadedDesign: false,
    spotCount: 0,
    stale: true,
  });
  expect(view.result.current.phase).toBe("idle");
  expect(view.result.current.error).toBeNull();
  expect(view.result.current.identity.missionName).toBe("second");
});

test("a stale success cannot clear the newer request's error", async () => {
  const first = deferred();
  const currentFailure = new Error("second failed");
  const getMission = jest.fn()
    .mockReturnValueOnce(first.promise)
    .mockRejectedValueOnce(currentFailure);
  const { view } = renderLoader(loaderDependencies({ getMission }));

  let firstPromise;
  act(() => {
    firstPromise = view.result.current.load({ mapName: "warehouse", missionName: "first" });
  });
  await act(async () => {
    try {
      await view.result.current.load({ mapName: "warehouse", missionName: "second" });
    } catch {
      // The state assertion below verifies that this current failure is retained.
    }
  });
  first.resolve(manifest("old_wp"));

  let firstResult;
  await act(async () => {
    firstResult = await firstPromise;
  });

  expect(firstResult).toMatchObject({ stale: true });
  expect(view.result.current.phase).toBe("error");
  expect(view.result.current.error).toBe(currentFailure);
  expect(view.result.current.identity.missionName).toBe("second");
});

test("a current failure is reported without publishing a snapshot", async () => {
  const failure = new Error("manifest unavailable");
  const { view } = renderLoader(loaderDependencies({
    getMission: jest.fn().mockRejectedValue(failure),
  }));

  let caught;
  await act(async () => {
    try {
      await view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
    } catch (error) {
      caught = error;
    }
  });

  expect(caught).toBe(failure);
  expect(view.result.current.phase).toBe("error");
  expect(view.result.current.error).toBe(failure);
  act(() => view.result.current.clearError());
  expect(view.result.current.phase).toBe("idle");
  expect(view.result.current.error).toBeNull();
});

test("invalidate revokes existing leases and prevents a pending load from committing", async () => {
  const pending = deferred();
  const { view } = renderLoader(loaderDependencies({
    getMission: jest.fn().mockReturnValue(pending.promise),
  }));

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
  });
  const loadLease = view.result.current.captureLease();
  expect(view.result.current.isCurrent(loadLease)).toBe(true);

  let nextLease;
  act(() => {
    nextLease = view.result.current.invalidate({
      mapName: "warehouse",
      missionName: "inspection",
    });
  });
  expect(view.result.current.isCurrent(loadLease)).toBe(false);
  expect(view.result.current.isCurrent(nextLease)).toBe(true);
  pending.resolve(manifest("late_wp"));

  let result;
  await act(async () => {
    result = await loadPromise;
  });

  expect(result).toMatchObject({ stale: true });
  expect(result).not.toHaveProperty("snapshot");
});

test("a successful result lease must still be current at the caller's commit point", async () => {
  const { view } = renderLoader();
  let result;
  await act(async () => {
    result = await view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
  });

  let callerDocument = { id: "previous" };
  act(() => {
    view.result.current.invalidate({ mapName: "warehouse", missionName: "replacement" });
  });
  if (view.result.current.isCurrent(result.lease)) {
    callerDocument = result.snapshot;
  }

  expect(view.result.current.isCurrent(result.lease)).toBe(false);
  expect(callerDocument).toEqual({ id: "previous" });
});

test("unmount invalidates an in-flight load without a late state update", async () => {
  const pending = deferred();
  const { view } = renderLoader(loaderDependencies({
    getMission: jest.fn().mockReturnValue(pending.promise),
  }));

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
  });
  const lease = view.result.current.captureLease();
  view.unmount();
  expect(view.result.current.isCurrent(lease)).toBe(false);

  pending.resolve(manifest("late_wp"));
  let result;
  await act(async () => {
    result = await loadPromise;
  });

  expect(result).toMatchObject({ stale: true });
  expect(result).not.toHaveProperty("snapshot");
});

test("a rejection after unmount is also reduced to a stale result", async () => {
  const pending = deferred();
  const { view } = renderLoader(loaderDependencies({
    getMission: jest.fn().mockReturnValue(pending.promise),
  }));

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load({ mapName: "warehouse", missionName: "inspection" });
  });
  view.unmount();
  pending.reject(new Error("late failure"));

  let result;
  await act(async () => {
    result = await loadPromise;
  });

  expect(result).toEqual({
    exists: false,
    loadedDesign: false,
    spotCount: 0,
    stale: true,
  });
});
