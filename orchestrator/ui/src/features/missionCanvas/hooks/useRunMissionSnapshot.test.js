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
import { getNavigationMission } from "../../../utils/navigationMissionsApi";
import useRunMissionSnapshot, { emptyRunMissionSnapshot } from "./useRunMissionSnapshot";

jest.mock("../../../utils/navigationMissionsApi", () => ({
  getNavigationMission: jest.fn(),
}));

const loadLegacySpotsForMap = jest.fn();
const loadMissionBtFileOrDefault = jest.fn();

function renderSnapshotHook(options = {}) {
  return renderHook(() => useRunMissionSnapshot({
    loadLegacySpotsForMap,
    loadMissionBtFileOrDefault,
    ...options,
  }));
}

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

function waypoint(id, label = id, localBt = `locals/${id}/main.xml`) {
  return {
    id,
    label,
    pose: { frame_id: "map", x: 1, y: 2, yaw: 0.5 },
    local_bt: localBt,
    local_bt_files: [localBt, `locals/${id}/alternate.xml`],
  };
}

function manifest(id, options = {}) {
  return {
    exists: true,
    revision: options.revision ?? 4,
    global_bt: options.globalBt ?? "global.xml",
    waypoints: [waypoint(id)],
    metadata: options.metadata ?? {},
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  loadLegacySpotsForMap.mockResolvedValue([]);
  loadMissionBtFileOrDefault.mockImplementation(async (
    _mapName,
    _missionName,
    path,
  ) => `server:${path}`);
});

test("loads and commits one complete Run snapshot", async () => {
  getNavigationMission.mockResolvedValue(manifest("wp_1", {
    metadata: {
      mission_flow: {
        nodes: [{ id: "wp_1", position: { x: 32, y: 48 } }],
        edges: [],
      },
    },
  }));
  const view = renderSnapshotHook();

  let result;
  await act(async () => {
    result = await view.result.current.load("warehouse", "inspection", {
      catalogNames: ["inspection", "delivery"],
    });
  });

  expect(result).toEqual({ exists: true, loadedDesign: false, spotCount: 1 });
  expect(getNavigationMission).toHaveBeenCalledWith("warehouse", "inspection");
  expect(loadLegacySpotsForMap).not.toHaveBeenCalled();
  expect(loadMissionBtFileOrDefault.mock.calls.map((call) => call.slice(0, 3))).toEqual([
    ["warehouse", "inspection", "global.xml"],
    ["warehouse", "inspection", "locals/wp_1/main.xml"],
  ]);
  expect(view.result.current.snapshot).toMatchObject({
    mapName: "warehouse",
    missionName: "inspection",
    catalog: { mapName: "warehouse", names: ["inspection", "delivery"] },
    invalid: false,
    btFiles: {
      "global.xml": "server:global.xml",
      "locals/wp_1/main.xml": "server:locals/wp_1/main.xml",
    },
  });
  expect(view.result.current.snapshot.flowNodes[0].position).toEqual({ x: 32, y: 48 });
});

test("uses the default mission request name and generated BT fallbacks", async () => {
  getNavigationMission.mockResolvedValue(manifest("wp_default"));
  loadMissionBtFileOrDefault.mockImplementation(async (
    _mapName,
    _missionName,
    _path,
    fallback,
  ) => fallback);
  const view = renderSnapshotHook();

  await act(async () => {
    await view.result.current.load("warehouse", "default");
  });

  expect(getNavigationMission).toHaveBeenCalledWith("warehouse", "");
  expect(loadMissionBtFileOrDefault).toHaveBeenCalledWith(
    "warehouse",
    "default",
    "global.xml",
    expect.any(String),
    4,
  );
  expect(view.result.current.snapshot.btFiles["global.xml"]).toContain("GlobalMission");
  expect(view.result.current.snapshot.btFiles["locals/wp_default/main.xml"]).toContain(
    "BehaviorTree",
  );
});

test("loads legacy spots when no mission manifest exists", async () => {
  getNavigationMission.mockResolvedValue({ exists: false });
  loadLegacySpotsForMap.mockResolvedValue([
    {
      id: "legacy_2",
      map_name: "legacy_map",
      label: "B",
      pose: { frame_id: "map", x: 2, y: 0, yaw: 0 },
      linked_bt_tree: "legacy.xml",
    },
    {
      id: "legacy_1",
      map_name: "legacy_map",
      label: "A",
      pose: { frame_id: "map", x: 1, y: 0, yaw: 0 },
      linked_bt_tree: "first.xml",
    },
  ]);
  const view = renderSnapshotHook();

  let result;
  await act(async () => {
    result = await view.result.current.load("legacy_map", "default");
  });

  expect(result).toEqual({ exists: false, loadedDesign: false, spotCount: 2 });
  expect(loadLegacySpotsForMap).toHaveBeenCalledWith("legacy_map", { apply: false });
  expect(loadMissionBtFileOrDefault).not.toHaveBeenCalled();
  expect(view.result.current.snapshot.spots.map((spot) => spot.id)).toEqual([
    "legacy_2",
    "legacy_1",
  ]);
  expect(view.result.current.snapshot.flowNodes.map((node) => node.id)).toEqual([
    "legacy_1",
    "legacy_2",
  ]);
  expect(view.result.current.snapshot.btFiles).toEqual(expect.objectContaining({
    "global.xml": expect.stringContaining("GlobalMission"),
    "first.xml": expect.stringContaining("BehaviorTree"),
    "legacy.xml": expect.stringContaining("BehaviorTree"),
  }));
});

test("keeps the previous snapshot when one BT file fails revision validation", async () => {
  getNavigationMission.mockResolvedValueOnce(manifest("stable"));
  const view = renderSnapshotHook();
  await act(async () => {
    await view.result.current.load("map_a", "stable");
  });
  const stableSnapshot = view.result.current.snapshot;

  getNavigationMission.mockResolvedValueOnce(manifest("changed", { revision: 9 }));
  loadMissionBtFileOrDefault.mockRejectedValue(new Error(
    "Mission changed while loading global.xml; reload the mission before editing or running it",
  ));

  await expect(act(async () => {
    await view.result.current.load("map_b", "changed");
  })).rejects.toThrow(
    "Mission changed while loading global.xml; reload the mission before editing or running it",
  );
  expect(view.result.current.snapshot).toBe(stableSnapshot);
});

test("a slower load cannot overwrite a newer mission", async () => {
  const slowManifest = deferred();
  getNavigationMission.mockImplementation((mapName) => (
    mapName === "slow_map" ? slowManifest.promise : Promise.resolve(manifest("fast"))
  ));
  const view = renderSnapshotHook();

  let slowLoad;
  act(() => {
    slowLoad = view.result.current.load("slow_map", "slow");
  });
  await act(async () => {
    await view.result.current.load("fast_map", "fast");
  });
  slowManifest.resolve(manifest("slow"));

  let slowResult;
  await act(async () => {
    slowResult = await slowLoad;
  });

  expect(slowResult).toEqual({
    exists: true,
    loadedDesign: false,
    spotCount: 1,
    stale: true,
  });
  expect(view.result.current.snapshot.mapName).toBe("fast_map");
  expect(view.result.current.snapshot.missionName).toBe("fast");
  expect(view.result.current.snapshot.spots[0].id).toBe("fast");
});

test("clear resets state but intentionally does not cancel an in-flight load", async () => {
  const pendingManifest = deferred();
  getNavigationMission.mockReturnValue(pendingManifest.promise);
  const view = renderSnapshotHook();

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load("warehouse", "inspection");
    view.result.current.clear();
  });
  expect(view.result.current.snapshot).toEqual(emptyRunMissionSnapshot());

  pendingManifest.resolve(manifest("wp_1"));
  await act(async () => {
    await loadPromise;
  });

  expect(view.result.current.snapshot.invalid).toBe(false);
  expect(view.result.current.snapshot.missionName).toBe("inspection");
});

test("invalidate cancels an in-flight load without clearing the current snapshot", async () => {
  getNavigationMission.mockResolvedValueOnce(manifest("stable"));
  const view = renderSnapshotHook();
  await act(async () => {
    await view.result.current.load("stable-map", "stable");
  });
  const stableSnapshot = view.result.current.snapshot;

  const pendingManifest = deferred();
  getNavigationMission.mockReturnValueOnce(pendingManifest.promise);
  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load("late-map", "late");
    view.result.current.invalidate();
  });
  pendingManifest.resolve(manifest("late"));
  let result;
  await act(async () => {
    result = await loadPromise;
  });

  expect(result.stale).toBe(true);
  expect(view.result.current.snapshot).toBe(stableSnapshot);
});

test("cancelAndClear prevents a late load from repopulating the snapshot", async () => {
  const pendingManifest = deferred();
  getNavigationMission.mockReturnValue(pendingManifest.promise);
  const view = renderSnapshotHook();

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load("warehouse", "inspection");
    view.result.current.cancelAndClear();
  });
  pendingManifest.resolve(manifest("wp_1"));

  let result;
  await act(async () => {
    result = await loadPromise;
  });

  expect(result.stale).toBe(true);
  expect(view.result.current.snapshot).toEqual(emptyRunMissionSnapshot());
});

test("cancelAndClear makes a late rejected request inert", async () => {
  const pendingManifest = deferred();
  getNavigationMission.mockReturnValue(pendingManifest.promise);
  const view = renderSnapshotHook();

  let loadPromise;
  act(() => {
    loadPromise = view.result.current.load("warehouse", "inspection");
    view.result.current.cancelAndClear();
  });
  pendingManifest.reject(new Error("network lost"));

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
  expect(view.result.current.snapshot.invalid).toBe(true);
});
