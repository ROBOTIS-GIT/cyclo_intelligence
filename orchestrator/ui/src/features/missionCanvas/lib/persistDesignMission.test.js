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
import useDesignMissionDocumentLedger from "../hooks/useDesignMissionDocumentLedger";
import persistDesignMission from "./persistDesignMission";

const MAP_NAME = "warehouse";
const MISSION_NAME = "inspection";
const GLOBAL_PATH = "global.xml";
const LOCAL_PATH = "locals/waypoint_1/main.xml";
const ORPHAN_PATH = "locals/orphan/main.xml";

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, reject, resolve };
}

function waypoint(overrides = {}) {
  return {
    id: "waypoint_1",
    map_name: MAP_NAME,
    label: "Waypoint 1",
    _missionManifest: true,
    pose: { frame_id: "map", x: 1, y: 2, yaw: 0.5 },
    linked_bt_tree: LOCAL_PATH,
    local_bt_files: [LOCAL_PATH],
    metadata: {
      source: "mission_manifest",
      coordinate_space: "map",
      local_bt: LOCAL_PATH,
      local_bt_files: [LOCAL_PATH],
    },
    ...overrides,
  };
}

function renderLedger({
  files = {
    [GLOBAL_PATH]: "global-old",
    [LOCAL_PATH]: "local-old",
  },
  localPaths = [LOCAL_PATH],
  revision = 7,
} = {}) {
  const view = renderHook(() => useDesignMissionDocumentLedger());
  act(() => {
    view.result.current.commitSnapshot({
      btFiles: files,
      baseline: {
        persistedBtFiles: files,
        persistedLocalBtPaths: localPaths,
        revision,
      },
    });
  });
  return view;
}

function request(view, overrides = {}) {
  const spot = waypoint();
  return {
    mapName: MAP_NAME,
    targetMissionName: MISSION_NAME,
    targetKnown: true,
    visibleSpots: [spot],
    behaviorNodes: [{ id: "behavior_1", map_name: MAP_NAME, tag: "Rotate" }],
    missionFlowNodes: [{
      id: spot.id,
      type: "missionWaypoint",
      position: { x: 80, y: 72 },
      data: { label: spot.label, localBt: LOCAL_PATH },
    }],
    missionFlowEdges: [],
    historyAtStart: {
      spots: [spot],
      missionFlowNodes: [],
      missionFlowEdges: [],
      selectedSpotId: spot.id,
      ...view.result.current.getHistorySlice(),
    },
    ledger: view.result.current,
    ...overrides,
  };
}

const getMission = jest.fn();
const saveBtFile = jest.fn();
const saveMission = jest.fn();
const deleteBtFile = jest.fn();
const saveBehaviorNodes = jest.fn();

function dependencies() {
  return {
    getMission,
    saveBtFile,
    saveMission,
    deleteBtFile,
    saveBehaviorNodes,
  };
}

beforeEach(() => {
  jest.clearAllMocks();
  getMission.mockResolvedValue({ exists: false, revision: 0 });
  saveBtFile.mockResolvedValue({});
  saveMission.mockResolvedValue({});
  deleteBtFile.mockResolvedValue({});
});

test("persists one canonical mission through a sequential revision chain", async () => {
  const view = renderLedger();
  saveBtFile
    .mockResolvedValueOnce({ revision: 8 })
    .mockResolvedValueOnce({ revision: 9 });
  saveMission.mockResolvedValue({ revision: 10 });

  let result;
  await act(async () => {
    result = await persistDesignMission(request(view), dependencies());
  });

  expect(getMission).not.toHaveBeenCalled();
  expect(saveBehaviorNodes).toHaveBeenCalledWith(
    MAP_NAME,
    [{ id: "behavior_1", map_name: MAP_NAME, tag: "Rotate" }],
  );
  expect(saveBtFile).toHaveBeenNthCalledWith(
    1,
    MAP_NAME,
    GLOBAL_PATH,
    expect.stringContaining("GlobalMission"),
    MISSION_NAME,
    { expectedRevision: 7 },
  );
  expect(saveBtFile).toHaveBeenNthCalledWith(
    2,
    MAP_NAME,
    LOCAL_PATH,
    "local-old",
    MISSION_NAME,
    { waypointId: "waypoint_1", expectedRevision: 8 },
  );
  expect(saveMission).toHaveBeenCalledWith(
    MAP_NAME,
    expect.objectContaining({
      expected_revision: 9,
      global_bt: GLOBAL_PATH,
      waypoints: [expect.objectContaining({
        id: "waypoint_1",
        local_bt: LOCAL_PATH,
      })],
      metadata: expect.objectContaining({
        source: "mission_canvas",
        behavior_node_count: 1,
      }),
    }),
    MISSION_NAME,
  );
  expect(deleteBtFile).not.toHaveBeenCalled();
  expect(result.canonicalMissionSpots[0].linked_bt_tree).toBe(LOCAL_PATH);
  expect(result.syncedMissionFlowNodes[0].id).toBe("waypoint_1");
  expect(result.syncedMissionFlowEdges).toEqual([]);
  expect(JSON.parse(result.savedHistorySnapshot)).toEqual(expect.objectContaining({
    selectedSpotId: "waypoint_1",
    missionBtFiles: expect.objectContaining({ [LOCAL_PATH]: "local-old" }),
    deletedMissionBtPaths: [],
    designDirty: false,
    nonBtDesignDirty: false,
  }));
  expect(result.saveResult).toMatchObject({
    hasNewerEdits: false,
    revision: 10,
  });
  expect(view.result.current.getPersistedRevision()).toBe(10);
  expect(view.result.current.hasActiveSave()).toBe(false);
});

test("uses a missing target's tombstone revision before the first upload", async () => {
  const view = renderLedger({ revision: 3 });
  getMission.mockResolvedValue({ exists: false, revision: 21 });
  saveBtFile
    .mockResolvedValueOnce({ revision: 22 })
    .mockResolvedValueOnce({ revision: 23 });
  saveMission.mockResolvedValue({ revision: 24 });

  await act(async () => {
    await persistDesignMission(request(view, {
      targetMissionName: "reused-name",
      targetKnown: false,
    }), dependencies());
  });

  expect(getMission).toHaveBeenCalledWith(MAP_NAME, "reused-name");
  expect(saveBtFile.mock.calls[0][4]).toEqual({ expectedRevision: 21 });
  expect(view.result.current.getPersistedRevision()).toBe(24);
});

test("rejects an occupied target and releases the save transaction", async () => {
  const view = renderLedger();
  getMission.mockResolvedValue({ exists: true, revision: 30 });
  let failure;

  await act(async () => {
    try {
      await persistDesignMission(request(view, {
        targetMissionName: "already-there",
        targetKnown: false,
      }), dependencies());
    } catch (error) {
      failure = error;
    }
  });

  expect(failure).toEqual(new Error(
    "Mission already-there already exists. Reload it before saving.",
  ));
  expect(saveBehaviorNodes).not.toHaveBeenCalled();
  expect(saveBtFile).not.toHaveBeenCalled();
  expect(view.result.current.hasActiveSave()).toBe(false);
  expect(view.result.current.getPersistedRevision()).toBe(7);
});

test("commits the manifest before pruning stale BT files", async () => {
  const files = {
    [GLOBAL_PATH]: "global-old",
    [LOCAL_PATH]: "local-old",
    [ORPHAN_PATH]: "orphan-old",
  };
  const view = renderLedger({
    files,
    localPaths: [LOCAL_PATH, ORPHAN_PATH],
  });
  act(() => {
    view.result.current.replaceDeletedBtPaths([ORPHAN_PATH]);
  });
  saveBtFile
    .mockResolvedValueOnce({ revision: 8 })
    .mockResolvedValueOnce({ revision: 9 });
  saveMission.mockResolvedValue({ revision: 10 });
  deleteBtFile.mockResolvedValue({ revision: 11 });

  let result;
  await act(async () => {
    result = await persistDesignMission(request(view), dependencies());
  });

  expect(saveMission).toHaveBeenCalledTimes(1);
  expect(deleteBtFile).toHaveBeenCalledWith(
    MAP_NAME,
    ORPHAN_PATH,
    MISSION_NAME,
    { expectedRevision: 10 },
  );
  expect(view.result.current.getPersistedRevision()).toBe(11);
  expect(view.result.current.getPersistedBtFiles()).not.toHaveProperty(ORPHAN_PATH);
  expect(view.result.current.deletedMissionBtPaths).toEqual([]);
  expect(result.saveResult.hasNewerEdits).toBe(false);
});

test("retains an upload checkpoint and retries from its revision after failure", async () => {
  const view = renderLedger();
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "local-edited");
  });
  saveBtFile
    .mockResolvedValueOnce({ revision: 8 })
    .mockRejectedValueOnce(new Error("second upload failed"));
  let failure;

  await act(async () => {
    try {
      await persistDesignMission(request(view), dependencies());
    } catch (error) {
      failure = error;
    }
  });

  expect(failure).toEqual(new Error("second upload failed"));
  expect(view.result.current.hasActiveSave()).toBe(false);
  expect(view.result.current.getPersistedRevision()).toBe(8);
  expect(view.result.current.getPersistedBtFiles()[GLOBAL_PATH]).toContain(
    "GlobalMission",
  );
  expect(view.result.current.designDirty).toBe(true);

  saveBtFile.mockReset();
  saveBtFile
    .mockResolvedValueOnce({ revision: 9 })
    .mockResolvedValueOnce({ revision: 10 });
  saveMission.mockResolvedValue({ revision: 11 });
  await act(async () => {
    await persistDesignMission(request(view), dependencies());
  });

  expect(saveBtFile.mock.calls[0][4]).toEqual({ expectedRevision: 8 });
  expect(view.result.current.getPersistedRevision()).toBe(11);
  expect(view.result.current.designDirty).toBe(false);
});

test("preserves a newer editor value while an older snapshot is saving", async () => {
  const view = renderLedger();
  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "save-snapshot");
  });
  const firstUpload = deferred();
  saveBtFile
    .mockImplementationOnce(() => firstUpload.promise)
    .mockResolvedValueOnce({ revision: 9 });
  saveMission.mockResolvedValue({ revision: 10 });

  let persistence;
  act(() => {
    persistence = persistDesignMission(request(view), dependencies());
  });
  await waitFor(() => expect(saveBtFile).toHaveBeenCalledTimes(1));

  act(() => {
    view.result.current.recordBtEdit(LOCAL_PATH, "newer-editor-value");
  });
  firstUpload.resolve({ revision: 8 });

  let result;
  await act(async () => {
    result = await persistence;
  });

  expect(saveBtFile.mock.calls[1][2]).toBe("save-snapshot");
  expect(result.saveResult).toMatchObject({
    hasNewerBtEdits: true,
    hasNewerEdits: true,
  });
  expect(view.result.current.getLiveBtFiles()[LOCAL_PATH]).toBe(
    "newer-editor-value",
  );
  expect(view.result.current.getPersistedBtFiles()[LOCAL_PATH]).toBe(
    "save-snapshot",
  );
  expect(view.result.current.designDirty).toBe(true);
});
