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

import { LAYER_PRESETS } from "../lib/layers";
import {
  STAGE_AUTHORING,
  STAGE_MAPPING,
  STAGE_MAP_EDIT,
  STAGE_NAVIGATE,
  STAGE_RUN,
} from "../lib/stages";
import {
  deriveMissionRosSceneGates,
  deriveMissionTopicRows,
} from "./useMissionRosScene";

const baseInputs = {
  busy: "",
  designMapPath: "maps/design.pgm",
  layersByStage: LAYER_PRESETS,
  mapName: "design-map",
  missionMapLoaded: true,
  navigationRuntimeMode: "idle",
  runBtVisualizationActive: false,
  runMapName: "run-map",
  running: false,
  workspaceStage: STAGE_MAPPING,
};

describe("deriveMissionRosSceneGates", () => {
  test("activates Mapping topics only for an active Mapping runtime", () => {
    const gates = deriveMissionRosSceneGates({
      ...baseInputs,
      navigationRuntimeMode: "mapping",
      running: true,
    });

    expect(gates.mappingRuntimeActive).toBe(true);
    expect(gates.mappingTopicsActive).toBe(true);
    expect(gates.mappingPoseSubscriptionActive).toBe(true);
    expect(gates.runTopicsActive).toBe(false);
    expect(gates.currentMapName).toBe("design-map");
  });

  test("keeps Map Edit offline even while Mapping is reported up", () => {
    const gates = deriveMissionRosSceneGates({
      ...baseInputs,
      navigationRuntimeMode: "mapping",
      running: true,
      workspaceStage: STAGE_MAP_EDIT,
    });

    expect(gates.mappingEditorActive).toBe(true);
    expect(gates.mappingRuntimeActive).toBe(true);
    expect(gates.mappingTopicsActive).toBe(false);
    expect(gates.mappingPoseSubscriptionActive).toBe(false);
  });

  test("treats Navigation as a Run-family map and localization session", () => {
    const gates = deriveMissionRosSceneGates({
      ...baseInputs,
      navigationRuntimeMode: "run",
      running: true,
      workspaceStage: STAGE_NAVIGATE,
    });

    expect(gates.runFamilyStage).toBe(true);
    expect(gates.runSessionActive).toBe(true);
    expect(gates.runRuntimeActive).toBe(true);
    expect(gates.runTopicsActive).toBe(true);
    expect(gates.currentMapName).toBe("run-map");
  });

  test("enables temporary Design localization only with a loaded design map", () => {
    const localized = deriveMissionRosSceneGates({
      ...baseInputs,
      navigationRuntimeMode: "localization",
      running: true,
      workspaceStage: STAGE_AUTHORING,
    });
    const unloaded = deriveMissionRosSceneGates({
      ...baseInputs,
      designMapPath: "",
      navigationRuntimeMode: "localization",
      running: true,
      workspaceStage: STAGE_AUTHORING,
    });

    expect(localized.designMapActive).toBe(true);
    expect(localized.designLocalizationActive).toBe(true);
    expect(localized.robotPoseCaptureActive).toBe(true);
    expect(unloaded.designLocalizationActive).toBe(false);
  });

  test("marks the Run BT split view as a lightweight map scene", () => {
    const gates = deriveMissionRosSceneGates({
      ...baseInputs,
      runBtVisualizationActive: true,
      workspaceStage: STAGE_RUN,
    });

    expect(gates.runFamilyStage).toBe(true);
    expect(gates.runBtMapLightweight).toBe(true);
  });

  test("keeps runtime ownership visible while Stop suppresses topic work", () => {
    const gates = deriveMissionRosSceneGates({
      ...baseInputs,
      busy: "Stop",
      navigationRuntimeMode: "run",
      running: true,
      workspaceStage: STAGE_RUN,
    });

    expect(gates.runRuntimeActive).toBe(true);
    expect(gates.runTopicsActive).toBe(false);
    expect(gates.stageNavigationTopicsActive).toBe(false);
  });
});

const topicInputs = {
  activeLayers: LAYER_PRESETS[STAGE_MAPPING],
  amclPose: null,
  btActiveNodesData: null,
  btNodeIsUp: false,
  btStatusData: null,
  designLocalizationActive: false,
  footprint: null,
  globalCostmap: null,
  localCostmap: null,
  map: null,
  odometry: null,
  plan: null,
  runFamilyStage: false,
  scan: null,
  slamPose: null,
  tf: null,
  tfStatic: null,
  workspaceStage: STAGE_MAPPING,
};

describe("deriveMissionTopicRows", () => {
  test("adds Mapping pose anchors and preserves the canonical topic order", () => {
    const rows = deriveMissionTopicRows({
      ...topicInputs,
      map: {},
      odometry: {},
      slamPose: {},
    });

    expect(rows.map(({ topic }) => topic)).toEqual([
      "/map",
      "/scan",
      "/pose",
      "/odom",
      "/tf",
      "/tf_static",
      "/local_costmap/published_footprint",
    ]);
    expect(rows.find(({ topic }) => topic === "/pose").isLive).toBe(true);
    expect(rows.find(({ topic }) => topic === "/scan").isLive).toBe(false);
  });

  test("hides live Design pose topics until temporary localization starts", () => {
    const idleRows = deriveMissionTopicRows({
      ...topicInputs,
      activeLayers: LAYER_PRESETS[STAGE_AUTHORING],
      workspaceStage: STAGE_AUTHORING,
    });
    const localizationRows = deriveMissionTopicRows({
      ...topicInputs,
      activeLayers: LAYER_PRESETS[STAGE_AUTHORING],
      designLocalizationActive: true,
      workspaceStage: STAGE_AUTHORING,
    });

    expect(idleRows).toEqual([]);
    expect(localizationRows.map(({ topic }) => topic)).toEqual([
      "/map",
      "/scan",
      "/amcl_pose",
      "/tf",
      "/tf_static",
      "/local_costmap/published_footprint",
    ]);
  });

  test("shows Run BT topics as live while the managed BT node is up", () => {
    const rows = deriveMissionTopicRows({
      ...topicInputs,
      activeLayers: LAYER_PRESETS[STAGE_RUN],
      btNodeIsUp: true,
      runFamilyStage: true,
      workspaceStage: STAGE_RUN,
    });

    expect(rows.slice(-2)).toEqual([
      { topic: "/bt/status", isLive: true },
      { topic: "/bt/active_nodes", isLive: true },
    ]);
  });

  test("does not expose Run-only BT topics on Navigation", () => {
    const rows = deriveMissionTopicRows({
      ...topicInputs,
      activeLayers: LAYER_PRESETS[STAGE_NAVIGATE],
      runFamilyStage: true,
      workspaceStage: STAGE_NAVIGATE,
    });

    expect(rows.some(({ topic }) => topic.startsWith("/bt/"))).toBe(false);
    expect(rows.some(({ topic }) => topic === "/amcl_pose")).toBe(true);
  });
});
