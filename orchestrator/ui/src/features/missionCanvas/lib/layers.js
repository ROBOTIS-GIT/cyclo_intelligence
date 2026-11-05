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

import { STAGE_AUTHORING, STAGE_MAPPING, STAGE_MAP_EDIT, STAGE_NAVIGATE, STAGE_RUN } from "./stages";

export const LAYER_DEFINITIONS = {
  map: "Map",
  scan: "Lidar",
  robotModel: "Robot Footprint",
  tf: "TF",
  globalCostmap: "Global costmap",
  localCostmap: "Local costmap",
  globalPlan: "Global plan",
  mapAreas: "Map areas",
};

export const STAGE_LAYER_IDS = {
  [STAGE_MAPPING]: ["map", "scan", "robotModel", "tf"],
  // The map editor draws the PGM itself; no live layers to toggle.
  [STAGE_MAP_EDIT]: [],
  [STAGE_AUTHORING]: ["map", "mapAreas", "scan", "robotModel", "tf"],
  [STAGE_NAVIGATE]: [
    "map",
    "mapAreas",
    "scan",
    "robotModel",
    "globalCostmap",
    "localCostmap",
    "globalPlan",
    "tf",
  ],
  [STAGE_RUN]: [
    "map",
    "mapAreas",
    "scan",
    "robotModel",
    "globalCostmap",
    "localCostmap",
    "globalPlan",
    "tf",
  ],
};

export const LAYER_PRESETS = {
  [STAGE_MAP_EDIT]: {
    map: false,
    scan: false,
    robotModel: false,
    tf: false,
    globalCostmap: false,
    localCostmap: false,
    globalPlan: false,
    mapAreas: false,
  },
  [STAGE_MAPPING]: {
    map: true,
    scan: true,
    robotModel: true,
    tf: true,
    globalCostmap: false,
    localCostmap: false,
    globalPlan: false,
    mapAreas: false,
  },
  [STAGE_AUTHORING]: {
    map: true,
    scan: false,
    robotModel: false,
    tf: false,
    globalCostmap: false,
    localCostmap: false,
    globalPlan: false,
    mapAreas: true,
  },
  [STAGE_NAVIGATE]: {
    map: true,
    scan: true,
    robotModel: true,
    tf: false,
    globalCostmap: true,
    localCostmap: true,
    globalPlan: true,
    mapAreas: true,
  },
  [STAGE_RUN]: {
    map: true,
    scan: true,
    robotModel: true,
    tf: false,
    globalCostmap: true,
    localCostmap: true,
    globalPlan: true,
    mapAreas: true,
  },
};

export const LAYER_TOPIC_IDS = {
  map: ["/map"],
  scan: ["/scan"],
  robotModel: ["/local_costmap/published_footprint"],
  tf: ["/tf", "/tf_static"],
  globalCostmap: ["/global_costmap/costmap"],
  localCostmap: ["/local_costmap/costmap"],
  globalPlan: ["/plan"],
};

export const STAGE_EXTRA_TOPIC_IDS = {
  [STAGE_MAPPING]: [],
  [STAGE_MAP_EDIT]: [],
  [STAGE_AUTHORING]: [],
  [STAGE_RUN]: ["/bt/status", "/bt/active_nodes"],
};

export const TOPIC_ORDER = [
  "/map",
  "/scan",
  "/pose",
  "/odom",
  "/amcl_pose",
  "/tf",
  "/tf_static",
  "/local_costmap/published_footprint",
  "/global_costmap/costmap",
  "/local_costmap/costmap",
  "/plan",
  "/bt/status",
  "/bt/active_nodes",
];
