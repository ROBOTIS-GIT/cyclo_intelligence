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
// Author: Howon Kim, Seongwoo Kim

"use client";

export const CAMERA_NEAR = 0.05;
export const CAMERA_FAR = 2000;
export const MAP_DISPLAY_ROTATION = Math.PI;
export const CLICK_DRAG_THRESHOLD_PX = 8;
export const TF_AXIS_LENGTH = 0.2;
export const WAYPOINT_RING_INNER_RADIUS = 2.25;
export const WAYPOINT_RING_OUTER_RADIUS = 3;
export const WAYPOINT_SELECTED_HALO_INNER_RADIUS = 3.28;
export const WAYPOINT_SELECTED_HALO_OUTER_RADIUS = 3.72;
export const WAYPOINT_CENTER_RADIUS = 1.25;
export const WAYPOINT_BODY_HIT_RADIUS = WAYPOINT_SELECTED_HALO_OUTER_RADIUS;
export const WAYPOINT_HEADING_LENGTH = 4.4;
export const WAYPOINT_HEADING_SHAFT_WIDTH = 1.12;
export const WAYPOINT_HEADING_HEAD_LENGTH = 1.05;
export const WAYPOINT_HEADING_HEAD_WIDTH = 2.84;
export const WAYPOINT_ROTATE_HIT_RADIUS = 1.4;
export const WAYPOINT_ROTATE_HIT_OFFSET = 0.1;
export const WAYPOINT_LABEL_SCALE_Y = 6;
export const WAYPOINT_LABEL_BG_ALPHA = 0.5;
export const WAYPOINT_LABEL_SELECTED_BG_ALPHA = 0.5;
export const WAYPOINT_LABEL_CLEARANCE = 1.5;
export const WAYPOINT_LABEL_OFFSET_Y = WAYPOINT_HEADING_LENGTH + (WAYPOINT_LABEL_SCALE_Y / 2) + WAYPOINT_LABEL_CLEARANCE;
export const BT_FOCUS_VISIBLE_HEIGHT_MIN = 5;
export const BT_FOCUS_VISIBLE_HEIGHT_MAX = 11;
export const BT_FOCUS_WAYPOINT_NDC_X = -0.75;
export const SCAN_VISUALIZATION_STRIDE = 2;
export const RENDER_INTERVAL_ACTIVE_MS = 16;
export const RENDER_INTERVAL_IDLE_MS = 100;
export const RENDER_INTERVAL_HIDDEN_MS = 500;
export const GLOBAL_COSTMAP_FULL_UPLOAD_RATIO = 0.25;
export const EMPTY_LIST = Object.freeze([]);
export const MARKER_COLORS = {
  idle: 0x5b8266,
  // sage — waypoints/behavior actions (walls stay ink)
  selected: 0xc96442,
  // clay
  control: 0x1c1a17,
  // ink — behavior control chips
  action: 0x5b8266,
  // sage
  decorator: 0xb4762f,
  // amber
  badgeTextIdle: "#ffffff",
  badgeTextSelected: "#ffffff",
  badgeStroke: "rgba(250,248,243,0.95)",
  labelBg: "rgba(243,241,234,0.92)",
  labelBorder: "rgba(226,221,209,1)",
  labelText: "#1c1a17",
  labelSelBg: "rgba(244,229,220,0.96)",
  labelSelBorder: "#c96442",
  labelSelText: "#b5563a",
  poseArrow: "#f3f1ea"
};
export const markerPalette = MARKER_COLORS;
export const SCENE_BG = 0xefece3;
