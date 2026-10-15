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

export const FREE_VALUE = 254;
export const OCCUPIED_VALUE = 0;
export const UNKNOWN_VALUE = 205;
export const FREE_THRESHOLD = 250;
export const OCCUPIED_THRESHOLD = 50;
export const DEFAULT_BRUSH_SIZE_CELLS = 1;
export const MAX_BRUSH_SIZE_CELLS = 10;
export const BRUSH_SIZE_OPTIONS = [{
  label: "Small",
  value: 1
}, {
  label: "Medium",
  value: 3
}, {
  label: "Large",
  value: 6
}, {
  label: "XL",
  value: 10
}];
export const EDIT_TOOLS = [{
  id: "erase_black",
  label: "Clear Space"
}, {
  id: "draw_black",
  label: "Add Obstacle"
}, {
  id: "draw_unknown",
  label: "Mark Unknown"
}];
export const PIXEL_TOOL_IDS = new Set(EDIT_TOOLS.map(tool => tool.id));
export const ANNOTATION_TOOL = {
  id: "label_marker",
  label: "Area"
};
export const ANNOTATION_EXTEND_TOOL = {
  id: "extend_area",
  label: "Extend"
};
export const ANNOTATION_ERASE_TOOL = {
  id: "erase_area",
  label: "Erase Area"
};
export const ANNOTATION_COLORS = [{
  id: "dark_brown",
  label: "Warm sand",
  value: "#CBB9A4"
}, {
  id: "red_wine",
  label: "Rose",
  value: "#D9A4AB"
}, {
  id: "navy",
  label: "Powder blue",
  value: "#A9BFD9"
}, {
  id: "gray",
  label: "Silver gray",
  value: "#BFC3C9"
}, {
  id: "plum",
  label: "Lavender",
  value: "#C0ABD6"
}, {
  id: "teal",
  label: "Soft teal",
  value: "#9CC5C7"
}, {
  id: "indigo",
  label: "Periwinkle",
  value: "#A9AEDB"
}, {
  id: "berry",
  label: "Dusty berry",
  value: "#CFA3BE"
}];
