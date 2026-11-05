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

export const DEFAULT_MAP_NAME = "map";

export const DEFAULT_MISSION_NAME = "default";

export function missionRequestName(missionName) {
  return missionName === DEFAULT_MISSION_NAME ? "" : missionName;
}

// Mirrors the server's _SAFE_NAME charset; invalid names would 400 on save.
export const MISSION_NAME_PATTERN = /^[A-Za-z0-9_.-]+$/;

export const MISSION_NAME_MAX_LENGTH = 128;

export function isValidMissionName(name) {
  return (
    name.length > 0 &&
    name.length <= MISSION_NAME_MAX_LENGTH &&
    MISSION_NAME_PATTERN.test(name)
  );
}

export function uniqueMissionName(base, existingNames) {
  if (!existingNames.includes(base)) return base;
  let index = 2;
  while (existingNames.includes(`${base}-${index}`)) index += 1;
  return `${base}-${index}`;
}

export function mapNameFromPgmPath(path) {
  const fileName = String(path || "").split("/").filter(Boolean).pop() || "";
  return fileName.replace(/\.pgm$/i, "") || DEFAULT_MAP_NAME;
}
