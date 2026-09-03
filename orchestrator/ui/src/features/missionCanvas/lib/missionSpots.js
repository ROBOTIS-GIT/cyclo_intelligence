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

import { yawFromPose } from "../../../utils/navigationTf";
import { initialLocalBtPathForSpot, localBtPathForSpot, localBtPathsForSpot } from "./missionBtFiles";

export function spotPoseFromMapPose(x, y, yaw) {
  return {
    frame_id: "map",
    x,
    y,
    yaw,
  };
}

export function mapPlacementMeta(grid) {
  const info = grid?.info;
  const width = Number(info?.width ?? 0);
  const height = Number(info?.height ?? 0);
  const resolution = Number(info?.resolution ?? 0);
  const origin = info?.origin ?? {};
  const originX = Number(origin.position?.x ?? 0);
  const originY = Number(origin.position?.y ?? 0);
  const originYaw = yawFromPose(origin);
  if (!width || !height || !resolution) return null;
  return {
    width,
    height,
    resolution,
    widthMeters: width * resolution,
    heightMeters: height * resolution,
    originX,
    originY,
    originYaw,
  };
}

export function pointInMapBounds(x, y, meta) {
  const dx = x - meta.originX;
  const dy = y - meta.originY;
  const cos = Math.cos(meta.originYaw);
  const sin = Math.sin(meta.originYaw);
  const localX = cos * dx + sin * dy;
  const localY = -sin * dx + cos * dy;
  const padding = meta.resolution * 4;
  return (
    localX >= -padding &&
    localX <= meta.widthMeters + padding &&
    localY >= -padding &&
    localY <= meta.heightMeters + padding
  );
}

export function legacyCellPointToMap(x, y, meta) {
  const localX = x * meta.resolution;
  const localY = y * meta.resolution;
  const cos = Math.cos(meta.originYaw);
  const sin = Math.sin(meta.originYaw);
  return {
    x: meta.originX + cos * localX - sin * localY,
    y: meta.originY + sin * localX + cos * localY,
  };
}

export function spotForMapDisplay(spot, grid) {
  const pose = spot?.pose;
  const meta = mapPlacementMeta(grid);
  if (!pose || !meta || spot.metadata?.coordinate_space === "map") return spot;
  const x = Number(pose.x ?? 0);
  const y = Number(pose.y ?? 0);
  const looksLikeLegacyCell = x >= 0 && x <= meta.width && y >= 0 && y <= meta.height;
  if (!looksLikeLegacyCell || pointInMapBounds(x, y, meta)) return spot;
  const converted = legacyCellPointToMap(x, y, meta);
  return {
    ...spot,
    pose: {
      ...pose,
      x: converted.x,
      y: converted.y,
    },
    metadata: {
      ...(spot.metadata ?? {}),
      coordinate_space: "legacy_cell_display",
    },
  };
}

export function nextWaypointLabel(spots) {
  const occupied = new Set();
  (spots || []).forEach((spot) => {
    const label = String(spot?.label || "").trim().toLowerCase();
    if (label) occupied.add(label);
    // A display rename (for example Waypoint 1 -> Start) must not release the
    // stable ordinal embedded in the waypoint ID. Reusing it would generate a
    // second tokenized ID that collapses onto the same readable BT directory.
    const idMatch = String(spot?.id || "").match(/^waypoint_(\d+)(?:_[0-9a-f]{8})?$/i);
    if (idMatch) occupied.add(`waypoint ${Number(idMatch[1])}`);
  });
  let index = 1;
  while (occupied.has(`waypoint ${index}`)) index += 1;
  return `Waypoint ${index}`;
}

export function missionWaypointsFromSpots(spots) {
  return spots.map((spot) => {
    const localBt = localBtPathForSpot(spot);
    const localBtFiles = localBtPathsForSpot(spot);
    return {
      id: spot.id,
      label: spot.label || spot.id,
      pose: {
        frame_id: spot.pose?.frame_id || "map",
        x: Number(spot.pose?.x ?? 0),
        y: Number(spot.pose?.y ?? 0),
        yaw: Number(spot.pose?.yaw ?? 0),
      },
      local_bt: localBt,
      local_bt_files: localBtFiles,
      metadata: {
        ...(spot.metadata ?? {}),
        linked_bt_tree: localBt,
        local_bt: localBt,
        local_bt_files: localBtFiles,
      },
    };
  });
}

export function orderedMissionSpots(spots) {
  return [...spots].sort((a, b) => (
    String(a.label || a.id).localeCompare(String(b.label || b.id))
  ));
}

export function spotsFromMissionWaypoints(mapName, waypoints) {
  if (!Array.isArray(waypoints)) return [];
  return waypoints.map((waypoint) => {
    const id = String(waypoint?.id || "").trim();
    if (!id) return null;
    const localBt = String(
      waypoint.local_bt
        || waypoint.metadata?.linked_bt_tree
        || initialLocalBtPathForSpot({ id }),
    ).trim();
    const pose = waypoint.pose || {};
    const metadata = waypoint.metadata && typeof waypoint.metadata === "object"
      ? waypoint.metadata
      : {};
    const localBtFiles = [
      localBt,
      ...(Array.isArray(waypoint.local_bt_files)
        ? waypoint.local_bt_files
        : Array.isArray(metadata.local_bt_files)
          ? metadata.local_bt_files
          : []),
    ]
      .map((path) => String(path || "").trim())
      .filter((path, index, paths) => path && paths.indexOf(path) === index);
    return {
      id,
      map_name: mapName,
      label: String(waypoint.label || id).trim() || id,
      _missionManifest: true,
      pose: {
        frame_id: pose.frame_id || "map",
        x: Number(pose.x ?? 0),
        y: Number(pose.y ?? 0),
        yaw: Number(pose.yaw ?? 0),
      },
      linked_bt_tree: localBt,
      local_bt_files: localBtFiles,
      metadata: {
        ...metadata,
        source: metadata.source || "mission_manifest",
        coordinate_space: metadata.coordinate_space || "map",
        local_bt: localBt,
        local_bt_files: localBtFiles,
      },
    };
  }).filter(Boolean);
}

export function isMissionManifestSpot(spot) {
  return spot?._missionManifest === true;
}
