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

export function xmlAttr(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/"/g, "&quot;");
}

export function btXmlName(value, fallback = "Node") {
  const cleaned = String(value || fallback)
    .trim()
    .replace(/[^A-Za-z0-9_]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || fallback;
}

export function localBtDirectoryBaseForSpot(spot) {
  const waypointId = btXmlName(spot?.id, "waypoint")
    .replace(/_[0-9a-f]{8}$/i, "")
    .toLowerCase();
  return `locals/${waypointId || "waypoint"}`;
}

export function storedLocalBtPathsForSpot(spot) {
  const defaultPath = existingLocalBtPathForSpot(spot);
  const configured = Array.isArray(spot?.local_bt_files)
    ? spot.local_bt_files
    : Array.isArray(spot?.metadata?.local_bt_files)
      ? spot.metadata.local_bt_files
      : [];
  return [defaultPath, ...configured]
    .map((path) => String(path || "").trim())
    .filter((path, index, paths) => path && paths.indexOf(path) === index);
}

export function storedTokenFreeLocalBtDirectory(spot) {
  const directories = storedLocalBtPathsForSpot(spot)
    .map((path) => path.match(/^(locals\/[^/]+)\/[^/]+\.xml$/i)?.[1] || "")
    .filter(Boolean);
  const unique = [...new Set(directories.map((path) => path.toLowerCase()))];
  if (unique.length !== 1) return "";
  const directory = directories[0];
  const name = directory.split("/").pop() || "";
  return /_[0-9a-f]{8}$/i.test(name) ? "" : directory;
}

export function localBtDirectoryForSpot(spot) {
  return storedTokenFreeLocalBtDirectory(spot) || localBtDirectoryBaseForSpot(spot);
}

export function localBtDirectoriesForSpots(spots) {
  const directories = new Map();
  const used = new Set();
  // Preserve folders that have already been committed under the token-free
  // layout. They take priority over yet-to-be-migrated waypoint IDs.
  (spots || []).forEach((spot) => {
    const stored = storedTokenFreeLocalBtDirectory(spot);
    if (!stored) return;
    const key = stored.toLowerCase();
    if (used.has(key)) {
      throw new Error(`Multiple waypoints reference the same Waypoint Task directory: ${stored}`);
    }
    used.add(key);
    directories.set(spot.id, stored);
  });
  (spots || []).forEach((spot) => {
    if (directories.has(spot.id)) return;
    const base = localBtDirectoryBaseForSpot(spot);
    let directory = base;
    let suffix = 1;
    while (used.has(directory.toLowerCase())) {
      suffix += 1;
      directory = `${base}_${suffix}`;
    }
    used.add(directory.toLowerCase());
    directories.set(spot.id, directory);
  });
  return directories;
}

export function initialLocalBtPathForSpot(spot) {
  return `${localBtDirectoryForSpot(spot)}/main.xml`;
}

export function localBtSaveAsPath(spot, fileName, directory = localBtDirectoryForSpot(spot)) {
  const rawName = String(fileName || "").trim();
  const stem = rawName.toLowerCase().endsWith(".xml")
    ? rawName.slice(0, -4)
    : rawName;
  if (!stem || !/^[A-Za-z0-9_.-]+$/.test(stem)) {
    throw new Error("XML name may contain only letters, numbers, '.', '_' and '-'");
  }
  const normalizedName = `${stem}.xml`;
  return `${directory}/${normalizedName}`;
}

export function existingLocalBtPathForSpot(spot) {
  return String(
    spot?.linked_bt_tree
      || spot?.metadata?.local_bt
      || spot?.metadata?.linked_bt_tree
      || "",
  ).trim();
}

export function localBtPathForSpot(spot) {
  const existing = existingLocalBtPathForSpot(spot);
  if (existing) return existing;
  return initialLocalBtPathForSpot(spot);
}

export function localBtPathsForSpot(spot) {
  const defaultPath = localBtPathForSpot(spot);
  const configured = Array.isArray(spot?.local_bt_files)
    ? spot.local_bt_files
    : Array.isArray(spot?.metadata?.local_bt_files)
      ? spot.metadata.local_bt_files
      : [];
  return [defaultPath, ...configured]
    .map((path) => String(path || "").trim())
    .filter((path, index, paths) => path && paths.indexOf(path) === index);
}

export function uniqueLocalBtTargetPath(directory, preferredName, usedPaths) {
  const rawName = String(preferredName || "bt.xml").split("/").filter(Boolean).pop() || "bt.xml";
  const rawStem = rawName.replace(/\.xml$/i, "");
  const stem = rawStem && /^[A-Za-z0-9_.-]+$/.test(rawStem)
    ? rawStem
    : btXmlName(rawStem, "bt").toLowerCase();
  let suffix = 1;
  let target = `${directory}/${stem}.xml`;
  while (usedPaths.has(target.toLowerCase())) {
    suffix += 1;
    target = `${directory}/${stem}_${suffix}.xml`;
  }
  usedPaths.add(target.toLowerCase());
  return target;
}

export function canonicalLocalBtPathMappingsForSpot(
  spot,
  directory = localBtDirectoryForSpot(spot),
) {
  // Every waypoint owns one directory. A new or legacy default becomes
  // main.xml; alternate filenames are preserved whenever possible. Once a
  // file is inside that directory, changing the Run BT pointer does not rename
  // the file on the next Save Mission.
  const sourceDefault = localBtPathForSpot(spot);
  const directoryPrefix = `${directory}/`;
  const usedPaths = new Set();
  return localBtPathsForSpot(spot).map((sourcePath, index) => {
    const isAlreadyOwned = sourcePath.startsWith(directoryPrefix)
      && !sourcePath.slice(directoryPrefix.length).includes("/");
    const isNestedLocalBt = /^locals\/[^/]+\/[^/]+\.xml$/i.test(sourcePath);
    const preferredName = isAlreadyOwned
      ? sourcePath.slice(directoryPrefix.length)
      : isNestedLocalBt
        ? sourcePath.split("/").pop()
      : index === 0 && sourcePath === sourceDefault
        ? "main.xml"
        : sourcePath.split("/").filter(Boolean).pop() || `bt_${index + 1}.xml`;
    return {
      sourcePath,
      targetPath: uniqueLocalBtTargetPath(directory, preferredName, usedPaths),
    };
  });
}

export function canonicalLocalBtPathForSpot(spot, directory = localBtDirectoryForSpot(spot)) {
  const sourceDefault = localBtPathForSpot(spot);
  return canonicalLocalBtPathMappingsForSpot(spot, directory)
    .find(({ sourcePath }) => sourcePath === sourceDefault)?.targetPath
    || `${directory}/main.xml`;
}

export function canonicalLocalBtPathsForSpot(spot, directory = localBtDirectoryForSpot(spot)) {
  return canonicalLocalBtPathMappingsForSpot(spot, directory)
    .map(({ targetPath }) => targetPath);
}

export function migrateCanonicalLocalBtFileKeys(spots, files) {
  const next = { ...(files || {}) };
  const original = { ...(files || {}) };
  const directories = localBtDirectoriesForSpots(spots);
  const mappings = (spots || []).flatMap((spot) => (
    canonicalLocalBtPathMappingsForSpot(spot, directories.get(spot.id))
  ));
  const targetPaths = new Set(mappings.map(({ targetPath }) => targetPath));
  mappings.forEach(({ sourcePath, targetPath }) => {
    if (sourcePath === targetPath || original[sourcePath] === undefined) return;
    // Preserve the newest in-memory edit when a legacy path is migrated while
    // a full Save is in flight.
    next[targetPath] = original[sourcePath];
  });
  mappings.forEach(({ sourcePath, targetPath }) => {
    if (sourcePath !== targetPath && !targetPaths.has(sourcePath)) delete next[sourcePath];
  });
  return next;
}

export function withLocalBtLibrary(spot, defaultPath, paths) {
  const normalizedPaths = [defaultPath, ...(paths || [])]
    .map((path) => String(path || "").trim())
    .filter((path, index, values) => path && values.indexOf(path) === index);
  return {
    ...spot,
    linked_bt_tree: defaultPath,
    local_bt_files: normalizedPaths,
    metadata: {
      ...(spot?.metadata ?? {}),
      local_bt: defaultPath,
      linked_bt_tree: defaultPath,
      local_bt_files: normalizedPaths,
    },
  };
}

export function changedLocalBtPaths(files, persistedFiles) {
  const paths = new Set([
    ...Object.keys(files || {}),
    ...Object.keys(persistedFiles || {}),
  ]);
  return new Set([...paths].filter((path) => (
    (path.startsWith("locals/") || (/^[^/]+\.xml$/.test(path) && path !== "global.xml"))
    && files?.[path] !== persistedFiles?.[path]
  )));
}

export function defaultLocalBtXml() {
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree"/>',
    '</root>',
    '',
  ].join("\n");
}

export function initializeCreatedWaypointLocalBt(existingSpots, createdSpot, reservedPaths = []) {
  const existingDirectories = localBtDirectoriesForSpots(existingSpots || []);
  const usedDirectories = new Set(
    [...existingDirectories.values()].map((path) => path.toLowerCase()),
  );
  (reservedPaths || []).forEach((path) => {
    const directory = String(path || "").match(/^(locals\/[^/]+)\/[^/]+\.xml$/i)?.[1];
    if (directory) usedDirectories.add(directory.toLowerCase());
  });

  const base = localBtDirectoryBaseForSpot(createdSpot);
  let directory = base;
  let suffix = 1;
  while (usedDirectories.has(directory.toLowerCase())) {
    suffix += 1;
    directory = `${base}_${suffix}`;
  }
  const defaultPath = canonicalLocalBtPathForSpot(createdSpot, directory);
  const paths = canonicalLocalBtPathsForSpot(createdSpot, directory);
  return {
    spot: withLocalBtLibrary(createdSpot, defaultPath, paths),
    defaultPath,
    paths,
  };
}

export function missionStepXmlLines(spots, tagName) {
  return spots.map((spot, index) => {
    const pose = spot.pose || {};
    const localBt = localBtPathForSpot(spot);
    const stepName = btXmlName(`Step ${index + 1} ${spot.label || spot.id}`, `Step_${index + 1}`);
    return [
      `      <${tagName}`,
      `        name="${xmlAttr(stepName)}"`,
      `        waypoint_id="${xmlAttr(spot.id)}"`,
      `        label="${xmlAttr(spot.label || spot.id)}"`,
      `        local_bt="${xmlAttr(localBt)}"`,
      `        x="${xmlAttr(Number(pose.x ?? 0).toFixed(6))}"`,
      `        y="${xmlAttr(Number(pose.y ?? 0).toFixed(6))}"`,
      `        yaw="${xmlAttr(Number(pose.yaw ?? 0).toFixed(6))}"/>`,
    ].join("\n");
  });
}

export function missionSequenceXml(rootName, spots, stepTag) {
  const stepLines = missionStepXmlLines(spots, stepTag);
  return [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<root BTCPP_format="4" main_tree_to_execute="MainTree">',
    '  <BehaviorTree ID="MainTree">',
    stepLines.length
      ? `    <Sequence name="${xmlAttr(rootName)}">`
      : `    <Sequence name="${xmlAttr(rootName)}"/>`,
    ...stepLines,
    ...(stepLines.length ? ['    </Sequence>'] : []),
    '  </BehaviorTree>',
    '</root>',
    '',
  ].join("\n");
}

export function buildGlobalMissionXml(spots) {
  return missionSequenceXml("GlobalMission", spots, "MissionStep");
}

export function missionBtFileDefaultsForSpots(spots) {
  const entries = {
    "global.xml": buildGlobalMissionXml(spots),
  };
  spots.forEach((spot) => {
    localBtPathsForSpot(spot).forEach((path) => {
      entries[path] = defaultLocalBtXml(spot);
    });
  });
  return entries;
}

export function missionBtFileDefaultsForRunSpots(spots) {
  const entries = {
    "global.xml": buildGlobalMissionXml(spots),
  };
  spots.forEach((spot) => {
    entries[localBtPathForSpot(spot)] = defaultLocalBtXml(spot);
  });
  return entries;
}

// Assemble every BT file owned by every waypoint. local_bt is only the runtime
// default pointer; local_bt_files is the durable library and display-label
// changes must never rename or discard any of its entries.
// Returns the files to write plus the now-orphaned local paths to delete.
export function assembleMissionBtFilesForSave(spots, missionBtFiles, deletedPaths, globalPath, globalXml) {
  const files = { [globalPath]: globalXml };
  const activePaths = new Map();
  const directories = localBtDirectoriesForSpots(spots);
  const mappings = spots.flatMap((spot) => (
    canonicalLocalBtPathMappingsForSpot(
      spot,
      directories.get(spot.id),
    ).map((mapping) => ({ ...mapping, spot }))
  ));
  mappings.forEach(({ spot, targetPath }) => {
    const ownershipKey = targetPath.toLowerCase();
    if (activePaths.has(ownershipKey)) {
      throw new Error(`Multiple waypoints reference the same Waypoint Task path: ${targetPath}`);
    }
    activePaths.set(ownershipKey, spot.id);
  });
  mappings.forEach(({ sourcePath, targetPath, spot }) => {
    const sourceOwner = activePaths.get(sourcePath.toLowerCase());
    const sourceOwnedByAnotherWaypoint = sourceOwner && sourceOwner !== spot.id;
    // A token-free fallback may point at another waypoint's canonical target.
    // Treat that as an uninitialized new tree instead of copying its XML.
    const content = !sourceOwnedByAnotherWaypoint
      && missionBtFiles[sourcePath] !== undefined
      ? missionBtFiles[sourcePath]
      : missionBtFiles[targetPath] !== undefined
        ? missionBtFiles[targetPath]
        : defaultLocalBtXml(spot);
    files[targetPath] = content;
  });
  const stale = new Set();
  const considerStale = (path) => {
    const isLocalBt = path.startsWith("locals/")
      || (/^[^/]+\.xml$/.test(path) && path !== globalPath);
    if (isLocalBt && !activePaths.has(path.toLowerCase())) stale.add(path);
  };
  Object.keys(missionBtFiles).forEach(considerStale);
  (deletedPaths || []).forEach(considerStale);
  return { files, stalePaths: [...stale] };
}
