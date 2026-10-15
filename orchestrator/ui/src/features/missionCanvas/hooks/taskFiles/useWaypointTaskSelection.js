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

import { useCallback, useMemo, useState } from "react";
import {
  localBtDirectoriesForSpots,
  localBtDirectoryForSpot,
  localBtPathForSpot,
  localBtPathsForSpot,
} from "../../../mapWorkspace/lib/missionBtFiles";

export default function useWaypointTaskSelection({ selection }) {
  const { spots, selectedSpotId } = selection;
  const [selectedPathBySpotId, setSelectedPathBySpotId] = useState({});
  const selectedSpot = useMemo(
    () => spots.find((spot) => spot.id === selectedSpotId) || null,
    [selectedSpotId, spots],
  );
  const directoryBySpotId = useMemo(
    () => localBtDirectoriesForSpots(spots),
    [spots],
  );
  const selectedDirectory = selectedSpot
    ? directoryBySpotId.get(selectedSpot.id)
      || localBtDirectoryForSpot(selectedSpot)
    : "";
  const defaultPath = selectedSpot ? localBtPathForSpot(selectedSpot) : "";
  const filePaths = useMemo(
    () => (selectedSpot ? localBtPathsForSpot(selectedSpot) : []),
    [selectedSpot],
  );
  const requestedPath = selectedSpot
    ? selectedPathBySpotId[selectedSpot.id]
    : "";
  const selectedPath = filePaths.includes(requestedPath)
    ? requestedPath
    : defaultPath;
  const selectXml = useCallback((path) => {
    if (!selectedSpot || !filePaths.includes(path)) {
      throw new Error("This XML does not belong to the selected waypoint");
    }
    setSelectedPathBySpotId((current) => ({
      ...current,
      [selectedSpot.id]: path,
    }));
  }, [filePaths, selectedSpot]);

  const resetSelectedPaths = useCallback(() => {
    setSelectedPathBySpotId({});
  }, []);

  const forgetSpotSelection = useCallback((spotId) => {
    setSelectedPathBySpotId((current) => {
      if (!Object.prototype.hasOwnProperty.call(current, spotId)) return current;
      const next = { ...current };
      delete next[spotId];
      return next;
    });
  }, []);

  return { selectedSpot, selectedDirectory, defaultPath, filePaths, selectedPath,
    setSelectedPathBySpotId, selectXml, resetSelectedPaths, forgetSpotSelection };
}
