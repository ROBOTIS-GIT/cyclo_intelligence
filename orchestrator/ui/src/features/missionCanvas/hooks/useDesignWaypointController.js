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

import { useCallback, useRef } from 'react';
import {
  createNavigationSpot,
  updateNavigationSpot,
  deleteNavigationSpot,
} from '../../../utils/navigationSpotsApi';
import createWaypointMutationQueue from '../lib/createWaypointMutationQueue';
import useDesignWaypointCreation from './waypoint/useDesignWaypointCreation';
import useDesignWaypointEdits from './waypoint/useDesignWaypointEdits';
import useDesignWaypointDeletion from './waypoint/useDesignWaypointDeletion';
export default function useDesignWaypointController({
  spots,
  setSpots,
  editingSpotId,
  editingSpotLabel,
  setEditingSpotId,
  setEditingSpotLabel,
  mapName,
  captureDocumentLease,
  isDocumentLeaseCurrent,
  getCurrentIdentity,
  runCommand,
  markDirty,
  ledger,
  routeView,
  setFlowEdges,
  tryAcquireRouteMutationLock,
  releaseRouteMutationLock,
  setRouteSourceAfterExternalMutation,
  forgetTaskSelection,
  setSelectedSpotId,
  clearBehaviorSelection,
  setTaskLayerSpotId,
  setInteractionMode,
  setShowWaypointOptions,
  setBusy,
  onMessage,
  api = {},
}) {
  const queueRef = useRef(null);
  if (queueRef.current === null) queueRef.current = createWaypointMutationQueue();
  const documentIsCurrent = useCallback((lease, identity) => {
    const current = getCurrentIdentity();
    return (
      isDocumentLeaseCurrent(lease)
      && current.mapName === identity.mapName
      && current.missionName === identity.missionName
    );
  }, [getCurrentIdentity, isDocumentLeaseCurrent]);

  const document = { mapName, captureDocumentLease, isDocumentLeaseCurrent,
    getCurrentIdentity, documentIsCurrent };
  const waypoints = { spots, setSpots, editingSpotId, editingSpotLabel,
    setEditingSpotId, setEditingSpotLabel };
  const selection = { setSelectedSpotId, clearBehaviorSelection, setTaskLayerSpotId,
    setInteractionMode, setShowWaypointOptions, forgetTaskSelection };
  const persistence = { markDirty, ledger };
  const route = { routeView, setFlowEdges, tryAcquireRouteMutationLock,
    releaseRouteMutationLock, setRouteSourceAfterExternalMutation };
  const execution = { runCommand, setBusy, onMessage };
  const services = {
    createSpot: api.createSpot || createNavigationSpot,
    updateSpot: api.updateSpot || updateNavigationSpot,
    deleteSpot: api.deleteSpot || deleteNavigationSpot,
  };
  const coordinator = { mutationQueue: queueRef.current };
  const creation = useDesignWaypointCreation({
    waypoints,
    persistence,
    selection,
    document,
    execution,
    api: services,
  });
  const edits = useDesignWaypointEdits({
    waypoints,
    document,
    coordinator,
    persistence,
    execution,
    api: services,
    selection,
  });
  const deleteWaypoint = useDesignWaypointDeletion({
    route,
    execution,
    document,
    api: services,
    persistence,
    waypoints,
    selection,
  });
  return { ...creation, ...edits, deleteWaypoint };
}
