// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.

// Serialize writes per waypoint field while preserving the newest optimistic
// edit. A document identity change starts a separate queue and confirmed value.
export default function createWaypointMutationQueue() {
  const queues = new Map();
  const generations = new Map();
  return {
    capture(kind, spotId, lease, identity, initialValue) {
      const leaseToken = typeof lease === 'object'
        ? lease?.generation ?? lease?.id ?? JSON.stringify(lease)
        : lease;
      const identityKey = `${identity.mapName}::${identity.missionName}::${leaseToken}`;
      const key = `${kind}:${spotId}`;
      const existing = queues.get(key);
      const coordinator = existing?.identityKey === identityKey
        ? existing
        : { identityKey, confirmed: initialValue, tail: Promise.resolve() };
      const generation = (generations.get(key) || 0) + 1;
      generations.set(key, generation);
      queues.set(key, coordinator);
      return {
        isLatest: () => generations.get(key) === generation,
        confirm: (value) => { coordinator.confirmed = value; },
        confirmed: () => coordinator.confirmed,
        enqueue(request) {
          coordinator.tail = coordinator.tail.catch(() => {}).then(request);
          return coordinator.tail;
        },
      };
    },
  };
}
