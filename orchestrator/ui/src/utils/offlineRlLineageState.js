// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

export const OFFLINE_RL_LINEAGE_STORAGE_KEY =
  'cyclo_intelligence.offline_rl.lineage';

export const DEFAULT_OFFLINE_RL_LINEAGE_STATE = Object.freeze({
  policyEpoch: 0,
  policyPath: '',
  forceFresh: false,
  lineageId: '',
});

const getSessionStorage = () => {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage;
  } catch (_error) {
    return null;
  }
};

const defaultState = () => ({ ...DEFAULT_OFFLINE_RL_LINEAGE_STATE });

const normalizeLineageState = (value) => {
  if (
    !value ||
    Array.isArray(value) ||
    typeof value !== 'object' ||
    !Number.isInteger(value.policyEpoch) ||
    value.policyEpoch < 0 ||
    typeof value.policyPath !== 'string' ||
    typeof value.forceFresh !== 'boolean' ||
    typeof value.lineageId !== 'string'
  ) {
    return defaultState();
  }

  return {
    policyEpoch: value.policyEpoch,
    policyPath: value.policyPath,
    forceFresh: value.forceFresh,
    lineageId: value.lineageId,
  };
};

const generateLineageId = () => {
  try {
    if (
      typeof window !== 'undefined' &&
      typeof window.crypto?.randomUUID === 'function'
    ) {
      return window.crypto.randomUUID();
    }
  } catch (_error) {
    // Fall through to a browser-compatible local identifier.
  }
  return `rl-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2, 10)}`;
};

export function resolveOfflineRLLineageState(storage = getSessionStorage()) {
  if (!storage) return defaultState();

  try {
    const storedValue = storage.getItem(OFFLINE_RL_LINEAGE_STORAGE_KEY);
    if (!storedValue) return defaultState();
    return normalizeLineageState(JSON.parse(storedValue));
  } catch (_error) {
    return defaultState();
  }
}

export function persistOfflineRLLineageState(
  state,
  storage = getSessionStorage()
) {
  const normalizedState = normalizeLineageState(state);
  if (!storage) return normalizedState;

  try {
    storage.setItem(
      OFFLINE_RL_LINEAGE_STORAGE_KEY,
      JSON.stringify(normalizedState)
    );
  } catch (_error) {
    // Storage can be blocked in private/browser-restricted contexts.
  }
  return normalizedState;
}

export function createOfflineRLLineage(
  policyPath,
  {
    lineageId = generateLineageId(),
    storage = getSessionStorage(),
  } = {}
) {
  const state = {
    policyEpoch: 0,
    policyPath: typeof policyPath === 'string' ? policyPath : '',
    forceFresh: true,
    lineageId:
      typeof lineageId === 'string' && lineageId ? lineageId : generateLineageId(),
  };
  return persistOfflineRLLineageState(state, storage);
}

export function advanceOfflineRLLineage(
  currentState,
  {
    policyEpoch,
    policyPath,
    storage = getSessionStorage(),
  } = {}
) {
  const current = normalizeLineageState(currentState);
  const nextState = {
    policyEpoch,
    policyPath,
    forceFresh: false,
    lineageId: current.lineageId,
  };
  return persistOfflineRLLineageState(nextState, storage);
}
