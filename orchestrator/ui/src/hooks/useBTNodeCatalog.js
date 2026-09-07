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

import { useCallback, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';

import {
  catalogFetchFailed,
  catalogFetchStarted,
  catalogFetchSucceeded,
} from '../features/actionCanvas/btCatalogSlice';
import { useRosServiceCaller } from './useRosServiceCaller';

const API_BASE = '/api';

// Fire-once-per-app guard. The slice's `source` field also tracks this,
// but if a component mounts → fetch starts → component unmounts before
// the response arrives, `source` stays at 'loading' and the next mount
// would re-fetch. The module-level ref keeps a single in-flight fetch
// per page load — good enough for PR1.
let fetchStarted = false;

async function readJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

async function getBtNodeStatus() {
  const response = await fetch(`${API_BASE}/services/bt_node/status`);
  const data = await readJsonResponse(response);
  if (!response.ok) {
    throw new Error(data.detail || `bt_node status failed (${response.status})`);
  }
  return data;
}

async function canFetchLiveCatalog() {
  try {
    const status = await getBtNodeStatus();
    return status.state === 'up';
  } catch {
    // Supervisor API may be unavailable in test/dev setups. Let the ROS service
    // call below decide whether the catalog can be fetched, but never start the
    // BT node implicitly from this hook.
    return true;
  }
}

/**
 * Subscribe to the BT node catalog Redux slice and lazy-fetch it from
 * `/bt/nodes/catalog` on first mount. Returns the same shape every time:
 *
 *   { catalog: META[], source: 'fallback'|'loading'|'ros', schemaVersion, error }
 *
 * Components can read `catalog` directly without waiting — the slice is
 * seeded with the static fallback so the palette renders immediately.
 */
export function useBTNodeCatalog() {
  const dispatch = useDispatch();
  const slice = useSelector((state) => state.btCatalog);
  const { getNodeCatalog } = useRosServiceCaller();

  const refreshCatalog = useCallback(async ({ force = false } = {}) => {
    if (fetchStarted && !force) return;
    fetchStarted = true;

    dispatch(catalogFetchStarted());
    try {
      const canFetch = await canFetchLiveCatalog();
      if (!canFetch) {
        fetchStarted = false;
        dispatch(catalogFetchFailed('Task Engine is off. Turn it on to refresh the task steps.'));
        return;
      }

      const result = await getNodeCatalog();
      if (!result || !result.success) {
        const msg = (result && result.message) || 'Service returned failure';
        fetchStarted = false;
        dispatch(catalogFetchFailed(msg));
        return;
      }
      let parsed = [];
      try {
        parsed = JSON.parse(result.catalog_json || '[]');
      } catch (e) {
        fetchStarted = false;
        dispatch(catalogFetchFailed(`Invalid catalog JSON: ${e.message}`));
        return;
      }
      if (!Array.isArray(parsed)) {
        fetchStarted = false;
        dispatch(catalogFetchFailed('Catalog JSON is not an array'));
        return;
      }
      dispatch(catalogFetchSucceeded({
        catalog: parsed,
        schemaVersion: result.schema_version || '',
      }));
    } catch (err) {
      // Reset the guard so a later manual BT Node ON can try again.
      fetchStarted = false;
      dispatch(catalogFetchFailed(err.message || String(err)));
    }
  }, [dispatch, getNodeCatalog]);

  useEffect(() => {
    refreshCatalog();
  }, [refreshCatalog]);

  return { ...slice, refreshCatalog };
}
