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

import { createSlice } from '@reduxjs/toolkit';
import {
  FALLBACK_CATALOG,
  FALLBACK_SCHEMA_VERSION,
} from '../../constants/btNodeCatalogFallback';

// Catalog source: 'fallback' = static file shipped with the UI,
// 'ros' = fetched from /bt/nodes/catalog, 'loading' = in-flight.
// The slice boots with the fallback so the palette renders even before
// rosbridge connects; useBTNodeCatalog flips it to 'ros' on success.
const initialState = {
  catalog: FALLBACK_CATALOG,
  schemaVersion: FALLBACK_SCHEMA_VERSION,
  source: 'fallback',
  error: null,
};

const btCatalogSlice = createSlice({
  name: 'btCatalog',
  initialState,
  reducers: {
    catalogFetchStarted: (state) => {
      state.source = 'loading';
      state.error = null;
    },
    catalogFetchSucceeded: (state, action) => {
      const { catalog, schemaVersion } = action.payload;
      state.catalog = catalog;
      state.schemaVersion = schemaVersion;
      state.source = 'ros';
      state.error = null;
    },
    catalogFetchFailed: (state, action) => {
      // Keep whatever's currently in `catalog` (fallback or previously
      // fetched ROS copy) — just record the failure for diagnostics.
      state.source = 'fallback';
      state.error = action.payload || 'Unknown error';
    },
  },
});

export const {
  catalogFetchStarted,
  catalogFetchSucceeded,
  catalogFetchFailed,
} = btCatalogSlice.actions;

export default btCatalogSlice.reducer;
