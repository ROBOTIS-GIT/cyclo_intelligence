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
  DEFAULT_BT_SUPPORTED_ROBOT_TYPES,
  normalizeBtSupportedRobotTypes,
} from '../../constants/btSupport';

// Robots the behavior-tree engine supports, as reported by the supervisor API
// (/api/bt/support, which reads shared.robot_configs.schema). Until the
// request answers the default list applies, so gating behaves as before.
const initialState = {
  robotTypes: [...DEFAULT_BT_SUPPORTED_ROBOT_TYPES],
  loaded: false,
  error: null,
};

const btSupportSlice = createSlice({
  name: 'btSupport',
  initialState,
  reducers: {
    btSupportLoaded: (state, action) => {
      state.robotTypes = normalizeBtSupportedRobotTypes(action.payload);
      state.loaded = true;
      state.error = null;
    },
    btSupportFailed: (state, action) => {
      state.error = action.payload || 'unavailable';
    },
  },
});

export const { btSupportLoaded, btSupportFailed } = btSupportSlice.actions;

export const selectBtSupportedRobotTypes = (state) => (
  state.btSupport?.robotTypes ?? DEFAULT_BT_SUPPORTED_ROBOT_TYPES
);

// True once the supervisor has answered (or the request failed and the
// default list is in force). Session-restore gating waits for this so a robot
// the supervisor supports is not bounced on the fallback list.
export const selectBtSupportSettled = (state) => (
  state.btSupport ? Boolean(state.btSupport.loaded || state.btSupport.error) : true
);

export async function fetchBtSupport(dispatch) {
  if (typeof fetch !== 'function') {
    dispatch(btSupportFailed('fetch unavailable'));
    return;
  }
  try {
    const response = await fetch('/api/bt/support');
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail || `bt support request failed (${response.status})`);
    }
    dispatch(btSupportLoaded(data.supported_robot_types));
  } catch (error) {
    dispatch(btSupportFailed(error?.message || String(error)));
  }
}

export default btSupportSlice.reducer;
