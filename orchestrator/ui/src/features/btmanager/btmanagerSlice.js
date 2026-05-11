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

const initialState = {
  treeXml: null,
  treeFileName: '',
  btStatus: 'stopped', // 'stopped' | 'running'
  activeNodeNames: [], // names of currently active BT nodes
  selectedNodeId: null, // ID of the node selected for param editing
};

const btmanagerSlice = createSlice({
  name: 'btmanager',
  initialState,
  reducers: {
    setTreeXml: (state, action) => {
      state.treeXml = action.payload;
    },
    setTreeFileName: (state, action) => {
      state.treeFileName = action.payload;
    },
    setBtStatus: (state, action) => {
      state.btStatus = action.payload;
    },
    setActiveNodeNames: (state, action) => {
      state.activeNodeNames = action.payload;
    },
    setSelectedNodeId: (state, action) => {
      state.selectedNodeId = action.payload;
    },
  },
});

export const { setTreeXml, setTreeFileName, setBtStatus, setActiveNodeNames, setSelectedNodeId } = btmanagerSlice.actions;
export default btmanagerSlice.reducer;
