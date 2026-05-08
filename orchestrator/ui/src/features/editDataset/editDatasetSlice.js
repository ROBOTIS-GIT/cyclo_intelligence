/*
 * Copyright 2025 ROBOTIS CO., LTD.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Author: Kiwoong Park
 */

import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  // --- Merge (rosbag task folders) -----------------------------------------
  mergeSourceTaskDirs: [],
  mergeOutputPath: '',
  mergeOutputFolderName: '',
  mergeMoveSources: false,

  // --- Delete (rosbag task folder) ----------------------------------------
  deleteTaskDir: '',
  deleteEpisodeNums: [],
  deleteCompact: true,

  // --- Task info displayed by Delete section ------------------------------
  datasetInfo: {
    robotType: '',
    taskInstruction: '',
    episodeCount: 0,
    totalDurationS: 0.0,
    fps: 0,
  },

  // --- Hugging Face section -----------------------------------------------
  // Currently active endpoint URL (matches one of HF_ENDPOINT_PRESETS or a
  // user-registered custom URL). Empty when nothing has been registered yet.
  hfActiveEndpoint: '',
  // List of registered endpoints from the server, shape:
  //   [{ endpoint: string, label: string, userId: string }]
  hfEndpoints: [],
  hfUserId: '',
  hfRepoIdUpload: '',
  hfRepoIdDownload: '',
  hfStatus: 'Idle',
  hfDataType: 'dataset',
  uploadStatus: {
    current: 0,
    total: 0,
    percentage: 0.0,
  },
  downloadStatus: {
    current: 0,
    total: 0,
    percentage: 0.0,
  },

  // --- Conversion (rosbag → MP4 → LeRobot) progress -----------------------
  // Sourced from /data/status (DataOperationStatus) filtered by
  // operation_type == OP_CONVERSION. status mirrors DataOperationStatus
  // enum names lower-cased.
  conversionStatus: {
    status: 'idle',  // 'idle' | 'running' | 'completed' | 'failed' | 'cancelled'
    progress: 0,
    stage: '',
    message: '',
    jobId: '',
  },
};

const editDatasetSlice = createSlice({
  name: 'editDataset',
  initialState,
  reducers: {
    // Merge
    setMergeSourceTaskDirs: (state, action) => {
      state.mergeSourceTaskDirs = action.payload;
    },
    setMergeOutputPath: (state, action) => {
      state.mergeOutputPath = action.payload;
    },
    setMergeOutputFolderName: (state, action) => {
      state.mergeOutputFolderName = action.payload;
    },
    setMergeMoveSources: (state, action) => {
      state.mergeMoveSources = action.payload;
    },
    // Delete
    setDeleteTaskDir: (state, action) => {
      state.deleteTaskDir = action.payload;
    },
    setDeleteEpisodeNums: (state, action) => {
      state.deleteEpisodeNums = action.payload;
    },
    setDeleteCompact: (state, action) => {
      state.deleteCompact = action.payload;
    },
    // Info
    setDatasetInfo: (state, action) => {
      state.datasetInfo = action.payload;
    },
    // Hugging Face
    setHFActiveEndpoint: (state, action) => {
      state.hfActiveEndpoint = action.payload || '';
    },
    setHFEndpoints: (state, action) => {
      state.hfEndpoints = action.payload || [];
    },
    setHFUserId: (state, action) => {
      state.hfUserId = action.payload;
    },
    setHFRepoIdUpload: (state, action) => {
      state.hfRepoIdUpload = action.payload;
    },
    setHFRepoIdDownload: (state, action) => {
      state.hfRepoIdDownload = action.payload;
    },
    setHFStatus: (state, action) => {
      state.hfStatus = action.payload;
    },
    setHFDataType: (state, action) => {
      state.hfDataType = action.payload;
    },
    setUploadStatus: (state, action) => {
      state.uploadStatus = action.payload;
    },
    setDownloadStatus: (state, action) => {
      state.downloadStatus = action.payload;
    },
    setConversionStatus: (state, action) => {
      state.conversionStatus = { ...state.conversionStatus, ...action.payload };
    },
  },
});

export const {
  setMergeSourceTaskDirs,
  setMergeOutputPath,
  setMergeOutputFolderName,
  setMergeMoveSources,
  setDeleteTaskDir,
  setDeleteEpisodeNums,
  setDeleteCompact,
  setDatasetInfo,
  setHFActiveEndpoint,
  setHFEndpoints,
  setHFUserId,
  setHFRepoIdUpload,
  setHFRepoIdDownload,
  setHFStatus,
  setHFDataType,
  setUploadStatus,
  setDownloadStatus,
  setConversionStatus,
} = editDatasetSlice.actions;

export default editDatasetSlice.reducer;
