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
import { RecordPhase, InferencePhase } from '../../constants/taskPhases';

const initialState = {
  // Hoisted shared field — same value drives both flows. Owned by
  // SetRobotType + the latest snapshot from either status topic.
  robotType: '',

  taskInfo: {
    taskNum: '',
    taskName: '',
    taskType: '',
    taskInstruction: [],
    policyPath: '',
    recordInferenceMode: false,
    controlHz: 100,
    inferenceHz: 15,
    chunkAlignWindowS: 0.3,
    // Off by default — recording outputs are user-owned, not ROBOTIS'.
    // Tick on at the Record page when the dataset is a ROBOTIS internal
    // capture; the recorder then bakes the Apache 2.0 license header
    // into the task-folder README.
    includeRobotisLicense: false,
    // Inference backend (TaskInfo.service_type) + policy class. The two
    // are chosen together via the Model dropdown — see
    // components/InferenceModelSelector.js MODEL_OPTIONS for the list.
    // Empty serviceType falls back to orchestrator's config.json type
    // detection (backward-compat). policyType is UI-only (drives instruction
    // visibility via constants/policyCapabilities.js).
    serviceType: 'lerobot',
    policyType: 'act',
  },

  // Record-side snapshot from /data/recording/status (cyclo_data direct,
  // 5 Hz during a recording session). Conversion progress is its own
  // flow on /data/status (DataOperationStatus, OP_CONVERSION) routed
  // through editDatasetSlice.conversionStatus — distinct from this
  // record-session state.
  recordStatus: {
    taskName: 'idle',
    running: false,
    recordPhase: RecordPhase.READY,
    progress: 0,
    encodingProgress: 0,
    proceedTime: 0,
    currentEpisodeNumber: 0,
    currentScenarioNumber: 0,
    currentTaskInstruction: '',
    userId: '',
    usedStorageSize: 0,
    totalStorageSize: 0,
    usedCpu: 0,
    usedRamSize: 0,
    totalRamSize: 0,
    topicReceived: false,
  },

  // Inference-side snapshot from /task/inference_status (orchestrator
  // direct, one-shot per phase transition).
  inferenceStatus: {
    inferencePhase: InferencePhase.READY,
    error: '',
    topicReceived: false,
  },

  availableRobots: [],
  availableCameras: [],
  policyList: [],
  datasetList: [],
  heartbeatStatus: 'disconnected',
  lastHeartbeatTime: 0,
  joystickMode: '',
  // Per-topic live monitor snapshot from rosbag_recorder (1 Hz while recording).
  recordingMonitor: {
    topics: [],         // [{name, rateHz, baselineHz, secondsSinceLast, status}]
    totalReceived: 0,
    totalWritten: 0,
  },
};

const taskSlice = createSlice({
  name: 'tasks',
  initialState,
  reducers: {
    setTaskInfo: (state, action) => {
      state.taskInfo = { ...state.taskInfo, ...action.payload };
    },
    resetTaskInfo: (state) => {
      state.taskInfo = initialState.taskInfo;
    },
    setRecordStatus: (state, action) => {
      state.recordStatus = { ...state.recordStatus, ...action.payload };
    },
    resetRecordStatus: (state) => {
      state.recordStatus = initialState.recordStatus;
    },
    setInferenceStatus: (state, action) => {
      state.inferenceStatus = { ...state.inferenceStatus, ...action.payload };
    },
    resetInferenceStatus: (state) => {
      state.inferenceStatus = initialState.inferenceStatus;
    },
    selectRobotType: (state, action) => {
      state.robotType = action.payload;
    },
    setTaskType: (state, action) => {
      state.taskInfo.taskType = action.payload;
    },
    setTaskInstruction: (state, action) => {
      state.taskInfo.taskInstruction = action.payload;
    },
    setPolicyPath: (state, action) => {
      state.taskInfo.policyPath = action.payload;
    },
    setRecordInferenceMode: (state, action) => {
      state.taskInfo.recordInferenceMode = action.payload;
    },
    setHeartbeatStatus: (state, action) => {
      state.heartbeatStatus = action.payload;
    },
    setLastHeartbeatTime: (state, action) => {
      state.lastHeartbeatTime = action.payload;
    },
    setJoystickMode: (state, action) => {
      state.joystickMode = action.payload || '';
    },
    setRecordingMonitor: (state, action) => {
      state.recordingMonitor = action.payload;
    },
  },
});

export const {
  setTaskInfo,
  resetTaskInfo,
  setRecordStatus,
  resetRecordStatus,
  setInferenceStatus,
  resetInferenceStatus,
  selectRobotType,
  setTaskType,
  setTaskInstruction,
  setPolicyPath,
  setRecordInferenceMode,
  setHeartbeatStatus,
  setLastHeartbeatTime,
  setJoystickMode,
  setRecordingMonitor,
} = taskSlice.actions;

export default taskSlice.reducer;
