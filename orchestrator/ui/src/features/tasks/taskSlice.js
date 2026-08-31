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

import { createSelector, createSlice } from '@reduxjs/toolkit';
import { RecordPhase, InferencePhase } from '../../constants/taskPhases';
import {
  getInferenceTaskInfoKey,
  getRecordTaskInfoKey,
} from '../../utils/taskInfoSync';

const SYNCED_MESSAGE = 'Session task info synced.';
const CONFLICT_MESSAGE = 'Server task info changed while editing; local draft not synced.';
const FAILED_MESSAGE = 'Task info not synced; robot button may use old task.';
export const ROBOT_TYPE_STORAGE_KEY = 'cyclo_intelligence.robot_type';
export const ROBOT_TYPE_STATUS_GUARD_MS = 30000;

const getSessionStorage = () => {
  if (typeof window === 'undefined') {
    return null;
  }
  try {
    return window.sessionStorage;
  } catch (_error) {
    return null;
  }
};

export const resolveInitialRobotType = (storage = getSessionStorage()) => {
  if (!storage) {
    return '';
  }
  try {
    return String(storage.getItem(ROBOT_TYPE_STORAGE_KEY) || '').trim();
  } catch (_error) {
    return '';
  }
};

export const persistRobotType = (robotType, storage = getSessionStorage()) => {
  if (!storage) {
    return;
  }
  try {
    const normalizedRobotType = String(robotType || '').trim();
    if (normalizedRobotType) {
      storage.setItem(ROBOT_TYPE_STORAGE_KEY, normalizedRobotType);
    } else {
      storage.removeItem(ROBOT_TYPE_STORAGE_KEY);
    }
  } catch (_error) {
    // Storage can be disabled in private/browser-restricted contexts.
  }
};

const syncInitialState = {
  serverTaskKey: '',
  editBaseServerTaskKey: '',
  dirty: false,
  conflict: false,
  syncStatus: 'idle',
  syncMessage: '',
  serverTaskInfo: null,
};

const inferenceSyncInitialState = {
  serverTaskKey: '',
  editBaseServerTaskKey: '',
  staleEchoTaskKey: '',
  dirty: false,
  syncStatus: 'idle',
  syncMessage: '',
  serverTaskInfo: null,
};

const sharedTaskInfoInitialState = {
  taskInstruction: [],
};

const recordTaskInfoInitialState = {
  taskNum: '',
  taskName: '',
  taskType: 'record',
  subtaskInstruction: [],
  includeRobotisLicense: false,
  warmupTime: 0,
  episodeTime: 0,
  resetTime: 0,
  numEpisodes: 0,
  pushToHub: false,
  privateMode: false,
  useOptimizedSave: false,
  recordRosBag2: false,
};

const inferenceTaskInfoInitialState = {
  taskType: 'inference',
  policyPath: '',
  recordInferenceMode: false,
  controlHz: 100,
  inferenceHz: 15,
  chunkAlignWindowS: 0.3,
  serviceType: 'lerobot',
  policyType: 'act',
  inferenceMode: 'simulation',
  actionRequestMode: 'async',
  accelerationMode: 'pytorch',
  accelerationEnginePath: '',
  initialPoseSync: false,
  initialPoseSyncDurationS: 5.0,
};

const stringArray = (items) => (
  Array.isArray(items) ? items.map((item) => String(item ?? '')) : []
);

const copySharedTaskInfo = (sharedTaskInfo = sharedTaskInfoInitialState) => ({
  taskInstruction: stringArray(sharedTaskInfo.taskInstruction),
});

const copyRecordTaskInfo = (recordTaskInfo = recordTaskInfoInitialState) => ({
  ...recordTaskInfoInitialState,
  ...recordTaskInfo,
  subtaskInstruction: stringArray(recordTaskInfo.subtaskInstruction),
});

const copyInferenceTaskInfo = (
  inferenceTaskInfo = inferenceTaskInfoInitialState
) => ({
  ...inferenceTaskInfoInitialState,
  ...inferenceTaskInfo,
  actionRequestMode:
    String(inferenceTaskInfo.actionRequestMode || '').trim().toLowerCase() === 'sync'
      ? 'sync'
      : 'async',
});

const selectTasksState = (state) => state.tasks || state;

const buildRecordTaskInfo = (tasks) => {
  const shared = copySharedTaskInfo(
    tasks.sharedTaskInfo || {
      taskInstruction: tasks.taskInfo?.taskInstruction,
    }
  );
  const record = copyRecordTaskInfo(tasks.recordTaskInfo || tasks.taskInfo);
  return {
    ...record,
    taskType: 'record',
    taskInstruction: shared.taskInstruction,
  };
};

const buildInferenceTaskInfo = (tasks) => {
  const shared = copySharedTaskInfo(
    tasks.sharedTaskInfo || {
      taskInstruction: tasks.taskInfo?.taskInstruction,
    }
  );
  const inference = copyInferenceTaskInfo(tasks.inferenceTaskInfo || tasks.taskInfo);
  return {
    ...inference,
    taskType: 'inference',
    taskInstruction: shared.taskInstruction,
    subtaskInstruction: [],
  };
};

const getRecordIdentityKey = (taskInfo = {}) => JSON.stringify({
  taskNum: String(taskInfo.taskNum ?? '').trim(),
  taskName: String(taskInfo.taskName ?? '').trim(),
  subtaskInstruction: stringArray(taskInfo.subtaskInstruction),
});

const hasRecordTaskIdentity = (taskInfo = {}) => (
  Boolean(String(taskInfo.taskNum ?? '').trim()) &&
  Boolean(String(taskInfo.taskName ?? '').trim())
);

const getInstructionKey = (taskInfo = {}) => JSON.stringify(
  stringArray(taskInfo.taskInstruction)
);

const hasLocalInferenceTaskInfoEdit = (state) => (
  Boolean(state.inferenceTaskInfoSync.dirty) ||
  ['pending', 'syncing'].includes(state.inferenceTaskInfoSync.syncStatus)
);

const hasLocalRecordTaskInfoEdit = (state) => (
  Boolean(state.taskInfoSync.dirty) ||
  ['pending', 'syncing', 'conflict'].includes(state.taskInfoSync.syncStatus)
);

const omitTaskInstruction = (taskInfo = {}) => {
  const { taskInstruction, ...rest } = taskInfo;
  return rest;
};

const buildLegacyTaskInfo = (state, source = 'record') => {
  const record = buildRecordTaskInfo(state);
  const inference = buildInferenceTaskInfo(state);
  return {
    ...record,
    ...inference,
    taskNum: record.taskNum,
    taskName: record.taskName,
    taskType: source === 'inference' ? 'inference' : record.taskType,
    taskInstruction: stringArray(state.sharedTaskInfo.taskInstruction),
    subtaskInstruction: record.subtaskInstruction,
    includeRobotisLicense: record.includeRobotisLicense,
  };
};

const syncLegacyTaskInfo = (state, source = 'record') => {
  state.taskInfo = buildLegacyTaskInfo(state, source);
};

const applySharedTaskInfo = (state, taskInfo = {}) => {
  if (Object.prototype.hasOwnProperty.call(taskInfo, 'taskInstruction')) {
    state.sharedTaskInfo.taskInstruction = stringArray(taskInfo.taskInstruction);
  }
};

const applyRecordTaskInfo = (state, taskInfo = {}, options = {}) => {
  applySharedTaskInfo(state, taskInfo);
  state.recordTaskInfo = {
    ...state.recordTaskInfo,
    taskNum: String(taskInfo.taskNum ?? state.recordTaskInfo.taskNum ?? ''),
    taskName: String(taskInfo.taskName ?? state.recordTaskInfo.taskName ?? ''),
    taskType: 'record',
    subtaskInstruction: Object.prototype.hasOwnProperty.call(taskInfo, 'subtaskInstruction')
      ? stringArray(taskInfo.subtaskInstruction)
      : stringArray(state.recordTaskInfo.subtaskInstruction),
    includeRobotisLicense: Object.prototype.hasOwnProperty.call(taskInfo, 'includeRobotisLicense')
      ? Boolean(taskInfo.includeRobotisLicense)
      : Boolean(state.recordTaskInfo.includeRobotisLicense),
    warmupTime: taskInfo.warmupTime ?? state.recordTaskInfo.warmupTime ?? 0,
    episodeTime: taskInfo.episodeTime ?? state.recordTaskInfo.episodeTime ?? 0,
    resetTime: taskInfo.resetTime ?? state.recordTaskInfo.resetTime ?? 0,
    numEpisodes: taskInfo.numEpisodes ?? state.recordTaskInfo.numEpisodes ?? 0,
    pushToHub: Object.prototype.hasOwnProperty.call(taskInfo, 'pushToHub')
      ? Boolean(taskInfo.pushToHub)
      : Boolean(state.recordTaskInfo.pushToHub),
    privateMode: Object.prototype.hasOwnProperty.call(taskInfo, 'privateMode')
      ? Boolean(taskInfo.privateMode)
      : Boolean(state.recordTaskInfo.privateMode),
    useOptimizedSave: Object.prototype.hasOwnProperty.call(taskInfo, 'useOptimizedSave')
      ? Boolean(taskInfo.useOptimizedSave)
      : Boolean(state.recordTaskInfo.useOptimizedSave),
    recordRosBag2: Object.prototype.hasOwnProperty.call(taskInfo, 'recordRosBag2')
      ? Boolean(taskInfo.recordRosBag2)
      : Boolean(state.recordTaskInfo.recordRosBag2),
  };

  if (options.resetSegmentPlan !== false) {
    const subtasks = stringArray(state.recordTaskInfo.subtaskInstruction);
    state.plannedSubTasks = subtasks;
    state.plannedCount = subtasks.length;
    state.slotToServerIdx = subtasks.map(() => -1);
    state.activeSlotIndex = 0;
  }

  syncLegacyTaskInfo(state, 'record');
};

const applyInferenceTaskInfo = (state, taskInfo = {}) => {
  applySharedTaskInfo(state, taskInfo);
  state.inferenceTaskInfo = {
    ...state.inferenceTaskInfo,
    taskType: 'inference',
    policyPath: String(taskInfo.policyPath ?? state.inferenceTaskInfo.policyPath ?? ''),
    recordInferenceMode: Object.prototype.hasOwnProperty.call(taskInfo, 'recordInferenceMode')
      ? Boolean(taskInfo.recordInferenceMode)
      : Boolean(state.inferenceTaskInfo.recordInferenceMode),
    controlHz: taskInfo.controlHz ?? state.inferenceTaskInfo.controlHz ?? 100,
    inferenceHz: taskInfo.inferenceHz ?? state.inferenceTaskInfo.inferenceHz ?? 15,
    chunkAlignWindowS:
      taskInfo.chunkAlignWindowS ?? state.inferenceTaskInfo.chunkAlignWindowS ?? 0.3,
    serviceType: String(taskInfo.serviceType ?? state.inferenceTaskInfo.serviceType ?? ''),
    policyType: String(taskInfo.policyType ?? state.inferenceTaskInfo.policyType ?? 'act'),
    inferenceMode:
      String(taskInfo.inferenceMode ?? state.inferenceTaskInfo.inferenceMode ?? 'simulation') ||
      'simulation',
    actionRequestMode:
      String(
        taskInfo.actionRequestMode ?? state.inferenceTaskInfo.actionRequestMode ?? ''
      ).trim().toLowerCase() === 'sync'
        ? 'sync'
        : 'async',
    accelerationMode: String(
      taskInfo.accelerationMode ?? state.inferenceTaskInfo.accelerationMode ?? 'pytorch'
    ),
    accelerationEnginePath: String(
      taskInfo.accelerationEnginePath ?? state.inferenceTaskInfo.accelerationEnginePath ?? ''
    ),
    initialPoseSync: Object.prototype.hasOwnProperty.call(taskInfo, 'initialPoseSync')
      ? Boolean(taskInfo.initialPoseSync)
      : Boolean(state.inferenceTaskInfo.initialPoseSync),
    initialPoseSyncDurationS:
      taskInfo.initialPoseSyncDurationS ??
      state.inferenceTaskInfo.initialPoseSyncDurationS ??
      5.0,
  };
  syncLegacyTaskInfo(state, 'inference');
};

const initialState = {
  robotType: resolveInitialRobotType(),
  robotTypeStatusGuardUntilMs: 0,

  sharedTaskInfo: { ...sharedTaskInfoInitialState },
  recordTaskInfo: copyRecordTaskInfo(),
  inferenceTaskInfo: copyInferenceTaskInfo(),
  taskInfo: {},
  taskInfoSync: { ...syncInitialState },
  inferenceTaskInfoSync: { ...inferenceSyncInitialState },

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
    currentSubtaskIndex: 0,
    subtaskCount: 0,
    currentSubtaskInstruction: '',
    subtaskInstructions: [],
    savedSubtaskIndices: null,
    userId: '',
    usedStorageSize: 0,
    totalStorageSize: 0,
    usedCpu: 0,
    usedRamSize: 0,
    totalRamSize: 0,
    recordingWarnings: [],
    recordingOperationStatus: 'idle',
    recordingOperationStage: '',
    recordingOperationMessage: '',
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
    cameraTopics: [],   // [{name, cameraName, rateHz, baselineHz, secondsSinceLast, status}]
    totalReceived: 0,
    totalWritten: 0,
  },

  plannedCount: 0,
  plannedSubTasks: [],
  slotToServerIdx: [],
  activeSlotIndex: 0,
};

initialState.taskInfo = buildLegacyTaskInfo(initialState);

const taskSlice = createSlice({
  name: 'tasks',
  initialState,
  reducers: {
    setTaskInfo: (state, action) => {
      const payload = action.payload || {};
      if (Object.prototype.hasOwnProperty.call(payload, 'taskInstruction')) {
        applySharedTaskInfo(state, payload);
      }
      applyRecordTaskInfo(state, payload, { resetSegmentPlan: false });
      applyInferenceTaskInfo(state, payload);
      syncLegacyTaskInfo(state, payload.taskType === 'inference' ? 'inference' : 'record');
    },
    setRecordTaskInfo: (state, action) => {
      applyRecordTaskInfo(state, action.payload || {}, { resetSegmentPlan: false });
    },
    setInferenceTaskInfo: (state, action) => {
      applyInferenceTaskInfo(state, action.payload || {});
    },
    setSharedTaskInstruction: (state, action) => {
      state.sharedTaskInfo.taskInstruction = stringArray(action.payload);
      syncLegacyTaskInfo(state);
    },
    resetTaskInfo: (state) => {
      state.sharedTaskInfo = { ...sharedTaskInfoInitialState };
      state.recordTaskInfo = copyRecordTaskInfo();
      state.inferenceTaskInfo = copyInferenceTaskInfo();
      state.taskInfoSync = { ...syncInitialState };
      state.inferenceTaskInfoSync = { ...inferenceSyncInitialState };
      syncLegacyTaskInfo(state);
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
      const payload = action.payload || '';
      const robotType = typeof payload === 'object'
        ? payload.robotType
        : payload;
      const normalizedRobotType = String(robotType || '').trim();
      const source = typeof payload === 'object'
        ? payload.source || 'local'
        : 'local';
      const receivedAtMs = Number(
        typeof payload === 'object' ? payload.receivedAtMs || 0 : 0
      );

      if (source === 'status') {
        if (
          state.robotTypeStatusGuardUntilMs &&
          receivedAtMs < state.robotTypeStatusGuardUntilMs &&
          normalizedRobotType &&
          normalizedRobotType !== state.robotType
        ) {
          return;
        }
        if (normalizedRobotType === state.robotType) {
          state.robotTypeStatusGuardUntilMs = 0;
        }
      } else if (source === 'user') {
        const selectedAtMs = Number(
          typeof payload === 'object' ? payload.selectedAtMs || 0 : 0
        );
        state.robotTypeStatusGuardUntilMs =
          selectedAtMs > 0 ? selectedAtMs + ROBOT_TYPE_STATUS_GUARD_MS : 0;
      } else {
        state.robotTypeStatusGuardUntilMs = 0;
      }

      state.robotType = normalizedRobotType;
    },
    setTaskType: (state, action) => {
      state.recordTaskInfo.taskType = action.payload || 'record';
      syncLegacyTaskInfo(state);
    },
    setTaskInstruction: (state, action) => {
      state.sharedTaskInfo.taskInstruction = stringArray(action.payload);
      syncLegacyTaskInfo(state);
    },
    setPolicyPath: (state, action) => {
      state.inferenceTaskInfo.policyPath = action.payload || '';
      syncLegacyTaskInfo(state, 'inference');
    },
    setRecordInferenceMode: (state, action) => {
      state.inferenceTaskInfo.recordInferenceMode = Boolean(action.payload);
      syncLegacyTaskInfo(state, 'inference');
    },
    setInferenceMode: (state, action) => {
      state.inferenceTaskInfo.inferenceMode = action.payload || 'simulation';
      syncLegacyTaskInfo(state, 'inference');
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
      state.recordingMonitor = {
        ...state.recordingMonitor,
        ...action.payload,
        cameraTopics: action.payload.cameraTopics ?? state.recordingMonitor.cameraTopics,
      };
    },
    setCameraRecordingMonitor: (state, action) => {
      state.recordingMonitor.cameraTopics = action.payload || [];
    },
    setPlannedCount: (state, action) => {
      state.plannedCount = action.payload;
    },
    setPlannedSubTasks: (state, action) => {
      const subtasks = stringArray(action.payload);
      state.plannedSubTasks = subtasks;
      state.recordTaskInfo.subtaskInstruction = subtasks;
      syncLegacyTaskInfo(state);
    },
    setPlannedSubTaskAt: (state, action) => {
      const { index, value } = action.payload;
      if (index >= 0 && index < state.plannedSubTasks.length) {
        state.plannedSubTasks[index] = value;
        state.recordTaskInfo.subtaskInstruction = state.plannedSubTasks;
        syncLegacyTaskInfo(state);
      }
    },
    setSlotToServerIdx: (state, action) => {
      state.slotToServerIdx = action.payload;
    },
    setActiveSlotIndex: (state, action) => {
      state.activeSlotIndex = action.payload;
    },
    resetSegmentPlan: (state) => {
      state.plannedCount = 0;
      state.plannedSubTasks = [];
      state.slotToServerIdx = [];
      state.activeSlotIndex = 0;
      state.recordTaskInfo.subtaskInstruction = [];
      syncLegacyTaskInfo(state);
    },
    resetSegmentProgress: (state) => {
      state.slotToServerIdx = state.plannedSubTasks.map(() => -1);
      state.activeSlotIndex = 0;
    },
    markLocalTaskInfoEdited: (state, action) => {
      const source = action.payload?.source || 'record';
      if (source === 'inference') {
        if (!state.inferenceTaskInfoSync.dirty) {
          state.inferenceTaskInfoSync.editBaseServerTaskKey =
            state.inferenceTaskInfoSync.serverTaskKey;
        }
        state.inferenceTaskInfoSync.staleEchoTaskKey = '';
        state.inferenceTaskInfoSync.dirty = true;
        state.inferenceTaskInfoSync.syncStatus = 'pending';
        state.inferenceTaskInfoSync.syncMessage = 'Task info changed; syncing soon...';
        return;
      }
      if (!state.taskInfoSync.dirty) {
        state.taskInfoSync.editBaseServerTaskKey = state.taskInfoSync.serverTaskKey;
      }
      state.taskInfoSync.dirty = true;
      state.taskInfoSync.syncStatus = state.taskInfoSync.conflict ? 'conflict' : 'pending';
      state.taskInfoSync.syncMessage = state.taskInfoSync.conflict
        ? CONFLICT_MESSAGE
        : 'Task info changed; syncing soon...';
    },
    markTaskInfoSyncPending: (state) => {
      state.taskInfoSync.syncStatus = 'pending';
      state.taskInfoSync.syncMessage = 'Task info changed; syncing soon...';
    },
    markTaskInfoSyncing: (state) => {
      state.taskInfoSync.syncStatus = 'syncing';
      state.taskInfoSync.syncMessage = 'Syncing task info...';
    },
    markTaskInfoSyncSuccess: (state) => {
      const taskInfo = buildRecordTaskInfo(state);
      const taskKey = getRecordTaskInfoKey(taskInfo);
      state.taskInfoSync.serverTaskKey = taskKey;
      state.taskInfoSync.editBaseServerTaskKey = taskKey;
      state.taskInfoSync.dirty = false;
      state.taskInfoSync.conflict = false;
      state.taskInfoSync.syncStatus = 'synced';
      state.taskInfoSync.syncMessage = SYNCED_MESSAGE;
      state.taskInfoSync.serverTaskInfo = taskInfo;
    },
    markTaskInfoSyncFailed: (state, action) => {
      state.taskInfoSync.syncStatus = 'failed';
      state.taskInfoSync.syncMessage = action.payload || FAILED_MESSAGE;
    },
    markTaskInfoSyncMissing: (state) => {
      state.taskInfoSync.syncStatus = 'missing';
      state.taskInfoSync.syncMessage = 'Fill Task Num and Task Name to sync.';
    },
    markInferenceTaskInfoSyncPending: (state) => {
      state.inferenceTaskInfoSync.syncStatus = 'pending';
      state.inferenceTaskInfoSync.syncMessage = 'Task info changed; syncing soon...';
    },
    markInferenceTaskInfoSyncing: (state) => {
      state.inferenceTaskInfoSync.syncStatus = 'syncing';
      state.inferenceTaskInfoSync.syncMessage = 'Syncing task info...';
    },
    markInferenceTaskInfoSyncSubmitted: (state) => {
      state.inferenceTaskInfoSync.syncStatus = 'syncing';
      state.inferenceTaskInfoSync.syncMessage = 'Waiting for synced task info...';
    },
    markInferenceTaskInfoSyncSuccess: (state, action) => {
      const taskInfo = action.payload?.taskInfo || buildInferenceTaskInfo(state);
      const taskKey = action.payload?.taskKey || getInferenceTaskInfoKey(taskInfo);
      const editBaseServerTaskKey =
        state.inferenceTaskInfoSync.editBaseServerTaskKey ||
        state.inferenceTaskInfoSync.serverTaskKey;
      state.inferenceTaskInfoSync.serverTaskKey = taskKey;
      state.inferenceTaskInfoSync.editBaseServerTaskKey = taskKey;
      state.inferenceTaskInfoSync.staleEchoTaskKey =
        editBaseServerTaskKey && editBaseServerTaskKey !== taskKey
          ? editBaseServerTaskKey
          : '';
      state.inferenceTaskInfoSync.dirty = false;
      state.inferenceTaskInfoSync.syncStatus = 'synced';
      state.inferenceTaskInfoSync.syncMessage = SYNCED_MESSAGE;
      state.inferenceTaskInfoSync.serverTaskInfo = taskInfo;
    },
    markInferenceTaskInfoSyncFailed: (state, action) => {
      state.inferenceTaskInfoSync.syncStatus = 'failed';
      state.inferenceTaskInfoSync.syncMessage =
        action.payload || 'Inference task info not synced.';
    },
    receiveServerRecordTaskInfo: (state, action) => {
      const serverTaskInfo = action.payload || {};
      const isInferenceEcho = serverTaskInfo.taskType === 'inference';
      if (isInferenceEcho) {
        const currentInferenceTaskInfo = buildInferenceTaskInfo(state);
        const currentRecordTaskInfo = buildRecordTaskInfo(state);
        const currentRecordTaskKey = getRecordTaskInfoKey(currentRecordTaskInfo);
        const hasLocalRecordEdit = hasLocalRecordTaskInfoEdit(state);
        const protectRecordSharedInstruction = Boolean(
          hasLocalRecordEdit &&
          Object.prototype.hasOwnProperty.call(serverTaskInfo, 'taskInstruction') &&
          getInstructionKey(serverTaskInfo) !== getInstructionKey(currentInferenceTaskInfo)
        );
        const inferenceServerTaskInfo = protectRecordSharedInstruction
          ? omitTaskInstruction(serverTaskInfo)
          : serverTaskInfo;
        const nextInferenceTaskInfo = {
          ...currentInferenceTaskInfo,
          ...inferenceServerTaskInfo,
          taskType: 'inference',
        };
        const currentInferenceTaskKey = getInferenceTaskInfoKey(currentInferenceTaskInfo);
        const nextInferenceTaskKey = getInferenceTaskInfoKey(nextInferenceTaskInfo);
        if (
          hasLocalInferenceTaskInfoEdit(state) &&
          nextInferenceTaskKey !== currentInferenceTaskKey
        ) {
          return;
        }
        if (
          !hasLocalInferenceTaskInfoEdit(state) &&
          state.inferenceTaskInfoSync.staleEchoTaskKey &&
          nextInferenceTaskKey === state.inferenceTaskInfoSync.staleEchoTaskKey &&
          state.inferenceTaskInfoSync.serverTaskKey === currentInferenceTaskKey &&
          nextInferenceTaskKey !== currentInferenceTaskKey
        ) {
          return;
        }
        applyInferenceTaskInfo(state, nextInferenceTaskInfo);
        state.inferenceTaskInfoSync.serverTaskKey = nextInferenceTaskKey;
        state.inferenceTaskInfoSync.editBaseServerTaskKey = nextInferenceTaskKey;
        state.inferenceTaskInfoSync.staleEchoTaskKey = '';
        state.inferenceTaskInfoSync.serverTaskInfo = nextInferenceTaskInfo;
        state.inferenceTaskInfoSync.dirty = false;
        state.inferenceTaskInfoSync.syncStatus = 'synced';
        state.inferenceTaskInfoSync.syncMessage = SYNCED_MESSAGE;
        const nextRecordTaskInfo = buildRecordTaskInfo(state);
        const nextRecordTaskKey = getRecordTaskInfoKey(nextRecordTaskInfo);
        if (
          !hasLocalRecordEdit &&
          hasRecordTaskIdentity(nextRecordTaskInfo) &&
          nextRecordTaskKey !== currentRecordTaskKey &&
          nextRecordTaskKey !== state.taskInfoSync.serverTaskKey
        ) {
          state.taskInfoSync.editBaseServerTaskKey = state.taskInfoSync.serverTaskKey;
          state.taskInfoSync.dirty = true;
          state.taskInfoSync.conflict = false;
          state.taskInfoSync.syncStatus = 'pending';
          state.taskInfoSync.syncMessage = 'Task info changed; syncing soon...';
        }
        return;
      }

      const currentRecordTaskInfo = buildRecordTaskInfo(state);
      const protectSharedInstruction = Boolean(
        hasLocalInferenceTaskInfoEdit(state) &&
        Object.prototype.hasOwnProperty.call(serverTaskInfo, 'taskInstruction') &&
        getInstructionKey(serverTaskInfo) !== getInstructionKey(currentRecordTaskInfo)
      );
      const recordServerTaskInfo = protectSharedInstruction
        ? omitTaskInstruction(serverTaskInfo)
        : serverTaskInfo;
      const nextRecordTaskInfo = {
        ...currentRecordTaskInfo,
        ...recordServerTaskInfo,
        taskType: 'record',
      };
      const previousRecordTaskInfo = state.taskInfoSync.serverTaskInfo;
      const staleRecordSharedEcho = Boolean(
        previousRecordTaskInfo &&
        getRecordIdentityKey(nextRecordTaskInfo) === getRecordIdentityKey(previousRecordTaskInfo) &&
        getRecordIdentityKey(nextRecordTaskInfo) === getRecordIdentityKey(currentRecordTaskInfo) &&
        getInstructionKey(nextRecordTaskInfo) === getInstructionKey(previousRecordTaskInfo) &&
        getInstructionKey(nextRecordTaskInfo) !== getInstructionKey(currentRecordTaskInfo)
      );
      if (staleRecordSharedEcho) {
        return;
      }
      const serverTaskKey = getRecordTaskInfoKey(nextRecordTaskInfo);
      const localTaskKey = getRecordTaskInfoKey(currentRecordTaskInfo);
      if (
        serverTaskKey === state.taskInfoSync.serverTaskKey &&
        serverTaskKey === localTaskKey &&
        !state.taskInfoSync.dirty &&
        !state.taskInfoSync.conflict &&
        state.taskInfoSync.syncStatus === 'synced'
      ) {
        return;
      }
      state.taskInfoSync.serverTaskKey = serverTaskKey;
      state.taskInfoSync.serverTaskInfo = nextRecordTaskInfo;

      if (state.taskInfoSync.dirty) {
        if (serverTaskKey === localTaskKey) {
          state.taskInfoSync.editBaseServerTaskKey = serverTaskKey;
          state.taskInfoSync.dirty = false;
          state.taskInfoSync.conflict = false;
          state.taskInfoSync.syncStatus = 'synced';
          state.taskInfoSync.syncMessage = SYNCED_MESSAGE;
        } else if (serverTaskKey !== state.taskInfoSync.editBaseServerTaskKey) {
          state.taskInfoSync.conflict = true;
          state.taskInfoSync.syncStatus = 'conflict';
          state.taskInfoSync.syncMessage = CONFLICT_MESSAGE;
        }
        return;
      }

      if (serverTaskKey !== localTaskKey) {
        applyRecordTaskInfo(state, nextRecordTaskInfo);
      }
      state.taskInfoSync.editBaseServerTaskKey = serverTaskKey;
      state.taskInfoSync.conflict = false;
      state.taskInfoSync.syncStatus = 'synced';
      state.taskInfoSync.syncMessage = SYNCED_MESSAGE;
    },
    applyServerTaskInfo: (state) => {
      if (!state.taskInfoSync.serverTaskInfo) return;
      applyRecordTaskInfo(state, state.taskInfoSync.serverTaskInfo);
      state.taskInfoSync.editBaseServerTaskKey = state.taskInfoSync.serverTaskKey;
      state.taskInfoSync.dirty = false;
      state.taskInfoSync.conflict = false;
      state.taskInfoSync.syncStatus = state.taskInfoSync.serverTaskKey ? 'synced' : 'idle';
      state.taskInfoSync.syncMessage = state.taskInfoSync.serverTaskKey ? SYNCED_MESSAGE : '';
    },
  },
});

export const selectRecordTaskInfo = createSelector(
  [selectTasksState],
  (tasks) => buildRecordTaskInfo(tasks)
);

export const selectInferenceTaskInfo = createSelector(
  [selectTasksState],
  (tasks) => buildInferenceTaskInfo(tasks)
);

export const {
  setTaskInfo,
  setRecordTaskInfo,
  setInferenceTaskInfo,
  setSharedTaskInstruction,
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
  setInferenceMode,
  setHeartbeatStatus,
  setLastHeartbeatTime,
  setJoystickMode,
  setRecordingMonitor,
  setCameraRecordingMonitor,
  setPlannedCount,
  setPlannedSubTasks,
  setPlannedSubTaskAt,
  setSlotToServerIdx,
  setActiveSlotIndex,
  resetSegmentPlan,
  resetSegmentProgress,
  markLocalTaskInfoEdited,
  markTaskInfoSyncPending,
  markTaskInfoSyncing,
  markTaskInfoSyncSuccess,
  markTaskInfoSyncFailed,
  markTaskInfoSyncMissing,
  markInferenceTaskInfoSyncPending,
  markInferenceTaskInfoSyncing,
  markInferenceTaskInfoSyncSubmitted,
  markInferenceTaskInfoSyncSuccess,
  markInferenceTaskInfoSyncFailed,
  receiveServerRecordTaskInfo,
  applyServerTaskInfo,
} = taskSlice.actions;

export default taskSlice.reducer;
