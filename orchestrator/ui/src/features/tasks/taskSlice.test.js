import reducer, {
  applyServerTaskInfo,
  markInferenceTaskInfoSyncSubmitted,
  markInferenceTaskInfoSyncSuccess,
  markLocalTaskInfoEdited,
  markTaskInfoSyncSuccess,
  persistRobotType,
  receiveServerRecordTaskInfo,
  resolveInitialRobotType,
  ROBOT_TYPE_STORAGE_KEY,
  ROBOT_TYPE_STATUS_GUARD_MS,
  selectInferenceTaskInfo,
  selectRecordTaskInfo,
  selectRobotType,
  setCameraRecordingMonitor,
  setInferenceMode,
  setInferenceTaskInfo,
  setRecordTaskInfo,
  setRecordingMonitor,
} from './taskSlice';
import {
  getInferenceTaskInfoKey,
  getRecordTaskInfoKey,
} from '../../utils/taskInfoSync';

const makeStorage = (initial = {}) => {
  const values = { ...initial };
  return {
    getItem: jest.fn((key) => (
      Object.prototype.hasOwnProperty.call(values, key) ? values[key] : null
    )),
    setItem: jest.fn((key, value) => {
      values[key] = value;
    }),
    removeItem: jest.fn((key) => {
      delete values[key];
    }),
    values,
  };
};

describe('taskSlice task ownership', () => {
  test('defaults inference to simulation mode', () => {
    const state = reducer(undefined, { type: '@@INIT' });
    const inferenceInfo = selectInferenceTaskInfo({ tasks: state });

    expect(inferenceInfo.inferenceMode).toBe('simulation');
    expect(inferenceInfo.actionRequestMode).toBe('async');
    expect(inferenceInfo.accelerationMode).toBe('pytorch');
  });

  test('sets inference mode without changing record identity', () => {
    const withRecordInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo({
        taskNum: '7',
        taskName: 'record-task',
        taskType: 'record',
        taskInstruction: ['record prompt'],
      })
    );

    const next = reducer(withRecordInfo, setInferenceMode('robot'));

    expect(selectInferenceTaskInfo({ tasks: next }).inferenceMode).toBe('robot');
    expect(selectRecordTaskInfo({ tasks: next }).taskNum).toBe('7');
    expect(selectRecordTaskInfo({ tasks: next }).taskName).toBe('record-task');
  });

  test('record echo hydrates record identity, shared instruction, and segment plan', () => {
    const next = reducer(
      undefined,
      receiveServerRecordTaskInfo({
        taskNum: '11',
        taskName: 'task',
        taskType: 'record',
        taskInstruction: ['main'],
        subtaskInstruction: ['sub 1', 'sub 2', 'sub 3'],
      })
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskNum).toBe('11');
    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual(['main']);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual(['main']);
    expect(next.plannedCount).toBe(3);
    expect(next.plannedSubTasks).toEqual(['sub 1', 'sub 2', 'sub 3']);
    expect(next.slotToServerIdx).toEqual([-1, -1, -1]);
    expect(next.activeSlotIndex).toBe(0);
  });

  test('inference echo updates inference settings and shared instruction only', () => {
    const withRecordInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo({
        taskNum: '7',
        taskName: 'record-task',
        taskType: 'record',
        taskInstruction: ['old prompt'],
        subtaskInstruction: ['record sub 1', 'record sub 2'],
      })
    );

    const next = reducer(
      withRecordInfo,
      receiveServerRecordTaskInfo({
        taskNum: '999',
        taskName: 'inference-task',
        taskType: 'inference',
        taskInstruction: ['new prompt'],
        subtaskInstruction: ['should not replace record subtasks'],
        serviceType: 'groot',
        actionRequestMode: 'sync',
      })
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskNum).toBe('7');
    expect(selectRecordTaskInfo({ tasks: next }).taskName).toBe('record-task');
    expect(selectRecordTaskInfo({ tasks: next }).subtaskInstruction).toEqual([
      'record sub 1',
      'record sub 2',
    ]);
    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual(['new prompt']);
    expect(selectInferenceTaskInfo({ tasks: next }).serviceType).toBe('groot');
    expect(selectInferenceTaskInfo({ tasks: next }).actionRequestMode).toBe('sync');
    expect(next.plannedSubTasks).toEqual(['record sub 1', 'record sub 2']);
  });

  test('stale record echo does not revert an inference-owned instruction edit', () => {
    const baseRecordTaskInfo = {
      taskNum: '7',
      taskName: 'record-task',
      taskType: 'record',
      taskInstruction: ['old prompt'],
      subtaskInstruction: ['record sub'],
    };
    const withRecordInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(baseRecordTaskInfo)
    );
    const edited = reducer(
      reducer(
        withRecordInfo,
        setInferenceTaskInfo({ taskInstruction: ['new prompt'] })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );
    const submitted = reducer(edited, markInferenceTaskInfoSyncSubmitted());

    const next = reducer(
      submitted,
      receiveServerRecordTaskInfo(baseRecordTaskInfo)
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual(['new prompt']);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual(['new prompt']);
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('syncing');
  });

  test('stale inference echo does not revert an inference-owned instruction edit', () => {
    const oldInferenceTaskInfo = {
      taskType: 'inference',
      taskInstruction: ['old prompt'],
      policyPath: '/policy/old',
      serviceType: 'groot',
    };
    const withInferenceInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );
    const edited = reducer(
      reducer(
        withInferenceInfo,
        setInferenceTaskInfo({ taskInstruction: ['new prompt'] })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );

    const next = reducer(
      edited,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );

    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(next.inferenceTaskInfoSync.dirty).toBe(true);
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('pending');
  });

  test('matching inference echo accepts an inference-owned instruction edit', () => {
    const edited = reducer(
      reducer(
        undefined,
        setInferenceTaskInfo({
          taskInstruction: ['new prompt'],
          policyPath: '/policy/new',
          serviceType: 'groot',
        })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );

    const next = reducer(
      edited,
      receiveServerRecordTaskInfo({
        taskType: 'inference',
        taskInstruction: ['new prompt'],
        policyPath: '/policy/new',
        serviceType: 'groot',
      })
    );

    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(next.inferenceTaskInfoSync.dirty).toBe(false);
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
  });

  test('successful inference sync marks local task info synced without echo', () => {
    const edited = reducer(
      reducer(
        undefined,
        setInferenceTaskInfo({
          taskInstruction: ['new prompt'],
          policyPath: '/policy/new',
          serviceType: 'groot',
        })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );
    const taskKey = getInferenceTaskInfoKey(selectInferenceTaskInfo({ tasks: edited }));

    const next = reducer(
      edited,
      markInferenceTaskInfoSyncSuccess({ taskKey })
    );

    expect(next.inferenceTaskInfoSync.dirty).toBe(false);
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
    expect(next.inferenceTaskInfoSync.serverTaskKey).toBe(taskKey);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
  });

  test('stale inference echo after successful sync does not revert instruction', () => {
    const oldInferenceTaskInfo = {
      taskType: 'inference',
      taskInstruction: ['old prompt'],
      policyPath: '/policy/old',
      serviceType: 'groot',
    };
    const withInferenceInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );
    const edited = reducer(
      reducer(
        withInferenceInfo,
        setInferenceTaskInfo({
          taskInstruction: ['new prompt'],
          policyPath: '/policy/new',
          serviceType: 'groot',
        })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );
    const synced = reducer(
      edited,
      markInferenceTaskInfoSyncSuccess({
        taskKey: getInferenceTaskInfoKey(selectInferenceTaskInfo({ tasks: edited })),
      })
    );

    const next = reducer(
      synced,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );

    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).policyPath).toBe('/policy/new');
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
  });

  test('new inference echo after successful sync can still update other tabs', () => {
    const oldInferenceTaskInfo = {
      taskType: 'inference',
      taskInstruction: ['old prompt'],
      policyPath: '/policy/old',
      serviceType: 'groot',
    };
    const withInferenceInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );
    const edited = reducer(
      reducer(
        withInferenceInfo,
        setInferenceTaskInfo({
          taskInstruction: ['new prompt'],
          policyPath: '/policy/new',
          serviceType: 'groot',
        })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );
    const synced = reducer(
      edited,
      markInferenceTaskInfoSyncSuccess({
        taskKey: getInferenceTaskInfoKey(selectInferenceTaskInfo({ tasks: edited })),
      })
    );

    const next = reducer(
      synced,
      receiveServerRecordTaskInfo({
        taskType: 'inference',
        taskInstruction: ['external prompt'],
        policyPath: '/policy/external',
        serviceType: 'groot',
      })
    );

    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'external prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).policyPath).toBe('/policy/external');
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
  });

  test('matching inference echo accepts backend-normalized numeric defaults', () => {
    const edited = reducer(
      reducer(
        undefined,
        setInferenceTaskInfo({
          taskInstruction: ['new prompt'],
          policyPath: '/policy/new',
          serviceType: 'groot',
          controlHz: '',
          inferenceHz: '',
          chunkAlignWindowS: '',
        })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );

    const next = reducer(
      edited,
      receiveServerRecordTaskInfo({
        taskType: 'inference',
        taskInstruction: ['new prompt'],
        policyPath: '/policy/new',
        serviceType: 'groot',
        controlHz: 100,
        inferenceHz: 15,
        chunkAlignWindowS: 0.3,
      })
    );

    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(next.inferenceTaskInfoSync.dirty).toBe(false);
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
  });

  test('first stale record echo does not revert an inference-owned instruction edit', () => {
    const edited = reducer(
      reducer(
        undefined,
        setInferenceTaskInfo({ taskInstruction: ['new prompt'] })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );

    const next = reducer(
      edited,
      receiveServerRecordTaskInfo({
        taskNum: '7',
        taskName: 'record-task',
        taskType: 'record',
        taskInstruction: ['old prompt'],
        subtaskInstruction: ['record sub'],
      })
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskNum).toBe('7');
    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(next.plannedSubTasks).toEqual(['record sub']);
  });

  test('stale inference echo does not revert a record-owned instruction edit', () => {
    const oldInferenceTaskInfo = {
      taskType: 'inference',
      taskInstruction: ['old prompt'],
      policyPath: '/policy/old',
      serviceType: 'groot',
    };
    const withInferenceInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );
    const edited = reducer(
      reducer(
        withInferenceInfo,
        setRecordTaskInfo({
          taskNum: '7',
          taskName: 'record-task',
          taskInstruction: ['record prompt'],
        })
      ),
      markLocalTaskInfoEdited({ source: 'record' })
    );

    const next = reducer(
      edited,
      receiveServerRecordTaskInfo(oldInferenceTaskInfo)
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'record prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'record prompt',
    ]);
    expect(next.taskInfoSync.dirty).toBe(true);
  });

  test('record sync key ignores inference-only fields from server echo', () => {
    const local = reducer(
      reducer(
        undefined,
        setRecordTaskInfo({
          taskNum: '7',
          taskName: 'record-task',
          taskInstruction: ['prompt'],
          subtaskInstruction: ['sub'],
        })
      ),
      markTaskInfoSyncSuccess()
    );

    const next = reducer(
      local,
      receiveServerRecordTaskInfo({
        taskNum: '7',
        taskName: 'record-task',
        taskType: 'record',
        taskInstruction: ['prompt'],
        subtaskInstruction: ['sub'],
        serviceType: 'lerobot',
        inferenceMode: 'simulation',
        actionRequestMode: 'async',
        accelerationMode: 'pytorch',
      })
    );

    expect(next.taskInfoSync.syncStatus).toBe('synced');
    expect(next.taskInfoSync.dirty).toBe(false);
    expect(next.taskInfoSync.serverTaskKey).toBe(
      getRecordTaskInfoKey(selectRecordTaskInfo({ tasks: next }))
    );
  });

  test('new record echo can still update shared instruction after inference edit is synced', () => {
    const baseRecordTaskInfo = {
      taskNum: '7',
      taskName: 'record-task',
      taskType: 'record',
      taskInstruction: ['old prompt'],
      subtaskInstruction: ['record sub'],
    };
    const withRecordInfo = reducer(
      undefined,
      receiveServerRecordTaskInfo(baseRecordTaskInfo)
    );
    const edited = reducer(
      reducer(
        withRecordInfo,
        setInferenceTaskInfo({ taskInstruction: ['new prompt'] })
      ),
      markLocalTaskInfoEdited({ source: 'inference' })
    );
    const afterStaleEcho = reducer(
      edited,
      receiveServerRecordTaskInfo(baseRecordTaskInfo)
    );

    const synced = reducer(
      afterStaleEcho,
      receiveServerRecordTaskInfo({
        taskType: 'inference',
        taskInstruction: ['new prompt'],
      })
    );

    const next = reducer(
      synced,
      receiveServerRecordTaskInfo({
        ...baseRecordTaskInfo,
        taskInstruction: ['record page prompt'],
      })
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'record page prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'record page prompt',
    ]);
  });

  test('inference echo queues record sync when shared instruction changes record payload', () => {
    const withRecordInfo = reducer(
      reducer(
        undefined,
        setRecordTaskInfo({
          taskNum: '7',
          taskName: 'record-task',
          taskInstruction: ['old prompt'],
          subtaskInstruction: ['record sub'],
        })
      ),
      markTaskInfoSyncSuccess()
    );
    const oldRecordTaskKey = getRecordTaskInfoKey(
      selectRecordTaskInfo({ tasks: withRecordInfo })
    );

    const next = reducer(
      withRecordInfo,
      receiveServerRecordTaskInfo({
        taskType: 'inference',
        taskInstruction: ['new prompt'],
        policyPath: '/policy/new',
        serviceType: 'groot',
      })
    );

    expect(selectRecordTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(selectInferenceTaskInfo({ tasks: next }).taskInstruction).toEqual([
      'new prompt',
    ]);
    expect(next.taskInfoSync.serverTaskKey).toBe(oldRecordTaskKey);
    expect(next.taskInfoSync.editBaseServerTaskKey).toBe(oldRecordTaskKey);
    expect(next.taskInfoSync.dirty).toBe(true);
    expect(next.taskInfoSync.conflict).toBe(false);
    expect(next.taskInfoSync.syncStatus).toBe('pending');
    expect(next.inferenceTaskInfoSync.syncStatus).toBe('synced');
  });
});

describe('taskSlice robot type session state', () => {
  test('restores robot type from tab session storage', () => {
    const storage = makeStorage({
      [ROBOT_TYPE_STORAGE_KEY]: ' ffw_sg2_rev2 ',
    });

    expect(resolveInitialRobotType(storage)).toBe('ffw_sg2_rev2');
  });

  test('persists and clears robot type', () => {
    const storage = makeStorage();

    persistRobotType(' ffw_sg2_rev2 ', storage);
    persistRobotType('', storage);

    expect(storage.setItem).toHaveBeenCalledWith(
      ROBOT_TYPE_STORAGE_KEY,
      'ffw_sg2_rev2'
    );
    expect(storage.removeItem).toHaveBeenCalledWith(ROBOT_TYPE_STORAGE_KEY);
  });

  test('user-selected robot type ignores stale status during guard window', () => {
    const selectedAtMs = 1000;
    const selected = reducer(
      undefined,
      selectRobotType({
        robotType: 'ffw_sg2_rev2',
        source: 'user',
        selectedAtMs,
      })
    );

    const staleStatus = reducer(
      selected,
      selectRobotType({
        robotType: 'ffw_sg2_rev1',
        source: 'status',
        receivedAtMs: selectedAtMs + 100,
      })
    );

    expect(staleStatus.robotType).toBe('ffw_sg2_rev2');
    expect(staleStatus.robotTypeStatusGuardUntilMs).toBe(
      selectedAtMs + ROBOT_TYPE_STATUS_GUARD_MS
    );
  });

  test('matching robot status clears user-selection guard', () => {
    const selectedAtMs = 1000;
    const selected = reducer(
      undefined,
      selectRobotType({
        robotType: 'ffw_sg2_rev2',
        source: 'user',
        selectedAtMs,
      })
    );

    const confirmed = reducer(
      selected,
      selectRobotType({
        robotType: 'ffw_sg2_rev2',
        source: 'status',
        receivedAtMs: selectedAtMs + 100,
      })
    );

    expect(confirmed.robotType).toBe('ffw_sg2_rev2');
    expect(confirmed.robotTypeStatusGuardUntilMs).toBe(0);
  });

  test('robot status can update after user-selection guard expires', () => {
    const selectedAtMs = 1000;
    const selected = reducer(
      undefined,
      selectRobotType({
        robotType: 'ffw_sg2_rev2',
        source: 'user',
        selectedAtMs,
      })
    );

    const laterStatus = reducer(
      selected,
      selectRobotType({
        robotType: 'ffw_sg2_rev1',
        source: 'status',
        receivedAtMs: selectedAtMs + ROBOT_TYPE_STATUS_GUARD_MS + 1,
      })
    );

    expect(laterStatus.robotType).toBe('ffw_sg2_rev1');
  });
});

describe('taskSlice recording progress', () => {
  test('resets saved segment progress when applying a server task', () => {
    const initial = reducer(undefined, { type: '@@INIT' });
    const serverTaskInfo = {
      taskNum: '2',
      taskName: 'new',
      taskInstruction: ['new main'],
      subtaskInstruction: ['new 1', 'new 2'],
    };
    const state = {
      ...initial,
      plannedSubTasks: ['old 1', 'old 2'],
      plannedCount: 2,
      slotToServerIdx: [0, 1],
      activeSlotIndex: 1,
      taskInfoSync: {
        ...initial.taskInfoSync,
        serverTaskKey: '2:new',
        serverTaskInfo,
      },
    };

    const next = reducer(state, applyServerTaskInfo());

    expect(next.plannedSubTasks).toEqual(['new 1', 'new 2']);
    expect(next.slotToServerIdx).toEqual([-1, -1]);
    expect(next.activeSlotIndex).toBe(0);
  });
});

describe('taskSlice recording monitor', () => {
  test('preserves camera monitor rows when rosbag monitor updates', () => {
    const initial = reducer(undefined, { type: '@@INIT' });
    const withCamera = reducer(
      initial,
      setCameraRecordingMonitor([
        {
          name: '/camera/image/compressed',
          cameraName: 'cam_left_head',
          rateHz: 30,
          baselineHz: 30,
          secondsSinceLast: 0,
          status: 0,
          source: 'camera',
        },
      ])
    );

    const next = reducer(
      withCamera,
      setRecordingMonitor({
        topics: [{ name: '/joint_states', rateHz: 100, baselineHz: 100, status: 0 }],
        totalReceived: 10,
        totalWritten: 8,
      })
    );

    expect(next.recordingMonitor.topics).toHaveLength(1);
    expect(next.recordingMonitor.cameraTopics).toHaveLength(1);
    expect(next.recordingMonitor.cameraTopics[0].cameraName).toBe('cam_left_head');
    expect(next.recordingMonitor.totalReceived).toBe(10);
    expect(next.recordingMonitor.totalWritten).toBe(8);
  });

  test('updates camera monitor rows independently', () => {
    const state = reducer(
      undefined,
      setRecordingMonitor({
        topics: [{ name: '/joint_states', rateHz: 100, baselineHz: 100, status: 0 }],
      })
    );

    const next = reducer(
      state,
      setCameraRecordingMonitor([
        {
          name: '/camera/image/compressed',
          cameraName: 'cam_right_wrist',
          rateHz: 0,
          baselineHz: 30,
          secondsSinceLast: 2.5,
          status: 2,
          source: 'camera',
        },
      ])
    );

    expect(next.recordingMonitor.topics).toHaveLength(1);
    expect(next.recordingMonitor.cameraTopics).toHaveLength(1);
    expect(next.recordingMonitor.cameraTopics[0].status).toBe(2);
  });
});
