import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InferencePanel from './InferencePanel';
import taskReducer, { setInferenceCheckpointInstruction } from '../features/tasks/taskSlice';
import { InferencePhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('react-hot-toast', () => ({
  __esModule: true,
  default: {
    error: jest.fn(),
    success: jest.fn(),
  },
}));

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('./InferenceModelSelector', () => () => <div />);
jest.mock('./PolicyBackendControl', () => () => <div />);
jest.mock('./TrtEngineControl', () => () => <div />);
jest.mock('./FileBrowserModal', () => ({ isOpen, shortcuts, onFileSelect }) => isOpen && (
  <div>{shortcuts.map(({ label, path }) => <button key={path} onClick={() => onFileSelect({ full_path:path+'/checkpoint', is_directory:true })}>{label}</button>)}</div>
));
jest.mock('./Tooltip', () => ({ children }) => <>{children}</>);

const renderPanel = ({
  inferenceMode,
  inferencePhase = InferencePhase.READY,
  initialPoseSync = true,
  inferenceHz = 15,
  controlHz = 100,
  serviceType = 'lerobot',
  policyType = 'act',
  policyPath = '',
  taskInstruction = [],
} = {}) => {
  const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const initialTasks = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({
    reducer: { tasks: taskReducer },
    preloadedState: {
      tasks: {
        ...initialTasks,
        sharedTaskInfo: { taskInstruction },
        inferenceTaskInfo: {
          ...initialTasks.inferenceTaskInfo,
          inferenceMode,
          serviceType,
          policyType,
          policyPath,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
        },
        taskInfo: {
          ...initialTasks.taskInfo,
          inferenceMode,
          initialPoseSync,
          initialPoseSyncDurationS: 5.0,
          inferenceHz,
          controlHz,
        },
        inferenceStatus: {
          ...initialTasks.inferenceStatus,
          inferencePhase,
        },
      },
    },
  });

  render(
    <Provider store={store}>
      <InferencePanel />
    </Provider>
  );
  return { store, sendRecordCommand };
};

describe('InferencePanel initial pose sync settings', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('preserves but disables initial pose sync in simulation mode', () => {
    renderPanel({ inferenceMode: 'simulation' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeChecked();
    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('allows initial pose sync editing for an idle real robot session', () => {
    renderPanel({ inferenceMode: 'robot' });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeEnabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('shows duration only after initial pose sync is enabled', () => {
    renderPanel({ inferenceMode: 'robot', initialPoseSync: false });

    expect(screen.queryByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }));

    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeEnabled();
  });

  test('makes initial pose sync settings read-only while synchronizing', () => {
    renderPanel({
      inferenceMode: 'robot',
      inferencePhase: InferencePhase.SYNCING,
    });

    expect(screen.getByRole('checkbox', { name: 'Initial Pose Sync' }))
      .toBeDisabled();
    expect(screen.getByRole('spinbutton', { name: 'Initial Pose Sync duration' }))
      .toBeDisabled();
  });

  test('keeps Dataset FPS blank while the user replaces its value', () => {
    jest.useFakeTimers();
    const { sendRecordCommand } = renderPanel({ inferenceMode: 'robot' });
    const datasetFpsInput = screen.getByRole('spinbutton', { name: 'Dataset FPS' });

    fireEvent.change(datasetFpsInput, { target: { value: '' } });
    expect(datasetFpsInput).toHaveValue(null);

    act(() => {
      jest.advanceTimersByTime(1000);
    });
    expect(datasetFpsInput).toHaveValue(null);
    expect(sendRecordCommand).not.toHaveBeenCalled();
    jest.useRealTimers();
  });

  test('does not sync an empty or fractional Chunk Size', () => {
    jest.useFakeTimers();
    const { sendRecordCommand } = renderPanel();
    const input = screen.getByRole('spinbutton', { name: 'Chunk Size' });
    for (const value of ['', '1.5']) {
      fireEvent.change(input, { target: { value } });
      act(() => jest.advanceTimersByTime(1000));
      expect(sendRecordCommand).not.toHaveBeenCalled();
    }
    jest.useRealTimers();
  });

  test('shows a non-blocking warning for unusual Dataset FPS', () => {
    renderPanel({ inferenceMode: 'robot', inferenceHz: 1515, controlHz: 200 });

    expect(screen.getByRole('status', { name: 'Timing warnings' }))
      .toHaveTextContent('Dataset FPS is unusually high (1515)');
  });
});

test('checkpoint metadata fills the inference prompt while preserving the recording prompt', async () => {
  const originalFetch = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({
    available: true, conditioning: 'subtask', default_instruction: '1. Align.',
    task_instructions: ['Insert and release.'], subtask_instruction_choices: [['1. Align.'], ['2. Release.']],
  }) });
  try {
    const { store } = renderPanel({ serviceType: 'groot', policyType: 'n17', policyPath: '/workspace/model/groot/example' });
    await waitFor(() => expect(store.getState().tasks.inferenceTaskInfo.checkpointTaskInstruction).toEqual(['1. Align.']));
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction).toEqual([]);
    fireEvent.change(screen.getByLabelText('Checkpoint instruction selection'), { target: { value: '2. Release.' } });
    expect(store.getState().tasks.inferenceTaskInfo.checkpointTaskInstruction).toEqual(['2. Release.']);
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction).toEqual([]);
  } finally {
    global.fetch = originalFetch;
  }
});

test.each([true, false, 'automatic'])('subtask completion commits the next prompt only on success=%s', async (success) => {
  const originalFetch = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({
    available: true, conditioning: 'subtask', default_instruction: '1. Align.',
    task_instructions: ['Insert and release.'], subtask_instruction_choices: [['1. Align.'], ['2. Release.'], ['3. Retreat.']],
  }) });
  let resolve;
  try {
    const { store, sendRecordCommand } = renderPanel({ serviceType: 'groot', policyType: 'n17',
      policyPath: '/workspace/model/groot/example', inferencePhase: InferencePhase.INFERENCING,
      taskInstruction: ['1. Align.'] });
    sendRecordCommand.mockImplementation((command) => command === 'update_instruction'
      ? new Promise((done) => { resolve = done; }) : Promise.resolve({ success: true }));
    const button = await screen.findByRole('button', { name: 'Complete subtask & continue' });
    expect(screen.queryByRole('button', { name: 'Update Task Instruction' })).not.toBeInTheDocument();
    fireEvent.click(button);
    expect(sendRecordCommand).toHaveBeenCalledWith('update_instruction', {
      taskInstruction: ['2. Release.'], autofillEmptyTaskFields: false,
    });
    expect(screen.getByPlaceholderText('Enter Task Instruction')).toHaveValue('1. Align.');
    expect(screen.getByRole('button', { name: 'Switching subtask…' })).toBeDisabled();
    if (success === 'automatic') {
      act(() => store.dispatch(setInferenceCheckpointInstruction({
        policyPath: '/workspace/model/groot/example', serviceType: 'groot', instruction: '3. Retreat.',
      })));
    }
    await act(async () => resolve({ success: Boolean(success), message: success ? 'updated' : 'hold failed' }));
    expect(screen.getByPlaceholderText('Enter Task Instruction')).toHaveValue(success === 'automatic'
      ? '3. Retreat.' : success ? '2. Release.' : '1. Align.');
    expect(store.getState().tasks.sharedTaskInfo.taskInstruction).toEqual(['1. Align.']);
  } finally { global.fetch = originalFetch; }
});

test('whole-task checkpoints retain automatic trained instruction and legacy update control', async () => {
  const originalFetch = global.fetch;
  global.fetch = jest.fn().mockResolvedValue({ ok: true, json: async () => ({
    available: true, conditioning: 'task', default_instruction: 'Trained whole task.',
    task_instructions: ['Trained whole task.'], subtask_instruction_choices: [],
  }) });
  try {
    renderPanel({ serviceType: 'groot', policyType: 'n17', policyPath: '/workspace/model/groot/single' });
    await waitFor(() => expect(screen.getByPlaceholderText('Enter Task Instruction')).toHaveValue('Trained whole task.'));
    expect(screen.getByRole('button', { name: 'Update Task Instruction' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Complete subtask & continue' })).not.toBeInTheDocument();
  } finally { global.fetch = originalFetch; }
});

test.each([
  ['lerobot','hero_act_inference','/workspace/model/lerobot/hero_act_inference'],
  ['groot','hero_groot_inference','/workspace/model/groot/hero_groot_inference'],
])('Policy Path selects a shared checkpoint through the %s workspace shortcut', (serviceType, label, path) => {
  const { store } = renderPanel({ inferenceMode:'robot', serviceType });
  fireEvent.click(screen.getByRole('button', { name:'Browse for policy model folder' }));
  fireEvent.click(screen.getByRole('button', { name:label }));
  expect(store.getState().tasks.inferenceTaskInfo.policyPath).toBe(path+'/checkpoint');
});
