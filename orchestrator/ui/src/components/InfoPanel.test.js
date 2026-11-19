import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import InfoPanel from './InfoPanel';
import taskReducer from '../features/tasks/taskSlice';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('react-hot-toast', () => ({
  __esModule: true, default: { error: jest.fn(), success: jest.fn() },
}));

function setup(record = {}, inferencePhase = InferencePhase.PAUSED) {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer }, preloadedState: { tasks: {
    ...initial,
    recordTaskInfo: { ...initial.recordTaskInfo, taskNum: 'record-1', taskName: 'record-task' },
    taskInfoSync: { ...initial.taskInfoSync, dirty: false },
    recordStatus: { ...initial.recordStatus, topicReceived: true, ...record },
    inferenceStatus: { ...initial.inferenceStatus, inferencePhase, loadedModelPath: '/loaded-model' },
  } } });
  const sendRecordCommand = jest.fn().mockResolvedValue({ success: true });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  render(<Provider store={store}><InfoPanel /></Provider>);
  return { store, sendRecordCommand };
}

beforeEach(() => jest.clearAllMocks());

test('Record inputs can shrink within a narrow settings panel', () => {
  setup();
  expect(screen.queryByText('Add License')).not.toBeInTheDocument();
  expect(screen.queryByRole('checkbox')).not.toBeInTheDocument();
  for (const placeholder of ['Enter Task Num', 'Enter Task Name', 'Enter Task Instruction']) {
    expect(screen.getByPlaceholderText(placeholder)).toHaveClass('min-w-0');
  }
});

test.each([InferencePhase.INFERENCING, InferencePhase.PAUSED])(
  'Record Prepare selects record context while inference phase %s is retained', async (phase) => {
    const { store, sendRecordCommand } = setup({}, phase);
    expect(screen.getByPlaceholderText('Enter Task Num')).toBeEnabled();
    fireEvent.click(screen.getByTitle(/Sync task info now/));
    await waitFor(() => expect(sendRecordCommand).toHaveBeenCalledWith('prepare_session'));
    expect(sendRecordCommand.mock.calls).toEqual([['set_task_info'], ['prepare_session']]);
    expect(useRosServiceCaller).toHaveBeenCalledWith('record');
    expect(store.getState().tasks.inferenceStatus.loadedModelPath).toBe('/loaded-model');
  },
);

test.each([
  { recordPhase: RecordPhase.RECORDING },
  { recordPhase: RecordPhase.SAVING },
  { recordPhase: RecordPhase.READY, running: true },
])('recording settings stay locked across inference Clear: %j', (record) => {
  const { sendRecordCommand } = setup(record, InferencePhase.READY);
  expect(screen.getByPlaceholderText('Enter Task Num')).toBeDisabled();
  expect(screen.getByPlaceholderText('Enter Task Name')).toBeDisabled();
  expect(screen.getByPlaceholderText('Enter Task Instruction')).toBeDisabled();
  expect(screen.getByTitle(/Task is running/)).toBeDisabled();
  expect(sendRecordCommand).not.toHaveBeenCalled();
});
