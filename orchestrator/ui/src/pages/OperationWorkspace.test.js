import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import OperationWorkspace from './OperationWorkspace';
import taskReducer, { setRecordStatus, setInferenceStatus } from '../features/tasks/taskSlice';
import uiReducer, { moveToPage } from '../features/ui/uiSlice';
import rosReducer from '../features/ros/rosSlice';
import PageType from '../constants/pageType';
import { InferencePhase, RecordPhase } from '../constants/taskPhases';
import { PolicyCatalogProvider } from '../contexts/PolicyCatalogContext';
import { testPolicyCatalog } from '../testUtils/policyCatalog';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { ThemeProvider } from '../contexts/ThemeContext';
import ThemeToggle from '../components/ThemeToggle';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('../hooks/usePolicyBackendStatus', () => ({
  ...jest.requireActual('../hooks/usePolicyBackendStatus'),
  __esModule: true,
  default: () => ({ readiness: { ready: true, state: 'ready' }, refreshStatus: jest.fn() }),
  getPolicyBackendReadiness: (status) => status,
}));
jest.mock('../components/HeartbeatStatus', () => () => null);
jest.mock('../components/InlineSystemStatus', () => () => null);
jest.mock('../components/ImageGrid', () => ({ headerActions, isRecording }) => <div role="region" aria-label="Camera feeds" data-recording={Boolean(isRecording)}>{headerActions}</div>);
jest.mock('../components/RobotViewer3D', () => () => <div data-testid="viewer-content" />);
jest.mock('../components/RecordTopicMonitor', () => ({ actions, collapsed }) => <div>Topic Monitor{actions}<div hidden={collapsed} data-testid="topic-content" /></div>);
jest.mock('../components/PolicyBackendControl', () => () => null);
jest.mock('react-hot-toast', () => {
  const toast = Object.assign(jest.fn(), { error: jest.fn(), success: jest.fn(), dismiss: jest.fn() });
  return { __esModule: true, default: toast, useToasterStore: () => ({ toasts: [] }) };
});

beforeEach(() => localStorage.clear());

function setup({ recording = true, withTheme = false, command = jest.fn().mockResolvedValue({ success: true, message: 'ok' }) } = {}) {
  const initial = taskReducer(undefined, { type: '@@INIT' });
  const store = configureStore({ reducer: { tasks: taskReducer, ui: uiReducer, ros: rosReducer }, preloadedState: { tasks: {
    ...initial, robotType: 'ffw_sg2_rev1',
    recordTaskInfo: { ...initial.recordTaskInfo, taskNum: '42', taskName: 'demo' },
    inferenceTaskInfo: { ...initial.inferenceTaskInfo, policyId: 'groot:n17', serviceType: 'groot', policyPath: '/model', inferenceMode: 'simulation' },
    recordStatus: { ...initial.recordStatus, taskNum: '42', taskName: 'demo', topicReceived: true, running: recording,
      recordPhase: recording ? RecordPhase.RECORDING : RecordPhase.READY, currentEpisodeNumber: 7 },
    inferenceStatus: { ...initial.inferenceStatus, topicReceived: true, runtimeState: 'running',
      inferencePhase: InferencePhase.INFERENCING, loadedModelPath: '/model', loadedPolicyId: 'lerobot:act' },
  } } });
  store.dispatch(moveToPage(PageType.RECORD));
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: command, getInferenceStatus: jest.fn(), sendRobotPoseCommand: jest.fn() });
  const workspace = <Provider store={store}><PolicyCatalogProvider initialCatalog={testPolicyCatalog}>
    <OperationWorkspace />
  </PolicyCatalogProvider></Provider>;
  const view = render(withTheme ? <ThemeProvider><ThemeToggle />{workspace}</ThemeProvider> : workspace);
  return { store, command, ...view };
}

test('header identifies the robot and inference state with bordered non-interactive labels', () => {
  setup();
  expect(screen.queryByRole('heading', { name: 'Operate' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Robot type')).toHaveTextContent('ffw_sg2_rev1');
  expect(screen.getByRole('button', { name: 'Mute recording announcements' })).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByLabelText('Robot type')).toHaveClass('border', 'border-gray-300');
  expect(screen.getByLabelText('Inference status')).toHaveClass('border', 'border-gray-300');
  expect(screen.getByLabelText('Inference status')).toHaveTextContent('Inference');
  expect(screen.getByLabelText('Inference status')).toHaveTextContent('running');
});

test('recording actions use three equal columns and visible command labels', () => {
  setup();
  const group = screen.getByRole('group', { name: 'Recording actions' });
  expect(group).toHaveClass('grid', 'grid-cols-3', 'w-full');
  const buttons = within(group).getAllByRole('button');
  expect(buttons).toHaveLength(3);
  expect(buttons.map((button) => button.textContent)).toEqual(['Record', 'Save', 'Discard']);
  buttons.forEach((button) => expect(button).toHaveClass('recording-action', 'h-10'));
});

test('camera recording indication follows confirmed recorder phase independently of inference and tabs', () => {
  const { store } = setup({ recording: false });
  const cameras = screen.getByRole('region', { name: 'Camera feeds' });
  expect(cameras).toHaveAttribute('data-recording', 'false');
  act(() => store.dispatch(setRecordStatus({ running: true, recordPhase: RecordPhase.RECORDING })));
  expect(cameras).toHaveAttribute('data-recording', 'true');
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  act(() => store.dispatch(setInferenceStatus({ runtimeState: 'unloaded', inferencePhase: InferencePhase.READY })));
  expect(cameras).toHaveAttribute('data-recording', 'true');
  for (const phase of [RecordPhase.SAVING, RecordPhase.PAUSED, RecordPhase.READY]) {
    act(() => store.dispatch(setRecordStatus({ running: true, recordPhase: phase })));
    expect(cameras).toHaveAttribute('data-recording', 'false');
  }
  act(() => store.dispatch(setRecordStatus({ topicReceived: false, recordPhase: RecordPhase.RECORDING })));
  expect(cameras).toHaveAttribute('data-recording', 'false');
  act(() => store.dispatch(setRecordStatus({ topicReceived: true })));
  expect(cameras).toHaveAttribute('data-recording', 'true');
});

test('save pending across tabs retains the sequence and locks discard; Stop and Clear never mutate recording', async () => {
  let resolveSave;
  const command = jest.fn((name) => name === 'stop_segment'
    ? new Promise((resolve) => { resolveSave = resolve; })
    : Promise.resolve({ success: true }));
  const { store } = setup({ command });
  fireEvent.click(screen.getByRole('button', { name: 'Save recording' }));
  expect(command.mock.calls.map(([name]) => name)).toEqual(['stop_segment']);
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  expect(screen.queryByRole('button', { name: 'Save recording' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Record' }));
  expect(screen.getByRole('button', { name: 'Save recording' })).toBeDisabled();
  expect(screen.getByRole('button', { name: 'Discard recording episode' })).toBeDisabled();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  fireEvent.click(screen.getByRole('button', { name: 'Pause inference (model stays loaded)' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith('stop_inference', {}));
  fireEvent.click(screen.getByRole('button', { name: 'Stop inference and unload model' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith('finish', {}));
  expect(store.getState().tasks.recordStatus.running).toBe(true);
  await act(async () => resolveSave({ success: true }));
  expect(command.mock.calls.map(([name]) => name)).toEqual(['stop_segment', 'stop_inference', 'finish', 'finish_episode']);
  fireEvent.click(screen.getByRole('tab', { name: 'Record' }));
  expect(screen.getByRole('button', { name: 'Save recording' })).toBeDisabled();
  act(() => store.dispatch(setRecordStatus({ running: false, recordPhase: RecordPhase.READY, currentEpisodeNumber: 8 })));
  expect(screen.getByRole('button', { name: 'Start recording' })).toBeEnabled();
});

test('only the active tab handles Escape and focused inputs suppress operation shortcuts', async () => {
  const { command } = setup();
  fireEvent.keyUp(window, { key: 'Escape' });
  await waitFor(() => expect(command).toHaveBeenCalledWith('cancel_segment', expect.any(Object)));
  expect(command.mock.calls.map(([name]) => name)).toEqual(['cancel_segment']);
  command.mockClear();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  const input = screen.getAllByPlaceholderText('Enter Task Instruction').find((element) => !element.disabled);
  input.focus();
  fireEvent.keyUp(window, { key: 'Escape' });
  expect(command).not.toHaveBeenCalled();
  input.blur();
  fireEvent.keyUp(window, { key: 'Escape' });
  await waitFor(() => expect(command).toHaveBeenCalledWith('finish', {}));
  expect(command.mock.calls.map(([name]) => name)).toEqual(['finish']);
});

test('record actions remain in Record after Clear while the Inference tab hides them', async () => {
  const { store, command } = setup();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  act(() => store.dispatch(setInferenceStatus({ runtimeState: 'unloaded', inferencePhase: InferencePhase.READY, loadedModelPath: '' })));
  expect(screen.queryByRole('button', { name: 'Discard recording episode' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Record' }));
  fireEvent.click(screen.getByRole('button', { name: 'Discard recording episode' }));
  await waitFor(() => expect(command).toHaveBeenCalledWith('discard_episode', expect.objectContaining({ segmentIndex: 8 })));
  expect(command.mock.calls.map(([name]) => name)).toEqual(['discard_episode']);
  expect(screen.queryByRole('button', { name: /inference recording/i })).not.toBeInTheDocument();
});

test('Record contains every recording action and subtask controls; inference Clear stays accessible', () => {
  setup({ recording: false });
  const panel = screen.getByRole('tabpanel', { name: 'Record' });
  for (const name of ['Start recording', 'Save recording', 'Discard recording episode']) {
    expect(within(panel).getByRole('button', { name })).toBeVisible();
  }
  fireEvent.change(within(panel).getByRole('spinbutton'), { target: { value: '2' } });
  const instructions = within(panel).getAllByPlaceholderText('Sub Task Instruction');
  expect(instructions).toHaveLength(2);
  fireEvent.change(instructions[0], { target: { value: 'Pick up the vegetable' } });
  expect(within(panel).getByRole('button', { name: 'Reset' })).toBeVisible();
  expect(within(panel).getByRole('button', { name: 'Save subtask 1' })).toBeVisible();
  expect(within(panel).getByRole('button', { name: 'Discard subtask 2' })).toBeVisible();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  expect(screen.getByRole('button', { name: 'Stop inference and unload model' })).toBeVisible();
  fireEvent.click(screen.getByRole('tab', { name: 'Record' }));
  expect(instructions[0]).toHaveValue('Pick up the vegetable');
});

test('independent monitor toggles preserve mounted content and issue no commands', () => {
  const { command } = setup();
  const viewer = screen.getByTestId('viewer-content');
  const topics = screen.getByTestId('topic-content');
  command.mockClear();
  fireEvent.click(screen.getByRole('button', { name: 'Close Topic Monitor' }));
  expect(topics).not.toBeVisible();
  expect(screen.queryByText('Topic Monitor')).not.toBeVisible();
  expect(viewer).toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Close Robot Viewer' }));
  expect(screen.getByRole('button', { name: 'Expand Robot Viewer' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Expand Topic Monitor' })).toBeVisible();
  expect(screen.getByRole('region', { name: 'Camera feeds' })).toContainElement(screen.getByRole('button', { name: 'Expand Robot Viewer' }));
  expect(screen.getByRole('region', { name: 'Camera feeds' })).toContainElement(screen.getByRole('button', { name: 'Expand Topic Monitor' }));
  expect(screen.getByLabelText('Observation monitors')).not.toBeVisible();
  expect(viewer).not.toBeVisible();
  fireEvent.click(screen.getByRole('button', { name: 'Expand Topic Monitor' }));
  expect(topics).toBeVisible();
  expect(screen.getByLabelText('Observation monitors')).toBeVisible();
  expect(screen.getByTestId('topic-content')).toBe(topics);
  expect(screen.getByTestId('viewer-content')).toBe(viewer);
  expect(command).not.toHaveBeenCalled();
});

test('monitor visibility survives remount, but never hides the toggle controls', () => {
  const { unmount } = setup();
  fireEvent.click(screen.getByRole('button', { name: 'Close Topic Monitor' }));
  fireEvent.click(screen.getByRole('button', { name: 'Close Robot Viewer' }));
  unmount();
  setup();
  expect(screen.getByTestId('topic-content')).not.toBeVisible();
  expect(screen.getByTestId('viewer-content')).not.toBeVisible();
  expect(screen.getByRole('button', { name: 'Expand Topic Monitor' })).toHaveAttribute('aria-expanded', 'false');
  expect(screen.getByRole('button', { name: 'Expand Robot Viewer' })).toHaveAttribute('aria-expanded', 'false');
});

test('split ratios persist independently without remounting monitors or sending robot commands', () => {
  const { command, unmount } = setup();
  const rows = () => screen.getByRole('separator', { name: 'Resize cameras and observation monitors' });
  const columns = () => screen.getByRole('separator', { name: 'Resize Robot Viewer and Topic Monitor' });
  const viewer = screen.getByTestId('viewer-content');
  command.mockClear();
  fireEvent.keyDown(rows(), { key: 'ArrowDown' });
  fireEvent.keyDown(columns(), { key: 'Home' });
  expect(rows()).toHaveAttribute('aria-valuenow', '65');
  expect(columns()).toHaveAttribute('aria-valuenow', '25');
  fireEvent.click(screen.getByRole('button', { name: 'Close Topic Monitor' }));
  expect(screen.queryByRole('separator', { name: 'Resize Robot Viewer and Topic Monitor' })).toBeNull();
  fireEvent.click(screen.getByRole('button', { name: 'Expand Topic Monitor' }));
  expect(screen.getByTestId('viewer-content')).toBe(viewer);
  expect(columns()).toHaveAttribute('aria-valuenow', '25');
  expect(command).not.toHaveBeenCalled();
  unmount();
  setup();
  expect(rows()).toHaveAttribute('aria-valuenow', '65');
  expect(columns()).toHaveAttribute('aria-valuenow', '25');
  fireEvent.doubleClick(rows());
  fireEvent.doubleClick(columns());
  expect(rows()).toHaveAttribute('aria-valuenow', '60');
  expect(columns()).toHaveAttribute('aria-valuenow', '50');
});

test('hidden Record settings cannot receive recording shortcuts', () => {
  const { command } = setup();
  fireEvent.click(screen.getByRole('button', { name: 'Collapse operation settings' }));
  command.mockClear();
  fireEvent.keyUp(window, { key: 'Escape' });
  expect(command).not.toHaveBeenCalled();
});

test('blocked browser storage does not disable visibility controls', () => {
  const read = jest.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('blocked'); });
  const write = jest.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('blocked'); });
  try {
    setup();
    fireEvent.click(screen.getByRole('button', { name: 'Close Topic Monitor' }));
    expect(screen.getByTestId('topic-content')).not.toBeVisible();
  } finally {
    read.mockRestore();
    write.mockRestore();
  }
});

test('inference commands live only in the Inference tab and collapsing it preserves the session', () => {
  const { store, command } = setup();
  expect(screen.queryByRole('button', { name: 'Stop inference and unload model' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  const panel = screen.getByRole('tabpanel', { name: 'Inference' });
  const clear = within(panel).getByRole('button', { name: 'Stop inference and unload model' });
  expect(clear).toBeVisible();
  const taskState = store.getState().tasks;
  command.mockClear();
  fireEvent.click(screen.getByRole('button', { name: 'Collapse operation settings' }));
  expect(clear).not.toBeVisible();
  expect(screen.getByLabelText('Inference status')).toHaveTextContent('running');
  fireEvent.keyUp(window, { key: 'Escape' });
  expect(command).not.toHaveBeenCalled();
  fireEvent.click(screen.getByRole('button', { name: 'Expand operation settings' }));
  expect(clear).toBeVisible();
  expect(within(screen.getByRole('tabpanel', { name: 'Inference' })).getByRole('button', { name: 'Stop inference and unload model' })).toBe(clear);
  expect(store.getState().tasks).toBe(taskState);
  expect(command).not.toHaveBeenCalled();
});

test('Operate respects the saved dark theme and live theme changes without resetting the session', async () => {
  localStorage.setItem('cyclo_intelligence_theme', 'dark');
  try {
    const { unmount, store, command } = setup({ withTheme: true });
    await act(async () => {});
    expect(document.documentElement).toHaveClass('dark');
    const tasks = store.getState().tasks;
    const viewer = screen.getByTestId('viewer-content');
    command.mockClear();
    for (const theme of ['Light', 'Dark']) {
      fireEvent.click(screen.getByRole('button', { name: theme, exact: true }));
      await act(async () => {});
      expect(document.documentElement.classList.contains('dark')).toBe(theme === 'Dark');
      expect(localStorage.getItem('cyclo_intelligence_theme')).toBe(theme.toLowerCase());
      expect(screen.getByTestId('viewer-content')).toBe(viewer);
      expect(store.getState().tasks).toBe(tasks);
    }
    expect(command).not.toHaveBeenCalled();
    unmount();
    expect(document.documentElement).toHaveClass('dark');
  } finally {
    document.documentElement.classList.remove('dark');
    document.documentElement.removeAttribute('data-theme');
  }
});
