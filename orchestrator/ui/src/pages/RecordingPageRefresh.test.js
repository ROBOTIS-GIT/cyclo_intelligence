import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import OperationWorkspace from './OperationWorkspace';
import taskReducer, { setRecordStatus } from '../features/tasks/taskSlice';
import uiReducer, { moveToPage } from '../features/ui/uiSlice';
import PageType from '../constants/pageType';
import { RecordPhase } from '../constants/taskPhases';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('react-hot-toast', () => ({
  __esModule: true, default: { dismiss: jest.fn() }, useToasterStore: () => ({ toasts: [] }),
}));
jest.mock('../components/RecordControlPanel', () => () => null);
jest.mock('../components/InferenceControlPanel', () => () => null);
jest.mock('../components/HeartbeatStatus', () => () => null);
jest.mock('../components/InlineSystemStatus', () => () => null);
jest.mock('../components/ImageGrid', () => ({ responsive }) => <div data-testid="cameras" data-responsive={responsive} />);
jest.mock('../components/RobotViewer3D', () => ({ showSourceSelector }) => <select aria-label="Visualization source" hidden={!showSourceSelector}><option>State</option><option>Action</option></select>);
jest.mock('../components/SegmentPanel', () => () => <input aria-label="Record draft" defaultValue="record" />);
jest.mock('../components/InferencePanel', () => () => <input aria-label="Inference draft" defaultValue="inference" />);
jest.mock('../components/RecordTopicMonitor', () => () => null);

test('tabs keep cameras, visualization selection and both settings mounted without commands', () => {
  const store = configureStore({ reducer: { tasks: taskReducer, ui: uiReducer } });
  store.dispatch(moveToPage(PageType.RECORD));
  const command = jest.fn().mockResolvedValue({ success: true });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand: command });
  render(<Provider store={store}><OperationWorkspace /></Provider>);
  const cameras = screen.getByTestId('cameras');
  expect(cameras).toHaveAttribute('data-responsive', 'true');
  expect(screen.getByLabelText('System metrics')).toHaveClass('hidden', 'lg:block');
  const source = screen.getByRole('combobox', { name: 'Visualization source' });
  const record = screen.getByLabelText('Record draft');
  const inference = screen.getByLabelText('Inference draft');
  fireEvent.change(record, { target: { value: 'retained record' } });
  fireEvent.change(source, { target: { value: 'Action' } });
  command.mockClear();
  fireEvent.click(screen.getByRole('tab', { name: 'Inference' }));
  expect(record).not.toBeVisible();
  expect(inference).toBeVisible();
  fireEvent.change(inference, { target: { value: 'retained model' } });
  fireEvent.click(screen.getByRole('tab', { name: 'Record' }));
  expect(record).toBeVisible();
  expect(record).toHaveValue('retained record');
  expect(inference).toHaveValue('retained model');
  expect(screen.getByTestId('cameras')).toBe(cameras);
  expect(screen.getByRole('combobox', { name: 'Visualization source' })).toBe(source);
  expect(source).toHaveValue('Action');
  expect(command).not.toHaveBeenCalled();
});

test.each([PageType.RECORD, PageType.INFERENCE])(
  '%s tab entry does not reset topics during recording or saving', (page) => {
    const initial = taskReducer(undefined, { type: '@@INIT' });
    const store = configureStore({ reducer: { tasks: taskReducer, ui: uiReducer }, preloadedState: { tasks: {
      ...initial, recordStatus: { ...initial.recordStatus, recordPhase: RecordPhase.RECORDING, running: true },
    } } });
    store.dispatch(moveToPage(page));
    const command = jest.fn().mockResolvedValue({ success: true });
    useRosServiceCaller.mockReturnValue({ sendRecordCommand: command });
    render(<Provider store={store}><OperationWorkspace /></Provider>);
    expect(command).not.toHaveBeenCalled();
    act(() => store.dispatch(setRecordStatus({ recordPhase: RecordPhase.SAVING, running: false })));
    expect(command).not.toHaveBeenCalled();
    act(() => store.dispatch(setRecordStatus({ recordPhase: RecordPhase.READY, running: false })));
    expect(command).toHaveBeenCalledTimes(1);
    expect(command).toHaveBeenCalledWith('refresh_topics');
  },
);
