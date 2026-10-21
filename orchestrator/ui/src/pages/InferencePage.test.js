import { fireEvent, render, screen } from '@testing-library/react';
import InferencePage from './InferencePage';
import taskReducer from '../features/tasks/taskSlice';

const mockRefresh = jest.fn().mockResolvedValue({ success: true });
const mockState = { tasks: { ...taskReducer(undefined, { type: '@@INIT' }), robotType: 'ffw_sg2_rev1' },
  ui: { isFirstLoad: { inference: false } } };
beforeEach(() => mockRefresh.mockResolvedValue({ success: true }));
jest.mock('react-redux', () => ({ useSelector: (fn) => fn(mockState), useDispatch: () => jest.fn() }));
jest.mock('react-hot-toast', () => ({ __esModule: true, default: { dismiss: jest.fn() }, useToasterStore: () => ({ toasts: [] }) }));
jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: () => ({ sendRecordCommand: mockRefresh }) }));
jest.mock('../components/InferencePanel', () => () => <div>Model settings</div>);
jest.mock('../components/InferenceGripperPanel', () => ({ isActive, externalBusy, onBusyChange }) => (
  <div data-testid="gripper-tools" data-active={isActive} data-locked={externalBusy}>
    <button onClick={() => onBusyChange(true)}>Move test gripper</button>
  </div>
));
jest.mock('../components/InferencePosePanel', () => ({ isActive, externalBusy, onBusyChange }) => (
  <div data-testid="pose-tools" data-active={isActive} data-locked={externalBusy}>
    <button onClick={() => onBusyChange(true)}>Move test pose</button>
  </div>
));
jest.mock('../components/InferenceControlPanel', () => ({ manualControlBusy }) => <button disabled={manualControlBusy}>Start inference</button>);
jest.mock('../components/InferenceDaggerPanel', () => ({ onBusyChange }) => <button onClick={() => onBusyChange(true)}>Start test DAgger</button>);
jest.mock('../components/HeartbeatStatus', () => () => null);
jest.mock('../components/InlineSystemStatus', () => () => null);
jest.mock('../components/ImageGrid', () => () => null);
jest.mock('../components/RobotViewer3D', () => () => null);
jest.mock('../components/RecordTopicMonitor', () => () => null);

test('gripper tab preserves model settings and forwards movement lock to Start across tab switches', () => {
  render(<InferencePage />);
  expect(screen.getByRole('tab', { name: 'Settings' })).toHaveAttribute('aria-selected', 'true');
  fireEvent.click(screen.getByRole('tab', { name: 'Gripper' }));
  expect(screen.getByRole('tab', { name: 'Gripper' })).toHaveAttribute('aria-selected', 'true');
  expect(screen.getByTestId('gripper-tools')).toHaveAttribute('data-active', 'true');
  fireEvent.click(screen.getByRole('button', { name: 'Move test gripper' }));
  expect(screen.getByRole('button', { name: 'Start inference' })).toBeDisabled();
  expect(screen.getByTestId('pose-tools')).toHaveAttribute('data-locked', 'true');
  fireEvent.click(screen.getByRole('tab', { name: 'Settings' }));
  expect(screen.getByText('Model settings')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Start inference' })).toBeDisabled();
});

test('Pose tab locks Start and the gripper even after switching back to Settings', () => {
  render(<InferencePage />);
  fireEvent.click(screen.getByRole('tab', { name:'Pose' }));
  expect(screen.getByTestId('pose-tools')).toHaveAttribute('data-active', 'true');
  fireEvent.click(screen.getByRole('button', { name:'Move test pose' }));
  fireEvent.click(screen.getByRole('tab', { name:'Settings' }));
  expect(screen.getByRole('button', { name:'Start inference' })).toBeDisabled();
  expect(screen.getByTestId('gripper-tools')).toHaveAttribute('data-locked', 'true');
});

test('tab keyboard navigation does not trigger the inference Space shortcut', () => {
  render(<InferencePage />);
  const gripper = screen.getByRole('tab', { name: 'Gripper' });
  const settings = screen.getByRole('tab', { name: 'Settings' });
  settings.focus();
  fireEvent.keyDown(settings, { key: 'ArrowRight' });
  expect(gripper).toHaveFocus();
  expect(gripper).toHaveAttribute('aria-selected', 'true');
  const pose = screen.getByRole('tab', { name:'Pose' });
  fireEvent.keyDown(gripper, { key:'ArrowRight' });
  expect(pose).toHaveFocus();
  fireEvent.keyDown(pose, { key:'ArrowRight' });
  const dagger = screen.getByRole('tab', { name:'DAgger' });
  expect(dagger).toHaveFocus();
  fireEvent.keyDown(dagger, { key:'ArrowRight' });
  expect(settings).toHaveFocus();
  fireEvent.keyDown(settings, { key:'End' });
  expect(dagger).toHaveFocus();
  const shortcut = jest.fn();
  window.addEventListener('keyup', shortcut);
  fireEvent.keyUp(gripper, { key: ' ', code: 'Space' });
  expect(shortcut).not.toHaveBeenCalled();
  window.removeEventListener('keyup', shortcut);
});


test('DAgger tab ownership locks Start, Pose and Gripper after tab switches', () => {
  render(<InferencePage />);
  fireEvent.click(screen.getByRole('tab', { name:'DAgger' }));
  fireEvent.click(screen.getByRole('button', { name:'Start test DAgger' }));
  fireEvent.click(screen.getByRole('tab', { name:'Settings' }));
  expect(screen.getByRole('button', { name:'Start inference' })).toBeDisabled();
  expect(screen.getByTestId('pose-tools')).toHaveAttribute('data-locked','true');
  expect(screen.getByTestId('gripper-tools')).toHaveAttribute('data-locked','true');
});
