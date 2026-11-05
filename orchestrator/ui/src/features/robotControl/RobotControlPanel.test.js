import { act, fireEvent, render, screen } from '@testing-library/react';
import RobotControlPanel, { RobotControls, RobotControlProvider } from './RobotControlPanel';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import useRobotControl from './useRobotControl';

jest.mock('./useRobotControl');
let control;
beforeEach(() => {
  jest.clearAllMocks();
  control = {
    connected: true, source: 'external', mode: 'head_control', modeAvailable: true,
    canSetSource: true, canSetPreset: true, pending: null, error: '',
    selected: { left: null, right: null }, setSource: jest.fn(), setPreset: jest.fn().mockResolvedValue(true),
  };
  useRobotControl.mockImplementation(() => control);
});
const open = (name = 'Hand presets') => fireEvent.click(screen.getByRole('button', { name }));

test('status buttons are immediately visible without a Mode toggle or popup', () => {
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeVisible();
  expect(screen.getByRole('button', { name: 'Joystick: Head' })).toBeVisible();
  expect(screen.queryByRole('button', { name: 'Robot controls' })).not.toBeInTheDocument();
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(control.setSource).not.toHaveBeenCalled();
  expect(control.setPreset).not.toHaveBeenCalled();
});

test('clicking Source switches while matching cyan External and white Teleop/Head', () => {
  const { rerender } = render(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  const source = screen.getByRole('button', { name: 'Source: External' });
  expect(source).toHaveClass('is-cyan');
  fireEvent.click(source);
  expect(control.setSource).toHaveBeenCalledWith('teleop');
  control.source = 'teleop';
  rerender(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: Teleop' })).toHaveClass('is-white');
  fireEvent.click(screen.getByRole('button', { name: 'Source: Teleop' }));
  expect(control.setSource).toHaveBeenLastCalledWith('external');
});

test('Head/Base updates independently and does not offer an unsupported robot command', () => {
  const { rerender } = render(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Joystick: Head' })).toBeDisabled();
  control.mode = 'swerve';
  rerender(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Joystick: Base' })).toHaveClass('is-yellow');
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeEnabled();
});

test('applies only the edited hand, without an extra target selector', async () => {
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" section="presets" />);
  open('Hand presets');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  expect(control.setPreset).not.toHaveBeenCalled();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('left', '2', '');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
});

test('changing both preset fields applies both hands once', async () => {
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" section="presets" />);
  open('Hand presets');
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: '3' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledTimes(1);
  expect(control.setPreset).toHaveBeenCalledWith('both', '2', '3');
});

test('supports a right custom preset and keeps the draft after a failed request', async () => {
  control.setPreset.mockResolvedValue(false);
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" section="presets" />);
  open('Hand presets');
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: 'custom' } });
  fireEvent.change(screen.getByLabelText('Right custom preset ID'), { target: { value: '42' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('right', '', '42');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
});

test('Escape returns focus to the toggle; outside click dismisses the popup', () => {
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" section="presets" />);
  open('Hand presets');
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Hand presets' })).toHaveFocus();
  open('Hand presets');
  fireEvent.pointerDown(document.body);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('missing services and pending requests cannot send commands', () => {
  control.canSetSource = false;
  control.canSetPreset = false;
  const { rerender } = render(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
  control.canSetSource = true;
  control.pending = 'source';
  control.error = 'Source change rejected';
  rerender(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Applying');
  expect(screen.getByRole('alert')).toHaveTextContent('Source change rejected');
});


test.each(['f1', 'f2', 'ffw_bg2_rev4', 'ffw_sg2_rev1'])(
  '%s hides hand presets even when the connected backend advertises them', (robotType) => {
    render(<RobotControls control={control} robotType={robotType} />);
      expect(screen.getByRole('button', { name: 'Source: External' })).toBeEnabled();
    expect(screen.getByRole('button', { name: 'Joystick: Head' })).toBeVisible();
    expect(screen.queryByLabelText('Left preset')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Right preset')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Source: External' }));
    expect(control.setSource).toHaveBeenCalledWith('teleop');
    expect(control.setPreset).not.toHaveBeenCalled();
  }
);

test.each([
  ['external', 'head_control', 'cyan', false],
  ['external', 'swerve', 'cyan', false],
  ['teleop', 'head_control', 'white', false],
  ['teleop', 'swerve', 'yellow', true],
  ['teleop', null, 'unknown', false],
  [null, 'swerve', 'unknown', false],
])('source=%s, mode=%s uses LED color %s and conditional Base blinking', (source, mode, color, blinking) => {
  control.source = source;
  control.mode = mode;
  render(<RobotControls control={control} robotType="ffw_sh5_rev1" />);
  const badge = screen.getByRole('button', { name: /^Source:/ });
  expect(badge).toHaveClass('is-' + color);
  expect(badge.classList.contains('is-base-blinking')).toBe(blinking);
  expect(screen.getByRole('button', { name: /^Joystick:/ }).classList.contains('is-base-blinking')).toBe(blinking);
});

function renderPanels(robotType) {
  const store = configureStore({ reducer: (state = { tasks: { robotType } }, action) =>
    action.type === 'select' ? { tasks: { robotType: action.payload } } : state });
  render(<Provider store={store}><RobotControlProvider>
    <header><RobotControlPanel /></header>
    <aside><RobotControlPanel section="presets" /></aside>
  </RobotControlProvider></Provider>);
  return store;
}

test('status and presets occupy separate locations and share one control session', () => {
  renderPanels('ffw_sh5_rev1');
  expect(useRobotControl).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('button', { name: 'Source: External' }).closest('header')).not.toBeNull();
  expect(screen.getByRole('button', { name: 'Hand presets' }).closest('aside')).not.toBeNull();
  expect(screen.queryByLabelText('Left preset')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeVisible();
  fireEvent.pointerDown(screen.getByRole('button', { name: 'Hand presets' }));
  open('Hand presets');
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeVisible();
  expect(screen.getByLabelText('Left preset')).toBeVisible();
});

test.each(['f1', 'f2', 'ffw_bg2_rev4', 'ffw_sg2_rev1'])(
  '%s mounts status controls without a preset toggle', robot => {
    renderPanels(robot);
    expect(screen.getByRole('button', { name: 'Source: External' })).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Hand presets' })).not.toBeInTheDocument();
  });

test.each(['omx_f', 'omy_f3m', '', 'unknown_robot'])(
  '%s mounts no added controls or control subscription', robot => {
    renderPanels(robot);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(useRobotControl).not.toHaveBeenCalled();
  });

test('switching to OMX removes an open popup; switching back starts closed', () => {
  const store = renderPanels('ffw_sh5_rev1');
  open('Hand presets');
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  act(() => store.dispatch({ type: 'select', payload: 'omx_f' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.queryByRole('button')).not.toBeInTheDocument();
  act(() => store.dispatch({ type: 'select', payload: 'ffw_sh5_rev1' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  open('Hand presets');
  expect(screen.getByLabelText('Left preset')).toHaveValue('');
});
