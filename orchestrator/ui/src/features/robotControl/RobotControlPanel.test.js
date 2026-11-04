import { act, fireEvent, render, screen } from '@testing-library/react';
import { RobotControls } from './RobotControlPanel';
import useRobotControl from './useRobotControl';

jest.mock('./useRobotControl');
let control;
beforeEach(() => {
  control = {
    connected: true, source: 'external', mode: 'head_control', modeAvailable: true,
    canSetSource: true, canSetPreset: true, pending: null, error: '',
    selected: { left: null, right: null }, setSource: jest.fn(), setPreset: jest.fn().mockResolvedValue(true),
  };
  useRobotControl.mockImplementation(() => control);
});
const open = () => fireEvent.click(screen.getByRole('button', { name: 'Robot controls' }));

test('starts as a small toggle; the popup is outside the recorder layout', () => {
  const { container } = render(<RobotControls robotType="ffw_sh5_rev1" />);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(container.querySelector('.robot-control-anchor')).toBeInTheDocument();
  open();
  const dialog = screen.getByRole('dialog', { name: 'Robot controls' });
  expect(dialog.parentElement).toBe(document.body);
  expect(container.contains(dialog)).toBe(false);
  expect(screen.queryByRole('button', { name: 'Use Teleop' })).not.toBeInTheDocument();
  expect(screen.queryByLabelText('Target hands')).not.toBeInTheDocument();
  expect(control.setSource).not.toHaveBeenCalled();
  expect(control.setPreset).not.toHaveBeenCalled();
});

test('clicking Source switches while matching cyan External and white Teleop/Head', () => {
  const { rerender } = render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  const source = screen.getByRole('button', { name: 'Source: External' });
  expect(source).toHaveClass('is-cyan');
  fireEvent.click(source);
  expect(control.setSource).toHaveBeenCalledWith('teleop');
  control.source = 'teleop';
  rerender(<RobotControls robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: Teleop' })).toHaveClass('is-white');
  fireEvent.click(screen.getByRole('button', { name: 'Source: Teleop' }));
  expect(control.setSource).toHaveBeenLastCalledWith('external');
});

test('Head/Base updates independently and does not offer an unsupported robot command', () => {
  const { rerender } = render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  expect(screen.getByRole('button', { name: 'Joystick: Head' })).toBeDisabled();
  control.mode = 'swerve';
  rerender(<RobotControls robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Joystick: Base' })).toHaveClass('is-yellow');
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeEnabled();
});

test('applies only the edited hand, without an extra target selector', async () => {
  render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  expect(control.setPreset).not.toHaveBeenCalled();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('left', '2', '');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
});

test('changing both preset fields applies both hands once', async () => {
  render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: '3' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledTimes(1);
  expect(control.setPreset).toHaveBeenCalledWith('both', '2', '3');
});

test('supports a right custom preset and keeps the draft after a failed request', async () => {
  control.setPreset.mockResolvedValue(false);
  render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: 'custom' } });
  fireEvent.change(screen.getByLabelText('Right custom preset ID'), { target: { value: '42' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('right', '', '42');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
});

test('Escape returns focus to the toggle; outside click dismisses the popup', () => {
  render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  fireEvent.keyDown(document, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Robot controls' })).toHaveFocus();
  open();
  fireEvent.pointerDown(document.body);
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});

test('missing services and pending requests cannot send commands', () => {
  control.canSetSource = false;
  control.canSetPreset = false;
  const { rerender } = render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
  control.canSetSource = true;
  control.pending = 'source';
  control.error = 'Source change rejected';
  rerender(<RobotControls robotType="ffw_sh5_rev1" />);
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Applying');
  expect(screen.getByRole('alert')).toHaveTextContent('Source change rejected');
});


test.each(['f1', 'f2', 'ffw_bg2_rev4', 'ffw_sg2_rev1', 'omx_f', 'omy_f3m', 'unknown_robot'])(
  '%s hides hand presets even when the connected backend advertises them', (robotType) => {
    render(<RobotControls robotType={robotType} />);
    open();
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
  const { container } = render(<RobotControls robotType="ffw_sh5_rev1" />);
  open();
  const badge = screen.getByRole('button', { name: /^Source:/ });
  const dot = container.querySelector('.robot-source-dot');
  for (const element of [badge, dot]) {
    expect(element).toHaveClass('is-' + color);
    expect(element.classList.contains('is-base-blinking')).toBe(blinking);
  }
  expect(screen.getByRole('button', { name: /^Joystick:/ }).classList.contains('is-base-blinking')).toBe(blinking);
});
