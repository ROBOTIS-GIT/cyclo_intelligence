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
  const { container } = render(<RobotControls robotType="robot" />);
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

test('clicking the source status switches between blue External and green Teleop', () => {
  const { rerender } = render(<RobotControls robotType="robot" />);
  open();
  const source = screen.getByRole('button', { name: 'Source: External' });
  expect(source).toHaveClass('is-external');
  fireEvent.click(source);
  expect(control.setSource).toHaveBeenCalledWith('teleop');
  control.source = 'teleop';
  rerender(<RobotControls robotType="robot" />);
  expect(screen.getByRole('button', { name: 'Source: Teleop' })).toHaveClass('is-teleop');
  fireEvent.click(screen.getByRole('button', { name: 'Source: Teleop' }));
  expect(control.setSource).toHaveBeenLastCalledWith('external');
});

test('Head/Base updates independently and does not offer an unsupported robot command', () => {
  const { rerender } = render(<RobotControls robotType="robot" />);
  open();
  expect(screen.getByRole('button', { name: 'Joystick: Head' })).toBeDisabled();
  control.mode = 'swerve';
  rerender(<RobotControls robotType="robot" />);
  expect(screen.getByRole('button', { name: 'Joystick: Base' })).toHaveClass('is-swerve');
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeEnabled();
});

test('applies only the edited hand, without an extra target selector', async () => {
  render(<RobotControls robotType="robot" />);
  open();
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  expect(control.setPreset).not.toHaveBeenCalled();
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('left', '2', '');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeDisabled();
});

test('changing both preset fields applies both hands once', async () => {
  render(<RobotControls robotType="robot" />);
  open();
  fireEvent.change(screen.getByLabelText('Left preset'), { target: { value: '2' } });
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: '3' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledTimes(1);
  expect(control.setPreset).toHaveBeenCalledWith('both', '2', '3');
});

test('supports a right custom preset and keeps the draft after a failed request', async () => {
  control.setPreset.mockResolvedValue(false);
  render(<RobotControls robotType="robot" />);
  open();
  fireEvent.change(screen.getByLabelText('Right preset'), { target: { value: 'custom' } });
  fireEvent.change(screen.getByLabelText('Right custom preset ID'), { target: { value: '42' } });
  await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Apply' })));
  expect(control.setPreset).toHaveBeenCalledWith('right', '', '42');
  expect(screen.getByRole('button', { name: 'Apply' })).toBeEnabled();
});

test('Escape returns focus to the toggle; outside click dismisses the popup', () => {
  render(<RobotControls robotType="robot" />);
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
  const { rerender } = render(<RobotControls robotType="robot" />);
  open();
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.queryByRole('button', { name: 'Apply' })).not.toBeInTheDocument();
  control.canSetSource = true;
  control.pending = 'source';
  control.error = 'Source change rejected';
  rerender(<RobotControls robotType="robot" />);
  expect(screen.getByRole('button', { name: 'Source: External' })).toBeDisabled();
  expect(screen.getByRole('status')).toHaveTextContent('Applying');
  expect(screen.getByRole('alert')).toHaveTextContent('Source change rejected');
});
