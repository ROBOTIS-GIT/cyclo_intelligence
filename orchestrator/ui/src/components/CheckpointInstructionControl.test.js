import { act, fireEvent, render, screen } from '@testing-library/react';
import CheckpointInstructionControl from './CheckpointInstructionControl';

const profile = {
  available: true, conditioning: 'subtask', default_instruction: '1. Align 3–5 cm.',
  task_instructions: ['Insert, release and retreat.'],
  subtask_instruction_choices: [['1. Align 3–5 cm.'], ['2. Release.'], ['3. Retreat.']],
};
const info = { serviceType: 'groot', policyPath: '/workspace/model/groot/first', taskInstruction: ['record instruction'] };
const reply = (data = profile) => ({ ok: true, json: async () => data });
const startFetch = async () => { await act(async () => jest.advanceTimersByTime(250)); };
const originalFetch = global.fetch;
beforeEach(() => { jest.useFakeTimers(); global.fetch = jest.fn().mockResolvedValue(reply()); });
afterEach(() => { jest.useRealTimers(); global.fetch = originalFetch; });

test('running subtask completion uses the common callback without changing the prompt early', async () => {
  const onSelect = jest.fn(), onComplete = jest.fn();
  render(<CheckpointInstructionControl info={{ ...info, taskInstruction: [profile.default_instruction] }}
    running inferencing languageConditioned onSelect={onSelect} onComplete={onComplete} />);
  await startFetch();
  expect(screen.getByLabelText('Checkpoint instruction selection')).toBeDisabled();
  fireEvent.click(screen.getByRole('button', { name: 'Complete subtask & continue' }));
  expect(onComplete).toHaveBeenCalledWith('2. Release.');
  expect(onSelect).not.toHaveBeenCalled();
});

test('completion is locked while paused or transitioning', async () => {
  const props = { info: { ...info, taskInstruction: [profile.default_instruction] },
    running: true, languageConditioned: true, onSelect: jest.fn(), onComplete: jest.fn() };
  const { rerender } = render(<CheckpointInstructionControl {...props} />);
  await startFetch();
  expect(screen.getByRole('button', { name: 'Complete subtask & continue' })).toBeDisabled();
  rerender(<CheckpointInstructionControl {...props} inferencing transitioning />);
  expect(screen.getByRole('button', { name: 'Switching subtask…' })).toBeDisabled();
});

test('loads overall instruction and separate steps, and selects exact first phase', async () => {
  const onSelect = jest.fn();
  render(<CheckpointInstructionControl info={info} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  expect(screen.getByText('Insert, release and retreat.')).toBeInTheDocument();
  expect(screen.getByText('Subtasks (3)')).toBeInTheDocument();
  expect(onSelect).toHaveBeenCalledWith('1. Align 3–5 cm.');
  fireEvent.change(screen.getByLabelText('Checkpoint instruction selection'), { target: { value: '2. Release.' } });
  expect(onSelect).toHaveBeenLastCalledWith('2. Release.');
});

test('ignores the previous checkpoint reply after switching models', async () => {
  let resolveOld;
  global.fetch.mockImplementationOnce(() => new Promise((resolve) => { resolveOld = resolve; }));
  const onSelect = jest.fn();
  const { rerender } = render(<CheckpointInstructionControl info={info} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  rerender(<CheckpointInstructionControl info={{ ...info, policyPath: '/workspace/model/groot/second' }} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  await act(async () => resolveOld(reply({ ...profile, default_instruction: 'Stale prompt.' })));
  expect(onSelect).toHaveBeenCalledTimes(1);
  expect(onSelect).toHaveBeenCalledWith(profile.default_instruction);
});

test.each(['edited', 'running'])('does not replace a prompt when %s while metadata is loading', async (state) => {
  let resolve;
  global.fetch.mockImplementationOnce(() => new Promise((done) => { resolve = done; }));
  const onSelect = jest.fn();
  const { rerender } = render(<CheckpointInstructionControl info={info} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  rerender(<CheckpointInstructionControl info={state === 'edited' ? { ...info, taskInstruction: ['My edit.'] } : info}
    autoApply={state !== 'running'} languageConditioned onSelect={onSelect} />);
  await act(async () => resolve(reply()));
  expect(onSelect).not.toHaveBeenCalled();
  expect(screen.getByText('Subtasks (3)')).toBeInTheDocument();
});

test('retains a manually selected phase after remounting', async () => {
  const onSelect = jest.fn();
  render(<CheckpointInstructionControl info={{ ...info, taskInstruction: ['My custom prompt.'],
    checkpointInstructionKey: JSON.stringify([info.serviceType, info.policyPath]) }} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  expect(onSelect).not.toHaveBeenCalled();
});

test('shows metadata for ACT without changing its instruction', async () => {
  const onSelect = jest.fn();
  render(<CheckpointInstructionControl info={{ ...info, serviceType: 'lerobot' }} autoApply languageConditioned={false} onSelect={onSelect} />);
  await startFetch();
  expect(screen.getByText('Subtasks (3)')).toBeInTheDocument();
  expect(onSelect).not.toHaveBeenCalled();
  expect(screen.queryByLabelText('Checkpoint instruction selection')).not.toBeInTheDocument();
});

test('handles unavailable metadata without replacing manual instruction', async () => {
  global.fetch.mockResolvedValue(reply({ available: false }));
  const onSelect = jest.fn();
  render(<CheckpointInstructionControl info={info} autoApply languageConditioned onSelect={onSelect} />);
  await startFetch();
  expect(screen.getByRole('status')).toHaveTextContent('metadata unavailable');
  expect(onSelect).not.toHaveBeenCalled();
});
