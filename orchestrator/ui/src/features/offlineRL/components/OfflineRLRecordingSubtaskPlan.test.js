import React, { useState } from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import OfflineRLRecordingSubtaskPlan, {
  MAX_OFFLINE_RL_SUBTASKS,
} from './OfflineRLRecordingSubtaskPlan';

function ControlledPlan({ disabled = false, initialInstructions = [] }) {
  const [instructions, setInstructions] = useState(initialInstructions);
  const count = instructions.length;

  const handleCountChange = (nextCount) => {
    setInstructions((current) => current
      .slice(0, nextCount)
      .concat(Array(Math.max(0, nextCount - current.length)).fill('')));
  };

  return (
    <OfflineRLRecordingSubtaskPlan
      count={count}
      instructions={instructions}
      disabled={disabled}
      onCountChange={handleCountChange}
      onInstructionChange={(index, value) => {
        setInstructions((current) => current.map((instruction, itemIndex) => (
          itemIndex === index ? value : instruction
        )));
      }}
      onReset={() => setInstructions([])}
    />
  );
}

describe('OfflineRLRecordingSubtaskPlan', () => {
  test('resizes the ordered plan from 0 to 50 through controlled callbacks', () => {
    render(<ControlledPlan />);
    const count = screen.getByLabelText('Count');

    fireEvent.change(count, { target: { value: '3' } });

    expect(count).toHaveValue(3);
    expect(screen.getAllByPlaceholderText(/Subtask \d+ instruction/))
      .toHaveLength(3);

    fireEvent.change(count, {
      target: { value: String(MAX_OFFLINE_RL_SUBTASKS + 12) },
    });
    expect(count).toHaveValue(MAX_OFFLINE_RL_SUBTASKS);
    expect(screen.getAllByPlaceholderText(/Subtask \d+ instruction/))
      .toHaveLength(MAX_OFFLINE_RL_SUBTASKS);

    fireEvent.change(count, { target: { value: '-4' } });
    expect(count).toHaveValue(0);
    expect(screen.queryAllByPlaceholderText(/Subtask \d+ instruction/))
      .toHaveLength(0);
  });

  test('edits instructions in order and truncates them when count decreases', () => {
    render(<ControlledPlan initialInstructions={['Approach', '', 'Retreat']} />);

    fireEvent.change(screen.getByLabelText('Subtask 2 instruction'), {
      target: { value: 'Grasp' },
    });

    expect(screen.getByLabelText('Subtask 1 instruction')).toHaveValue('Approach');
    expect(screen.getByLabelText('Subtask 2 instruction')).toHaveValue('Grasp');
    expect(screen.getByLabelText('Subtask 3 instruction')).toHaveValue('Retreat');

    fireEvent.change(screen.getByLabelText('Count'), {
      target: { value: '2' },
    });
    expect(screen.getByLabelText('Subtask 1 instruction')).toHaveValue('Approach');
    expect(screen.getByLabelText('Subtask 2 instruction')).toHaveValue('Grasp');
    expect(screen.queryByLabelText('Subtask 3 instruction')).not.toBeInTheDocument();
  });

  test('resets the controlled plan', () => {
    render(<ControlledPlan initialInstructions={['Approach', 'Grasp']} />);

    fireEvent.click(screen.getByRole('button', {
      name: 'Reset subtask plan',
    }));

    expect(screen.getByLabelText('Count')).toHaveValue(0);
    expect(screen.getByText(/Set Count to add/)).toBeInTheDocument();
  });

  test('locks count, reset, and instruction editing when disabled', () => {
    render(
      <ControlledPlan
        disabled
        initialInstructions={['Approach', 'Grasp']}
      />
    );

    expect(screen.getByLabelText('Count')).toBeDisabled();
    expect(screen.getByRole('button', {
      name: 'Decrease subtask count',
    })).toBeDisabled();
    expect(screen.getByRole('button', {
      name: 'Increase subtask count',
    })).toBeDisabled();
    expect(screen.getByRole('button', {
      name: 'Reset subtask plan',
    })).toBeDisabled();
    expect(screen.getByLabelText('Subtask 1 instruction')).toBeDisabled();
    expect(screen.getByLabelText('Subtask 2 instruction')).toBeDisabled();
  });

  test('shows saved progress and advances only before the final subtask', () => {
    const onSaveAndNext = jest.fn();
    const { rerender } = render(
      <OfflineRLRecordingSubtaskPlan
        count={3}
        instructions={['Approach', 'Grasp', 'Retreat']}
        disabled
        activeIndex={1}
        savedIndices={[0]}
        recordingActive
        onSaveAndNext={onSaveAndNext}
      />
    );

    expect(screen.getByLabelText('Subtask 1 saved')).toBeInTheDocument();
    expect(screen.getByText('Subtask 2 / 3')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Save & Next/i }));
    expect(onSaveAndNext).toHaveBeenCalledTimes(1);

    rerender(
      <OfflineRLRecordingSubtaskPlan
        count={3}
        instructions={['Approach', 'Grasp', 'Retreat']}
        disabled
        activeIndex={2}
        savedIndices={[0, 1]}
        recordingActive
        onSaveAndNext={onSaveAndNext}
      />
    );
    expect(screen.queryByRole('button', { name: /Save & Next/i }))
      .not.toBeInTheDocument();
    expect(screen.getByText('Finish with Success or Fail')).toBeInTheDocument();
  });
});
