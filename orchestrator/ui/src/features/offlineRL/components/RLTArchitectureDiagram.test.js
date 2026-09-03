import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import RLTArchitectureDiagram, {
  DEFAULT_RLT_TRAINABLE_GROUPS,
} from './RLTArchitectureDiagram';

function ControlledDiagram() {
  const [trainableGroups, setTrainableGroups] = useState(
    DEFAULT_RLT_TRAINABLE_GROUPS
  );
  return (
    <RLTArchitectureDiagram
      policyLabel="GR00T"
      trainableGroups={trainableGroups}
      onChange={setTrainableGroups}
    />
  );
}

describe('RLTArchitectureDiagram', () => {
  test('shows the 50:50 action policy and twin-Q Stage-2 structure', () => {
    render(<ControlledDiagram />);

    expect(screen.queryByText('Language Model')).not.toBeInTheDocument();
    expect(screen.queryByText('Flow-Matching DiT')).not.toBeInTheDocument();
    expect(screen.queryByText('Reference action · 16 × 19')).not.toBeInTheDocument();
    expect(screen.getByLabelText('RL Token Encoder to Action MLP')).toBeInTheDocument();
    expect(screen.getByLabelText('Action MLP to 10 by 19 action chunk'))
      .toBeInTheDocument();
    expect(screen.getByText('10 × 19 action chunk')).toBeInTheDocument();
    expect(screen.getByTestId('rlt-action-policy-diagram')).toBeInTheDocument();
    expect(screen.getByTestId('rlt-action-policy-diagram'))
      .toHaveAttribute('data-loop-policy-update-source', 'top-center');
    expect(screen.getByTestId('rlt-q-critic-diagram')).toBeInTheDocument();
    expect(screen.getByLabelText('RLT independent twin Q critic flow')).toBeInTheDocument();
    expect(screen.getByLabelText('Independent twin Q critics')).toBeInTheDocument();
    expect(screen.getByLabelText('Q1 MLP: Trainable')).toBeInTheDocument();
    expect(screen.getByLabelText('Q2 MLP: Trainable')).toBeInTheDocument();
    expect(screen.getByText('min(Q1, Q2) · Bellman target')).toBeInTheDocument();
  });

  test('toggles the RL Token Encoder and Action MLP independently', () => {
    render(<ControlledDiagram />);

    const tokenEncoder = screen.getByRole('button', {
      name: 'RL Token Encoder: Frozen; make trainable',
    });
    const actionMlp = screen.getByRole('button', {
      name: 'Action MLP: Trainable; freeze',
    });
    expect(tokenEncoder).toHaveAttribute('aria-pressed', 'false');
    expect(actionMlp).toHaveAttribute('aria-pressed', 'true');

    fireEvent.click(tokenEncoder);

    expect(screen.getByRole('button', {
      name: 'RL Token Encoder: Trainable; freeze',
    })).toHaveAttribute('aria-pressed', 'true');
    expect(screen.getByRole('button', {
      name: 'Action MLP: Trainable; freeze',
    })).toHaveAttribute('aria-pressed', 'true');
  });
});
