import { render, screen } from '@testing-library/react';
import RLTArchitectureDiagram from './RLTArchitectureDiagram';

function ControlledDiagram() {
  return (
    <RLTArchitectureDiagram
      policyLabel="GR00T"
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

  test('shows the paper-faithful fixed Stage-2 trainability contract', () => {
    render(<ControlledDiagram />);

    expect(screen.getByLabelText('RL Token Encoder: Frozen')).toBeInTheDocument();
    expect(screen.getByLabelText('Action MLP: Trainable')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /RL Token Encoder/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Action MLP/i }))
      .not.toBeInTheDocument();
  });
});
