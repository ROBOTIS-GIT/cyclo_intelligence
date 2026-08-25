import { render, screen } from '@testing-library/react';
import MultiTaskDiTArchitectureDiagram from './MultiTaskDiTArchitectureDiagram';

describe('MultiTaskDiTArchitectureDiagram', () => {
  test('shows the real frozen-conditioning and trainable action-head boundary', () => {
    render(<MultiTaskDiTArchitectureDiagram />);

    expect(screen.getByTestId('multi-task-dit-architecture-diagram'))
      .toHaveTextContent('MultiTaskDiT · Flow-Matching action policy');
    expect(screen.getByLabelText('Visual + language encoder: Frozen; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText('Robot-state encoder: Frozen; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText('Frozen observation conditioning: Frozen; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText('Flow-Matching DiT action head: Fire; Trainable; fixed'))
      .toBeInTheDocument();
  });

  test('documents the deployed three-camera, state, and chunk contract', () => {
    render(<MultiTaskDiTArchitectureDiagram />);

    expect(screen.getByText('3 cameras + task tokens')).toBeInTheDocument();
    expect(screen.getByText('22D proprioceptive state')).toBeInTheDocument();
    expect(screen.getByText(/16 × 22D velocity/)).toBeInTheDocument();
  });
});
