import { render, screen } from '@testing-library/react';
import MultiTaskDiTArchitectureDiagram from './MultiTaskDiTArchitectureDiagram';

describe('MultiTaskDiTArchitectureDiagram', () => {
  test('compacts conditioning and the trainable DiT into one action module', () => {
    render(<MultiTaskDiTArchitectureDiagram />);

    expect(screen.getByTestId('multi-task-dit-architecture-diagram'))
      .toHaveTextContent('MultiTaskDiT · Flow-Matching action policy');
    expect(screen.getByLabelText('Visual + task encoder: Frozen; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText('Robot-state encoder: Frozen; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText(
      'Action Module: conditioning Frozen; Flow-Matching DiT Fire; Trainable; fixed'
    ))
      .toBeInTheDocument();
    expect(screen.getByText('Frozen conditioning → trainable Flow-Matching DiT'))
      .toBeInTheDocument();
    expect(screen.getByText('DiT · Trainable')).toBeInTheDocument();
    expect(screen.queryByText('Frozen observation conditioning')).not.toBeInTheDocument();
    expect(screen.queryByText('Flow-Matching DiT action head')).not.toBeInTheDocument();
  });

  test('documents the deployed three-camera, state, and chunk contract', () => {
    render(<MultiTaskDiTArchitectureDiagram />);

    expect(screen.getByTestId('multi-task-dit-policy-inputs'))
      .toHaveTextContent('3 camera images + task');
    expect(screen.getByTestId('multi-task-dit-policy-inputs'))
      .toHaveTextContent('22D robot state');
    expect(screen.getByTestId('multi-task-dit-policy-output'))
      .toHaveTextContent('16 × 22D');
  });

  test('uses the ACT frozen palette for both fixed observation encoders', () => {
    render(<MultiTaskDiTArchitectureDiagram />);

    for (const label of ['Visual + task encoder', 'Robot-state encoder']) {
      expect(screen.getByLabelText(`${label}: Frozen; fixed`))
        .toHaveClass('border-[#d9d2c5]', 'bg-[#f1eee7]', 'text-[#7d7569]');
    }
  });
});
