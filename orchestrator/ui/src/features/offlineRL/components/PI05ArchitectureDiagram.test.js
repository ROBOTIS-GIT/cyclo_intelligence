import { render, screen, within } from '@testing-library/react';
import PI05ArchitectureDiagram from './PI05ArchitectureDiagram';

describe('PI05ArchitectureDiagram', () => {
  test('shows the Pi0.5 inputs, policy topology, and action output', () => {
    render(<PI05ArchitectureDiagram />);

    const inputs = screen.getByTestId('pi05-policy-inputs');
    expect(within(inputs).getByText('Camera images + task instruction')).toBeInTheDocument();
    expect(within(inputs).getByText('Robot state')).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /Vision-language encoder: Frozen/i }))
      .toHaveTextContent('SigLIP + PaliGemma');
    expect(screen.getByRole('button', { name: /Action conditioning: Trainable/i }))
      .toHaveTextContent('Robot state + noisy action + time');
    expect(screen.getByRole('button', { name: /Action Module: Trainable/i }))
      .toHaveTextContent('Flow-matching velocity prediction');
    expect(screen.getByTestId('pi05-policy-output')).toHaveTextContent('Action chunk');
  });

  test('shows the locked fine-tuning boundary without editable controls', () => {
    render(<PI05ArchitectureDiagram />);

    const frozenVlm = screen.getByRole('button', {
      name: 'Vision-language encoder: Frozen; locked',
    });
    const trainableConditioning = screen.getByRole('button', {
      name: 'Action conditioning: Trainable; locked',
    });
    const trainableAction = screen.getByRole('button', {
      name: 'Action Module: Trainable; locked',
    });

    expect(frozenVlm).toBeDisabled();
    expect(trainableConditioning).toBeDisabled();
    expect(trainableAction).toBeDisabled();
    expect(trainableAction).toHaveClass('opacity-100');
    expect(screen.getByText(/Fine-tuning boundary · view only/i)).toBeInTheDocument();
  });

  test.each([
    ['RLT mode', { mode: 'rlt' }],
    ['allFrozen override', { allFrozen: true }],
  ])('freezes the complete Pi0.5 base policy in %s', (_label, props) => {
    render(<PI05ArchitectureDiagram {...props} />);

    for (const label of ['Vision-language encoder', 'Action conditioning', 'Action Module']) {
      expect(screen.getByRole('button', { name: `${label}: Frozen; locked` }))
        .toBeDisabled();
    }
    expect(screen.getByTestId('pi05-architecture-diagram'))
      .toHaveAttribute('data-architecture-mode', 'all-frozen');
    expect(screen.getByText(/RLT base policy · all modules frozen/i)).toBeInTheDocument();
  });
});
