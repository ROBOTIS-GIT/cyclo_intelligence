import { render, screen, within } from '@testing-library/react';
import PI05ArchitectureDiagram from './PI05ArchitectureDiagram';

describe('PI05ArchitectureDiagram', () => {
  test('shows the Pi0.5 inputs, policy topology, and action output', () => {
    render(<PI05ArchitectureDiagram />);

    const inputs = screen.getByTestId('pi05-policy-inputs');
    expect(within(inputs).getByText('Camera images + task instruction')).toBeInTheDocument();
    expect(within(inputs).getByText('Robot state')).toBeInTheDocument();

    expect(screen.getByRole('group', { name: 'Vision-language encoder' }))
      .toHaveTextContent('SigLIP + PaliGemma');
    expect(screen.getByRole('group', { name: 'Action conditioning' }))
      .toHaveTextContent('Robot state + noisy action + time');
    expect(screen.getByRole('group', { name: 'Action Module' }))
      .toHaveTextContent('Flow-matching velocity prediction');
    expect(screen.getByTestId('pi05-policy-output')).toHaveTextContent('Action chunk');
  });

  test('does not expose preview-only fine-tune controls', () => {
    render(<PI05ArchitectureDiagram disabled />);

    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.queryByText(/Fire|Frozen/)).not.toBeInTheDocument();
    expect(screen.getByText(/training controls pending integration/i)).toBeInTheDocument();
  });
});
