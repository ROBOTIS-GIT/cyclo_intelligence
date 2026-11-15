import { render, screen } from '@testing-library/react';
import TD3ArchitectureDiagram from './TD3ArchitectureDiagram';

describe('TD3ArchitectureDiagram', () => {
  test('shows the executed-prefix inputs and two fixed trainable critics', () => {
    render(<TD3ArchitectureDiagram />);

    expect(screen.getByTestId('td3-architecture-diagram')).toHaveTextContent(
      '3 images + robot state'
    );
    expect(screen.getByTestId('td3-architecture-diagram')).toHaveTextContent(
      'Action chunk + prefix mask'
    );
    expect(screen.getByLabelText('Independent twin critics')).toBeInTheDocument();
    expect(screen.getByLabelText('Q1 critic: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByLabelText('Q2 critic: Fire; Trainable; fixed')).toBeInTheDocument();
  });

  test('shows frozen Polyak targets, clipped target, and the actor Q1 gradient', () => {
    render(<TD3ArchitectureDiagram />);

    expect(screen.getByText('Target networks')).toBeInTheDocument();
    expect(screen.getByLabelText('Target ACT: Frozen; Polyak target; fixed'))
      .toBeInTheDocument();
    expect(screen.getByLabelText('Target Q1 / Q2: Frozen; Polyak target; fixed'))
      .toBeInTheDocument();
    expect(screen.getByText('min(target Q1, target Q2) → Bellman target'))
      .toBeInTheDocument();
    expect(screen.getByText('ACT ← maximize Q1')).toBeInTheDocument();
  });

  test('does not present critic warm-up as a user-configurable stage', () => {
    render(<TD3ArchitectureDiagram />);

    expect(screen.queryByText(/critic warm-?up/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
  });

  test('shows the ACT actor as frozen in critic-only warm-up mode', () => {
    render(<TD3ArchitectureDiagram criticOnly />);

    expect(screen.getByText('TD3 critic warm-up')).toBeInTheDocument();
    expect(screen.getByLabelText('ACT actor: Frozen; no gradients')).toBeInTheDocument();
    expect(screen.queryByText('ACT ← maximize Q1')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Q1 critic: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByLabelText('Q2 critic: Fire; Trainable; fixed')).toBeInTheDocument();
  });

  test('fills the training column with enlarged architecture labels', () => {
    render(<TD3ArchitectureDiagram />);

    expect(screen.getByTestId('td3-architecture-diagram'))
      .toHaveClass('flex-1', 'flex-col');
    expect(screen.getByTestId('td3-architecture-flow'))
      .toHaveClass('min-h-0', 'flex-1');
    expect(screen.getByText('Observation')).toHaveClass('text-[11px]');
    expect(screen.getByText('3 images + robot state')).toHaveClass('text-[9px]');
  });
});
