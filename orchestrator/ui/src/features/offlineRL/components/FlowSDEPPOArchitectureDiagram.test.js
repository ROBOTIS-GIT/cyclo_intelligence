import { render, screen } from '@testing-library/react';
import FlowSDEPPOArchitectureDiagram from './FlowSDEPPOArchitectureDiagram';

describe('FlowSDEPPOArchitectureDiagram', () => {
  test('shows the Flow-SDE rollout and PPO actor/value contract', () => {
    render(<FlowSDEPPOArchitectureDiagram />);

    expect(screen.getByText('Flow-SDE rollout')).toBeInTheDocument();
    expect(screen.getByText('GAE advantages')).toBeInTheDocument();
    expect(screen.getByLabelText('DiT actor: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByLabelText('Value head: Fire; Trainable; fixed')).toBeInTheDocument();
    expect(screen.getByText('PPO checkpoint → standard Flow-Matching inference'))
      .toBeInTheDocument();
  });

  test('reports backend readiness without changing the algorithm label', () => {
    const { rerender } = render(<FlowSDEPPOArchitectureDiagram />);
    expect(screen.getByText('Backend pending')).toBeInTheDocument();

    rerender(<FlowSDEPPOArchitectureDiagram backendReady />);
    expect(screen.getByText('Backend ready')).toBeInTheDocument();
    expect(screen.getByText('Flow-SDE PPO')).toBeInTheDocument();
  });
});
