import { render, screen } from '@testing-library/react';
import GrootArchitectureDiagram from './GrootArchitectureDiagram';

describe('GrootArchitectureDiagram', () => {
  test('shows the official N1.7 default frozen and trainable boundaries as locked', () => {
    render(<GrootArchitectureDiagram />);

    for (const name of ['Visual encoder', 'Language model']) {
      const node = screen.getByRole('button', { name: new RegExp(`${name}: Frozen`, 'i') });
      expect(node).toHaveAttribute('aria-pressed', 'false');
      expect(node).toBeDisabled();
    }

    const actionModule = screen.getByRole('button', {
      name: /Action Module: Trainable/i,
    });
    expect(actionModule).toHaveAttribute('aria-pressed', 'true');
    expect(actionModule).toHaveAttribute(
      'data-member-groups',
      'vl_adapter state_action_projectors flow_matching_dit'
    );
    expect(actionModule).toBeDisabled();

    expect(screen.getByText(/Official fine-tuning defaults/i)).toBeInTheDocument();
    expect(screen.getByText('Locked policy')).toBeInTheDocument();
  });

  test('uses the common policy visual grammar and exposes the real GR00T flow', () => {
    const { container } = render(<GrootArchitectureDiagram />);

    expect(screen.getByTestId('groot-architecture-diagram'))
      .toHaveClass('rounded-2xl', 'bg-white');
    expect(screen.getByTestId('groot-architecture-flow'))
      .toHaveClass('min-h-0', 'flex-1');
    expect(screen.getByTestId('groot-policy-inputs')).toHaveClass('grid-cols-2');
    expect(screen.getByText('3 camera images + task instruction')).toBeInTheDocument();
    expect(screen.getByText('Robot state')).toBeInTheDocument();
    expect(container.querySelector('[data-trainable-group="visual_encoder"]'))
      .toHaveClass('bg-[#f1eee7]', 'border-[#d9d2c5]');
    expect(container.querySelector('[data-trainable-group="action_module"]'))
      .toHaveClass('bg-[#f8f1e6]', 'border-[#d8c4a5]');
    expect(screen.getByTestId('groot-policy-output'))
      .toHaveClass('bg-[#e9edfa]', 'border-[#9faacf]');
    expect(screen.getByText('Action chunk')).toBeInTheDocument();
  });

  test.each([
    ['rlt mode', { mode: 'rlt' }],
    ['allFrozen override', { allFrozen: true }],
  ])('renders every base-policy module frozen for %s', (_label, props) => {
    render(<GrootArchitectureDiagram {...props} />);

    expect(screen.getByTestId('groot-architecture-diagram'))
      .toHaveAttribute('data-architecture-mode', 'all-frozen');
    expect(screen.getByText(/RLT base policy · all modules frozen/i)).toBeInTheDocument();
    expect(screen.getAllByRole('button')).toHaveLength(3);
    for (const node of screen.getAllByRole('button')) {
      expect(node).toHaveAttribute('aria-pressed', 'false');
      expect(node).toBeDisabled();
    }
  });
});
