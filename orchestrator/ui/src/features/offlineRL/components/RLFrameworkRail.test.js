import { fireEvent, render, screen } from '@testing-library/react';
import RLFrameworkRail from './RLFrameworkRail';

describe('RLFrameworkRail', () => {
  test('shows the three RL sections and marks the active section', () => {
    const { container } = render(
      <RLFrameworkRail
        activeSection="replay"
        sectionControls={{ replay: 'offline-rl-replay-drawer' }}
      />
    );

    expect(container.firstChild).toHaveClass('w-[220px]');
    expect(screen.getByTestId('rl-framework-title'))
      .toHaveClass('text-[#302d27]');
    expect(screen.getByTestId('rl-framework-title'))
      .not.toHaveClass('bg-[#302d27]', 'text-[#f8f6f0]');
    expect(screen.getByTestId('rl-framework-title'))
      .toHaveTextContent('PLAYGROUND');
    expect(screen.getByTestId('rl-framework-title')
      .querySelector('[data-robot-lab-icon="true"]')).toBeInTheDocument();
    expect(screen.getByTestId('rl-framework-toggle-glyph'))
      .toHaveClass('rounded-[5px]', 'border', 'bg-[#f3f0e8]');
    expect(screen.getByTestId('rl-framework-toggle-accent'))
      .toHaveClass('w-[20%]', 'bg-[#627d68]');
    expect(screen.getByText('Environment')).toBeInTheDocument();
    expect(screen.getByText('Replay Buffer')).toBeInTheDocument();
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replay Buffer' }))
      .toHaveAttribute('aria-current', 'page');
    expect(screen.getByRole('button', { name: 'Replay Buffer' }))
      .toHaveAttribute('aria-controls', 'offline-rl-replay-drawer');
    expect(screen.getByRole('button', { name: 'Replay Buffer' }))
      .toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: 'Environment' }))
      .not.toHaveAttribute('aria-current');
    expect(screen.getByRole('button', { name: 'Environment' })
      .querySelector('[data-robot-lab-icon="true"]')).toBeInTheDocument();
  });

  test('dispatches back, collapse, and section callbacks', () => {
    const onBack = jest.fn();
    const onSectionChange = jest.fn();
    const onToggleCollapsed = jest.fn();

    render(
      <RLFrameworkRail
        onBack={onBack}
        onSectionChange={onSectionChange}
        onToggleCollapsed={onToggleCollapsed}
      />
    );

    fireEvent.click(screen.getByRole('button', {
      name: 'Back to main navigation',
    }));
    fireEvent.click(screen.getByRole('button', {
      name: 'Collapse Playground menu',
    }));
    fireEvent.click(screen.getByRole('button', { name: 'Training' }));

    expect(onBack).toHaveBeenCalledTimes(1);
    expect(onToggleCollapsed).toHaveBeenCalledTimes(1);
    expect(onSectionChange).toHaveBeenCalledWith('training');
  });

  test('collapses to an icon-only rail while keeping accessible controls', () => {
    const { container } = render(<RLFrameworkRail collapsed />);

    expect(container.firstChild).toHaveClass('w-[68px]');
    expect(container.firstChild).toHaveAttribute('data-collapsed', 'true');
    expect(screen.getByTestId('rl-framework-rail-header'))
      .toHaveClass('flex-col');
    expect(screen.queryByTestId('rl-framework-title')).not.toBeInTheDocument();
    expect(screen.getByTestId('rl-framework-toggle-glyph')).toBeInTheDocument();
    expect(screen.queryByText('PLAYGROUND')).not.toBeInTheDocument();
    expect(screen.queryByText('Environment')).not.toBeInTheDocument();
    expect(screen.queryByText('Replay Buffer')).not.toBeInTheDocument();
    expect(screen.queryByText('Training')).not.toBeInTheDocument();
    expect(screen.getByRole('button', {
      name: 'Expand Playground menu',
    })).toHaveAttribute('aria-expanded', 'false');
    expect(screen.getByRole('button', { name: 'Environment' }))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Replay Buffer' }))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Training' }))
      .toBeInTheDocument();
  });
});
