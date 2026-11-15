import { fireEvent, render, screen } from '@testing-library/react';
import TrainingMetricsModal, { normalizeRlMetricHistory } from './TrainingMetricsModal';

jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-chart">{children}</div>,
  LineChart: ({ children, data }) => (
    <div data-testid="rl-metrics-line-chart" data-points={JSON.stringify(data)}>{children}</div>
  ),
  CartesianGrid: () => <div data-testid="chart-grid" />,
  Legend: () => <div data-testid="chart-legend" />,
  Tooltip: () => <div data-testid="chart-tooltip" />,
  XAxis: ({ label }) => <div data-testid="chart-x-axis">{label?.value}</div>,
  YAxis: ({ label, yAxisId }) => <div data-testid={`chart-y-axis-${yAxisId}`}>{label?.value}</div>,
  Line: ({ dataKey, name, yAxisId }) => (
    <div data-testid={`chart-line-${dataKey}`} data-axis={yAxisId}>{name}</div>
  ),
}));

describe('TrainingMetricsModal', () => {
  test('normalizes finite RL epoch means, merges duplicates, and sorts by integer epoch', () => {
    expect(normalizeRlMetricHistory([
      {
        rl_epoch: 3,
        actor_loss_mean: -0.3,
        critic_loss_mean: 0.7,
        replay_average_reward: 0.4,
      },
      {
        rl_epoch: 1,
        actor_loss_mean: -0.1,
        critic_loss_mean: 1.2,
        replay_average_reward: 0.2,
      },
      { rl_epoch: 3, actor_loss_mean: -0.35 },
      { rl_epoch: 2.5, actor_loss_mean: 9 },
      { rl_epoch: 0, actor_loss_mean: 9 },
      { rl_epoch: 4, actor_loss_mean: Number.POSITIVE_INFINITY },
      { rl_epoch: '5', actor_loss_mean: 1 },
      null,
    ])).toEqual([
      {
        rl_epoch: 1,
        actor_loss_mean: -0.1,
        critic_loss_mean: 1.2,
        replay_average_reward: 0.2,
      },
      {
        rl_epoch: 3,
        actor_loss_mean: -0.35,
        critic_loss_mean: 0.7,
        replay_average_reward: 0.4,
      },
    ]);
  });

  test('renders the three requested series against RL epoch and dual average axes', () => {
    render(
      <TrainingMetricsModal
        open
        onBack={jest.fn()}
        history={[{
          rl_epoch: 2,
          actor_loss_mean: -0.2,
          critic_loss_mean: 0.8,
          replay_average_reward: 0.6,
        }]}
      />
    );

    expect(screen.getByRole('dialog', { name: 'Training Metrics' })).toBeInTheDocument();
    expect(screen.getByText('1 RL epoch')).toBeInTheDocument();
    expect(screen.getByTestId('chart-x-axis')).toHaveTextContent('RL epoch');
    expect(screen.getByTestId('chart-y-axis-loss')).toHaveTextContent('Average loss');
    expect(screen.getByTestId('chart-y-axis-reward')).toHaveTextContent('Average reward');
    expect(screen.getByTestId('chart-line-actor_loss_mean')).toHaveTextContent('Actor loss');
    expect(screen.getByTestId('chart-line-critic_loss_mean')).toHaveTextContent('Critic loss');
    expect(screen.getByTestId('chart-line-replay_average_reward'))
      .toHaveTextContent('Replay average reward');
    expect(JSON.parse(screen.getByTestId('rl-metrics-line-chart').dataset.points))
      .toEqual([{
        rl_epoch: 2,
        actor_loss_mean: -0.2,
        critic_loss_mean: 0.8,
        replay_average_reward: 0.6,
      }]);
  });

  test('shows a safe empty state and closes with Escape or the backdrop', () => {
    const onBack = jest.fn();
    const { rerender } = render(
      <TrainingMetricsModal open onBack={onBack} history={[]} />
    );

    expect(screen.getByTestId('training-metrics-empty')).toHaveTextContent(
      'No RL epoch metrics yet.'
    );
    fireEvent.keyDown(window, { key: 'Escape' });
    expect(onBack).toHaveBeenCalledTimes(1);

    fireEvent.mouseDown(screen.getByTestId('training-metrics-backdrop'));
    expect(onBack).toHaveBeenCalledTimes(2);

    rerender(<TrainingMetricsModal open={false} onBack={onBack} history={[]} />);
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('locks body scroll, traps focus, and restores the expansion trigger', () => {
    const trigger = document.createElement('button');
    document.body.appendChild(trigger);
    trigger.focus();
    const returnFocusRef = { current: trigger };
    const { rerender } = render(
      <TrainingMetricsModal
        open
        onBack={jest.fn()}
        history={[]}
        returnFocusRef={returnFocusRef}
      />
    );

    const backButton = screen.getByRole('button', { name: 'Back to Training' });
    expect(backButton).toHaveFocus();
    expect(document.body.style.overflow).toBe('hidden');
    fireEvent.keyDown(window, { key: 'Tab', shiftKey: true });
    expect(backButton).toHaveFocus();

    rerender(
      <TrainingMetricsModal
        open={false}
        onBack={jest.fn()}
        history={[]}
        returnFocusRef={returnFocusRef}
      />
    );
    expect(document.body.style.overflow).toBe('');
    expect(trigger).toHaveFocus();
    trigger.remove();
  });
});
