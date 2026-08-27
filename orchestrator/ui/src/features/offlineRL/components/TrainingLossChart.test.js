import { fireEvent, render, screen } from '@testing-library/react';
import TrainingLossChart from './TrainingLossChart';

jest.mock('./TrainingMetricsModal', () => ({ open, onBack }) => (
  open ? (
    <div role="dialog" aria-label="Training Metrics">
      <button type="button" onClick={onBack}>Back to Training</button>
    </div>
  ) : null
));

describe('TrainingLossChart', () => {
  test('renders only the latest actor and critic losses with progress metadata', () => {
    render(
      <TrainingLossChart
        actorLossHistory={[
          { step: 10, loss: -0.2 },
          { step: 20, loss: -0.4 },
        ]}
        criticLossHistory={[
          { step: 10, loss: 1.2 },
          { step: 20, loss: 0.8 },
        ]}
        etaSeconds={125}
        percentage={42.25}
        status="running"
      />
    );

    expect(screen.getByLabelText('Latest critic loss')).toHaveTextContent('0.80');
    expect(screen.getByLabelText('Latest actor loss')).toHaveTextContent('-0.40');
    expect(screen.getByLabelText('Training percentage')).toHaveTextContent('42.3%');
    expect(screen.getByLabelText('Training ETA')).toHaveTextContent('ETA 2m 05s');
    expect(screen.getByRole('progressbar', { name: 'Training loss progress' }))
      .toHaveAttribute('aria-valuenow', '42.25');
    expect(screen.getByText('running')).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Expand training loss/i }))
      .not.toBeInTheDocument();
  });

  test('filters malformed values and shows an explicit empty loss', () => {
    render(
      <TrainingLossChart
        actorLossHistory={[
          { step: 4, loss: -0.5 },
          { step: 6, loss: Number.POSITIVE_INFINITY },
          { step: Number.NaN, loss: 2 },
        ]}
        criticLossHistory={[{ step: '4', loss: 0.75 }, null]}
      />
    );

    expect(screen.getByLabelText('Latest actor loss')).toHaveTextContent('-0.50');
    expect(screen.getByLabelText('Latest critic loss')).toHaveTextContent('—');
    expect(document.body.innerHTML).not.toMatch(/NaN|Infinity/);
  });

  test('selects the loss belonging to the greatest valid update step', () => {
    render(
      <TrainingLossChart
        criticLossHistory={[
          { step: 20, loss: 0.6 },
          { step: 5, loss: 1.4 },
          { step: 10, loss: 0.9 },
        ]}
      />
    );

    expect(screen.getByLabelText('Latest critic loss')).toHaveTextContent('0.60');
  });

  test('clamps percentage and safely formats invalid duration and status', () => {
    const { rerender } = render(
      <TrainingLossChart
        durationSeconds={Number.NaN}
        percentage={125}
        status="completed"
      />
    );

    expect(screen.getByLabelText('Training percentage')).toHaveTextContent('100.0%');
    expect(screen.getByLabelText('Training duration')).toHaveTextContent('—');
    expect(screen.getByText('completed')).toBeInTheDocument();

    rerender(
      <TrainingLossChart durationSeconds={8} percentage={-2} status="failed" />
    );
    expect(screen.getByLabelText('Training percentage')).toHaveTextContent('0.0%');
    expect(screen.getByLabelText('Training duration')).toHaveTextContent('8s');
    expect(screen.getByText('failed')).toBeInTheDocument();
  });

  test('reuses the ACT progress layout with policy-specific metrics', () => {
    render(
      <TrainingLossChart
        metrics={[
          { label: 'Actor loss', value: '-0.125000', tone: 'actor' },
          { label: 'Critic loss', value: '0.320000', tone: 'critic' },
          { label: 'Approx. KL', value: '0.004000', tone: 'neutral' },
        ]}
        percentage={60}
        status="running"
        progressLabel="Flow-SDE PPO training progress"
      />
    );

    expect(screen.getByTestId('training-progress-metrics')).toHaveClass('grid-cols-3');
    expect(screen.getByText('Actor loss').parentElement).toHaveTextContent('-0.125000');
    expect(screen.getByText('Critic loss').parentElement).toHaveTextContent('0.320000');
    expect(screen.getByText('Approx. KL').parentElement).toHaveTextContent('0.004000');
    expect(screen.getByRole('progressbar', { name: 'Flow-SDE PPO training progress' }))
      .toHaveAttribute('aria-valuenow', '60');
    expect(screen.getByLabelText('Latest critic loss')).toHaveTextContent('0.320000');
  });

  test('opens RL metric history from the far-right expansion action', () => {
    render(
      <TrainingLossChart
        expandable
        rlMetricHistory={[{
          rl_epoch: 1,
          actor_loss_mean: -0.25,
          critic_loss_mean: 0.75,
          replay_average_reward: 0.5,
        }]}
      />
    );

    const expandButton = screen.getByRole('button', { name: 'Expand training metrics' });
    fireEvent.click(expandButton);
    expect(screen.getByRole('dialog', { name: 'Training Metrics' })).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Back to Training' }));
    expect(screen.queryByRole('dialog', { name: 'Training Metrics' })).not.toBeInTheDocument();
  });
});
