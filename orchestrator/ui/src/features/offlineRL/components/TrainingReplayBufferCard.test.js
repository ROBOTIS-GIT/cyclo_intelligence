import { fireEvent, render, screen } from '@testing-library/react';
import TrainingReplayBufferCard, {
  normalizeTrainingReplayDatasets,
  TrainingReplayBufferCylinder,
} from './TrainingReplayBufferCard';

const datasets = [
  {
    dataset_path: '/workspace/lerobot/data_epoch_0001',
    name: 'data_epoch_0001',
    version: 'v3.0',
    total_episodes: 6,
    success_count: 4,
    failure_count: 2,
    unlabeled_count: 0,
  },
  {
    path: '/workspace/lerobot/data_epoch_0002',
    version: 'v3.0',
    episodes: [
      { episode_success: true },
      { outcome: 'failure' },
      { outcome: 'unlabeled' },
      { success: true },
    ],
  },
];

describe('TrainingReplayBufferCard', () => {
  test('normalizes inventory summaries and episode rows without mutating the source', () => {
    const original = JSON.parse(JSON.stringify(datasets));
    const normalized = normalizeTrainingReplayDatasets(datasets);

    expect(normalized).toEqual([
      expect.objectContaining({
        path: '/workspace/lerobot/data_epoch_0001',
        name: 'data_epoch_0001',
        metadataKnown: true,
        totalEpisodes: 6,
        successCount: 4,
        failureCount: 2,
        unlabeledCount: 0,
      }),
      expect.objectContaining({
        path: '/workspace/lerobot/data_epoch_0002',
        name: 'data_epoch_0002',
        metadataKnown: true,
        totalEpisodes: 4,
        successCount: 2,
        failureCount: 1,
        unlabeledCount: 1,
      }),
    ]);
    expect(datasets).toEqual(original);
  });

  test('keeps the current datasetSelections shape explicit when episode metadata is absent', () => {
    const normalized = normalizeTrainingReplayDatasets([{
      path: '/workspace/lerobot/data_epoch_0007',
      version: 'v3.0',
      dataEpoch: 7,
    }]);

    expect(normalized[0]).toEqual(expect.objectContaining({
      name: 'data_epoch_0007',
      dataEpoch: 7,
      metadataKnown: false,
      totalEpisodes: null,
    }));
  });

  test('renders continuous success, failure, and empty capacity proportions', () => {
    render(<TrainingReplayBufferCard datasets={datasets} capacityEpisodes={20} />);

    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('10 / 20 episodes')).toBeInTheDocument();
    const cylinder = screen.getByLabelText('Training replay buffer fill');
    expect(cylinder).toHaveClass('pg-buffer-vessel');
    expect(cylinder).toHaveAttribute('data-capacity-percent', '50');
    expect(screen.getByText('Success 6')).toHaveClass('text-[14px]');
    expect(screen.getByText('Failure 3')).toHaveClass('text-[14px]');
    expect(screen.getByText('Unlabeled 1')).toHaveClass('text-[14px]');
    expect(screen.getByLabelText('Inspect success datasets: 6 episodes'))
      .toHaveAttribute('data-episode-count', '6');
    expect(screen.getByLabelText('Inspect failure datasets: 3 episodes'))
      .toHaveAttribute('data-episode-count', '3');
    expect(screen.getByLabelText('Inspect unlabeled datasets: 1 episode'))
      .toHaveAttribute('data-episode-count', '1');
    expect(screen.queryByTestId('training-replay-buffer-grid')).not.toBeInTheDocument();
  });

  test('reveals dataset names and per-outcome episode counts on hover or click', () => {
    render(<TrainingReplayBufferCylinder datasets={datasets} capacityEpisodes={20} />);
    const successSegment = screen.getByLabelText('Inspect success datasets: 6 episodes');

    fireEvent.mouseEnter(successSegment);
    expect(screen.getByTestId('training-replay-outcome-detail')).toHaveTextContent(
      'data_epoch_0001'
    );
    expect(screen.getByTestId('training-replay-outcome-detail')).toHaveTextContent(
      '4 episodes'
    );
    expect(screen.getByTestId('training-replay-outcome-detail')).toHaveTextContent(
      'data_epoch_0002'
    );

    fireEvent.click(successSegment);
    fireEvent.mouseLeave(successSegment);
    expect(screen.getByTestId('training-replay-outcome-detail')).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByTestId('training-replay-outcome-detail'));
    expect(screen.getByTestId('training-replay-outcome-detail')).toBeInTheDocument();

    fireEvent.mouseDown(screen.getByText('buffer filled'));
    expect(screen.queryByTestId('training-replay-outcome-detail')).not.toBeInTheDocument();

    fireEvent.click(successSegment);
    fireEvent.click(successSegment);
    expect(screen.queryByTestId('training-replay-outcome-detail')).not.toBeInTheDocument();
  });

  test('exposes an optional read-only dataset inspection callback', () => {
    const onInspectDataset = jest.fn();
    render(
      <TrainingReplayBufferCylinder
        datasets={datasets}
        capacityEpisodes={20}
        onInspectDataset={onInspectDataset}
      />
    );

    fireEvent.click(screen.getByLabelText('Inspect failure datasets: 3 episodes'));
    fireEvent.click(screen.getByRole('button', { name: /data_epoch_0002.*1 episode/i }));

    expect(onInspectDataset).toHaveBeenCalledTimes(1);
    expect(onInspectDataset).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/workspace/lerobot/data_epoch_0002' }),
      'failure'
    );
  });

  test('does not invent fill for selected datasets whose metadata is unavailable', () => {
    render(
      <TrainingReplayBufferCylinder
        datasets={[{ path: '/workspace/lerobot/data_epoch_0007', version: 'v3.0' }]}
      />
    );

    expect(screen.getByText('0%')).toBeInTheDocument();
    expect(screen.getByText('1 dataset awaiting episode metadata')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Inspect success datasets/i }))
      .not.toBeInTheDocument();
  });
});
