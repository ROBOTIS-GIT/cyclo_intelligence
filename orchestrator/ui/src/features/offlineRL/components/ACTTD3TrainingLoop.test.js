import {
  act,
  fireEvent,
  render,
  screen,
} from '@testing-library/react';

import ACTTD3TrainingLoop, {
  buildLoopConnectorGeometry,
} from './ACTTD3TrainingLoop';

const datasets = [{
  path: '/workspace/lerobot/data_epoch_0001',
  total_episodes: 10,
  success_count: 7,
  failure_count: 3,
  unlabeled_count: 0,
}];

const renderLoop = (overrides = {}) => render(
  <ACTTD3TrainingLoop
    trainableGroups={[
      'visual_backbone',
      'cvae_encoder',
      'transformer_encoder',
      'action_decoder',
    ]}
    datasets={datasets}
    capacityEpisodes={20}
    actorObjective="td3_bc"
    criticEpochs="10"
    actorEpochs="5"
    batchSize="4"
    criticSource="policy_warmup"
    criticSourceLabel="Policy warm-up"
    criticCheckpoint="/workspace/checkpoint/act/critic/latest.pt"
    {...overrides}
  />
);

describe('ACTTD3TrainingLoop', () => {
  test('composes the policy, deployed replay buffer, and simple TD3 algorithm loop', () => {
    renderLoop();

    expect(screen.getByRole('region', { name: 'ACT TD3 training loop' })).toBeInTheDocument();
    expect(screen.getByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(screen.getByTestId('act-td3-algorithm-card')).toBeInTheDocument();
    expect(screen.getByText('Critic Network')).toBeInTheDocument();
    const connectors = screen.getByTestId('act-td3-loop-connectors');
    expect(connectors).toBeInTheDocument();
    const connectorPaths = [...connectors.querySelectorAll('path')]
      .filter((path) => path.getAttribute('stroke'));
    expect(connectorPaths).toHaveLength(3);
    const markerId = connectors.querySelector('marker').getAttribute('id');
    expect(markerId).toMatch(/^policy-training-loop-arrow-/);
    connectorPaths.forEach((path) => {
      expect(path).toHaveAttribute('stroke', '#8f887d');
      expect(path).toHaveAttribute('stroke-width', '1.6');
      expect(path).toHaveAttribute('stroke-linecap', 'round');
      expect(path).toHaveAttribute('stroke-linejoin', 'round');
      expect(path).toHaveAttribute('vector-effect', 'non-scaling-stroke');
      expect(path).not.toHaveAttribute('stroke-dasharray');
      expect(path).toHaveAttribute('marker-end', `url(#${markerId})`);
    });
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.queryByLabelText('TD3 critic initialization')).not.toBeInTheDocument();
  });

  test('anchors the responsive loop to the exact rendered card edge centers', () => {
    const geometry = buildLoopConnectorGeometry({
      containerRect: {
        left: 100, top: 50, right: 1100, bottom: 850, width: 1000, height: 800,
      },
      policyRect: {
        left: 120, top: 80, right: 480, bottom: 280, width: 360, height: 200,
      },
      replayRect: {
        left: 620, top: 90, right: 980, bottom: 310, width: 360, height: 220,
      },
      algorithmRect: {
        left: 350, top: 430, right: 750, bottom: 650, width: 400, height: 220,
      },
    });

    expect(geometry).toMatchObject({ width: 1000, height: 800 });
    expect(geometry.paths).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'policy-to-replay',
        start: { x: 380, y: 130 },
        end: { x: 520, y: 150 },
      }),
      expect.objectContaining({
        id: 'replay-to-algorithm',
        start: { x: 700, y: 260 },
        end: { x: 650, y: 490 },
      }),
      expect.objectContaining({
        id: 'algorithm-to-policy',
        start: { x: 250, y: 490 },
        end: { x: 200, y: 230 },
      }),
    ]));
  });

  test('remeasures connector paths when observed card bounds change', () => {
    const frames = [];
    let resizeCallback = null;
    const disconnect = jest.fn();
    const originalResizeObserver = global.ResizeObserver;
    const rects = {
      'act-td3-training-loop': {
        left: 100, top: 50, right: 1100, bottom: 850, width: 1000, height: 800,
      },
      'act-td3-policy-stage': {
        left: 120, top: 80, right: 480, bottom: 280, width: 360, height: 200,
      },
      'training-replay-buffer-stage': {
        left: 620, top: 90, right: 980, bottom: 310, width: 360, height: 220,
      },
      'training-algorithm-stage': {
        left: 350, top: 430, right: 750, bottom: 650, width: 400, height: 220,
      },
    };
    global.ResizeObserver = class ResizeObserverMock {
      constructor(callback) {
        resizeCallback = callback;
      }

      observe() {}

      disconnect() {
        disconnect();
      }
    };
    const animationFrame = jest.spyOn(window, 'requestAnimationFrame')
      .mockImplementation((callback) => {
        frames.push(callback);
        return frames.length;
      });
    const cancelAnimationFrame = jest.spyOn(window, 'cancelAnimationFrame')
      .mockImplementation(() => {});
    const bounds = jest.spyOn(HTMLElement.prototype, 'getBoundingClientRect')
      .mockImplementation(function getMockBounds() {
        return rects[this.getAttribute('data-testid')] || {
          left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0,
        };
      });

    const { unmount } = renderLoop();
    act(() => frames.shift()?.(0));
    const replayConnector = screen.getByTestId('act-td3-loop-connectors')
      .querySelector('[data-connector="replay-to-algorithm"]');
    const firstPath = replayConnector.getAttribute('d');
    expect(firstPath).toMatch(/^M 700 260 /);
    expect(firstPath).toMatch(/ 650 490$/);

    rects['training-replay-buffer-stage'] = {
      left: 700, top: 110, right: 1060, bottom: 350, width: 360, height: 240,
    };
    act(() => {
      resizeCallback();
      frames.shift()?.(16);
    });
    expect(replayConnector.getAttribute('d')).not.toBe(firstPath);
    expect(replayConnector.getAttribute('d')).toMatch(/^M 780 300 /);

    unmount();
    expect(disconnect).toHaveBeenCalledTimes(1);
    bounds.mockRestore();
    animationFrame.mockRestore();
    cancelAnimationFrame.mockRestore();
    global.ResizeObserver = originalResizeObserver;
  });

  test('keeps loss selection directly above the epoch settings and reports changes', () => {
    const onActorObjectiveChange = jest.fn();
    const onCriticEpochsChange = jest.fn();
    const onActorEpochsChange = jest.fn();
    const onBatchSizeChange = jest.fn();
    renderLoop({
      onActorObjectiveChange,
      onCriticEpochsChange,
      onActorEpochsChange,
      onBatchSizeChange,
    });

    expect(screen.getByRole('button', { name: 'TD3-BC loss' }))
      .toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(screen.getByRole('button', { name: 'TD3 loss' }));
    fireEvent.change(screen.getByLabelText('Critic epochs'), { target: { value: '20' } });
    fireEvent.change(screen.getByLabelText('Actor epochs'), { target: { value: '8' } });
    fireEvent.change(screen.getByLabelText('Batch size'), { target: { value: '16' } });

    expect(onActorObjectiveChange).toHaveBeenCalledWith('td3');
    expect(onCriticEpochsChange).toHaveBeenCalledWith('20');
    expect(onActorEpochsChange).toHaveBeenCalledWith('8');
    expect(onBatchSizeChange).toHaveBeenCalledWith('16');
  });

  test('forwards the compact fit-content contract to the shared loop shell', () => {
    renderLoop({ fitContent: true });

    const loop = screen.getByTestId('act-td3-training-loop');
    expect(loop).toHaveAttribute('data-fit-content', 'true');
    expect(loop).toHaveClass('self-start');
  });

  test('preserves controlled ACT trainability and dataset inspection callbacks', () => {
    const onTrainableGroupsChange = jest.fn();
    const onInspectDataset = jest.fn();
    renderLoop({ onTrainableGroupsChange, onInspectDataset });

    fireEvent.click(screen.getByRole('button', { name: /CVAE encoder/i }));
    expect(onTrainableGroupsChange).toHaveBeenCalledWith([
      'visual_backbone',
      'transformer_encoder',
      'action_decoder',
    ]);

    fireEvent.click(screen.getByLabelText('Inspect success datasets: 7 episodes'));
    fireEvent.click(screen.getByRole('button', { name: /data_epoch_0001.*7 episodes/i }));
    expect(onInspectDataset).toHaveBeenCalledWith(
      expect.objectContaining({ path: '/workspace/lerobot/data_epoch_0001' }),
      'success'
    );
  });

  test('locks controls while disabled and exposes the updated-policy state', () => {
    const onActorObjectiveChange = jest.fn();
    const onCriticEpochsChange = jest.fn();
    renderLoop({
      disabled: true,
      updated: true,
      onActorObjectiveChange,
      onCriticEpochsChange,
    });

    expect(screen.getByLabelText('ACT policy: updated')).toBeInTheDocument();
    expect(screen.getByText('Updated policy')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'TD3 loss' })).toBeDisabled();
    expect(screen.getByLabelText('Critic epochs')).toBeDisabled();
    expect(screen.getByRole('button', { name: /Visual backbone/i })).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: 'TD3 loss' }));
    fireEvent.change(screen.getByLabelText('Critic epochs'), { target: { value: '20' } });
    expect(onActorObjectiveChange).not.toHaveBeenCalled();
    expect(onCriticEpochsChange).not.toHaveBeenCalled();
  });

  test('keeps the policy and replay stages while switching the third card to imitation learning', () => {
    const onImitationStepsChange = jest.fn();
    const onImitationBatchSizeChange = jest.fn();
    const onImitationSaveFreqChange = jest.fn();
    const onImitationActionChunkSizeChange = jest.fn();
    renderLoop({
      mode: 'imitation',
      imitationSteps: '80000',
      onImitationStepsChange,
      imitationBatchSize: '8',
      onImitationBatchSizeChange,
      imitationSaveFreq: '10000',
      onImitationSaveFreqChange,
      imitationActionChunkSize: '30',
      onImitationActionChunkSizeChange,
    });

    expect(screen.getByRole('region', { name: 'ACT imitation training loop' })).toBeInTheDocument();
    expect(screen.getByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(screen.getByTestId('act-imitation-algorithm-card')).toBeInTheDocument();
    expect(screen.queryByTestId('act-td3-algorithm-card')).not.toBeInTheDocument();
    expect(screen.getByText('Action Chunk Reconstruction')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Imitation steps'), { target: { value: '90000' } });
    fireEvent.change(screen.getByLabelText('Imitation batch size'), { target: { value: '16' } });
    fireEvent.change(screen.getByLabelText('Imitation save frequency'), { target: { value: '5000' } });
    fireEvent.change(screen.getByLabelText('Imitation action chunk'), { target: { value: '20' } });

    expect(onImitationStepsChange).toHaveBeenCalledWith('90000');
    expect(onImitationBatchSizeChange).toHaveBeenCalledWith('16');
    expect(onImitationSaveFreqChange).toHaveBeenCalledWith('5000');
    expect(onImitationActionChunkSizeChange).toHaveBeenCalledWith('20');
  });

  test('freezes the ACT actor while keeping critic warm-up controls editable', () => {
    const onCriticWarmupUpdatesChange = jest.fn();
    const onCriticWarmupBatchSizeChange = jest.fn();
    renderLoop({
      mode: 'critic',
      criticWarmupUpdates: '1000',
      onCriticWarmupUpdatesChange,
      criticWarmupBatchSize: '256',
      onCriticWarmupBatchSizeChange,
      criticCheckpointPath: '/workspace/checkpoint/act/critic/warmup.pt',
      criticGuidance: 'The warmed critic will be reused by the next ACT TD3 round.',
    });

    expect(screen.getByRole('region', { name: 'ACT critic warm-up loop' })).toBeInTheDocument();
    expect(screen.getByTestId('act-architecture-diagram')).toBeInTheDocument();
    expect(screen.getByTestId('training-replay-buffer-card')).toBeInTheDocument();
    expect(screen.getByTestId('act-critic-warmup-card')).toBeInTheDocument();
    expect(screen.getAllByText('ACT Policy')).toHaveLength(2);
    expect(screen.getByText('Critic Network')).toBeInTheDocument();
    expect(screen.getAllByText('Frozen').length).toBeGreaterThanOrEqual(4);
    expect(screen.getByRole('button', { name: /Visual backbone: Frozen/i })).toBeDisabled();
    expect(screen.getByLabelText('Critic warm-up total updates')).not.toBeDisabled();
    expect(screen.getByLabelText('Critic warm-up batch size')).not.toBeDisabled();
    expect(screen.getByLabelText('ACT critic warm-up checkpoint')).toHaveTextContent(
      '/workspace/checkpoint/act/critic/warmup.pt'
    );
    expect(screen.getByText(/warmed critic will be reused/i)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Critic warm-up total updates'), { target: { value: '2000' } });
    fireEvent.change(screen.getByLabelText('Critic warm-up batch size'), { target: { value: '128' } });
    expect(onCriticWarmupUpdatesChange).toHaveBeenCalledWith('2000');
    expect(onCriticWarmupBatchSizeChange).toHaveBeenCalledWith('128');
  });

  test('global disabled state also locks imitation and critic settings', () => {
    const { unmount } = renderLoop({ mode: 'imitation', disabled: true });
    expect(screen.getByLabelText('Imitation steps')).toBeDisabled();
    expect(screen.getByLabelText('Imitation action chunk')).toBeDisabled();
    unmount();

    renderLoop({ mode: 'critic', disabled: true });
    expect(screen.getByLabelText('Critic warm-up total updates')).toBeDisabled();
    expect(screen.getByLabelText('Critic warm-up batch size')).toBeDisabled();
  });
});
