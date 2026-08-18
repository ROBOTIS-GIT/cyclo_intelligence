import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import taskReducer, { selectRobotType } from '../../tasks/taskSlice';
import editDatasetReducer, {
  setConversionStatus,
} from '../../editDataset/editDatasetSlice';
import OfflineRLTrainingSection from './OfflineRLTrainingSection';
import {
  getOfflineRLStatus,
  startOfflineRLTraining,
} from '../../../utils/offlineRlApi';

jest.mock('../../../utils/offlineRlApi', () => ({
  getOfflineRLStatus: jest.fn(),
  startOfflineRLTraining: jest.fn(),
}));

const deferred = () => {
  let resolve;
  const promise = new Promise((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
};

const renderSection = (props = {}) => {
  const testStore = configureStore({
    reducer: { tasks: taskReducer, editDataset: editDatasetReducer },
  });
  testStore.dispatch(selectRobotType('ffw_sg2_rev1'));
  const view = render(
    <Provider store={testStore}>
      <OfflineRLTrainingSection {...props} />
    </Provider>
  );
  return { ...view, testStore };
};

describe('OfflineRLTrainingSection', () => {
  beforeEach(() => {
    getOfflineRLStatus.mockResolvedValue({ status: 'idle', percentage: 0 });
    startOfflineRLTraining.mockResolvedValue({ status: 'starting', percentage: 0 });
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.clearAllMocks();
  });

  test('shows the bounded round and editable valid TD3 schedule', async () => {
    renderSection();

    expect(screen.getByText('200 episodes')).toBeInTheDocument();
    expect(screen.getByText('Auto · 1–50')).toBeInTheDocument();
    expect(screen.getByLabelText('Critic epochs')).toHaveValue(10);
    expect(screen.getByLabelText('Actor equivalent epochs')).toHaveValue(5);
    expect(screen.getByRole('option', { name: /SAC.*Coming soon/i })).toBeDisabled();
    expect(screen.getByRole('option', { name: /RLT.*Coming soon/i })).toBeDisabled();
    await screen.findByRole('button', { name: 'Start Offline RL' });
  });

  test('keeps inputs and Start locked until the first successful status response', async () => {
    const initialStatus = deferred();
    getOfflineRLStatus.mockReturnValueOnce(initialStatus.promise);

    renderSection();

    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByLabelText('Original ACT Checkpoint')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    await act(async () => {
      initialStatus.resolve({ status: 'idle', percentage: 0 });
      await initialStatus.promise;
    });

    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).not.toBeDisabled();
    expect(screen.getByLabelText('Original ACT Checkpoint')).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Start Offline RL' })).not.toBeDisabled();
  });

  test('starts TD3 with the dataset, original ACT, optional parent, and robot type', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Offline RL' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.change(screen.getByRole('textbox', { name: /Previous Round Checkpoint/ }), {
      target: { value: '/workspace/model/lerobot/round1/training_state/act_td3.pt' },
    });
    fireEvent.change(screen.getByLabelText('Critic epochs'), {
      target: { value: '6' },
    });
    fireEvent.change(screen.getByLabelText('Actor equivalent epochs'), {
      target: { value: '3' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Start Offline RL' }));

    await waitFor(() => expect(startOfflineRLTraining).toHaveBeenCalledWith({
      dataset_path: '/workspace/lerobot/task_lerobot_v30',
      act_checkpoint: '/workspace/model/lerobot/base/pretrained_model',
      parent_checkpoint: '/workspace/model/lerobot/round1/training_state/act_td3.pt',
      algorithm: 'td3',
      robot_type: 'ffw_sg2_rev1',
      critic_epochs: 6,
      actor_equivalent_epochs: 3,
    }));
  });

  test('rejects an actor schedule that violates TD3 policy delay 2', async () => {
    renderSection();
    await screen.findByRole('button', { name: 'Start Offline RL' });

    fireEvent.change(screen.getByLabelText('Actor equivalent epochs'), {
      target: { value: '4' },
    });

    expect(screen.getByRole('button', { name: 'Start Offline RL' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('blocks training while dataset conversion is running', async () => {
    const { testStore } = renderSection();
    await screen.findByRole('button', { name: 'Start Offline RL' });

    act(() => {
      testStore.dispatch(setConversionStatus({ status: 'running' }));
    });

    expect(await screen.findByText(/Dataset conversion is running/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start Offline RL' })).toBeDisabled();
    expect(startOfflineRLTraining).not.toHaveBeenCalled();
  });

  test('renders backend progress, metrics, and generated paths', async () => {
    getOfflineRLStatus.mockResolvedValue({
      status: 'complete',
      percentage: 100,
      episode_count: 50,
      round_index: 1,
      round_episode_count: 30,
      success_count: 42,
      failure_count: 8,
      completed_epochs: 10,
      total_epochs: 10,
      completed_critic_updates: 320,
      total_critic_updates: 320,
      completed_actor_updates: 160,
      total_actor_updates: 160,
      critic_loss: 0.001,
      actor_loss: 0.02,
      eta_seconds: 0,
      model_path: '/workspace/model/lerobot/result/pretrained_model',
      checkpoint_path: '/workspace/model/lerobot/result/training_state/act_td3.pt',
    });

    renderSection();

    expect(await screen.findByText('Complete')).toBeInTheDocument();
    expect(screen.getByText('100%')).toBeInTheDocument();
    expect(screen.getByText('50 / 200')).toBeInTheDocument();
    expect(screen.getByText('1 / 30')).toBeInTheDocument();
    expect(screen.getByText('42 / 8')).toBeInTheDocument();
    expect(screen.getByText('/workspace/model/lerobot/result/pretrained_model')).toBeInTheDocument();
    expect(screen.getByText('/workspace/model/lerobot/result/training_state/act_td3.pt')).toBeInTheDocument();
  });

  test('ignores an older GET that resolves after a newer POST result', async () => {
    jest.useFakeTimers();
    const staleStatus = deferred();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockReturnValueOnce(staleStatus.promise);
    startOfflineRLTraining.mockResolvedValue({
      status: 'running',
      percentage: 12,
      message: 'New training job',
    });

    const view = renderSection();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'Start Offline RL' })).not.toBeDisabled();

    act(() => {
      jest.advanceTimersByTime(2000);
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);
    act(() => {
      jest.advanceTimersByTime(6000);
    });
    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);

    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Offline RL' }));
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('12%')).toBeInTheDocument();

    await act(async () => {
      staleStatus.resolve({ status: 'idle', percentage: 0 });
      await staleStatus.promise;
    });
    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('12%')).toBeInTheDocument();

    view.unmount();
  });

  test('keeps controls locked after an ambiguous POST failure until status reconciliation reports running', async () => {
    const reconciliationStatus = deferred();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockReturnValueOnce(reconciliationStatus.promise);
    startOfflineRLTraining.mockRejectedValueOnce(new Error('Connection lost'));

    renderSection();
    await screen.findByRole('button', { name: 'Start Offline RL' });
    fireEvent.change(screen.getByLabelText('LeRobot v3 Dataset Path'), {
      target: { value: '/workspace/lerobot/task_lerobot_v30' },
    });
    fireEvent.change(screen.getByLabelText('Original ACT Checkpoint'), {
      target: { value: '/workspace/model/lerobot/base/pretrained_model' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Start Offline RL' }));

    await waitFor(() => expect(getOfflineRLStatus).toHaveBeenCalledTimes(2));
    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    await act(async () => {
      reconciliationStatus.resolve({ status: 'running', percentage: 7 });
      await reconciliationStatus.promise;
    });

    expect(screen.getByText('Training')).toBeInTheDocument();
    expect(screen.getByText('7%')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Training…' })).toBeDisabled();
  });

  test('locks controls again when a later current status request fails', async () => {
    jest.useFakeTimers();
    getOfflineRLStatus
      .mockResolvedValueOnce({ status: 'idle', percentage: 0 })
      .mockRejectedValueOnce(new Error('Status unavailable'));

    const view = renderSection();
    await act(async () => {
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(screen.getByRole('button', { name: 'Start Offline RL' })).not.toBeDisabled();

    await act(async () => {
      jest.advanceTimersByTime(2000);
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(getOfflineRLStatus).toHaveBeenCalledTimes(2);
    expect(screen.getByLabelText('LeRobot v3 Dataset Path')).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Checking status…' })).toBeDisabled();

    view.unmount();
  });
});
