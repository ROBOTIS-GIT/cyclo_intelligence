import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer from '../editDatasetSlice';
import taskReducer from '../../tasks/taskSlice';
import DatasetConvertSection from './DatasetConvertSection';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../../../components/FileBrowserModal', () => () => null);

const renderSection = () => {
  const store = configureStore({
    reducer: { tasks: taskReducer, editDataset: editDatasetReducer },
  });
  return render(
    <Provider store={store}>
      <DatasetConvertSection />
    </Provider>
  );
};

test('shows raw outcome counts and asks the server to prune oldest episodes', async () => {
  const getDatasetInfo = jest.fn().mockResolvedValue({
    success: true,
    dataset_info: {
      episode_count: 12,
      success_count: 8,
      failure_count: 3,
      unlabeled_count: 1,
    },
  });
  const sendEditDatasetCommand = jest.fn().mockResolvedValue({
    success: true,
    affected_count: 3,
    message: 'deleted',
  });
  useRosServiceCaller.mockReturnValue({
    getDatasetInfo,
    sendEditDatasetCommand,
    sendRecordCommand: jest.fn(),
  });
  jest.spyOn(window, 'confirm').mockReturnValue(true);

  renderSection();
  fireEvent.change(screen.getByPlaceholderText(/Task_1_1_MCAP/), {
    target: { value: 'Task_demo_inference_MCAP' },
  });

  await waitFor(() => expect(getDatasetInfo).toHaveBeenCalledWith(
    '/workspace/rosbag2/Task_demo_inference_MCAP'
  ));
  expect(await screen.findByText('12 / 200')).toBeInTheDocument();
  expect(screen.getByText('8')).toBeInTheDocument();
  expect(screen.getByText('3')).toBeInTheDocument();
  expect(screen.getByText('1')).toBeInTheDocument();

  fireEvent.change(screen.getByLabelText('Delete oldest success count'), {
    target: { value: '2' },
  });
  fireEvent.change(screen.getByLabelText('Delete oldest failure count'), {
    target: { value: '1' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Delete Oldest' }));

  await waitFor(() => expect(sendEditDatasetCommand).toHaveBeenCalledWith(
    'prune_oldest',
    {
      deleteTaskDir: '/workspace/rosbag2/Task_demo_inference_MCAP',
      pruneOldestSuccessCount: 2,
      pruneOldestFailureCount: 1,
    }
  ));
  expect(window.confirm).toHaveBeenCalled();

  window.confirm.mockRestore();
});
