import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer, {
  setConversionStatus,
} from '../../editDataset/editDatasetSlice';
import taskReducer from '../../tasks/taskSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import { reserveOfflineRLDataEpoch } from '../../../utils/offlineRlApi';
import offlineRLReducer, {
  setOfflineRLConversionDestinationPath,
  setOfflineRLReplayBufferPath,
} from '../offlineRLSlice';
import OfflineRLDatasetConversion, {
  conversionTaskName,
  deriveConvertedDatasetPaths,
  isAllowedConversionDestinationPath,
} from './OfflineRLDatasetConversion';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: jest.fn(),
}));

jest.mock('../../../utils/offlineRlApi', () => ({
  reserveOfflineRLDataEpoch: jest.fn(),
}));

jest.mock('../../../components/FileBrowserModal', () => {
  return function MockFileBrowserModal() {
    return null;
  };
});

const renderConversion = () => {
  reserveOfflineRLDataEpoch.mockResolvedValue({
    data_epoch: 0,
    epoch_name: 'data_epoch_0000',
    output_root: '/workspace/lerobot/round_1/data_epoch_0000',
    expected_outputs: {
      v21: '/workspace/lerobot/round_1/data_epoch_0000/Task_conversion_test_inference_MCAP_lerobot_v21',
      v30: '/workspace/lerobot/round_1/data_epoch_0000/Task_conversion_test_inference_MCAP_lerobot_v30',
    },
  });
  const sendRecordCommand = jest.fn().mockResolvedValue({
    success: true,
    job_id: 'job-1',
  });
  useRosServiceCaller.mockReturnValue({ sendRecordCommand });
  const store = configureStore({
    reducer: {
      tasks: taskReducer,
      editDataset: editDatasetReducer,
      offlineRL: offlineRLReducer,
    },
  });
  store.dispatch(setOfflineRLReplayBufferPath(
    '/workspace/rosbag2/Task_conversion_test_inference_MCAP'
  ));
  render(
    <Provider store={store}>
      <OfflineRLDatasetConversion />
    </Provider>
  );
  return { store, sendRecordCommand };
};

test('derives the Step 1 source task and destination-relative LeRobot outputs', () => {
  expect(conversionTaskName('/workspace/rosbag2/Task_01'))
    .toBe('Task_01');
  expect(deriveConvertedDatasetPaths(
    '/workspace/rosbag2/Task_01',
    '/workspace/lerobot/round_1'
  ))
    .toEqual({
      v21: '/workspace/lerobot/round_1/Task_01_lerobot_v21',
      v30: '/workspace/lerobot/round_1/Task_01_lerobot_v30',
    });
  expect(isAllowedConversionDestinationPath('/workspace/lerobot/round_1')).toBe(true);
  expect(isAllowedConversionDestinationPath('/workspace/model')).toBe(false);
  expect(isAllowedConversionDestinationPath('/workspace/lerobot/../model')).toBe(false);
});

test('shows the formats supported by the existing conversion backend', async () => {
  renderConversion();

  expect(screen.getByTestId('offline-rl-conversion-engine')).toBeInTheDocument();
  expect(screen.queryByText('MCAP Source')).not.toBeInTheDocument();
  expect(screen.queryByText('Validate & Align')).not.toBeInTheDocument();
  expect(screen.queryByText('LeRobot Output')).not.toBeInTheDocument();
  expect(screen.getByText('Conversion setup')).toBeInTheDocument();
  expect(screen.getByTestId('conversion-setup-surface'))
    .toHaveClass('bg-[#f8f5ef]');
  expect(screen.getByText('Validation')).toBeInTheDocument();
  expect(screen.getByText('MCAP linked')).toBeInTheDocument();
  expect(screen.getByText('Destination valid')).toBeInTheDocument();
  expect(screen.getByText('Output selected')).toBeInTheDocument();
  const pathRow = screen.getByTestId('conversion-path-row');
  expect(pathRow).toHaveClass('grid-cols-2');
  expect(within(pathRow).getByText('Conversion setup')).toBeInTheDocument();
  expect(within(pathRow).getByLabelText('LeRobot collection root')).toBeInTheDocument();
  const optionsRow = screen.getByTestId('conversion-options-row');
  const outputFormat = within(optionsRow).getByTestId('conversion-output-format');
  const validation = within(optionsRow).getByLabelText('Conversion validation');
  const dataEpoch = within(optionsRow).getByTestId('offline-rl-data-epoch-output');
  const convertButton = within(optionsRow).getByRole('button', { name: 'Convert Dataset' });
  expect(within(outputFormat).getByRole('button', { name: 'v3.0' }))
    .toBeInTheDocument();
  expect(validation).toBeInTheDocument();
  expect(dataEpoch.nextElementSibling).toBe(convertButton);
  expect(optionsRow.lastElementChild).toBe(convertButton);
  expect(screen.getAllByRole('button', { name: 'Convert Dataset' })).toHaveLength(1);
  expect(screen.queryByRole('progressbar', { name: 'Dataset conversion progress' }))
    .not.toBeInTheDocument();
  expect(within(validation).getByTestId('conversion-validation-summary'))
    .toBeInTheDocument();
  expect(screen.queryByText('Atomic output')).not.toBeInTheDocument();
  expect(screen.queryByText('Reserved at start')).not.toBeInTheDocument();
  expect(screen.queryByText('An immutable output folder is reserved when conversion starts'))
    .not.toBeInTheDocument();
  expect(screen.queryByText(
    'Source MCAP episodes are removed only after every selected LeRobot output passes verification.'
  )).not.toBeInTheDocument();
  expect(screen.queryByText('Ready to convert')).not.toBeInTheDocument();
  expect(await screen.findByRole('button', { name: 'v2.1' }))
    .toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByRole('button', { name: 'v3.0' }))
    .toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByRole('button', { name: 'v2.0' })).not.toBeInTheDocument();
  expect(screen.getByLabelText('Conversion FPS')).toHaveValue(15);
});

test('starts verified conversion with post-success MCAP cleanup and selects v3 output', async () => {
  const { store, sendRecordCommand } = renderConversion();
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  expect(screen.getByText('/workspace/rosbag2/Task_conversion_test_inference_MCAP'))
    .toBeInTheDocument();
  expect(screen.queryByLabelText('Conversion MCAP folder')).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText('LeRobot collection root'), {
    target: { value: '/workspace/lerobot/round_1' },
  });
  fireEvent.click(screen.getByRole('button', { name: 'Convert Dataset' }));

  await waitFor(() => {
    expect(reserveOfflineRLDataEpoch).toHaveBeenCalledWith({
      destination_root: '/workspace/lerobot/round_1',
      source_mcap: '/workspace/rosbag2/Task_conversion_test_inference_MCAP',
      behavior_policy_path: '',
      boundary_reason: 'manual_conversion',
      fps: 15,
      formats: ['v2.1', 'v3.0'],
    });
    expect(sendRecordCommand).toHaveBeenCalledWith('convert_mp4', expect.objectContaining({
      taskSource: 'record',
      conversionFps: 15,
      convertV21: true,
      convertV30: true,
      lerobotOutputRoot: '/workspace/lerobot/round_1/data_epoch_0000',
      deleteSourceAfterSuccess: true,
    }));
  });
  expect(screen.getByText('data_epoch_0000')).toBeInTheDocument();

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'running',
      jobId: 'job-1',
      progress: 50,
    }));
  });
  act(() => {
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'job-1',
      progress: 100,
    }));
  });

  await waitFor(() => {
    expect(store.getState().offlineRL.datasetPath).toBe(
      '/workspace/lerobot/round_1/data_epoch_0000/Task_conversion_test_inference_MCAP_lerobot_v30'
    );
  });
  expect(store.getState().offlineRL.datasetVersion).toBe('v3.0');
  expect(store.getState().offlineRL.convertedDatasetPaths.v21).toContain('_v21');
  expect(confirmSpy).toHaveBeenCalled();
  confirmSpy.mockRestore();
});

test('rejects a destination outside the LeRobot workspace', () => {
  const { store, sendRecordCommand } = renderConversion();
  act(() => {
    store.dispatch(setOfflineRLConversionDestinationPath('/workspace/model'));
  });

  expect(screen.getByText('Destination must be inside /workspace/lerobot'))
    .toBeInTheDocument();
  expect(screen.getByRole('button', { name: 'Convert Dataset' })).toBeDisabled();
  expect(sendRecordCommand).not.toHaveBeenCalled();
});

test('ignores conversion status from a job other than the accepted job ID', async () => {
  const { store, sendRecordCommand } = renderConversion();
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  fireEvent.click(screen.getByRole('button', { name: 'Convert Dataset' }));
  await waitFor(() => expect(sendRecordCommand).toHaveBeenCalled());
  await waitFor(() => {
    expect(screen.getByText('Conversion queued · job-1')).toBeInTheDocument();
  });

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'foreign-job',
      progress: 100,
      message: 'Foreign conversion completed',
    }));
  });
  expect(store.getState().offlineRL.datasetPath).toBe('');
  expect(screen.queryByText(/Complete · data_epoch_0000/)).not.toBeInTheDocument();

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'job-1',
      progress: 100,
    }));
  });
  await waitFor(() => {
    expect(store.getState().offlineRL.datasetPath).toBe(
      '/workspace/lerobot/round_1/data_epoch_0000/Task_conversion_test_inference_MCAP_lerobot_v30'
    );
  });
  confirmSpy.mockRestore();
});

test('latches the first new status ID when SendCommand omits StartConversion job_id', async () => {
  const { store, sendRecordCommand } = renderConversion();
  sendRecordCommand.mockResolvedValueOnce({ success: true });
  const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);

  fireEvent.click(screen.getByRole('button', { name: 'Convert Dataset' }));
  await waitFor(() => {
    expect(screen.getByText('Conversion queued · waiting for job ID'))
      .toBeInTheDocument();
  });

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'running',
      jobId: 'latched-job',
      progress: 25,
    }));
  });
  await waitFor(() => {
    expect(screen.getByText('Converting dataset…')).toBeInTheDocument();
  });

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'different-job',
      progress: 100,
    }));
  });
  expect(store.getState().offlineRL.datasetPath).toBe('');

  act(() => {
    store.dispatch(setConversionStatus({
      status: 'completed',
      jobId: 'latched-job',
      progress: 100,
    }));
  });
  await waitFor(() => {
    expect(store.getState().offlineRL.datasetVersion).toBe('v3.0');
  });
  confirmSpy.mockRestore();
});
