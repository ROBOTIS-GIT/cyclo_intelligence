import { configureStore } from '@reduxjs/toolkit';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import editDatasetReducer from '../editDatasetSlice';
import HuggingfaceSection from './DatasetHuggingfaceSection';

const mockControlHfServer = jest.fn();
const mockListHFEndpoints = jest.fn();
const mockGetRegisteredHFUser = jest.fn();
const endpoint = 'https://huggingface.co';

jest.mock('../../../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({
    controlHfServer: mockControlHfServer,
    listHFEndpoints: mockListHFEndpoints,
    getRegisteredHFUser: mockGetRegisteredHFUser,
  }),
}));
jest.mock('../../../hooks/useWorkspaceMount', () => () => null);
jest.mock('react-hot-toast', () => ({ success: jest.fn(), error: jest.fn() }));
jest.mock('../../../components/TokenInputPopup', () => () => null);
jest.mock('../../../components/FileBrowserModal', () => (props) => (
  props.isOpen ? (
    <button
      aria-label={props.title}
      data-initial-path={props.initialPath}
      data-default-path={props.defaultPath}
      onClick={() => props.onFileSelect({ full_path: `${props.initialPath}/owner/checkpoint` })}
    >
      Select checkpoint
    </button>
  ) : null
));

beforeEach(() => {
  jest.clearAllMocks();
  mockControlHfServer.mockResolvedValue({ success: true });
  mockListHFEndpoints.mockResolvedValue({
    success: true, active: endpoint, endpoints: [endpoint], user_ids: ['owner'],
  });
  mockGetRegisteredHFUser.mockResolvedValue({ success: true, user_id_list: ['owner'] });
});

async function showSection(section) {
  const store = configureStore({
    reducer: { editDataset: editDatasetReducer },
    preloadedState: {
      editDataset: {
        ...editDatasetReducer(undefined, { type: '@@init' }),
        hfUserId: 'owner', hfRepoIdUpload: 'checkpoint', hfRepoIdDownload: 'checkpoint',
        hfActiveEndpoint: endpoint,
      },
    },
  });
  render(<Provider store={store}><HuggingfaceSection /></Provider>);
  await waitFor(() => expect(mockGetRegisteredHFUser).toHaveBeenCalledWith(endpoint));
  fireEvent.click(screen.getByRole('button', { name: `Switch to ${section} section` }));
  fireEvent.click(screen.getByRole('radio', { name: 'Model' }));
  fireEvent.click(screen.getByRole('radio', { name: 'RLDX' }));
}

test('RLDX upload browses its checkpoint root and sends the selected folder as a model', async () => {
  await showSection('upload');
  expect(screen.getByDisplayValue('/workspace/model/rldx')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Browse files for local directory' }));
  const browser = screen.getByRole('button', { name: 'Select Local Directory for Upload' });
  expect(browser).toHaveAttribute('data-initial-path', '/workspace/model/rldx');
  expect(browser).toHaveAttribute('data-default-path', '/workspace/model/rldx');
  fireEvent.click(browser);
  fireEvent.click(screen.getByRole('button', { name: 'Upload', exact: true }));
  await waitFor(() => expect(mockControlHfServer).toHaveBeenCalledWith(
    'upload', 'owner/checkpoint', 'model', '/workspace/model/rldx/owner/checkpoint', endpoint
  ));
});

test('RLDX download sends the model root and switches back to existing backend defaults', async () => {
  await showSection('download');
  expect(screen.getByDisplayValue('/workspace/model/rldx')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: 'Download', exact: true }));
  await waitFor(() => expect(mockControlHfServer).toHaveBeenCalledWith(
    'download', 'owner/checkpoint', 'model', '/workspace/model/rldx', endpoint
  ));
  await waitFor(() => expect(screen.getByRole('radio', { name: 'GR00T' })).toBeEnabled());
  fireEvent.click(screen.getByRole('radio', { name: 'GR00T' }));
  expect(screen.getByDisplayValue('/workspace/model/groot')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('radio', { name: 'LeRobot' }));
  expect(screen.getByDisplayValue('/workspace/model/lerobot')).toBeInTheDocument();
  fireEvent.click(screen.getByRole('radio', { name: 'Dataset' }));
  expect(screen.getByDisplayValue('/workspace/rosbag2')).toBeInTheDocument();
});

test('selecting RLDX preserves a custom destination', async () => {
  await showSection('download');
  fireEvent.change(screen.getByDisplayValue('/workspace/model/rldx'), {
    target: { value: '/workspace/custom_models' },
  });
  fireEvent.click(screen.getByRole('radio', { name: 'LeRobot' }));
  fireEvent.click(screen.getByRole('radio', { name: 'RLDX' }));
  expect(screen.getByDisplayValue('/workspace/custom_models')).toBeInTheDocument();
});
