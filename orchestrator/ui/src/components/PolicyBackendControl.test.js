import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import PolicyBackendControl from './PolicyBackendControl';

jest.mock('react-hot-toast', () => ({
  success: jest.fn(),
  error: jest.fn(),
}));

const mockRegisterHFUser = jest.fn();
const mockListHFEndpoints = jest.fn();

jest.mock('../hooks/useRosServiceCaller', () => ({
  useRosServiceCaller: () => ({
    registerHFUser: mockRegisterHFUser,
    listHFEndpoints: mockListHFEndpoints,
  }),
}));

function mockResponse(data, options = {}) {
  return {
    ok: options.ok ?? true,
    status: options.status ?? 200,
    body: options.body,
    text: async () => (
      typeof data === 'string' ? data : JSON.stringify(data)
    ),
  };
}

describe('PolicyBackendControl', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
    mockRegisterHFUser.mockReset();
    mockListHFEndpoints.mockReset();
    mockListHFEndpoints.mockResolvedValue({ success: true, endpoints: [] });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('shows only pull controls until the backend image exists', async () => {
    global.fetch
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        container_state: 'not_created',
        image_pulled: false,
        services: [],
      }))
      .mockResolvedValueOnce(mockResponse('', { body: null }))
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        container_state: 'not_created',
        image_pulled: true,
        services: [],
      }));

    render(<PolicyBackendControl serviceType="groot" />);

    const pullButton = await screen.findByRole('button', {
      name: 'GR00T Docker pull image',
    });
    expect(screen.queryByRole('button', { name: 'GR00T Docker on' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker restart' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker off' }))
      .not.toBeInTheDocument();

    fireEvent.click(pullButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/backends/groot/pull',
        { method: 'POST' }
      );
    });
    await waitFor(() => {
      expect(screen.getByText('Image ready. Press ON.')).toBeInTheDocument();
    });
  });

  it('shows an explicit update action when the existing container image is stale', async () => {
    global.fetch
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        image: 'robotis/groot-zenoh:1.3.4-arm64',
        image_pulled: true,
        image_status: 'stale',
        container_state: 'exited',
        raw_state: 'stale_image',
        services: [],
      }))
      .mockResolvedValueOnce(mockResponse({ ok: true, message: 'recreated' }))
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        image: 'robotis/groot-zenoh:1.3.4-arm64',
        image_pulled: true,
        image_status: 'current',
        container_state: 'exited',
        services: [],
      }));

    render(<PolicyBackendControl serviceType="groot" />);

    const updateButton = await screen.findByRole('button', {
      name: 'GR00T Docker update container',
    });

    expect(screen.getByText('Update required')).toBeInTheDocument();
    expect(screen.getByText('Container outdated')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker on' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker restart' }))
      .not.toBeInTheDocument();

    fireEvent.click(updateButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/backends/groot/recreate',
        { method: 'POST' }
      );
    });
  });

  it('shows an explicit update action when the backend workspace mount is stale', async () => {
    global.fetch.mockResolvedValueOnce(mockResponse({
      name: 'lerobot',
      image: 'robotis/lerobot-zenoh:1.3.2-arm64',
      image_pulled: true,
      image_status: 'stale',
      container_state: 'exited',
      raw_state: 'workspace_mount_mismatch',
      services: [],
    }));

    render(<PolicyBackendControl serviceType="lerobot" />);

    await screen.findByRole('button', {
      name: 'LeRobot Docker update container',
    });

    expect(screen.getByText('Update required')).toBeInTheDocument();
    expect(screen.getByText('Container outdated')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'LeRobot Docker on' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'LeRobot Docker restart' }))
      .not.toBeInTheDocument();
  });

  it('lets GR00T users register a Hugging Face token from inference controls', async () => {
    mockRegisterHFUser.mockResolvedValue({ success: true });
    global.fetch.mockResolvedValueOnce(mockResponse({
      name: 'groot',
      image: 'robotis/groot-zenoh:1.3.4-arm64',
      image_pulled: true,
      image_status: 'current',
      container_state: 'running',
      services: [],
    }));

    render(<PolicyBackendControl serviceType="groot" />);

    const tokenButton = await screen.findByRole('button', {
      name: 'GR00T Docker Hugging Face token',
    });
    fireEvent.click(tokenButton);
    fireEvent.change(screen.getByPlaceholderText('Enter your Hugging Face token'), {
      target: { value: 'hf_test_token' },
    });
    fireEvent.click(screen.getByText('Submit'));

    await waitFor(() => {
      expect(mockRegisterHFUser).toHaveBeenCalledWith({
        endpoint: 'https://huggingface.co',
        label: 'Hugging Face',
        token: 'hf_test_token',
      });
    });
  });
});
