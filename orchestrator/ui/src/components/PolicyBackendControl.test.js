import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import PolicyBackendControl from './PolicyBackendControl';
import { testRuntime } from '../testUtils/policyCatalog';

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
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it('waits for a slow status response before scheduling another poll', async () => {
    jest.useFakeTimers();
    let finish;
    fetch.mockReturnValue(new Promise(resolve => { finish = resolve; }));
    render(<PolicyBackendControl serviceType="lerobot" runtime={testRuntime('lerobot')} />);
    await act(async () => { jest.advanceTimersByTime(15000); });
    expect(fetch).toHaveBeenCalledTimes(1);
    await act(async () => finish(mockResponse({container_state: 'not_created', image_pulled: false})));
    expect(screen.getByRole('button', {name: 'LeRobot Docker build image'})).toBeEnabled();
    await act(async () => { jest.advanceTimersByTime(5000); });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('builds the missing image without pulling or starting a worker', async () => {
    global.fetch
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        container_state: 'not_created',
        image_pulled: false,
        services: [],
      }))
      .mockResolvedValueOnce(mockResponse({ state: 'building', log_tail: [] }, { status: 202 }))
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        container_state: 'not_created',
        image_pulled: true,
        build: { state: 'succeeded', message: 'Image build complete', log_tail: ['exporting image'] },
        services: [],
      }));

    render(<PolicyBackendControl serviceType="groot" runtime={testRuntime('groot')} />);

    const buildButton = await screen.findByRole('button', {
      name: 'GR00T Docker build image',
    });
    expect(screen.queryByRole('button', { name: 'GR00T Docker on' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker restart' }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'GR00T Docker off' }))
      .not.toBeInTheDocument();

    fireEvent.click(buildButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/backends/groot/build',
        { method: 'POST' }
      );
    });
    await waitFor(() => {
      expect(screen.getByText('Image build complete')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: 'GR00T Docker on' })).toBeEnabled();
    expect(fetch.mock.calls.some(([url]) => url.endsWith('/pull') || url.endsWith('/start'))).toBe(false);
  });

  it('recovers build progress on mount and blocks lifecycle controls while building', async () => {
    fetch.mockResolvedValue(mockResponse({
      container_state: 'exited', image_pulled: true, services: [],
      build: { state: 'building', message: 'Building image', log_tail: ['compiling kernels'] },
    }));
    render(<PolicyBackendControl serviceType="lerobot" runtime={testRuntime('lerobot')} />);
    expect(await screen.findByText('Building image')).toBeInTheDocument();
    expect(screen.getByText('compiling kernels')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'LeRobot Docker on' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'LeRobot Docker rebuild image' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: 'LeRobot Docker on' }));
    expect(fetch.mock.calls.every(([, options]) => !options?.method)).toBe(true);
  });

  it('shows a failed build and offers retry without treating an old image as success', async () => {
    fetch.mockResolvedValue(mockResponse({
      container_state: 'exited', image_pulled: true, services: [],
      build: { state: 'failed', message: 'Image build failed (exit 1)', log_tail: ['compiler failed'] },
    }));
    render(<PolicyBackendControl serviceType="lerobot" runtime={testRuntime('lerobot')} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Image build failed');
    expect(screen.getByRole('button', { name: 'LeRobot Docker rebuild image' })).toBeEnabled();
  });

  it('does not put a late build response into a different worker panel', async () => {
    let finish;
    fetch.mockResolvedValueOnce(mockResponse({ container_state: 'not_created', image_pulled: false }))
      .mockReturnValueOnce(new Promise(resolve => { finish = resolve; }))
      .mockResolvedValue(mockResponse({ container_state: 'not_created', image_pulled: false }));
    const { rerender } = render(<PolicyBackendControl serviceType="lerobot" runtime={testRuntime('lerobot')} />);
    fireEvent.click(await screen.findByRole('button', { name: 'LeRobot Docker build image' }));
    rerender(<PolicyBackendControl serviceType="groot" runtime={testRuntime('groot')} />);
    await screen.findByRole('button', { name: 'GR00T Docker build image' });
    await act(async () => finish(mockResponse({ state: 'building', message: 'old worker build', log_tail: [] })));
    expect(screen.queryByText('old worker build')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'GR00T Docker build image' })).toBeEnabled();
  });

  it('shows an explicit update action when the existing container image is stale', async () => {
    global.fetch
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        image: 'robotis/groot-zenoh:1.3.5-arm64',
        image_pulled: true,
        image_status: 'stale',
        container_state: 'exited',
        raw_state: 'stale_image',
        services: [],
      }))
      .mockResolvedValueOnce(mockResponse({ ok: true, message: 'recreated' }))
      .mockResolvedValueOnce(mockResponse({
        name: 'groot',
        image: 'robotis/groot-zenoh:1.3.5-arm64',
        image_pulled: true,
        image_status: 'current',
        container_state: 'exited',
        services: [],
      }));

    render(<PolicyBackendControl serviceType="groot" runtime={testRuntime('groot')} />);

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
      image: 'robotis/lerobot-zenoh:1.4.2-arm64',
      image_pulled: true,
      image_status: 'stale',
      container_state: 'exited',
      raw_state: 'workspace_mount_mismatch',
      services: [],
    }));

    render(<PolicyBackendControl serviceType="lerobot" runtime={testRuntime('lerobot')} />);

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
      image: 'robotis/groot-zenoh:1.3.5-arm64',
      image_pulled: true,
      image_status: 'current',
      container_state: 'running',
      services: [],
    }));

    render(<PolicyBackendControl serviceType="groot" runtime={testRuntime('groot')} />);

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
