import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Provider } from 'react-redux';
import ImageGrid, { buildTopicRotationMap, getGridColumnCount, normalizeRotationDeg } from './ImageGrid';
import rosReducer, { setAssignedImageTopics } from '../features/ros/rosSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { requestSupervisorApi } from '../features/missionCanvas/lib/btNodeApi';

jest.mock('../features/missionCanvas/lib/btNodeApi', () => ({ requestSupervisorApi: jest.fn() }));

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('./ImageGridCell', () => ({ idx, topic, label, style, rotationDegrees, onClose, onRotateClick, dragHandleProps, resizeHandleProps }) => (
  <div data-testid="camera" data-camera-topic={topic} data-topic={topic} data-rotation={rotationDegrees} style={style}>
    <span>{label || topic}</span>
    <button {...dragHandleProps}>Move camera {idx}</button>
    <button {...resizeHandleProps}>Resize camera {idx}</button>
    <button onClick={() => onClose(idx)}>Close camera {idx}</button>
    <button onClick={() => onRotateClick(idx)}>Rotate camera {idx}</button>
  </div>
));
jest.mock('react-hot-toast', () => ({ __esModule: true, default: { success: jest.fn(), error: jest.fn() } }));
const topicsFor = (n) => Array.from({ length: n }, (_, i) => `/camera-${i}`);
const response = (topics) => ({ success: true, image_topic_list: topics, rotation_deg_list: [0, 90, 180, 270] });
const deferred = () => {
  let resolve;
  const promise = new Promise((yes) => { resolve = yes; });
  return { promise, resolve };
};
const makeStore = () => configureStore({ reducer: {
  ros: rosReducer,
  tasks: (state = { robotType: 'test' }, action) => action.type === 'test/robot' ? { robotType: action.payload } : state,
} });
const mount = (store = makeStore()) => ({ store, ...render(<Provider store={store}><ImageGrid responsive /></Provider>) });
const displayed = () => screen.queryAllByTestId('camera').map((cell) => cell.dataset.topic);
const setup = (n = 4) => {
  const call = jest.fn().mockResolvedValue(response(topicsFor(n)));
  useRosServiceCaller.mockReturnValue({ getImageTopicList: call });
  return call;
};
beforeEach(() => {
  jest.clearAllMocks();
  localStorage.clear();
  requestSupervisorApi.mockResolvedValue({});
});

test('recording adds a visible frame and status without remounting cameras or refetching topics', async () => {
  const call = setup(2);
  const store = makeStore();
  const { rerender } = mount(store);
  const cells = await screen.findAllByTestId('camera');
  const region = screen.getByRole('region', { name: 'Camera feeds' });
  expect(region).not.toHaveClass('image-grid-recording');
  for (const isRecording of [true, false, true]) {
    rerender(<Provider store={store}><ImageGrid responsive isRecording={isRecording} /></Provider>);
    expect(region.classList.contains('image-grid-recording')).toBe(isRecording);
    expect(screen.getByRole('status', { name: 'Camera recording status' }).textContent).toBe(isRecording ? 'Recording' : '');
    expect(screen.getAllByTestId('camera')[0]).toBe(cells[0]);
    expect(screen.getAllByTestId('camera')[1]).toBe(cells[1]);
  }
  expect(call).toHaveBeenCalledTimes(1);
});

test('camera labels use observation keys, including the hidden-camera menu', async () => {
  setup(2);
  requestSupervisorApi.mockResolvedValue({ topic_roles: {
    '/camera-0': { observation: ['cam_head'], action: ['not_a_camera'] },
  } });
  mount();
  await screen.findByText('cam_head');
  expect(screen.queryByText('not_a_camera')).not.toBeInTheDocument();
  expect(screen.getByText('/camera-1')).toBeVisible();
  fireEvent.click(screen.getByText('Close camera 0'));
  fireEvent.click(screen.getByRole('button', { name: 'Show hidden camera' }));
  expect(screen.getByRole('button', { name: 'cam_head' })).toHaveAttribute('title', '/camera-0');
});

test('metadata failure and late previous-robot metadata cannot relabel streams', async () => {
  setup(1);
  const old = deferred();
  requestSupervisorApi.mockReturnValueOnce(old.promise).mockRejectedValueOnce(new Error('offline'));
  const { store } = mount();
  await screen.findAllByTestId('camera');
  act(() => store.dispatch({ type: 'test/robot', payload: 'next' }));
  await act(async () => old.resolve({ topic_roles: { '/camera-0': { observation: ['stale'] } } }));
  expect(screen.queryByText('stale')).not.toBeInTheDocument();
  expect(screen.getByText('/camera-0')).toBeVisible();
});

test('manual resizing can exceed viewport width, preserves minimum height and restores automatic layout', async () => {
  setup(2);
  mount();
  const cells = await screen.findAllByTestId('camera');
  // Geometry is not provided by jsdom; resize uses measured tile and viewport bounds.
  // eslint-disable-next-line testing-library/no-node-access
  Object.defineProperty(document.querySelector('.image-grid-scroll'), 'clientWidth', { value: 800 });
  cells.forEach((cell) => { cell.getBoundingClientRect = () => ({ width: 790, height: 200 }); });
  fireEvent.keyDown(screen.getByText('Resize camera 0'), { key: 'ArrowRight' });
  expect(cells[0]).toHaveStyle({ width: '810px', height: '200px' });
  fireEvent.keyDown(screen.getByText('Resize camera 0'), { key: 'ArrowUp' });
  expect(cells[0]).toHaveStyle({ height: '200px' });
  expect(screen.getAllByTestId('camera')[0]).toBe(cells[0]);
  fireEvent.click(screen.getByRole('button', { name: 'Reset camera layout' }));
  expect(cells[0].style.width).toBe('');
  fireEvent.keyDown(screen.getByText('Resize camera 0'), { key: 'ArrowDown' });
  fireEvent.click(screen.getByText('Close camera 0'));
  expect(screen.getAllByTestId('camera')[0]).toBe(cells[1]);
  expect(cells[1].style.width).toBe('');
  expect(screen.getByRole('button', { name: 'Reset camera layout' })).toBeDisabled();
});

test.each([0, 1, 2, 4, 6])('shows all %i configured cameras', async (count) => {
  const call = setup(count);
  const { store } = mount();
  await waitFor(() => expect(store.getState().ros.assignedImageTopicsRobotType).toBe('test'));
  expect(displayed()).toEqual(topicsFor(count));
  expect(call).toHaveBeenCalledTimes(1);
});

test('closing removes the tile and restoring lists only hidden cameras', async () => {
  setup();
  const { store } = mount();
  await screen.findAllByTestId('camera');
  const retained = screen.getAllByTestId('camera')[2];
  fireEvent.click(screen.getByText('Close camera 1'));
  expect(displayed()).toEqual(['/camera-0', '/camera-2', '/camera-3']);
  expect(screen.getAllByTestId('camera')[1]).toBe(retained);
  fireEvent.click(screen.getByRole('button', { name: 'Show hidden camera' }));
  expect(screen.getByRole('dialog')).toBeVisible();
  expect(screen.queryByRole('button', { name: '/camera-0' })).not.toBeInTheDocument();
  fireEvent.click(screen.getByRole('button', { name: '/camera-1' }));
  expect(displayed()).toEqual(['/camera-0', '/camera-2', '/camera-3', '/camera-1']);
  expect(store.getState().ros.assignedImageTopics).toEqual(displayed());
});

test('keyboard reorder preserves identity and rotation; order and hidden selection survive remount', async () => {
  setup();
  const store = makeStore();
  const view = mount(store);
  await screen.findAllByTestId('camera');
  const moved = screen.getAllByTestId('camera')[1];
  fireEvent.click(screen.getByText('Rotate camera 1'));
  fireEvent.keyDown(screen.getByText('Move camera 1'), { key: 'ArrowRight' });
  expect(displayed()).toEqual(['/camera-0', '/camera-2', '/camera-1', '/camera-3']);
  expect(screen.getAllByTestId('camera')[2]).toBe(moved);
  expect(moved).toHaveAttribute('data-rotation', '180');
  fireEvent.click(screen.getByText('Close camera 0'));
  view.unmount();
  mount(store);
  await waitFor(() => expect(displayed()).toEqual(['/camera-2', '/camera-1', '/camera-3']));
});

test('all cameras hidden survive remount and can be restored', async () => {
  setup(2);
  const { store, unmount } = mount();
  await screen.findAllByTestId('camera');
  fireEvent.click(screen.getByText('Close camera 0'));
  fireEvent.click(screen.getByText('Close camera 0'));
  expect(displayed()).toEqual([]);
  expect(screen.getByRole('status', { name: '' })).toHaveTextContent('All cameras hidden');
  unmount();
  mount(store);
  await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh cameras' })).toBeEnabled());
  expect(displayed()).toEqual([]);
  fireEvent.click(screen.getByRole('button', { name: 'Show hidden camera' }));
  fireEvent.click(screen.getByRole('button', { name: '/camera-1' }));
  expect(displayed()).toEqual(['/camera-1']);
});

test('four rotation clicks complete a full turn from the configured angle', async () => {
  setup(2);
  mount();
  await screen.findAllByTestId('camera');
  const tile = screen.getAllByTestId('camera')[1];
  for (const angle of [180, 270, 0, 90]) {
    fireEvent.click(screen.getByText('Rotate camera 1'));
    expect(tile).toHaveAttribute('data-rotation', String(angle));
    expect(screen.getAllByTestId('camera')[1]).toBe(tile);
  }
});

test('fresh Redux stores restore hidden cameras, order, and rotation per robot', async () => {
  setup(4);
  const first = mount();
  await screen.findAllByTestId('camera');
  fireEvent.click(screen.getByText('Rotate camera 1'));
  fireEvent.keyDown(screen.getByText('Move camera 1'), { key: 'ArrowLeft' });
  fireEvent.click(screen.getByText('Close camera 3'));
  first.unmount();
  const next = mount();
  await waitFor(() => expect(displayed()).toEqual(['/camera-1', '/camera-0', '/camera-2']));
  expect(screen.getAllByTestId('camera')[0]).toHaveAttribute('data-rotation', '180');
  act(() => next.store.dispatch({ type: 'test/robot', payload: 'other' }));
  await waitFor(() => expect(displayed()).toEqual(topicsFor(4)));
  act(() => next.store.dispatch({ type: 'test/robot', payload: 'test' }));
  await waitFor(() => expect(displayed()).toEqual(['/camera-1', '/camera-0', '/camera-2']));
  expect(screen.getAllByTestId('camera')[0]).toHaveAttribute('data-rotation', '180');
});

test('all-hidden selection survives a new Redux store after reload', async () => {
  setup(1);
  const first = mount();
  await screen.findAllByTestId('camera');
  fireEvent.click(screen.getByText('Close camera 0'));
  first.unmount();
  mount();
  await screen.findByText('All cameras hidden');
  expect(displayed()).toEqual([]);
});

test('legacy null slots disappear', async () => {
  setup();
  const store = makeStore();
  store.dispatch(setAssignedImageTopics({ robotType: 'test', topics: ['/camera-3', null, '/camera-1'] }));
  mount(store);
  await waitFor(() => expect(store.getState().ros.assignedImageTopics).toEqual(['/camera-3', '/camera-1']));
  expect(displayed()).toEqual(['/camera-3', '/camera-1']);
});

test('robot replacement rejects old responses', async () => {
  const old = deferred(), next = deferred();
  useRosServiceCaller.mockReturnValue({ getImageTopicList: jest.fn().mockReturnValueOnce(old.promise).mockReturnValueOnce(next.promise) });
  const { store } = mount();
  act(() => store.dispatch({ type: 'test/robot', payload: 'next' }));
  await act(async () => next.resolve(response(['/next'])));
  await act(async () => old.resolve(response(['/stale'])));
  expect(displayed()).toEqual(['/next']);
  expect(store.getState().ros.imageTopicList).toEqual(['/next']);
});

test.each([() => Promise.resolve({ success: false, message: 'Not ready' }), () => Promise.reject(new Error('Offline'))])(
  'first-load failure offers refresh and recovers', async (failure) => {
    useRosServiceCaller.mockReturnValue({ getImageTopicList: jest.fn().mockImplementationOnce(failure).mockResolvedValue(response(['/recovered'])) });
    mount();
    await waitFor(() => expect(screen.getByRole('status', { name: '' })).toHaveTextContent(/Service error|Failed to load/));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh cameras' }));
    await waitFor(() => expect(displayed()).toEqual(['/recovered']));
  }
);

test('inventory refresh removes missing cameras and offers new ones without reopening hidden ones', async () => {
  const call = setup();
  mount();
  await screen.findAllByTestId('camera');
  fireEvent.click(screen.getByText('Close camera 1'));
  call.mockResolvedValue(response(['/camera-0', '/camera-1', '/new']));
  fireEvent.click(screen.getByRole('button', { name: 'Show hidden camera' }));
  fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
  await waitFor(() => expect(displayed()).toEqual(['/camera-0']));
  fireEvent.click(screen.getByRole('button', { name: '/new' }));
  expect(displayed()).toEqual(['/camera-0', '/new']);
});

test('late request after unmount does not change the store', async () => {
  const pending = deferred();
  useRosServiceCaller.mockReturnValue({ getImageTopicList: jest.fn().mockReturnValue(pending.promise) });
  const { store, unmount } = mount();
  unmount();
  const before = store.getState();
  await act(async () => pending.resolve(response(['/late'])));
  expect(store.getState()).toBe(before);
});

test('layout uses height and rotated ratios rather than width only', () => {
  expect(getGridColumnCount(4, 1144, 1000, [4/3, 3/4, 3/4, 4/3])).toBe(2);
  expect(getGridColumnCount(4, 1144, 260, [4/3, 3/4, 3/4, 4/3])).toBe(4);
  expect(getGridColumnCount(1, 1144, 1000)).toBe(1);
  expect(getGridColumnCount(4, 320, 800)).toBe(1);
  expect(getGridColumnCount(0, 0, 0)).toBe(1);
});

test('rotation helpers retain config alignment', () => {
  expect(normalizeRotationDeg(-90)).toBe(270);
  expect(normalizeRotationDeg(undefined)).toBe(0);
  expect(buildTopicRotationMap(['/a', '/b'], [90])).toEqual({ '/a': 90, '/b': 0 });
});
