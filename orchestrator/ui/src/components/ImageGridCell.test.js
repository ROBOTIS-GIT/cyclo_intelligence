import { StrictMode } from 'react';
import { configureStore } from '@reduxjs/toolkit';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { Provider } from 'react-redux';
import ImageGridCell from './ImageGridCell';
import ImageGrid from './ImageGrid';
import rosReducer, { setRosHost } from '../features/ros/rosSlice';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';
import { requestSupervisorApi } from '../features/missionCanvas/lib/btNodeApi';

jest.mock('../hooks/useRosServiceCaller', () => ({ useRosServiceCaller: jest.fn() }));
jest.mock('../features/missionCanvas/lib/btNodeApi', () => ({ requestSupervisorApi: jest.fn().mockResolvedValue({}) }));
jest.mock('react-hot-toast', () => ({ __esModule: true, default: { success: jest.fn(), error: jest.fn() } }));

let observers;
let originalResizeObserver;
beforeEach(() => {
  localStorage.clear();
  requestSupervisorApi.mockResolvedValue({});
  jest.useFakeTimers();
  observers = [];
  originalResizeObserver = global.ResizeObserver;
  global.ResizeObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target) { this.target = target; }
    disconnect = jest.fn();
  };
});
afterEach(() => {
  jest.useRealTimers();
  global.ResizeObserver = originalResizeObserver;
});

const mountCell = (props = {}) => {
  const store = configureStore({ reducer: { ros: rosReducer } });
  const defaults = { topic: '/camera/image/compressed', idx: 0, onClose: jest.fn(), onPlusClick: jest.fn(), onRotateClick: jest.fn() };
  const element = (overrides = {}) => <Provider store={store}><StrictMode><ImageGridCell {...defaults} {...props} {...overrides} /></StrictMode></Provider>;
  const view = render(element());
  act(() => jest.advanceTimersByTime(500));
  return { ...view, store, defaults, update: (overrides) => view.rerender(element(overrides)) };
};

const resize = (width, height) => act(() => {
  observers.filter((observer) => !observer.disconnect.mock.calls.length).forEach((observer) => {
    observer.target.getBoundingClientRect = () => ({ width, height });
    observer.callback([{ contentRect: { width, height } }]);
  });
});
const loadImage = (img, width, height) => {
  Object.defineProperties(img, { naturalWidth: { configurable: true, value: width }, naturalHeight: { configurable: true, value: height } });
  fireEvent.load(img);
};

test('config label and resize controls do not reconnect the image stream', () => {
  const { update } = mountCell({ dragHandleProps: {}, resizeHandleProps: {} });
  const img = screen.getByRole('img');
  const src = img.src;
  update({ label: 'cam_left_head', style: { width: 350, height: 260 } });
  expect(screen.getByRole('button', { name: 'Move camera 1' })).toHaveTextContent('cam_left_head');
  expect(screen.getByRole('button', { name: 'Move camera 1' }).title).toContain('/camera/image/compressed');
  expect(screen.getByRole('button', { name: 'Resize camera 1' })).toBeVisible();
  expect(screen.getByRole('img')).toBe(img);
  expect(img.src).toBe(src);
});

test.each([0, 90, 180, 270])('contains the entire natural image at %i degrees without reconnecting on resize', (rotationDegrees) => {
  const { update } = mountCell({ rotationDegrees });
  const img = screen.getByRole('img');
  const src = img.src;
  resize(320, 180);
  loadImage(img, 640, 480);
  expect(img.style.objectFit).toBe('contain');
  const quarterTurn = rotationDegrees === 90 || rotationDegrees === 270;
  expect(parseFloat(img.style.width)).toBeCloseTo(quarterTurn ? 180 : 240);
  expect(parseFloat(img.style.height)).toBeCloseTo(quarterTurn ? 135 : 180);
  resize(180, 320);
  expect(parseFloat(img.style.width)).toBeCloseTo(quarterTurn ? 240 : 180);
  expect(parseFloat(img.style.height)).toBeCloseTo(quarterTurn ? 180 : 135);
  update({ style: { width: '180px' } });
  expect(screen.getByRole('img')).toBe(img);
  expect(img.src).toBe(src);
});

test('rotation and changed incoming resolution resize the same stream, with toolbar outside its viewport', () => {
  const { update, container, defaults } = mountCell();
  const img = screen.getByRole('img');
  resize(320, 180);
  loadImage(img, 1920, 1080);
  const src = img.src;
  update({ rotationDegrees: 270 });
  expect(screen.getByRole('img')).toBe(img);
  expect(img.src).toBe(src);
  expect(img.style.transform).toContain('rotate(270deg)');
  expect(parseFloat(img.style.width)).toBeCloseTo(180);
  loadImage(img, 480, 640);
  expect(parseFloat(img.style.width)).toBeCloseTo(180);
  expect(parseFloat(img.style.height)).toBeCloseTo(240);
  // Structural checks ensure controls never overlay the image surface.
  // eslint-disable-next-line testing-library/no-node-access
  const viewport = container.querySelector('.image-grid-viewport');
  const rotate = screen.getByRole('button', { name: /rotate camera/i });
  const close = screen.getByRole('button', { name: /close camera/i });
  expect(viewport).not.toContainElement(rotate);
  expect(viewport).not.toContainElement(close);
  expect(viewport).not.toContainElement(screen.getByText('/camera/image/compressed'));
  // eslint-disable-next-line testing-library/no-node-access
  expect(container.querySelector('[class*="rounded"]')).toBeNull();
  fireEvent.click(rotate);
  expect(defaults.onRotateClick).toHaveBeenCalledWith(0);
});

test('topic/host changes and activation replace streams; teardown cancels retries and late callbacks', () => {
  const { update, store, unmount } = mountCell();
  const first = screen.getByRole('img');
  expect(first.src).toContain('default_transport=compressed&topic=/camera/image&');
  const lateError = first.onerror;
  update({ topic: '/other/image/compressed' });
  act(() => jest.advanceTimersByTime(500));
  const second = screen.getByRole('img');
  expect(second).not.toBe(first);
  expect(second.src).toContain('topic=/other/image&');
  expect(first.onerror).toBeNull();
  act(() => { lateError(); jest.advanceTimersByTime(10000); });
  expect(screen.getByRole('img')).toBe(second);
  act(() => store.dispatch(setRosHost('new-host')));
  act(() => jest.advanceTimersByTime(500));
  expect(screen.getByRole('img').src).toContain('http://new-host:');
  update({ isActive: false });
  act(() => jest.advanceTimersByTime(10000));
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
  update({ isActive: true });
  act(() => jest.advanceTimersByTime(500));
  const active = screen.getByRole('img');
  fireEvent.error(active);
  unmount();
  act(() => jest.advanceTimersByTime(10000));
  expect(active.onerror).toBeNull();
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
  expect(observers.every((observer) => observer.disconnect.mock.calls.length > 0)).toBe(true);
});

test('close stops the stream immediately and an empty cell can be reselected by keyboard', () => {
  const { update, defaults } = mountCell();
  const img = screen.getByRole('img');
  fireEvent.click(screen.getByRole('button', { name: /close camera/i }));
  expect(defaults.onClose).toHaveBeenCalledWith(0);
  expect(img.onerror).toBeNull();
  expect(screen.queryByRole('img')).not.toBeInTheDocument();
  update({ topic: null });
  fireEvent.click(screen.getByRole('button', { name: /select camera/i }));
  expect(defaults.onPlusClick).toHaveBeenCalledWith(0);
});

test('offscreen and background cameras release streams and retries across repeated visibility changes', () => {
  const original = global.IntersectionObserver;
  let intersect;
  const disconnect = jest.fn();
  global.IntersectionObserver = class {
    constructor(callback) { intersect = callback; }
    observe() {}
    disconnect = disconnect;
  };
  const { unmount } = mountCell();
  try {
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    for (let i = 0; i < 25; i += 1) {
      act(() => intersect([{ isIntersecting: true }]));
      const img = screen.getByRole('img');
      fireEvent.error(img);
      act(() => intersect([{ isIntersecting: false }]));
      act(() => jest.advanceTimersByTime(20000));
      expect(screen.queryByRole('img')).not.toBeInTheDocument();
      expect(img.onerror).toBeNull();
    }
    act(() => intersect([{ isIntersecting: true }]));
    expect(screen.getByRole('img')).toBeInTheDocument();
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    fireEvent(document, new Event('visibilitychange'));
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    fireEvent(document, new Event('visibilitychange'));
    expect(screen.getByRole('img')).toBeInTheDocument();
  } finally {
    unmount();
    delete document.hidden;
    global.IntersectionObserver = original;
  }
  expect(disconnect).toHaveBeenCalled();
});

test.each([4, 6])('grid width changes rebalance %i real camera cells without recreating streams', async (count) => {
  const topics = Array.from({ length: count }, (_, idx) => `/camera-${idx}/image/compressed`);
  const getImageTopicList = jest.fn().mockResolvedValue({ success: true, image_topic_list: topics });
  useRosServiceCaller.mockReturnValue({ getImageTopicList });
  const store = configureStore({ reducer: { ros: rosReducer, tasks: () => ({ robotType: 'test' }) } });
  const view = render(<Provider store={store}><ImageGrid responsive /></Provider>);
  const images = await screen.findAllByRole('img');
  expect(images).toHaveLength(count);
  const sources = images.map((img) => img.src);
  const assignments = store.getState().ros.assignedImageTopics;
  const observer = observers.find((item) => item.target.classList.contains('image-grid-scroll'));
  expect(observer).toBeDefined();
  // Inspect the measured viewport's layout, which has no interactive semantics.
  // eslint-disable-next-line testing-library/no-node-access
  const layout = observer.target.querySelector('.image-grid-layout');
  for (const [width, columns] of [[916, count === 4 ? 2 : 3], [488, 2], [320, 1], [1480, count]]) {
    act(() => observer.callback([{ contentRect: { width } }]));
    expect(layout.style.gridTemplateColumns).toBe(`repeat(${columns}, minmax(0, 1fr))`);
    screen.getAllByRole('img').forEach((img, idx) => {
      expect(img).toBe(images[idx]);
      expect(img.src).toBe(sources[idx]);
    });
    expect(store.getState().ros.assignedImageTopics).toBe(assignments);
  }
  expect(getImageTopicList).toHaveBeenCalledTimes(1);
  fireEvent.keyDown(screen.getByRole('button', { name: 'Move camera 1' }), { key: 'ArrowRight' });
  expect(screen.getAllByRole('img')[1]).toBe(images[0]);
  expect(images[0].src).toBe(sources[0]);
  fireEvent.click(screen.getByRole('button', { name: 'Close camera 1' }));
  const remaining = screen.getAllByRole('img');
  expect(remaining).toHaveLength(count - 1);
  expect(remaining[0]).toBe(images[0]);
  expect(remaining[0].src).toBe(sources[0]);
  view.unmount();
  expect(observer.disconnect).toHaveBeenCalledTimes(1);
});
