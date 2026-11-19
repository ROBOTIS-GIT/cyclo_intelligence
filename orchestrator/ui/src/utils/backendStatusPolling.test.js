import { readBackendStatus, subscribeBackendStatus } from './backendStatusPolling';

const flush = async () => { for (let i = 0; i < 8; i += 1) await Promise.resolve(); };
const response = (data) => ({ ok: true, text: async () => JSON.stringify(data) });

beforeEach(() => { jest.useFakeTimers(); global.fetch = jest.fn(); });
afterEach(() => { jest.useRealTimers(); delete global.fetch; });

test('two controls share a poll, never overlap slow requests, and stop after the last subscriber leaves', async () => {
  let resolve;
  fetch.mockImplementation(() => new Promise((done) => { resolve = done; }));
  const first = jest.fn(), second = jest.fn();
  const off1 = subscribeBackendStatus('slow', first, 2000);
  const off2 = subscribeBackendStatus('slow', second, 5000);
  expect(fetch).toHaveBeenCalledTimes(1);
  jest.advanceTimersByTime(9000);
  expect(fetch).toHaveBeenCalledTimes(1);
  resolve(response({ state: 'ready' }));
  await flush();
  expect(first).toHaveBeenCalledWith({ state: 'ready' }, undefined);
  expect(second).toHaveBeenCalledTimes(1);
  off1();
  jest.advanceTimersByTime(4999);
  expect(fetch).toHaveBeenCalledTimes(1);
  jest.advanceTimersByTime(1);
  expect(fetch).toHaveBeenCalledTimes(2);
  off2();
  resolve(response({ state: 'late' }));
  await flush();
  jest.advanceTimersByTime(60000);
  expect(first).toHaveBeenCalledTimes(1);
  expect(second).toHaveBeenCalledTimes(1);
  expect(fetch).toHaveBeenCalledTimes(2);
});

test('backend identity is isolated and failed requests can be retried', async () => {
  fetch.mockResolvedValueOnce({ ok: false, status: 503, text: async () => '{"detail":"offline"}' })
    .mockResolvedValue(response({ state: 'ready' }));
  await expect(readBackendStatus('a')).rejects.toThrow('offline');
  await expect(readBackendStatus('a')).resolves.toEqual({ state: 'ready' });
  await expect(readBackendStatus('b')).resolves.toEqual({ state: 'ready' });
  expect(fetch.mock.calls.map(([url]) => url)).toEqual([
    '/api/backends/a/status', '/api/backends/a/status', '/api/backends/b/status',
  ]);
});

test('a hung request is aborted and does not overlap its replacement', async () => {
  fetch.mockImplementation((_url, { signal }) => new Promise((_resolve, reject) => {
    signal.addEventListener('abort', () => reject(new Error('aborted')));
  }));
  const listener = jest.fn();
  const off = subscribeBackendStatus('hung', listener, 2000);
  jest.advanceTimersByTime(10000);
  await flush();
  expect(listener.mock.calls[0][1].message).toBe('aborted');
  jest.advanceTimersByTime(2000);
  expect(fetch).toHaveBeenCalledTimes(2);
  off();
  jest.advanceTimersByTime(10000);
  await flush();
  expect(listener).toHaveBeenCalledTimes(1);
});

test('synchronous fetch failure releases the request and allows a retry', async () => {
  fetch.mockImplementationOnce(() => { throw new Error('fetch failed'); })
    .mockResolvedValue(response({ ready: true }));
  await expect(readBackendStatus('sync-failure')).rejects.toThrow('fetch failed');
  await expect(readBackendStatus('sync-failure')).resolves.toEqual({ ready: true });
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(jest.getTimerCount()).toBe(0);
});

test('100 mount/unmount cycles release every listener and timer', async () => {
  fetch.mockResolvedValue(response({ ready: true }));
  for (let i = 0; i < 100; i += 1) {
    const listener = jest.fn();
    const off = subscribeBackendStatus('cycles', listener, 2000);
    await flush();
    off();
    expect(listener).toHaveBeenCalledTimes(1);
    expect(jest.getTimerCount()).toBe(0);
  }
  expect(fetch).toHaveBeenCalledTimes(100);
});
