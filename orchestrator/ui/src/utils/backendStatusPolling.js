// One in-flight request and polling timer per backend, shared by visible controls.
const backends = new Map();
const REQUEST_TIMEOUT_MS = 10000;

function entryFor(backend) {
  if (!backends.has(backend)) backends.set(backend, { listeners: new Map() });
  return backends.get(backend);
}

function schedule(backend, entry) {
  clearTimeout(entry.timer);
  if (!entry.listeners.size) {
    if (!entry.pending && backends.get(backend) === entry) backends.delete(backend);
    return;
  }
  const interval = Math.min(...entry.listeners.values());
  entry.timer = setTimeout(() => { readBackendStatus(backend).catch(() => {}); }, interval);
}

export function readBackendStatus(backend) {
  const entry = entryFor(backend);
  if (entry.pending) return entry.pending;
  clearTimeout(entry.timer);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let data, error;
  entry.pending = (async () => {
    try {
      const response = await fetch(`/api/backends/${backend}/status`, { signal: controller.signal });
      const text = await response.text();
      try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text }; }
      if (!response.ok) throw new Error(data.detail || `status failed (${response.status})`);
      return data;
    } catch (err) {
      error = err;
      throw err;
    }
  })().finally(() => {
    clearTimeout(timeout);
    entry.pending = null;
    entry.listeners.forEach((_, listener) => listener(data, error));
    schedule(backend, entry);
  });
  return entry.pending;
}

export function subscribeBackendStatus(backend, listener, intervalMs) {
  const entry = entryFor(backend);
  entry.listeners.set(listener, intervalMs);
  readBackendStatus(backend).catch(() => {});
  return () => {
    entry.listeners.delete(listener);
    schedule(backend, entry);
  };
}
