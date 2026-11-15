const API_BASE = '/api/intervention';

async function readPayload(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    return { detail: text };
  }
}

function errorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === 'string') return detail;
  if (detail?.message) return detail.message;
  if (Array.isArray(detail?.failures)) return detail.failures.join(' · ');
  return payload?.message || fallback;
}

export async function getInterventionState() {
  const response = await fetch(`${API_BASE}/state`, { cache: 'no-store' });
  const payload = await readPayload(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, 'Intervention state unavailable'));
  }
  return payload;
}

export async function sendInterventionAction(action, body = {}) {
  const response = await fetch(`${API_BASE}/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await readPayload(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, `Intervention ${action} failed`));
  }
  return payload;
}

export async function sendOperatorPose(body) {
  const response = await fetch(`${API_BASE}/pose`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await readPayload(response);
  if (!response.ok) {
    throw new Error(errorMessage(payload, 'Initial pose movement failed'));
  }
  return payload;
}
