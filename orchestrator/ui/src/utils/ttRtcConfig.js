// Translate saved Canvas nodes to the prefix-conditioned TT-RTC request mode.
// Prefix length is selected by the runtime from the trained model contract.
export function normalizeLegacyTtRtcParams(params = {}) {
  const { ttrtc_enabled, ttrtc_prefix_steps, ...next } = params;
  if ((ttrtc_enabled === true || ttrtc_enabled === 'true')
      && String(next.model || '').startsWith('groot')
      && String(next.action_request_mode || 'async') === 'async') {
    next.action_request_mode = 'tt_rtc';
    next.inference_hz = '15';
    next.control_hz = next.control_hz || '100';
    next.vla_rail_enabled = 'false';
  }
  return next;
}
