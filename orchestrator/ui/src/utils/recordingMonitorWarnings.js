const STATUS_STALLED = 2;
export const STATUS_IDLE = -1;
const ARBITER_FRESH_MS = 2500;
const ARM_COMMAND_TOPICS = new Set([
  '/command_source_arbiter/executed/arm_left',
  '/command_source_arbiter/executed/arm_right',
]);
const AUXILIARY_COMMAND_TOPICS = new Set([
  '/leader/joystick_controller_left/joint_trajectory',
  '/leader/joystick_controller_right/joint_trajectory',
  '/cmd_vel',
]);

// Silence is expected only when a fresh, fault-free control state confirms it.
// Unknown/stale control state and continuous sensor streams retain raw warnings.
export const classifyRecordingTopic = (topic, monitor, nowMs = Date.now()) => {
  const arbiter = monitor?.arbiter;
  const ageMs = nowMs - Number(arbiter?.receivedAtMs);
  const healthy = arbiter && Number.isFinite(ageMs) && ageMs >= 0
    && ageMs <= ARBITER_FRESH_MS && !arbiter.fault && !arbiter.publisher_conflict;
  const quiet = Number(topic.status) >= 1 && Number(topic.rateHz || 0) === 0;
  const expectedSilence = healthy && (
    (arbiter.mode === 'NONE' && (ARM_COMMAND_TOPICS.has(topic.name)
      || AUXILIARY_COMMAND_TOPICS.has(topic.name)))
    || (arbiter.mode === 'WAYPOINT' && AUXILIARY_COMMAND_TOPICS.has(topic.name)
      && Number(topic.secondsSinceLast) < 0)
  );
  if (quiet && expectedSilence && topic.source !== 'camera') {
    return { ...topic, status: STATUS_IDLE, reportedStatus: topic.status };
  }
  return topic;
};

const formatLastSeen = (secondsSinceLast) => {
  const value = Number(secondsSinceLast);
  if (!Number.isFinite(value) || value < 0) {
    return 'no messages yet';
  }
  return `last ${value.toFixed(1)}s ago`;
};

export const buildRosbagStartWarningMessage = (recordingMonitor, nowMs = Date.now()) => {
  const stalledTopics = (recordingMonitor?.topics || [])
    .map((topic) => classifyRecordingTopic(topic, recordingMonitor, nowMs))
    .filter((topic) => Number(topic.status) >= STATUS_STALLED);

  if (stalledTopics.length === 0) {
    return '';
  }

  const summary = stalledTopics
    .slice(0, 3)
    .map((topic) => `${topic.name}: ${formatLastSeen(topic.secondsSinceLast)}`)
    .join('\n');
  const moreText = stalledTopics.length > 3
    ? `\n+${stalledTopics.length - 3} more`
    : '';

  return `Rosbag topic warning\n${summary}${moreText}\nRecording will still start.`;
};
