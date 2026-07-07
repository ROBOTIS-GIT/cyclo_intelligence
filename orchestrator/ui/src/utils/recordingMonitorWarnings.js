const STATUS_STALLED = 2;

const formatLastSeen = (secondsSinceLast) => {
  const value = Number(secondsSinceLast);
  if (!Number.isFinite(value) || value < 0) {
    return 'no messages yet';
  }
  return `last ${value.toFixed(1)}s ago`;
};

export const buildRosbagStartWarningMessage = (recordingMonitor) => {
  const stalledTopics = (recordingMonitor?.topics || [])
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
