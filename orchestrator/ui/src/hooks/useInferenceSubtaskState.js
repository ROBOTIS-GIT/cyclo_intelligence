import { useEffect, useRef } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';

// Reconcile manual/automatic transitions and browser remounts with the runtime.
export function useInferenceSubtaskState({ enabled, modelPath, instruction, stages, onInstruction }) {
  const rosbridgeUrl = useSelector((state) => state.ros?.rosbridgeUrl);
  const latest = useRef();
  latest.current = { instruction, stages, onInstruction };
  useEffect(() => {
    if (!enabled || !rosbridgeUrl) return undefined;
    let active = true, topic;
    let lastTimestamp = 0;
    rosConnectionManager.getConnection(rosbridgeUrl).then((ros) => {
      if (!active || !ros) return;
      topic = new ROSLIB.Topic({ ros, name: '/groot/subtask_state',
        messageType: 'std_msgs/msg/String', queue_length: 1, throttle_rate: 500 });
      topic.subscribe((message) => {
        if (!active) return;
        try {
          const state = JSON.parse(message.data);
          const stamp = Number(state.timestamp) * 1000;
          const currentIndex = latest.current.stages?.findIndex((stage) => stage.includes(latest.current.instruction)) ?? -1;
          if (!state.available || state.model_path !== modelPath || !Number.isInteger(state.subtask_index)
            || state.subtask_index < 0 || !latest.current.stages?.[state.subtask_index]?.includes(state.instruction)
            || state.subtask_index < currentIndex
            || !Number.isFinite(stamp) || Math.abs(Date.now() - stamp) > 3000
            || stamp <= lastTimestamp || typeof state.instruction !== 'string') return;
          lastTimestamp = stamp;
          if (state.instruction !== latest.current.instruction) {
            latest.current.onInstruction(state.instruction);
          }
        } catch (_) { /* Ignore malformed or stale telemetry. */ }
      });
    }).catch(() => {});
    return () => { active = false; topic?.unsubscribe(); };
  }, [enabled, modelPath, rosbridgeUrl]);
}
