import { useEffect, useMemo, useRef, useCallback } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import rosConnectionManager from '../utils/rosConnectionManager';

const ACTION_CHUNK_TOPIC = {
  name: '/inference/trajectory_preview',
  type: 'trajectory_msgs/msg/JointTrajectory',
};

const DEFAULT_LIVE_UPDATE_HZ = 15;
const MIN_THROTTLE_MS = 1;
const TOPIC_QUEUE_LENGTH = 1;

function normalizeTopics(topics, defaultType) {
  if (!Array.isArray(topics)) return [];
  return topics
    .map((topic) => {
      if (typeof topic === 'string') {
        return { name: topic, type: defaultType };
      }
      return {
        name: topic?.name || '',
        type: topic?.type || defaultType,
      };
    })
    .filter((topic) => topic.name && topic.type);
}

export default function useJointStateSubscription(
  setJointValues,
  setActionChunk,
  enabled = true,
  options = {},
) {
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const subscribersRef = useRef([]);
  const lastJointUpdateRef = useRef(0);
  const lastActionUpdateRef = useRef(0);
  const visualizationSource = options.visualizationSource || 'state';
  const liveUpdateHz =
    typeof options.liveUpdateHz === 'number' && options.liveUpdateHz > 0
      ? options.liveUpdateHz
      : DEFAULT_LIVE_UPDATE_HZ;
  const throttleMs = Math.max(MIN_THROTTLE_MS, Math.round(1000 / liveUpdateHz));
  const actionSubscriptionsEnabled =
    Boolean(options.enableActionPreview) && visualizationSource === 'action';
  const stateTopics = useMemo(
    () => normalizeTopics(
      options.stateTopics,
      'sensor_msgs/msg/JointState'
    ),
    [options.stateTopics]
  );
  const actionCommandTopics = useMemo(
    () => normalizeTopics(
      options.actionTopics,
      'trajectory_msgs/msg/JointTrajectory'
    ),
    [options.actionTopics]
  );

  const handleJointState = useCallback(
    (msg) => {
      const now = Date.now();
      if (now - lastJointUpdateRef.current < throttleMs) return;
      lastJointUpdateRef.current = now;

      if (msg.name && msg.position) {
        setJointValues({ name: msg.name, position: msg.position });
      }
    },
    [setJointValues, throttleMs]
  );

  const handleActionChunk = useCallback(
    (msg) => {
      if (!msg.joint_names || !msg.points || msg.points.length === 0) return;

      const firstPoint = msg.points[0];
      const positions = firstPoint?.positions;
      if (visualizationSource === 'action' && positions && positions.length > 0) {
        const now = Date.now();
        if (now - lastActionUpdateRef.current >= throttleMs) {
          lastActionUpdateRef.current = now;
          setJointValues({ name: msg.joint_names, position: positions });
        }
      }

      if (setActionChunk) {
        setActionChunk({
          names: msg.joint_names,
          points: msg.points.map((p) => p.positions),
        });
      }
    },
    [setJointValues, setActionChunk, visualizationSource, throttleMs]
  );

  const handleActionCommand = useCallback(
    (msg) => {
      if (!msg.joint_names || !msg.points || msg.points.length === 0) return;

      const firstPoint = msg.points[0];
      const positions = firstPoint?.positions;
      if (!positions || positions.length === 0) return;

      if (visualizationSource === 'action') {
        const now = Date.now();
        if (now - lastActionUpdateRef.current >= throttleMs) {
          lastActionUpdateRef.current = now;
          setJointValues({ name: msg.joint_names, position: positions });
        }
      }

      if (setActionChunk) {
        setActionChunk({
          names: msg.joint_names,
          points: msg.points.map((p) => p.positions),
        });
      }
    },
    [setJointValues, setActionChunk, visualizationSource, throttleMs]
  );

  useEffect(() => {
    if (!enabled || !rosbridgeUrl) return;

    let cancelled = false;
    const subs = [];

    const unsubscribeAll = () => {
      subs.forEach((sub) => {
        try {
          sub.unsubscribe();
        } catch (_e) { /* ignore */ }
      });
      subscribersRef.current = [];
    };

    const subscribeTopic = (ros, topic, callback) => {
      const sub = new ROSLIB.Topic({
        ros,
        name: topic.name,
        messageType: topic.type,
        throttle_rate: throttleMs,
        queue_length: TOPIC_QUEUE_LENGTH,
      });
      sub.subscribe(callback);
      subs.push(sub);
    };

    const setupSubscriptions = async () => {
      try {
        const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
        if (cancelled || !ros) return;

        if (!actionSubscriptionsEnabled) {
          stateTopics.forEach((topic) => {
            subscribeTopic(ros, topic, handleJointState);
          });
        }

        if (actionSubscriptionsEnabled) {
          subscribeTopic(ros, ACTION_CHUNK_TOPIC, handleActionChunk);

          actionCommandTopics.forEach((topic) => {
            subscribeTopic(ros, topic, handleActionCommand);
          });
        }

        if (cancelled) {
          unsubscribeAll();
          return;
        }

        subscribersRef.current = subs;
      } catch (err) {
        if (!cancelled) {
          console.error('Joint state ROS connection error:', err);
        }
      }
    };

    setupSubscriptions();

    return () => {
      cancelled = true;
      unsubscribeAll();
    };
  }, [
    enabled,
    rosbridgeUrl,
    handleJointState,
    handleActionChunk,
    handleActionCommand,
    visualizationSource,
    actionSubscriptionsEnabled,
    stateTopics,
    actionCommandTopics,
    throttleMs,
  ]);
}
