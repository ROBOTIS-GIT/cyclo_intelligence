import { useEffect, useRef, useCallback } from 'react';
import { useSelector } from 'react-redux';
import ROSLIB from 'roslib';

// Single unified joint_states topic — carries all upper-body joints
// (arm_l × 8, arm_r × 8, head × 2, lift × 1 = 19 joints) at 100 Hz.
// Phase 4 yaml's observation.state.upper_body.topic. The legacy
// physical_ai_tools four-topic split (per-arm-controller) caused each
// joint group to update at only ~7.5 Hz on the URDF (one global throttle
// shared across four 100 Hz streams), which is why the 3D viewer used
// to lag the real robot.
const JOINT_STATE_TOPIC = {
  name: '/joint_states',
  type: 'sensor_msgs/msg/JointState',
};

const ACTION_CHUNK_TOPIC = {
  name: '/inference/trajectory_preview',
  type: 'trajectory_msgs/msg/JointTrajectory',
};

const THROTTLE_MS = 33;

export default function useJointStateSubscription(setJointValues, setActionChunk, enabled = true) {
  const rosHost = useSelector((state) => state.ros.rosHost);
  const subscribersRef = useRef([]);
  const lastJointUpdateRef = useRef(0);

  const handleJointState = useCallback(
    (msg) => {
      const now = Date.now();
      if (now - lastJointUpdateRef.current < THROTTLE_MS) return;
      lastJointUpdateRef.current = now;

      if (msg.name && msg.position) {
        setJointValues({ name: msg.name, position: msg.position });
      }
    },
    [setJointValues]
  );

  const handleActionChunk = useCallback(
    (msg) => {
      if (!msg.joint_names || !msg.points || msg.points.length === 0) return;

      if (setActionChunk) {
        setActionChunk({
          names: msg.joint_names,
          points: msg.points.map((p) => p.positions),
        });
      }
    },
    [setActionChunk]
  );

  useEffect(() => {
    if (!enabled || !rosHost) return;

    const ros = new ROSLIB.Ros({ url: `ws://${rosHost}:9090` });
    const subs = [];

    ros.on('connection', () => {
      const jointSub = new ROSLIB.Topic({
        ros,
        name: JOINT_STATE_TOPIC.name,
        messageType: JOINT_STATE_TOPIC.type,
        throttle_rate: THROTTLE_MS,
      });
      jointSub.subscribe(handleJointState);
      subs.push(jointSub);

      const chunkSub = new ROSLIB.Topic({
        ros,
        name: ACTION_CHUNK_TOPIC.name,
        messageType: ACTION_CHUNK_TOPIC.type,
      });
      chunkSub.subscribe(handleActionChunk);
      subs.push(chunkSub);

      subscribersRef.current = subs;
    });

    ros.on('error', (err) => {
      console.error('Joint state ROS connection error:', err);
    });

    return () => {
      subs.forEach((sub) => {
        try {
          sub.unsubscribe();
        } catch (_e) { /* ignore */ }
      });
      subscribersRef.current = [];
      try {
        ros.close();
      } catch (_e) { /* ignore */ }
    };
  }, [enabled, rosHost, handleJointState, handleActionChunk]);
}
