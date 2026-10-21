// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import ROSLIB from 'roslib';

export const GRIPPER_OPEN = 0.0;
export const GRIPPER_CLOSED = 1.1;
export const GRIPPER_MOVE_MS = 2000;
export const GRIPPERS = {
  right: { joint: 'gripper_r_joint1', controller: 'gripper_r_controller',
    topic: '/gripper_r_controller/joint_trajectory',
    leaderTopic: '/leader/joint_trajectory_command_broadcaster_right/joint_trajectory' },
  left: { joint: 'gripper_l_joint1', controller: 'gripper_l_controller',
    topic: '/gripper_l_controller/joint_trajectory',
    leaderTopic: '/leader/joint_trajectory_command_broadcaster_left/joint_trajectory' },
};

function connectedTopic(options) {
  const topic = new ROSLIB.Topic({ ...options, reconnect_on_close: false });
  // roslib 1.4.1 otherwise loses the Ros receiver with reconnect disabled.
  topic.callForSubscribeAndAdvertise = (message) => {
    if (options.ros.isConnected) options.ros.callOnConnection(message);
  };
  return topic;
}

export function trajectoryPublisher(ros, topic) {
  // roslib's Topic.advertise adds a new anonymous close listener on every
  // button press after unadvertise. Keep this one-shot publisher connection
  // bound instead, without reconnect listeners or deferred commands.
  return {
    isAdvertised: false,
    publish(message) {
      if (!ros.isConnected) throw new Error('ROS disconnected');
      if (!this.isAdvertised) {
        this.advertiseId = `advertise:${topic}:${++ros.idCounter}`;
        ros.callOnConnection({ op: 'advertise', id: this.advertiseId, topic,
          type: 'trajectory_msgs/msg/JointTrajectory', latch: false, queue_size: 1 });
        this.isAdvertised = true;
      }
      ros.callOnConnection({ op: 'publish', id: `publish:${topic}:${++ros.idCounter}`,
        topic, msg: message, latch: false });
    },
    unadvertise() {
      if (this.isAdvertised && ros.isConnected) {
        ros.callOnConnection({ op: 'unadvertise', id: this.advertiseId, topic });
      }
      this.isAdvertised = false;
    },
  };
}

export function gripperPosition(message, side) {
  const joint = GRIPPERS[side]?.joint;
  const index = message?.name?.indexOf(joint) ?? -1;
  const value = index >= 0 ? message.position?.[index] : null;
  return Number.isFinite(value) ? value : null;
}

export function gripperTrajectory(side, target, owner = '') {
  if (!GRIPPERS[side] || ![GRIPPER_OPEN, GRIPPER_CLOSED].includes(target)) {
    throw new Error('Unsupported gripper command');
  }
  return {
    header: { stamp: { sec: 0, nanosec: 0 }, frame_id: owner },
    joint_names: [GRIPPERS[side].joint],
    points: [{ positions: [target], velocities: [0],
      time_from_start: { sec: GRIPPER_MOVE_MS / 1000, nanosec: 0 } }],
  };
}

// One button press sends one finite follower-gripper trajectory. This client
// never publishes arm trajectories, repeatedly streams a target, activates a
// controller, or queues commands for a disconnected/reconnected robot.
export class GripperControlClient {
  constructor(ros, now = () => performance.now()) {
    this.ros = ros;
    this.now = now;
    this.closed = false;
    this.enabled = false;
    this.positions = { left: null, right: null };
    this.stateAt = -Infinity;
    this.controllers = {};
    this.controllerAt = -Infinity;
    this.otherCommandAt = -Infinity;
    this.lastPoll = -Infinity;
    this.pollId = 0;
    this.moving = null;
    this.failure = '';
    this.owner = `inference_gripper_${Math.random().toString(36).slice(2)}`;
    this.publishers = {};
    this.subscriptions = [];
    const subscribe = (name, type, callback) => {
      const topic = connectedTopic({ ros, name, messageType: type,
        throttle_rate: 50, queue_length: 1 });
      topic.subscribe(callback);
      this.subscriptions.push(topic);
    };
    subscribe('/joint_states', 'sensor_msgs/msg/JointState', (message) => {
      if (this.closed) return;
      this.positions = { left: gripperPosition(message, 'left'), right: gripperPosition(message, 'right') };
      this.stateAt = this.now();
    });
    Object.entries(GRIPPERS).forEach(([side, config]) => {
      this.publishers[side] = trajectoryPublisher(ros, config.topic);
      const observeCommand = (message) => {
        if (!this.closed && message.header?.frame_id !== this.owner) this.otherCommandAt = this.now();
      };
      subscribe(config.topic, 'trajectory_msgs/msg/JointTrajectory', observeCommand);
      subscribe(config.leaderTopic, 'trajectory_msgs/msg/JointTrajectory', observeCommand);
    });
    this.service = new ROSLIB.Service({ ros, name: '/controller_manager/list_controllers',
      serviceType: 'controller_manager_msgs/srv/ListControllers' });
    this.onClose = () => {
      this.enabled = false;
      this.controllerAt = -Infinity;
      this.stateAt = -Infinity;
      Object.values(this.publishers).forEach((publisher) => { publisher.isAdvertised = false; });
    };
    ros.on('close', this.onClose);
  }

  setEnabled(enabled) { this.enabled = enabled; }

  isBusy() { return !!this.moving && this.now() < this.moving.until; }

  blocker(side) {
    if (this.closed || !this.ros.isConnected) return 'ROS disconnected';
    if (!this.enabled) return 'Clear inference before using manual gripper controls';
    if (this.failure) return this.failure;
    if (this.now() - this.otherCommandAt < 1000) return 'Another robot control source is active';
    if (!Number.isFinite(this.positions[side]) || this.now() - this.stateAt > 750) return 'Waiting for fresh gripper feedback';
    if (this.positions[side] < -0.01 || this.positions[side] > GRIPPER_CLOSED + 0.01) return 'Gripper feedback is outside its range';
    if (this.now() - this.controllerAt > 2000) return 'Waiting for gripper controller status';
    const controller = this.controllers[GRIPPERS[side]?.controller];
    if (controller !== 'active') return `Gripper controller ${controller || 'unavailable'}`;
    return '';
  }

  snapshot() {
    return { positions: { ...this.positions },
      reasons: { left: this.blocker('left'), right: this.blocker('right') },
      busy: this.isBusy(), moving: this.isBusy() ? this.moving : null };
  }

  clearControllerRequest() {
    if (this.pendingServiceId) this.ros.removeAllListeners(this.pendingServiceId);
    this.pendingServiceId = null;
  }

  tick() {
    if (this.closed) return;
    if (this.moving && !this.isBusy()) {
      this.moving = null;
      this.releasePublishers();
    }
    const at = this.now();
    if (!this.ros.isConnected || at - this.lastPoll < 1000) return;
    this.lastPoll = at;
    this.clearControllerRequest();
    this.pendingServiceId = `call_service:${this.service.name}:${this.ros.idCounter + 1}`;
    const id = ++this.pollId;
    const valid = () => !this.closed && this.ros.isConnected && id === this.pollId && this.now() - at < 1000;
    try {
      this.service.callService(new ROSLIB.ServiceRequest({}), (response) => {
        if (!valid()) return;
        this.pendingServiceId = null;
        this.controllers = Object.fromEntries((response.controller || []).map((item) => [item.name, item.state]));
        this.controllerAt = this.now();
      }, () => {
        if (!valid()) return;
        this.pendingServiceId = null;
        this.controllers = {};
        this.controllerAt = this.now();
      });
    } catch (_error) {
      this.controllers = {};
      this.controllerAt = this.now();
    }
  }

  command(side, target) {
    if (!GRIPPERS[side] || this.isBusy() || this.blocker(side)) return false;
    try {
      const message = gripperTrajectory(side, target, this.owner);
      // No await between the connectivity/ownership check and publishing.
      this.publishers[side].publish(new ROSLIB.Message(message));
      this.moving = { side, target, until: this.now() + GRIPPER_MOVE_MS };
      return true;
    } catch (_error) {
      this.failure = 'Gripper command failed; reconnect before retrying';
      return false;
    }
  }

  releasePublishers() {
    Object.values(this.publishers).forEach((publisher) => publisher.unadvertise());
  }

  dispose() {
    this.closed = true;
    this.enabled = false;
    this.clearControllerRequest();
    this.releasePublishers();
    this.ros.off('close', this.onClose);
    this.subscriptions.forEach((topic) => {
      // Topic.unsubscribe uses callOnConnection even with reconnect disabled.
      // Remove local listeners only when disconnected to avoid deferred work.
      if (this.ros.isConnected) topic.unsubscribe();
      else { this.ros.off(topic.name, topic._messageCallback); topic.removeAllListeners(); }
    });
  }
}
