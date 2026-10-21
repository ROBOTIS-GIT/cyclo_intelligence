// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import ROSLIB from 'roslib';
import { connectedTopic } from '../features/missionCanvas/lib/jointTeleop';
import { trajectoryPublisher } from './gripperControl';

const armJoints = (side) => [...Array.from({ length: 7 }, (_, i) => `arm_${side}_joint${i + 1}`), `gripper_${side}_joint1`];
export const POSE_GROUPS = {
  left: { label: 'Left arm', joints: armJoints('l'), tolerance: .02, speed: .35,
    routes: [{ controller: 'arm_l_controller', topic: '/arm_l_controller/joint_trajectory', start: 0, end: 7 },
      { controller: 'gripper_l_controller', topic: '/gripper_l_controller/joint_trajectory', start: 7, end: 8 }],
    observed: ['/leader/joint_trajectory_command_broadcaster_left/joint_trajectory'],
    poses: [
      [0.2127, 0.52, 0.0685, -1.792, 0.1741, 0, -1.3972, 0],
      [-0.6273861580203218, 0.5307453683835265, -0.6734175658817965, -0.7148230629298035, 1.3759567982836094, 1.267823136962577, -1.5642494028451934, 0],
      [0.539937268885935, 0, -0.05366535912618673, -2.5617359315441153, 0.023009711818284616, 0.4601702679158816, 0.057463995994551084, 0],
      [-0.644259946687064, 0.3558715585645634, -0.7086991240031663, -0.7685123905058009, 0.7040732131896986, -0.3750463184131339, -1.3299253759818837, 0],
    ] },
  right: { label: 'Right arm', joints: armJoints('r'), tolerance: .02, speed: .35,
    routes: [{ controller: 'arm_r_controller', topic: '/arm_r_controller/joint_trajectory', start: 0, end: 7 },
      { controller: 'gripper_r_controller', topic: '/gripper_r_controller/joint_trajectory', start: 7, end: 8 }],
    observed: ['/leader/joint_trajectory_command_broadcaster_right/joint_trajectory'],
    poses: [
      [0.2127, -0.52, -0.0685, -1.792, -0.1741, 0, 1.3972, 0],
      [0.539937268885935, 0, 0.05366535912618673, -2.5617359315441153, -0.023009711818284616, 0.4601702679158816, -0.057463995994551084, 0],
    ] },
  head: { label: 'Head', joints: ['head_joint1', 'head_joint2'], tolerance: .01, speed: .15,
    routes: [{ controller: 'head_controller', topic: '/leader/joystick_controller_left/joint_trajectory', start: 0, end: 2 }],
    observed: ['/leader/mission_canvas_head/joint_trajectory'], poses: [[0.83, 0], [0.7286408742454724, 0]] },
  lift: { label: 'Lift', joints: ['lift_joint'], tolerance: .003, speed: .04,
    routes: [{ controller: 'lift_controller', topic: '/leader/joystick_controller_right/joint_trajectory', start: 0, end: 1 }],
    observed: [], poses: [[-0.12159]] },
};

export function groupPositions(message, group) {
  const joints = POSE_GROUPS[group]?.joints;
  if (!joints || !Array.isArray(message?.name) || !Array.isArray(message?.position)) return null;
  const values = joints.map((name) => {
    const index = message.name.indexOf(name);
    return index >= 0 && message.name.lastIndexOf(name) === index ? message.position[index] : NaN;
  });
  return values.every(Number.isFinite) ? values : null;
}

export function parseJointLimits(xml) {
  const document = new DOMParser().parseFromString(xml || '', 'application/xml');
  if (document.querySelector('parsererror') || !document.querySelector('robot')) throw new Error('Invalid robot description');
  const limits = {};
  document.querySelectorAll('robot > joint').forEach((joint) => {
    const limit = joint.querySelector('limit');
    if (!limit?.hasAttribute('lower') || !limit?.hasAttribute('upper')) return;
    const lower = Number(limit.getAttribute('lower'));
    const upper = Number(limit.getAttribute('upper'));
    if (Number.isFinite(lower) && Number.isFinite(upper) && lower < upper) limits[joint.getAttribute('name')] = [lower, upper];
  });
  return limits;
}

export function poseDuration(group, current, target, hold = false) {
  if (hold) return 1;
  const config = POSE_GROUPS[group];
  // Zero endpoint velocities give cubic interpolation; its peak is 1.5 * delta / T.
  const delta = Math.max(...current.map((value, i) => Math.abs(target[i] - value)));
  return Math.ceil(Math.max(3, 1.5 * delta / config.speed) * 10) / 10;
}

export function poseTrajectories(group, current, target, duration, owner) {
  const config = POSE_GROUPS[group];
  if (!config || current.length !== config.joints.length || target.length !== config.joints.length ||
      ![...current, ...target, duration].every(Number.isFinite) || duration <= 0) throw new Error('Invalid pose');
  const milliseconds = Math.round(duration * 1000);
  return config.routes.map(({ topic, start, end }) => ({ topic, message: {
    header: { stamp: { sec: 0, nanosec: 0 }, frame_id: owner },
    joint_names: config.joints.slice(start, end),
    points: [
      { positions: current.slice(start, end), velocities: Array(end - start).fill(0), time_from_start: { sec: 0, nanosec: 0 } },
      { positions: target.slice(start, end), velocities: Array(end - start).fill(0),
        time_from_start: { sec: Math.floor(milliseconds / 1000), nanosec: (milliseconds % 1000) * 1000000 } },
    ],
  } }));
}

// Sends finite trajectories on a click. Controllers retain the final target;
// the UI never streams a held pose over a subsequent inference command.
export class InferencePoseClient {
  constructor(ros, now = () => performance.now()) {
    this.ros = ros; this.now = now; this.enabled = false; this.closed = false;
    this.positions = {}; this.stateAt = -Infinity; this.controllers = {}; this.controllerAt = -Infinity;
    this.limits = {}; this.otherCommandAt = {}; this.holding = {}; this.failures = {};
    this.lastPoll = -Infinity; this.pollId = 0; this.pending = {}; this.moving = null;
    this.owner = `inference_pose_${Math.random().toString(36).slice(2)}`;
    this.publishers = {}; this.subscriptions = [];
    const subscribe = (name, type, callback) => {
      const topic = connectedTopic({ ros, name, messageType: type, queue_length: 1, throttle_rate: 50 });
      topic.subscribe(callback); this.subscriptions.push(topic);
    };
    subscribe('/joint_states', 'sensor_msgs/msg/JointState', (message) => {
      if (this.closed) return;
      this.positions = Object.fromEntries(Object.keys(POSE_GROUPS).map((group) => [group, groupPositions(message, group)]));
      this.stateAt = this.now();
      Object.entries(this.holding).forEach(([group, target]) => {
        if (!this.positions[group] || !this.atTarget(group, target)) delete this.holding[group];
      });
    });
    Object.entries(POSE_GROUPS).forEach(([group, config]) => {
      config.routes.forEach(({ topic }) => { this.publishers[topic] = trajectoryPublisher(ros, topic); });
      [...config.routes.map((route) => route.topic), ...config.observed].forEach((topic) => {
        subscribe(topic, 'trajectory_msgs/msg/JointTrajectory', (message) => {
          if (this.closed || message.header?.frame_id === this.owner) return;
          this.otherCommandAt[group] = this.now(); delete this.holding[group];
          if (this.moving?.group === group) this.failures[group] = 'Another control source replaced the pose command';
        });
      });
    });
    this.controllerService = new ROSLIB.Service({ ros, name: '/controller_manager/list_controllers',
      serviceType: 'controller_manager_msgs/srv/ListControllers' });
    this.descriptionService = new ROSLIB.Service({ ros, name: '/robot_state_publisher/get_parameters',
      serviceType: 'rcl_interfaces/srv/GetParameters' });
    this.onClose = () => {
      this.enabled = false; this.stateAt = -Infinity; this.controllerAt = -Infinity; this.limits = {}; this.holding = {};
      ++this.pollId; this.lastPoll = -Infinity;
      Object.keys(this.pending).forEach((key) => this.clearRequest(key));
      Object.values(this.publishers).forEach((publisher) => { publisher.isAdvertised = false; });
    };
    ros.on('close', this.onClose);
  }

  setEnabled(enabled) { this.enabled = enabled; }
  isBusy() { return !!this.moving; }
  atTarget(group, target) {
    return this.positions[group]?.every((value, i) => Math.abs(value - target[i]) <= POSE_GROUPS[group].tolerance);
  }
  blocker(group) {
    if (this.closed || !this.ros.isConnected) return 'ROS disconnected';
    if (!this.enabled) return 'Clear inference before preparing poses';
    if (this.now() - (this.otherCommandAt[group] ?? -Infinity) < 1000) return 'Stop the other robot control source first';
    if (!this.positions[group] || this.now() - this.stateAt > 750) return 'Waiting for fresh joint feedback';
    const config = POSE_GROUPS[group];
    if (!config?.joints.every((joint) => this.limits[joint])) return 'Waiting for live robot joint limits';
    if (this.positions[group].some((value, i) => value < this.limits[config.joints[i]][0] - .01 || value > this.limits[config.joints[i]][1] + .01)) return 'Measured pose is outside robot limits';
    if (this.now() - this.controllerAt > 2000) return 'Waiting for controller status';
    const inactive = config.routes.find((route) => this.controllers[route.controller] !== 'active');
    return inactive ? `${inactive.controller} is ${this.controllers[inactive.controller] || 'unavailable'}` : '';
  }
  snapshot() {
    return { positions: { ...this.positions }, reasons: Object.fromEntries(Object.keys(POSE_GROUPS).map((group) => [group, this.blocker(group)])),
      failures: { ...this.failures }, holding: { ...this.holding }, busy: this.isBusy(), moving: this.moving };
  }
  clearRequest(key) {
    if (this.pending[key]) this.ros.removeAllListeners(this.pending[key]);
    delete this.pending[key];
  }
  request(key, service, args, generation, at, callback) {
    this.clearRequest(key);
    this.pending[key] = `call_service:${service.name}:${this.ros.idCounter + 1}`;
    const valid = () => !this.closed && this.ros.isConnected && generation === this.pollId && this.now() - at < 1000;
    try {
      service.callService(new ROSLIB.ServiceRequest(args), (response) => {
        if (!valid()) return;
        delete this.pending[key]; callback(response);
      }, () => { if (valid()) delete this.pending[key]; });
    } catch (_error) { this.clearRequest(key); }
  }
  tick() {
    if (this.closed) return;
    const at = this.now();
    if (this.moving && at >= this.moving.until) {
      const { group, target } = this.moving;
      const fresh = this.ros.isConnected && at - this.stateAt < 750;
      if (fresh && this.atTarget(group, target) && !this.failures[group]) {
        this.holding[group] = target; this.finishMove();
      } else if (at >= this.moving.until + 5000 || this.failures[group]) {
        this.failures[group] = this.failures[group] || 'Target not confirmed; check feedback before retrying';
        this.finishMove();
      }
    }
    if (!this.ros.isConnected || at - this.lastPoll < 1000) return;
    this.lastPoll = at;
    const generation = ++this.pollId;
    this.request('controllers', this.controllerService, {}, generation, at, (response) => {
      this.controllers = Object.fromEntries((response.controller || []).map((controller) => [controller.name, controller.state]));
      this.controllerAt = this.now();
    });
    if (!Object.keys(this.limits).length) this.request('limits', this.descriptionService, { names: ['robot_description'] }, generation, at, (response) => {
      try { this.limits = parseJointLimits(response.values?.[0]?.string_value); } catch (_error) { this.limits = {}; }
    });
  }
  command(group, poseIndex, hold = false) {
    const config = POSE_GROUPS[group];
    if (!config || this.blocker(group) || (this.moving && (!hold || this.moving.group !== group))) return false;
    const current = this.positions[group];
    const target = hold ? [...current] : config.poses[poseIndex];
    if (!target || target.some((value, i) => !Number.isFinite(value) || value < this.limits[config.joints[i]][0] || value > this.limits[config.joints[i]][1])) {
      this.failures[group] = 'Requested pose exceeds the live robot limits'; return false;
    }
    const duration = poseDuration(group, current, target, hold);
    const trajectories = poseTrajectories(group, current, target, duration, this.owner);
    delete this.failures[group]; delete this.holding[group];
    this.moving = { group, poseIndex, target: [...target], hold, duration, until: this.now() + duration * 1000 };
    try {
      trajectories.forEach(({ topic, message }) => this.publishers[topic].publish(new ROSLIB.Message(message)));
      return true;
    } catch (_error) {
      this.failures[group] = 'Pose command failed; verify the robot before retrying';
      return false;
    }
  }
  finishMove() { this.moving = null; Object.values(this.publishers).forEach((publisher) => publisher.unadvertise()); }
  dispose() {
    this.closed = true; this.enabled = false;
    Object.keys(this.pending).forEach((key) => this.clearRequest(key));
    Object.values(this.publishers).forEach((publisher) => publisher.unadvertise());
    this.ros.off('close', this.onClose);
    this.subscriptions.forEach((topic) => {
      if (this.ros.isConnected) topic.unsubscribe();
      else { this.ros.off(topic.name, topic._messageCallback); topic.removeAllListeners(); }
    });
  }
}
