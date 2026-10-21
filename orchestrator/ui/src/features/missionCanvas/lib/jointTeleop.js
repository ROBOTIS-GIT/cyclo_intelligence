// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import ROSLIB from "roslib";

const STATE_MAX_AGE_MS = 750;
const CONTROLLER_MAX_AGE_MS = 2000;

export function connectedTopic(options) {
  const topic = new ROSLIB.Topic({ ...options, reconnect_on_close: false });
  // roslib 1.4.1 leaves callOnConnection unbound when reconnect_on_close=false.
  // Keep the Ros receiver and refuse deferred subscribe/advertise operations.
  topic.callForSubscribeAndAdvertise = (message) => {
    if (options.ros.isConnected) options.ros.callOnConnection(message);
  };
  return topic;
}

// Owns only subscriptions/publishers; never closes the shared rosbridge connection.
// A press sends finite, short trajectories based on measured positions. Commands
// are never queued waiting for a connection and do not resume after reconnect.
export class JointTeleopClient {
  constructor(ros, config, now = () => performance.now()) {
    this.config = config;
    this.ros = ros;
    this.now = now;
    this.positions = null;
    this.stateAt = -Infinity;
    this.controllerAt = -Infinity;
    this.controllerState = "unknown";
    this.lastPoll = -Infinity;
    this.pollId = 0;
    this.active = false;
    this.direction = null;
    this.closed = false;
    this.failure = "";
    this.otherCommandAt = -Infinity;
    this.knownCommandAt = -Infinity;
    this.owner = `mission_canvas_${config.id}_${Math.random().toString(36).slice(2)}`;
    this.publisher = connectedTopic({
      ros, name: config.topic, messageType: "trajectory_msgs/msg/JointTrajectory", reconnect_on_close: false,
    });
    this.subscription = connectedTopic({
      ros, name: "/joint_states", messageType: "sensor_msgs/msg/JointState",
      throttle_rate: 50, queue_length: 1, reconnect_on_close: false,
    });
    this.subscription.subscribe((message) => {
      if (this.closed) return;
      this.positions = config.positions(message);
      this.stateAt = this.now();
      if (this.blocker()) this.deactivate();
    });
    this.commandSubscription = connectedTopic({
      ros, name: config.commandTopic || config.topic,
      messageType: "trajectory_msgs/msg/JointTrajectory",
      queue_length: 1, reconnect_on_close: false,
    });
    this.commandSubscription.subscribe((message) => {
      if (this.closed) return;
      const frame = message.header?.frame_id;
      if (frame === this.owner || config.ignoredCommandFrames?.includes(frame)) {
        this.knownCommandAt = this.now();
        return;
      }
      this.otherCommandAt = this.now();
      this.deactivate();
    });
    this.service = new ROSLIB.Service({
      ros, name: "/controller_manager/list_controllers", serviceType: "controller_manager_msgs/srv/ListControllers",
    });
  }

  blocker() {
    if (this.closed || !this.ros.isConnected) return "ROS disconnected";
    if (this.failure) return this.failure;
    if (this.now() - this.otherCommandAt < 1000) return `Another ${this.config.id} control source is active`;
    if (this.config.requireKnownCommand && this.now() - this.knownCommandAt > 1000) {
      return `Waiting for leader ${this.config.id} command stream`;
    }
    if (!this.positions || this.now() - this.stateAt > STATE_MAX_AGE_MS) return `Waiting for fresh ${this.config.id} joint states`;
    const outside = this.positions.findIndex((value, i) => value < this.config.limits[i][0] || value > this.config.limits[i][1]);
    if (outside >= 0) return `${this.config.axisLabels[outside]} outside safe range — recover the joint before teleop`;
    if (this.now() - this.controllerAt > CONTROLLER_MAX_AGE_MS) return `Waiting for ${this.config.id} controller status`;
    if (this.controllerState !== "active") return `${this.config.label} controller ${this.controllerState}`;
    return "";
  }

  pollController() {
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
        this.controllerState = response.controller?.find((item) => item.name === this.config.controller)?.state || "unavailable";
        this.controllerAt = this.now();
        if (this.blocker()) this.deactivate();
      }, () => {
        if (!valid()) return;
        this.pendingServiceId = null;
        this.controllerState = "unavailable";
        this.controllerAt = this.now();
        this.deactivate();
      });
    } catch (_error) {
      this.controllerState = "unavailable";
      this.controllerAt = this.now();
      this.deactivate();
    }
  }

  clearControllerRequest() {
    if (this.pendingServiceId) this.ros.removeAllListeners?.(this.pendingServiceId);
    this.pendingServiceId = null;
  }

  snapshot() {
    return { positions: this.positions, reason: this.blocker(), active: this.active, direction: this.direction };
  }

  activate() {
    if (this.blocker()) return false;
    this.active = true;
    return true;
  }

  publish(direction) {
    // Synchronous check: no asynchronous reconnect may release an old command.
    if (this.blocker()) return false;
    try {
      const message = this.config.trajectory(this.positions, direction);
      message.header = { stamp: { sec: 0, nanosec: 0 }, frame_id: this.owner };
      this.publisher.publish(new ROSLIB.Message(message));
      return true;
    } catch (_error) {
      this.failure = `${this.config.label} command failed — reconnect before retrying`;
      this.active = false;
      this.direction = null;
      return false;
    }
  }

  start(direction) {
    if (!this.active || !this.config.directions[direction] || this.blocker()) return false;
    this.direction = direction;
    this.lastCommand = this.now();
    return this.publish(direction);
  }

  stop() {
    const moving = this.direction;
    this.direction = null;
    if (moving) this.publish(null);
  }

  deactivate() {
    this.stop();
    this.active = false;
    // Release the publisher so the follower's startup gate is not blocked by
    // an idle, deactivated panel. Never queue cleanup waiting for reconnect.
    if (this.ros.isConnected) this.publisher.unadvertise();
  }

  tick() {
    if (this.closed) return;
    this.pollController();
    if (this.blocker()) this.deactivate();
    if (this.active && this.direction && this.now() - this.lastCommand >= 150) {
      this.lastCommand = this.now();
      this.publish(this.direction);
    }
  }

  dispose() {
    this.deactivate();
    this.closed = true;
    this.clearControllerRequest();
    this.subscription.unsubscribe();
    this.commandSubscription.unsubscribe();
    this.publisher.unadvertise();
  }
}
