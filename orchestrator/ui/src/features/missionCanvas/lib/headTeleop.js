// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
import { JointTeleopClient } from "./jointTeleop";

// SG2 follower launch remaps head_controller/joint_trajectory to this topic.
export const HEAD_TOPIC = "/leader/joystick_controller_left/joint_trajectory";
export const HEAD_COMMAND_TOPIC = "/leader/mission_canvas_head/joint_trajectory";
export const HEAD_JOINTS = ["head_joint1", "head_joint2"];
// ffw_description/urdf/follower/body/ffw_body.urdf.xacro (1022 / SG2).
export const HEAD_LIMITS = [[-0.2317, 0.83], [-0.35, 0.35]];
export const HEAD_DIRECTIONS = {
  up: [0, -1], down: [0, 1], left: [1, 1], right: [1, -1],
};
const STEP_RAD = 0.015;
const LIMIT_MARGIN_RAD = 0.01;

export function headPositions(message) {
  if (!Array.isArray(message?.name) || !Array.isArray(message?.position)) return null;
  const values = HEAD_JOINTS.map((joint) => {
    const index = message.name.indexOf(joint);
    return index >= 0 && message.name.lastIndexOf(joint) === index ? message.position[index] : NaN;
  });
  return values.every(Number.isFinite) ? values : null;
}

export function headTrajectory(positions, direction) {
  const target = [...positions];
  if (direction) {
    const [axis, sign] = HEAD_DIRECTIONS[direction];
    const [min, max] = HEAD_LIMITS[axis];
    // Never reverse direction when starting inside the margin at a limit.
    target[axis] = sign > 0
      ? Math.max(target[axis], Math.min(max - LIMIT_MARGIN_RAD, target[axis] + STEP_RAD))
      : Math.min(target[axis], Math.max(min + LIMIT_MARGIN_RAD, target[axis] - STEP_RAD));
  }
  return {
    joint_names: HEAD_JOINTS,
    points: [{ positions: target, velocities: [0, 0], time_from_start: { sec: 0, nanosec: 300000000 } }],
  };
}

export class HeadTeleopClient extends JointTeleopClient {
  constructor(ros, now) {
    super(ros, {
      id: "head", label: "Head", topic: HEAD_COMMAND_TOPIC,
      commandTopic: HEAD_TOPIC, ignoredCommandFrames: ["leader_joystick_controller"],
      requireKnownCommand: true,
      controller: "head_controller",
      limits: HEAD_LIMITS, directions: HEAD_DIRECTIONS, axisLabels: ["Pitch", "Yaw"],
      positions: headPositions, trajectory: headTrajectory,
    }, now);
  }
}
